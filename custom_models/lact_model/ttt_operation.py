import torch.nn.functional as F
import torch

# Debug mode flag - set to True to enable intermediate output logging
# Usage: import custom_models.lact_model.ttt_operation as ttt_op; ttt_op.DEBUG_MODE = True
DEBUG_MODE = False

def debug_print(*args, **kwargs):
    """Print only when DEBUG_MODE is True."""
    if DEBUG_MODE:
        print(*args, **kwargs)

def debug_tensor(name, t):
    """Print tensor stats when DEBUG_MODE is True."""
    if DEBUG_MODE and t is not None:
        print(f"  [DEBUG] {name}: shape={t.shape}, mean={t.float().mean():.6f}, "
              f"std={t.float().std():.6f}, min={t.float().min():.6f}, max={t.float().max():.6f}")

# @torch.compile()
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


# @torch.compile()
def l2_norm(x: torch.Tensor):
    """
    x: [b, l, d]
    """
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)  # norm will upcast to float32
    return ret.type(x_type)


# @torch.compile()
def zeropower_via_newtonschulz5(G):
    """
    This is an updated version of the zeropower_via_newtonschulz5 function in here:
    https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L26
    The code is modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49, which contains the original muon implementation.
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Args:
        G: [b, d, d']
    Returns:
        X: [b, d, d']
    FLOPS:  When d=d', Total FLOPS=30 * b * d^3
    """
    assert len(G.shape) == 3
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X



# @torch.compile()
def block_causal_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int=2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None, # [b, s, 1]
    return_final_state: bool = False,
    update_last_chunk: bool = False,  # NEW: if True, also update weights for the last/remainder chunk
    momentum_state: tuple = None,  # NEW: (dw0_momentum, dw1_momentum, dw2_momentum) for chunked prefill
):
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))
    
    About precision:
        w0, w1, w2 are mostly likely fp32. 
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.
    
    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon, ignore last chunk)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B
    Outputs:
        o: [b, l, dv]
        If return_final_state is True and momentum is not None:
            Returns: (output, (w0, w1, w2), (dw0_momentum, dw1_momentum, dw2_momentum))
        If return_final_state is True and momentum is None:
            Returns: (output, (w0, w1, w2))
        
    Args:
        update_last_chunk: If True, update weights even for the last (possibly partial) chunk.
                          This is needed for chunked prefill where each chunk should contribute
                          to weight updates for subsequent chunks. Default False preserves
                          original behavior where remainder tokens don't update weights.
        momentum_state: Optional tuple of (dw0_momentum, dw1_momentum, dw2_momentum) from 
                       previous chunked prefill call. Used to maintain momentum continuity
                       across chunked prefill calls.
    """
    
    # adding detach here sometimes improves stability.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    if momentum is not None:
        if momentum_state is not None:
            # Use provided momentum state from previous chunk
            dw0_momentum, dw1_momentum, dw2_momentum = momentum_state
        else:
            # Initialize momentum to zeros
            dw1_momentum = torch.zeros_like(w1)
            dw0_momentum = torch.zeros_like(w0)
            dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    
    # [block_causal_lact_swiglu] Determine how many full chunks to process in the loop
    # If update_last_chunk is True and seq_len is a multiple of chunk_size,
    # we need to include the last chunk in the loop
    if update_last_chunk and seq_len > 0 and seq_len % chunk_size == 0:
        loop_end = seq_len
    else:
        loop_end = seq_len - chunk_size if seq_len > chunk_size else 0
    
    debug_print(f"[block_causal_lact_swiglu] seq_len={seq_len}, chunk_size={chunk_size}, "
                f"update_last_chunk={update_last_chunk}, loop_end={loop_end}, "
                f"momentum_state={'provided' if momentum_state is not None else 'None'}")
    debug_print(f"  INPUT k: mean={k.float().mean():.6f}, max={k.float().abs().max():.6f}")
    debug_print(f"  INPUT v: mean={v.float().mean():.6f}, max={v.float().abs().max():.6f}")
    debug_print(f"  INPUT q: mean={q.float().mean():.6f}, max={q.float().abs().max():.6f}")
    debug_print(f"  INPUT lr0: mean={lr0.float().mean():.6f}, max={lr0.float().abs().max():.6f}")
    debug_print(f"  w0 initial: mean={w0.float().mean():.6f}, max={w0.float().abs().max():.6f}")
    debug_print(f"  w1 initial: mean={w1.float().mean():.6f}, max={w1.float().abs().max():.6f}")
    debug_print(f"  w2 initial: mean={w2.float().mean():.6f}, max={w2.float().abs().max():.6f}")
    
    chunk_idx = 0
    for i in range(0, loop_end, chunk_size):
        s_index = i
        e_index = s_index + chunk_size
        
        debug_print(f"  [Chunk {chunk_idx}] range=[{s_index}:{e_index}]")

        # [b, l, dk]
        ki = k[:, s_index:e_index, :]  # bf16
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]  # bf16
        # [b, dh, l]
        qi = q[:, :, s_index:e_index]
        # [b, l, d/1] fp32
        lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr2i = lr2[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr0i = lr0[:, s_index:e_index, :]  # [b, l, d/1] fp32

        # use previous w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))

        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
        
        # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
        dhidden = torch.bmm(w1.transpose(1, 2), vi)

        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
        # in bmm two mat is fp32, but the result is bf16.
        # it's better to cast the mat to bf16 before bmm.
        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        # it's better to cast the mat to bf16 before bmm.
        dw1 = torch.bmm(
            vi, (hidden.transpose(1, 2) * lr1i).type_as(vi)
        )  # [b, d, d]
        # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :] 
            m_i = m_i.mean(dim=1, keepdim=True)
            
            debug_print(f"    [BEFORE mom] dw0 max={dw0.float().abs().max():.6f}, dw0_momentum max={dw0_momentum.float().abs().max():.6f}")
            debug_print(f"    [BEFORE mom] m_i mean={m_i.float().mean():.6f}")

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2
            debug_print(f"    [AFTER mom] dw0_m max={dw0_momentum.float().abs().max():.6f}")


        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)
            # legacy code for different global lr for muon. Conclusion: 1.0 is good
            # if muon_w0_lr is not None:
            #     # lr is fp32 (after softplus)
            #     # in future version, we can cast it before input. TODO
            #     dw1 = (dw1 * muon_w1_lr).type_as(w1)
            #     dw0 = (dw0 * muon_w0_lr).type_as(w0)
            #     dw2 = (dw2 * muon_w2_lr).type_as(w2)

        w1 = w1 + dw1
        w0 = w0 + dw0
        w2 = w2 + dw2
    
        # Do channel-wise l2 norm.  conceptually like post-norm.
        w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm
        
        debug_print(f"    [Chunk {chunk_idx}] after update: w0 max={w0.float().abs().max():.6f}, w1 max={w1.float().abs().max():.6f}")
        chunk_idx += 1
        debug_tensor(f"  w2_iter{i//chunk_size}", w2)
        debug_tensor(f"  output_chunk", output[:, :, s_index:e_index])
        
    # Handle the last (possibly partial) chunk
    # Only process if there are remaining tokens after the loop
    s_index = e_index
    e_index = seq_len
    
    debug_print(f"  [Last chunk] s_index={s_index}, e_index={e_index}, update_last_chunk={update_last_chunk}")
    
    if s_index < e_index:
        qi = q[:, :, s_index:e_index]
        # use the last w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)
        debug_print(f"    [Last chunk] output computed using w0 max={w0.float().abs().max():.6f}")
        
        # Optionally update weights for the last chunk (needed for chunked prefill)
        if update_last_chunk and return_final_state:
            debug_print(f"    [Last chunk] UPDATING weights (update_last_chunk=True)")
            ki = k[:, s_index:e_index, :]
            vi = v[:, :, s_index:e_index]
            lr0i = lr0[:, s_index:e_index, :]
            lr1i = lr1[:, s_index:e_index, :]
            lr2i = lr2[:, s_index:e_index, :]
            
            gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
            hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
            
            dhidden = torch.bmm(w1.transpose(1, 2), vi)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)
            
            dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))
            dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
            dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))
            
            if momentum is not None:
                m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True)
                dw0 = dw0 + dw0_momentum * m_i
                dw1 = dw1 + dw1_momentum * m_i
                dw2 = dw2 + dw2_momentum * m_i
                debug_print(f"    [Last chunk] momentum applied, m_i mean={m_i.float().mean():.6f}")
            
            if use_muon:
                dw1 = zeropower_via_newtonschulz5(dw1)
                dw0 = zeropower_via_newtonschulz5(dw0)
                dw2 = zeropower_via_newtonschulz5(dw2)
            
            w1 = w1 + dw1
            w0 = w0 + dw0
            w2 = w2 + dw2
            
            w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm
            debug_print(f"    [Last chunk] after update: w0 max={w0.float().abs().max():.6f}")
        else:
            debug_print(f"    [Last chunk] NOT updating weights (update_last_chunk={update_last_chunk})")
    else:
        debug_print(f"  [Last chunk] NO remaining tokens (s_index={s_index} >= e_index={e_index})")

    debug_print(f"  FINAL: w0 max={w0.float().abs().max():.6f}, w1 max={w1.float().abs().max():.6f}")
    debug_tensor("w0_final", w0)
    debug_tensor("w1_final", w1)
    debug_tensor("w2_final", w2)
    debug_tensor("output_final", output)
    
    if return_final_state:
        if momentum is not None:
            # Return momentum state for chunked prefill continuity
            return output.transpose(1, 2), (w0, w1, w2), (dw0_momentum, dw1_momentum, dw2_momentum)
        return output.transpose(1, 2), (w0, w1, w2)
    return output.transpose(1, 2)


# @torch.compile()
def prenorm_block_causal_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int=2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None, # [b, s, 1]
    return_final_state: bool = False,
    update_last_chunk: bool = False,  # NEW: consistent with block_causal_lact_swiglu
):
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))
    
    About precision:
        w0, w1, w2 are mostly likely fp32. 
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.
    
    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon, ignore last chunk)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B
    Outputs:
        o: [b, l, dv]
        
    Args:
        update_last_chunk: If True, update weights even for the last (possibly partial) chunk.
                          Note: prenorm version already updates on last chunk when return_final_state=True,
                          this flag makes it consistent with block_causal_lact_swiglu.
    """
    
    # adding detach here sometimes improves stability.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    w0_main, w1_main, w2_main = w0, w1, w2

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    
    # [prenorm_block_causal_lact_swiglu] Determine how many full chunks to process in the loop
    # If update_last_chunk is True and seq_len is a multiple of chunk_size,
    # we need to include the last chunk in the loop
    if update_last_chunk and seq_len > 0 and seq_len % chunk_size == 0:
        loop_end = seq_len
    else:
        loop_end = seq_len - chunk_size if seq_len > chunk_size else 0
    
    debug_print(f"[prenorm_block_causal_lact_swiglu] seq_len={seq_len}, chunk_size={chunk_size}, "
                f"update_last_chunk={update_last_chunk}, loop_end={loop_end}")
    debug_tensor("prenorm_w0_initial", w0)
    
    for i in range(0, loop_end, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        # [b, l, dk]
        ki = k[:, s_index:e_index, :]  # bf16
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]  # bf16
        # [b, dh, l]
        qi = q[:, :, s_index:e_index]
        # [b, l, d/1] fp32
        lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr2i = lr2[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr0i = lr0[:, s_index:e_index, :]  # [b, l, d/1] fp32

        # use previous w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))

        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
        
        # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
        dhidden = torch.bmm(w1.transpose(1, 2), vi)

        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
        # in bmm two mat is fp32, but the result is bf16.
        # it's better to cast the mat to bf16 before bmm.
        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        # it's better to cast the mat to bf16 before bmm.
        dw1 = torch.bmm(
            vi, (hidden.transpose(1, 2) * lr1i).type_as(vi)
        )  # [b, d, d]
        # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :] 
            m_i = m_i.mean(dim=1, keepdim=True)

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2


        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)
            # legacy code for different global lr for muon. Conclusion: 1.0 is good
            # if muon_w0_lr is not None:
            #     # lr is fp32 (after softplus)
            #     # in future version, we can cast it before input. TODO
            #     dw1 = (dw1 * muon_w1_lr).type_as(w1)
            #     dw0 = (dw0 * muon_w0_lr).type_as(w0)
            #     dw2 = (dw2 * muon_w2_lr).type_as(w2)

        w1_main = w1_main + dw1
        w0_main = w0_main + dw0
        w2_main = w2_main + dw2
    
        # Do channel-wise l2 norm.  conceptually like post-norm.
        w0 = w0_main / (w0_main.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2_main / (w2_main.norm(dim=2, keepdim=True) + 1e-5) * w2_norm
        
        # Update local vars for next iteration
        w0, w1, w2 = w0_main, w1_main, w2_main

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    if s_index < e_index:
        qi = q[:, :, s_index:e_index]
        # use the last w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        if return_final_state:
            ki = k[:, s_index:e_index, :]
            vi = v[:, :, s_index:e_index]
            lr1i = lr1[:, s_index:e_index, :]
            lr2i = lr2[:, s_index:e_index, :]
            lr0i = lr0[:, s_index:e_index, :]

            gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
            hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))

            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
            
            dhidden = torch.bmm(w1.transpose(1, 2), vi)

            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            dw1 = torch.bmm(
                vi, (hidden.transpose(1, 2) * lr1i).type_as(vi)
            )
            dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
            dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

            if momentum is not None:
                m_i = momentum[:, s_index:e_index, :] 
                m_i = m_i.mean(dim=1, keepdim=True)

                dw0 = dw0 + dw0_momentum * m_i
                dw1 = dw1 + dw1_momentum * m_i
                dw2 = dw2 + dw2_momentum * m_i
                dw0_momentum = dw0
                dw1_momentum = dw1
                dw2_momentum = dw2

            if use_muon:
                dw1 = zeropower_via_newtonschulz5(dw1)
                dw0 = zeropower_via_newtonschulz5(dw0)
                dw2 = zeropower_via_newtonschulz5(dw2)

            w1_main = w1_main + dw1
            w0_main = w0_main + dw0
            w2_main = w2_main + dw2
        
            w0 = w0_main / (w0_main.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            w1 = w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            w2 = w2_main / (w2_main.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    if return_final_state:
        return output.transpose(1, 2), (w0, w1, w2)
    return output.transpose(1, 2)


# === New (MFU-optimized): Vectorized fine-grained CSDM for LaCT + SwiGLU ===
# Computes all slice matmuls in parallel via batched bmm to improve MFU and reduce Python overhead.
# @torch.compile()
def block_causal_lact_swiglu_csdm_vec_kj(
    w0: torch.Tensor,   # [B, dh, dk]
    w1: torch.Tensor,   # [B, dv, dh]
    w2: torch.Tensor,   # [B, dh, dk]
    q: torch.Tensor,    # [B, l, dk]
    k: torch.Tensor,    # [B, l, dk]
    v: torch.Tensor,    # [B, l, dv]
    lr0: torch.Tensor,  # [B, l, 1] for w0
    lr1: torch.Tensor,  # [B, l, 1] for w1
    lr2: torch.Tensor,  # [B, l, 1] for w2
    gates: torch.Tensor,        # [B, l, E]
    slice_starts: torch.Tensor, # [E]
    slice_sizes: torch.Tensor,  # [E], sum==dh
    chunk_size: int = 2048,
    use_muon: bool = False,
    momentum: torch.Tensor = None,
):
    B, dh, dk = w0.shape
    _, dv, _ = w1.shape
    E = slice_starts.numel()

    # Guard: drop zero-width slices
    nz_mask = slice_sizes > 0
    if E == 0 or nz_mask.sum() == 0:
        return v.new_zeros(v.shape)
    if nz_mask.sum().item() < E:
        idx_nz = nz_mask.nonzero(as_tuple=False).squeeze(-1)
        slice_starts = slice_starts.index_select(0, idx_nz)
        slice_sizes  = slice_sizes.index_select(0, idx_nz)
        E = slice_starts.numel()

    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    qT = q.transpose(1, 2)
    vT = v.transpose(1, 2)
    out = torch.zeros_like(vT)
    
    seq_len = k.shape[1]
    max_sz = int(slice_sizes.max().item())
    device = w0.device

    # --- FIX: Initialize momentum buffers with PACKED shape ---
    if momentum is not None:
        dw0_momentum = torch.zeros(B, E, max_sz, dk, device=device, dtype=w0.dtype)
        dw1_momentum = torch.zeros(B, E, dv, max_sz, device=device, dtype=w0.dtype)
        dw2_momentum = torch.zeros(B, E, max_sz, dk, device=device, dtype=w0.dtype)

    # Pre-calculate indices and mask for vectorized operations
    slice_starts = slice_starts.to(device)
    slice_sizes = slice_sizes.to(device)
    base_indices = torch.arange(max_sz, device=device)
    indices_pack = slice_starts.unsqueeze(1) + base_indices.unsqueeze(0)
    mask_pack = base_indices.unsqueeze(0) < slice_sizes.unsqueeze(1)
    indices_flat = indices_pack.view(-1)

    for s_index in range(0, max(seq_len - chunk_size, 0), chunk_size):
        e_index = s_index + chunk_size

        ki = k[:, s_index:e_index, :]
        vi = vT[:, :, s_index:e_index]
        qi = qT[:, :, s_index:e_index]
        gi = gates[:, s_index:e_index, :]
        if gi.shape[-1] != E:
            gi = gi[:, :, :E]

        lr0i = lr0[:, s_index:e_index, :]
        lr1i = lr1[:, s_index:e_index, :]
        lr2i = lr2[:, s_index:e_index, :]
        
        # --- Apply Path ---
        gate_before_q = torch.bmm(w0, qi)
        hid_before_q  = torch.bmm(w2, qi)
        hidden_q = F.silu(gate_before_q, inplace=False) * hid_before_q

        W1p = (w1[:, :, indices_pack] * mask_pack).permute(0, 2, 1, 3)
        Hpack_q = hidden_q[:, indices_pack, :] * mask_pack.unsqueeze(-1)
        
        W1_be = W1p.reshape(B*E, dv, max_sz)
        H_be_q  = Hpack_q.reshape(B*E, max_sz, -1)
        y_be  = torch.bmm(W1_be, H_be_q)
        y_all = y_be.view(B, E, dv, -1)
        y = (y_all * gi.transpose(1, 2).unsqueeze(2)).sum(dim=1)
        out[:, :, s_index:e_index] = y

        # --- Update Path ---
        gate_before_k = torch.bmm(w0, ki.transpose(1,2))
        hid_before_k = torch.bmm(w2, ki.transpose(1,2))
        sw_k = F.silu(gate_before_k, inplace=False)
        hidden_k = sw_k * hid_before_k
        
        Hpack_k = hidden_k[:, indices_pack, :] * mask_pack.unsqueeze(-1)
        HBpack_k = hid_before_k[:, indices_pack, :] * mask_pack.unsqueeze(-1)
        GBpack_k = gate_before_k[:, indices_pack, :] * mask_pack.unsqueeze(-1)
        swpack_k = sw_k[:, indices_pack, :] * mask_pack.unsqueeze(-1)
        
        V_be = vi.unsqueeze(1).expand(B, E, dv, -1).reshape(B*E, dv, -1)
        dH_be = torch.bmm(W1_be.transpose(1,2), V_be)
        dH = dH_be.view(B, E, max_sz, -1)

        dgate_e = dH * HBpack_k
        dgate_bef = silu_backprop(dgate_e, GBpack_k)

        scaled_lr0 = (lr0i * gi).permute(0, 2, 1).unsqueeze(-1)
        scaled_lr1 = (lr1i * gi).permute(0, 2, 1).unsqueeze(-1)
        scaled_lr2 = (lr2i * gi).permute(0, 2, 1).unsqueeze(-1)

        dw1_all = torch.bmm(V_be, (Hpack_k.permute(0,1,3,2) * scaled_lr1).reshape(B*E, -1, max_sz).type_as(V_be)).view(B, E, dv, max_sz)
        dw0_all = torch.bmm(dgate_bef.reshape(B*E, max_sz, -1).type_as(ki), (ki.unsqueeze(1) * scaled_lr0).reshape(B*E, -1, dk)).view(B, E, max_sz, dk)
        dw2_all = torch.bmm((dH * swpack_k).reshape(B*E, max_sz, -1).type_as(ki), (ki.unsqueeze(1) * scaled_lr2).reshape(B*E, -1, dk)).view(B, E, max_sz, dk)

        if use_muon:
            dw1_all = zeropower_via_newtonschulz5(dw1_all.reshape(B*E, dv, max_sz)).view_as(dw1_all)
            dw0_all = zeropower_via_newtonschulz5(dw0_all.reshape(B*E, max_sz, dk)).view_as(dw0_all)
            dw2_all = zeropower_via_newtonschulz5(dw2_all.reshape(B*E, max_sz, dk)).view_as(dw2_all)

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True).view(B, 1, 1, 1)
            dw0_momentum = dw0_momentum * m_i + dw0_all
            dw1_momentum = dw1_momentum * m_i + dw1_all
            dw2_momentum = dw2_momentum * m_i + dw2_all
            dw0_all, dw1_all, dw2_all = dw0_momentum, dw1_momentum, dw2_momentum
        
        dw1_all_flat = dw1_all.permute(0, 2, 1, 3).reshape(B, dv, E*max_sz)
        dw0_all_flat = dw0_all.permute(0, 3, 1, 2).reshape(B, dk, E*max_sz)
        dw2_all_flat = dw2_all.permute(0, 3, 1, 2).reshape(B, dk, E*max_sz)
        
        dw0 = torch.zeros_like(w0).permute(0, 2, 1).index_add_(2, indices_flat, dw0_all_flat.to(w0.dtype)).permute(0, 2, 1)
        dw1 = torch.zeros_like(w1).index_add_(2, indices_flat, dw1_all_flat.to(w1.dtype))
        dw2 = torch.zeros_like(w2).permute(0, 2, 1).index_add_(2, indices_flat, dw2_all_flat.to(w2.dtype)).permute(0, 2, 1)
        
        w1 = w1 + dw1
        w0 = w0 + dw0
        w2 = w2 + dw2
        w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    s_index = e_index
    if s_index < seq_len:
        qi = qT[:, :, s_index:]
        gi = gates[:, s_index:, :]
        if gi.shape[-1] != E:
            gi = gi[:, :, :E]

        hidden = F.silu(torch.bmm(w0, qi), inplace=False) * torch.bmm(w2, qi)
        W1p = (w1[:, :, indices_pack] * mask_pack).permute(0, 2, 1, 3)
        Hpack = hidden[:, indices_pack, :] * mask_pack.unsqueeze(-1)
        
        y_be  = torch.bmm(W1p.reshape(B*E, dv, max_sz), Hpack.reshape(B*E, max_sz, -1))
        y_all = y_be.view(B, E, dv, -1)
        y = (y_all * gi.transpose(1, 2).unsqueeze(2)).sum(dim=1)
        out[:, :, s_index:] = y

    return out.transpose(1, 2)


# === New (MFU-optimized): Vectorized fine-grained CSDM for LaCT + SwiGLU ===
# Computes all slice matmuls in parallel via batched bmm to improve MFU and reduce Python overhead.
# @torch.compile()
def block_causal_lact_swiglu_csdm_vec(
    w0: torch.Tensor,   # [B, dh, dk]
    w1: torch.Tensor,   # [B, dv, dh]
    w2: torch.Tensor,   # [B, dh, dk]
    q: torch.Tensor,    # [B, l, dk]
    k: torch.Tensor,    # [B, l, dk]
    v: torch.Tensor,    # [B, l, dv]
    lr0: torch.Tensor,  # [B, l, 1] for w0
    lr1: torch.Tensor,  # [B, l, 1] for w1
    lr2: torch.Tensor,  # [B, l, 1] for w2
    gates: torch.Tensor,        # [B, l, E]
    slice_starts: torch.Tensor, # [E]
    slice_sizes: torch.Tensor,  # [E], sum==dh
    chunk_size: int = 2048,
    use_muon: bool = False,
    momentum: torch.Tensor = None,
):
    B, dh, dk = w0.shape
    _, dv, _ = w1.shape
    E = slice_starts.numel()

    # Guard: drop zero-width slices (can happen when num_slices > dh).
    # We only keep slices with size > 0 and work on them; zero-width slices contribute nothing.
    nz_mask = slice_sizes > 0
    if nz_mask.dim() == 0:
        nz_mask = nz_mask.unsqueeze(0)
    if nz_mask.sum().item() < E:
        idx_nz = nz_mask.nonzero(as_tuple=False).squeeze(-1)
        slice_starts = slice_starts.index_select(0, idx_nz)
        slice_sizes  = slice_sizes.index_select(0, idx_nz)
        E = slice_starts.numel()

    # Precompute norms for post-update renorm (channel-wise)
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    qT = q.transpose(1, 2)  # [B, dk, l]
    vT = v.transpose(1, 2)  # [B, dv, l]
    out = torch.zeros_like(vT)  # [B, dv, l]

    seq_len = k.shape[1]
    e_index = 0

    # Concatenate along a new slice dim: [B, E, dv, sz_e] (ragged sizes -> pack separately per ei)
    # Because sizes vary by at most 1, we still need per-ei packing. We'll process with a loop only for assignment,
    # while matmuls are done in a single batched bmm over B*E by padding to max_sz and masking.
    max_sz = int(slice_sizes.max().item())

    # --- OPTIMIZATION: Pre-calculate indices and mask for vectorized packing/scattering ---
    device = w0.device
    slice_starts = slice_starts.to(device)
    slice_sizes = slice_sizes.to(device)
    base_indices = torch.arange(max_sz, device=device)
    
    # Indices for gathering data from dense tensors into packed format
    indices_pack = slice_starts.unsqueeze(1) + base_indices.unsqueeze(0) # [E, max_sz]
    
    # Mask for zeroing out padded values in packed tensors
    mask_pack = base_indices.unsqueeze(0) < slice_sizes.unsqueeze(1) # [E, max_sz]
    indices_flat = indices_pack.view(-1)                             # [E * max_sz]

    # We'll reuse the same packing approach for dw1 and for activations.
    for s_index in range(0, max(seq_len - chunk_size, 0), chunk_size):
        e_index = s_index + chunk_size

        ki = k[:, s_index:e_index, :]         # [B, l, dk]
        vi = vT[:, :, s_index:e_index]        # [B, dv, l]
        qi = qT[:, :, s_index:e_index]        # [B, dk, l]
        gi = gates[:, s_index:e_index, :]     # [B, l, E]
        # Align gates with surviving slices if any zero-width were dropped.
        if gi.shape[-1] != E:
            gi = gi[:, :, :E]

        # Slice learning rates to current chunk
        lr0i = lr0[:, s_index:e_index, :]     # [B, l, 1]
        lr1i = lr1[:, s_index:e_index, :]     # [B, l, 1]
        lr2i = lr2[:, s_index:e_index, :]     # [B, l, 1]

        # Apply once for full dh (no Python loop over slices)
        gate_before = torch.bmm(w0, qi)       # [B, dh, l]
        hid_before  = torch.bmm(w2, qi)       # [B, dh, l]
        sw = F.silu(gate_before, inplace=False)
        hidden_q = sw * hid_before            # [B, dh, l]

        # w1 is [B, dv, dh], indices_pack is [E, max_sz] -> W1p is [B, dv, E, max_sz]
        W1p = w1[:, :, indices_pack] * mask_pack
        W1p = W1p.permute(0, 2, 1, 3) # -> [B, E, dv, max_sz]
    
        # hidden_q is [B, dh, l], indices_pack is [E, max_sz] -> Hpack_q is [B, E, max_sz, l]
        Hpack_q = hidden_q[:, indices_pack, :] * mask_pack.unsqueeze(-1)

        W1_be = W1p.reshape(B*E, dv, max_sz)
        H_be_q  = Hpack_q.reshape(B*E, max_sz, -1)
        y_be  = torch.bmm(W1_be, H_be_q)
        y_all = y_be.view(B, E, dv, -1)
        gi_t = gi.transpose(1, 2).unsqueeze(2)
        y = (y_all * gi_t).sum(dim=1)
        out[:, :, s_index:e_index] = y

        # === Update path (uses key `ki` and value `vi`) ===
        # We must recompute them using `ki` for the update, matching the baseline.
        gate_before_k = torch.bmm(w0, ki.transpose(1,2)) # [B, dh, l]
        hid_before_k = torch.bmm(w2, ki.transpose(1,2))  # [B, dh, l]
        sw_k = F.silu(gate_before_k, inplace=False)
        hidden_k = sw_k * hid_before_k                   # [B, dh, l]

        # === Updates (vectorized) ===
        # dhidden_e = (W1_e^T @ vi) for all slices
        Hpack_k = hidden_k[:, indices_pack, :] * mask_pack.unsqueeze(-1)
        HBpack_k = hid_before_k[:, indices_pack, :] * mask_pack.unsqueeze(-1)
        GBpack_k = gate_before_k[:, indices_pack, :] * mask_pack.unsqueeze(-1)
        swpack_k = sw_k[:, indices_pack, :] * mask_pack.unsqueeze(-1)
        
        V_be = vi.unsqueeze(1).expand(B, E, dv, -1).reshape(B*E, dv, -1)
        dH_be = torch.bmm(W1_be.transpose(1,2), V_be)
        dH = dH_be.view(B, E, max_sz, -1)

        dgate_e = dH * HBpack_k
        dgate_bef = silu_backprop(dgate_e, GBpack_k)

        # LR * gates per slice
        scaled_lr0 = (lr0i * gi).permute(0, 2, 1).unsqueeze(-1)
        scaled_lr1 = (lr1i * gi).permute(0, 2, 1).unsqueeze(-1)
        scaled_lr2 = (lr2i * gi).permute(0, 2, 1).unsqueeze(-1)

        # Compute dw1
        H_t_k = Hpack_k.permute(0,1,3,2)
        H_t_be_k = (H_t_k * scaled_lr1).reshape(B*E, -1, max_sz)
        dw1_be = torch.bmm(V_be, H_t_be_k.type_as(V_be))
        dw1_all = dw1_be.view(B, E, dv, max_sz)

        # Compute dw0
        Ki_scaled = (ki.unsqueeze(1) * scaled_lr0)
        Ki_be = Ki_scaled.reshape(B*E, -1, dk)
        dgate_bef_reshaped = dgate_bef.reshape(B*E, max_sz, -1)
        dw0_be = torch.bmm(dgate_bef_reshaped.type_as(Ki_be), Ki_be)
        dw0_all = dw0_be.view(B, E, max_sz, dk)

        # Compute dw2
        Ki_scaled2 = (ki.unsqueeze(1) * scaled_lr2)
        Ki_be2 = Ki_scaled2.reshape(B*E, -1, dk)
        dgrad_w2 = (dH * swpack_k).reshape(B*E, max_sz, -1)
        dw2_be = torch.bmm(dgrad_w2.type_as(Ki_be2), Ki_be2)
        dw2_all = dw2_be.view(B, E, max_sz, dk)

        # Optional muon orthogonalization
        if use_muon:
            dw1_all = zeropower_via_newtonschulz5(dw1_all.reshape(B*E, dv, max_sz)).view_as(dw1_all)
            dw0_all = zeropower_via_newtonschulz5(dw0_all.reshape(B*E, max_sz, dk)).view_as(dw0_all)
            dw2_all = zeropower_via_newtonschulz5(dw2_all.reshape(B*E, max_sz, dk)).view_as(dw2_all)
            
        dw1_all_flat = dw1_all.permute(0, 2, 1, 3).reshape(B, dv, E*max_sz)
        dw0_all_flat = dw0_all.permute(0, 3, 1, 2).reshape(B, dk, E*max_sz)
        dw2_all_flat = dw2_all.permute(0, 3, 1, 2).reshape(B, dk, E*max_sz)
        
        dw0 = torch.zeros_like(w0).permute(0, 2, 1).index_add_(2, indices_flat, dw0_all_flat.to(w0.dtype)).permute(0, 2, 1)
        dw1 = torch.zeros_like(w1).index_add_(2, indices_flat, dw1_all_flat.to(w1.dtype))
        dw2 = torch.zeros_like(w2).permute(0, 2, 1).index_add_(2, indices_flat, dw2_all_flat.to(w2.dtype)).permute(0, 2, 1)
        
        # Momentum (optional)
        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True)  # [B,1,1]
            # 1. Update the momentum state (velocity) with the current gradient
            # velocity = momentum * old_velocity + new_gradient
            dw0_momentum = dw0_momentum * m_i + dw0
            dw1_momentum = dw1_momentum * m_i + dw1
            dw2_momentum = dw2_momentum * m_i + dw2

             # 2. Use the updated momentum state as the final gradient for the weight update
            dw0, dw1, dw2 = dw0_momentum, dw1_momentum, dw2_momentum

        # Update + post-norm
        w1 = w1 + dw1
        w0 = w0 + dw0
        w2 = w2 + dw2
        w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    # Tail chunk (apply only)
    s_index = e_index
    e_index = seq_len
    if s_index < e_index:
        qi = qT[:, :, s_index:e_index]                # [B, dk, l]
        gi = gates[:, s_index:e_index, :]             # [B, l, E]
        if gi.shape[-1] != E:
            gi = gi[:, :, :E]

        gate_before = torch.bmm(w0, qi)
        hid_before  = torch.bmm(w2, qi)
        sw = F.silu(gate_before, inplace=False)
        hidden = sw * hid_before                      # [B, dh, l]

        W1p = (w1[:, :, indices_pack] * mask_pack).permute(0, 2, 1, 3) # [B, E, dv, max_sz]
        Hpack = hidden[:, indices_pack, :] * mask_pack.unsqueeze(-1) # [B, E, max_sz, l_tail]

        y_be  = torch.bmm(W1p.reshape(B*E, dv, max_sz), Hpack.reshape(B*E, max_sz, -1))
        y_all = y_be.view(B, E, dv, -1)
        gi_t = gi.transpose(1, 2).unsqueeze(2)
        y = (y_all * gi_t).sum(dim=1)
        out[:, :, s_index:e_index] = y

    return out.transpose(1, 2)

def ttt_apply_weights_only(q, w0, w1, w2):
    """
    O(1) Step: Applies current W to input q without any loop.
    q: [B, 1, dk]
    w0, w2: [B, dh, dk]
    w1: [B, dv, dh]
    """
    q_t = q.transpose(1, 2) # [B, dk, 1]
    h = torch.bmm(w2, q_t)
    gate = F.silu(torch.bmm(w0, q_t))
    out = torch.bmm(w1, gate * h)
    return out.transpose(1, 2) # [B, 1, dv]

def ttt_update_step_isolated(k, v, w0, w1, w2, lr0, lr1, lr2, w0_init_norm, w1_init_norm, w2_init_norm, use_muon=False):
    """
    O(1) Step: Takes exactly one chunk, runs update, returns NEW W.
    k: [B, chunk, dk]
    v: [B, chunk, dv]
    lr: [B, chunk, 1]
    """
    ki = k
    vi = v.transpose(1, 2) # [B, dv, chunk]
    
    gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
    hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))

    hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
    
    dhidden = torch.bmm(w1.transpose(1, 2), vi)

    dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

    dgate = dhidden * hidden_before_mul
    dgate_before_act = silu_backprop(dgate, gate_before_act)

    dw1 = torch.bmm(
        vi, (hidden.transpose(1, 2) * lr1).type_as(vi)
    )
    dw0 = torch.bmm(dgate_before_act, (ki * lr0).type_as(dgate_before_act))
    dw2 = torch.bmm(dhidden_before_mul, (ki * lr2).type_as(dhidden_before_mul))

    if use_muon:
        dw1 = zeropower_via_newtonschulz5(dw1)
        dw0 = zeropower_via_newtonschulz5(dw0)
        dw2 = zeropower_via_newtonschulz5(dw2)

    w1 = w1 + dw1
    w0 = w0 + dw0
    w2 = w2 + dw2
    
    w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_init_norm
    w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_init_norm
    w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_init_norm
    
    return w0, w1, w2
