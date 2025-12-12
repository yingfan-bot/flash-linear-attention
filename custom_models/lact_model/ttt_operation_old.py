import torch.nn.functional as F
import torch

@torch.compile()
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


@torch.compile()
def l2_norm(x: torch.Tensor):
    """
    x: [b, l, d]
    """
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)  # norm will upcast to float32
    return ret.type(x_type)


@torch.compile()
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



@torch.compile()
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
    """
    
    # adding detach here sometimes improves stability.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
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

        w1 = w1 + dw1
        w0 = w0 + dw0
        w2 = w2 + dw2
    
        # Do channel-wise l2 norm.  conceptually like post-norm.
        w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm
        
    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    qi = q[:, :, s_index:e_index]
    # use the last w0 and w1 to get the final output
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2)


@torch.compile()
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
    for i in range(0, seq_len - chunk_size, chunk_size):
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
        
    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    qi = q[:, :, s_index:e_index]
    # use the last w0 and w1 to get the final output
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2)


# === New: Fine-grained (parameter-budget-preserving) dense mixture for LaCT + SwiGLU ===
# This variant keeps the fast-weight parameter count unchanged by slicing the hidden dim (dh)
# into E channel blocks. All slices are active per token/head; router is non-competitive
# (activate then normalize). Updates are applied per slice, weighted by token gates.
@torch.compile()
def block_causal_lact_swiglu_csdm(
    w0: torch.Tensor,   # [B, dh, dk]
    w1: torch.Tensor,   # [B, dv, dh]
    w2: torch.Tensor,   # [B, dh, dk]
    q: torch.Tensor,    # [B, l, dk]
    k: torch.Tensor,    # [B, l, dk]
    v: torch.Tensor,    # [B, l, dv]
    lr0: torch.Tensor,  # [B, l, 1] for w0
    lr1: torch.Tensor,  # [B, l, 1] for w1
    lr2: torch.Tensor,  # [B, l, 1] for w2
    gates: torch.Tensor,        # [B, l, E]  per-token slice weights (already activated & normalized)
    slice_starts: torch.Tensor, # [E] int32  start index in dh for each slice
    slice_sizes: torch.Tensor,  # [E] int32  width in dh for each slice (sum == dh)
    chunk_size: int = 2048,
    use_muon: bool = False,
    momentum: torch.Tensor = None,  # [B, l, 1] or None
):
    B, dh, dk = w0.shape
    _, dv, _ = w1.shape

    # Precompute channel norms for post-update renorm (stability)
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    if momentum is not None:
        dw1_m = torch.zeros_like(w1)
        dw0_m = torch.zeros_like(w0)
        dw2_m = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [B, dk, l]
    v = v.transpose(1, 2)  # [B, dv, l]
    output = torch.zeros_like(v)

    seq_len = k.shape[1]
    e_index = 0

    # === Block-causal scan ===
    for s_index in range(0, max(seq_len - chunk_size, 0), chunk_size):
        e_index = s_index + chunk_size

        ki = k[:, s_index:e_index, :]      # [B, l, dk]
        vi = v[:, :, s_index:e_index]      # [B, dv, l]
        qi = q[:, :, s_index:e_index]      # [B, dk, l]
        gi = gates[:, s_index:e_index, :]  # [B, l, E]
        # Slice learning rates to the same chunk length to avoid broadcast mismatches
        lr0i = lr0[:, s_index:e_index, :]  # [B, l, 1]
        lr1i = lr1[:, s_index:e_index, :]  # [B, l, 1]
        lr2i = lr2[:, s_index:e_index, :]  # [B, l, 1]

        # Apply: compute once for full dh, then aggregate over slices
        gate_before = torch.bmm(w0, qi)      # [B, dh, l]
        hid_before  = torch.bmm(w2, qi)      # [B, dh, l]
        sw = F.silu(gate_before, inplace=False)
        hidden = sw * hid_before             # [B, dh, l]

        y_acc = torch.zeros_like(vi)         # [B, dv, l]

        # Gradient accumulators
        dw1 = torch.zeros_like(w1)
        dw0 = torch.zeros_like(w0)
        dw2 = torch.zeros_like(w2)

        # Per-slice apply + update
        for ei in range(slice_starts.numel()):
            st = int(slice_starts[ei].item())
            sz = int(slice_sizes[ei].item())
            ed = st + sz

            g_e = gi[:, :, ei:ei+1]                 # [B, l, 1]
            h_e = hidden[:, st:ed, :]               # [B, sz, l]

            # apply: y_e = W1[:, :, st:ed] @ h_e
            y_e = torch.bmm(w1[:, :, st:ed], h_e)   # [B, dv, l]
            y_acc = y_acc + y_e * g_e.transpose(1, 2)

            # update: compute partial grads for each slice
            dhidden_e = torch.bmm(w1[:, :, st:ed].transpose(1, 2), vi)  # [B, sz, l]
            dgate_e   = dhidden_e * hid_before[:, st:ed, :]
            dgate_bef = silu_backprop(dgate_e, gate_before[:, st:ed, :])

            # Per-token LR scaled by gates
            lr1e = lr1i * g_e  # for w1
            lr0e = lr0i * g_e  # for w0
            lr2e = lr2i * g_e  # for w2

            # dw1: [B, dv, l] @ [B, l, sz] -> [B, dv, sz]
            dw1[:, :, st:ed] = dw1[:, :, st:ed] + torch.bmm(
                vi, (h_e.transpose(1, 2) * lr1e).type_as(vi)
            )
            # dw0: [B, sz, l] @ [B, l, dk] -> [B, sz, dk]
            dw0[:, st:ed, :] = dw0[:, st:ed, :] + torch.bmm(
                dgate_bef, (ki * lr0e).type_as(dgate_bef)
            )
            # dw2: [B, sz, l] @ [B, l, dk] -> [B, sz, dk]
            dw2[:, st:ed, :] = dw2[:, st:ed, :] + torch.bmm(
                dhidden_e * sw[:, st:ed, :], (ki * lr2e).type_as(dhidden_e)
            )

        output[:, :, s_index:e_index] = y_acc

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True)  # [B,1,1]
            dw0 = dw0 + dw0_m * m_i
            dw1 = dw1 + dw1_m * m_i
            dw2 = dw2 + dw2_m * m_i
            dw0_m = dw0
            dw1_m = dw1
            dw2_m = dw2

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)

        # Parameter update + post-norm (channel-wise)
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
        qi = q[:, :, s_index:e_index]
        gi = gates[:, s_index:e_index, :]  # [B, l, E]

        gate_before = torch.bmm(w0, qi)
        hid_before  = torch.bmm(w2, qi)
        sw = F.silu(gate_before, inplace=False)
        hidden = sw * hid_before

        y_acc = torch.zeros(B, dv, e_index - s_index, device=w0.device, dtype=w0.dtype)
        for ei in range(slice_starts.numel()):
            st = int(slice_starts[ei].item())
            sz = int(slice_sizes[ei].item())
            ed = st + sz
            g_e = gi[:, :, ei:ei+1]               # [B, l, 1]
            h_e = hidden[:, st:ed, :]             # [B, sz, l]
            y_e = torch.bmm(w1[:, :, st:ed], h_e) # [B, dv, l]
            y_acc = y_acc + y_e * g_e.transpose(1, 2)
        output[:, :, s_index:e_index] = y_acc

    return output.transpose(1, 2)



# === New (MFU-optimized): Vectorized fine-grained CSDM for LaCT + SwiGLU ===
# Computes all slice matmuls in parallel via batched bmm to improve MFU and reduce Python overhead.
@torch.compile()
def block_causal_lact_swiglu_csdm_vec_old2(
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
        hidden = sw * hid_before              # [B, dh, l]

        # Pack W1 into [B,E,dv,max_sz] with zero padding at tail
        W1p = w1.new_zeros((B, E, dv, max_sz))
        for ei in range(E):
            sz = int(slice_sizes[ei].item())
            st = int(slice_starts[ei].item())
            W1p[:, ei, :, :sz] = w1[:, :, st:st+sz]

        # Pack hidden slices to [B,E,max_sz,l]
        Hpack = hidden.new_zeros((B, E, max_sz, hidden.shape[-1]))
        for ei in range(E):
            st = int(slice_starts[ei].item()); sz = int(slice_sizes[ei].item())
            Hpack[:, ei, :sz, :] = hidden[:, st:st+sz, :]

        # Batched apply for all slices: (B*E, dv, max_sz) @ (B*E, max_sz, l) -> (B*E, dv, l)
        W1_be = W1p.reshape(B*E, dv, max_sz)
        H_be  = Hpack.reshape(B*E, max_sz, hidden.shape[-1])
        y_be  = torch.bmm(W1_be, H_be)                      # [B*E, dv, l]
        y_all = y_be.view(B, E, dv, -1)                     # [B,E,dv,l]

        # Weight by gates and sum slices: gi: [B,l,E] -> [B,E,1,l]
        gi_t = gi.transpose(1, 2).unsqueeze(2)
        y = (y_all * gi_t).sum(dim=1)                       # [B,dv,l]
        out[:, :, s_index:e_index] = y

        # === Update path (uses key `ki` and value `vi`) ===
        # We must recompute them using `ki` for the update, matching the baseline.
        gate_before_k = torch.bmm(w0, ki.transpose(1,2)) # [B, dh, l]
        hid_before_k = torch.bmm(w2, ki.transpose(1,2))  # [B, dh, l]
        sw_k = F.silu(gate_before_k, inplace=False)
        hidden_k = sw_k * hid_before_k                   # [B, dh, l]

        # === Updates (vectorized) ===
        # dhidden_e = (W1_e^T @ vi) for all slices
        Vexp = vi.unsqueeze(1).expand(B, E, dv, vi.shape[-1])       # [B,E,dv,l] (no copy)
        V_be = Vexp.reshape(B*E, dv, vi.shape[-1])
        dH_be = torch.bmm(W1_be.transpose(1,2), V_be)               # [B*E, max_sz, l]
        dH = dH_be.view(B, E, max_sz, -1)                           # [B,E,max_sz,l]

        # dgate_before per slice: silu' * upstream
        # upstream per slice: dgate_e = dH * hid_before_slice
        HBpack_k = hidden_k.new_zeros((B, E, max_sz, hidden_k.shape[-1]))
        for ei in range(E):
            st = int(slice_starts[ei].item()); sz = int(slice_sizes[ei].item())
            HBpack_k[:, ei, :sz, :] = hid_before_k[:, st:st+sz, :]
        
        GBpack_k = hidden_k.new_zeros((B, E, max_sz, hidden_k.shape[-1]))
        for ei in range(E):
            st = int(slice_starts[ei].item()); sz = int(slice_sizes[ei].item())
            GBpack_k[:, ei, :sz, :] = gate_before_k[:, st:st+sz, :]

        dgate_e = dH * HBpack_k
        dgate_bef = silu_backprop(dgate_e, GBpack_k)                 # [B,E,max_sz,l]

        # LR * gates per slice
        scaled_lr0 = lr0i * gi
        scaled_lr1 = lr1i * gi
        scaled_lr2 = lr2i * gi

        # --- Compute dw1 ---
        # Reshape scaled_lr1 for multiplication: [B, l, E] -> [B, E, l, 1]
        lr1_multiplier = scaled_lr1.permute(0, 2, 1).unsqueeze(-1)

        # Pack hidden_k for dw1 calculation
        Hpack_k = hidden_k.new_zeros((B, E, max_sz, hidden_k.shape[-1]))
        for ei in range(E):
            st = int(slice_starts[ei].item()); sz = int(slice_sizes[ei].item())
            Hpack_k[:, ei, :sz, :] = hidden_k[:, st:st+sz, :]

        # Compute dw1 for all slices: (B*E, dv, l) @ (B*E, l, max_sz)
        H_t_k = Hpack_k.permute(0,1,3,2)                             # [B,E,l,max_sz]
        H_t_be_k = (H_t_k * lr1_multiplier).reshape(B*E, -1, max_sz) # [B*E,l,max_sz]
        dw1_be = torch.bmm(V_be, H_t_be_k.type_as(V_be))             # [B*E, dv, max_sz]
        dw1_all = dw1_be.view(B, E, dv, max_sz)

        # --- Compute dw0 ---
        # Reshape scaled_lr0 for multiplication: [B, l, E] -> [B, E, l, 1]
        lr0_multiplier = scaled_lr0.permute(0, 2, 1).unsqueeze(-1)

        # Compute dw0: (B*E, max_sz, l) @ (B*E, l, dk)
        Ki = ki  # [B,l,dk]
        Ki_be = (Ki.unsqueeze(1) * lr0_multiplier).reshape(B*E, -1, Ki.shape[-1])  # [B*E,l,dk]
        dw0_be = torch.bmm(dgate_bef.reshape(B*E, max_sz, -1).type_as(Ki_be), Ki_be)              # [B*E, max_sz, dk]
        dw0_all = dw0_be.view(B, E, max_sz, dk)

        # --- Compute dw2 ---
        # Reshape scaled_lr2 for multiplication: [B, l, E] -> [B, E, l, 1]
        lr2_multiplier = scaled_lr2.permute(0, 2, 1).unsqueeze(-1)

        # Compute dw2: (B*E, max_sz, l) @ (B*E, l, dk)
        swpack_k = hidden_k.new_zeros((B, E, max_sz, hidden_k.shape[-1]))
        for ei in range(E):
            st = int(slice_starts[ei].item()); sz = int(slice_sizes[ei].item())
            swpack_k[:, ei, :sz, :] = sw_k[:, st:st+sz, :]
        Ki_be2 = (Ki.unsqueeze(1) * lr2_multiplier).reshape(B*E, -1, Ki.shape[-1])
        dw2_be = torch.bmm((dH * swpack_k).reshape(B*E, max_sz, -1).type_as(Ki_be2), Ki_be2)
        dw2_all = dw2_be.view(B, E, max_sz, dk)

        # Momentum (optional)
        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True)  # [B,1,1]
            # 1. Update the momentum state (velocity) with the current gradient
            # velocity = momentum * old_velocity + new_gradient
            dw0 = dw0_momentum * m_i + dw0_all
            dw1 = dw1_momentum * m_i + dw1_all
            dw2 = dw2_momentum * m_i + dw2_all

             # 2. Use the updated momentum state as the final gradient for the weight update
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2

        # Optional muon orthogonalization
        if use_muon:
            dw1_all = zeropower_via_newtonschulz5(dw1_all.reshape(B*E, dv, max_sz)).view_as(dw1_all)
            dw0_all = zeropower_via_newtonschulz5(dw0_all.reshape(B*E, max_sz, dk)).view_as(dw0_all)
            dw2_all = zeropower_via_newtonschulz5(dw2_all.reshape(B*E, max_sz, dk)).view_as(dw2_all)

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

        # This part requires re-packing W1 for the final apply
        W1p = w1.new_zeros((B, E, dv, max_sz))
        for ei in range(E):
            st = int(slice_starts[ei].item()); sz = int(slice_sizes[ei].item())
            W1p[:, ei, :, :sz] = w1[:, :, st:st+sz]
            
        # Pack slices and batched apply
        Hpack = hidden.new_zeros((B, E, max_sz, hidden.shape[-1]))
        for ei in range(E):
            st = int(slice_starts[ei].item()); sz = int(slice_sizes[ei].item())
            Hpack[:, ei, :sz, :] = hidden[:, st:st+sz, :]
        y_be  = torch.bmm(W1p.reshape(B*E, dv, max_sz), Hpack.reshape(B*E, max_sz, -1))
        y_all = y_be.view(B, E, dv, -1)
        gi_t = gi.transpose(1, 2).unsqueeze(2)
        y = (y_all * gi_t).sum(dim=1)
        out[:, :, s_index:e_index] = y

    return out.transpose(1, 2)


# === New (MFU-optimized): Vectorized fine-grained CSDM for LaCT + SwiGLU ===
# Computes all slice matmuls in parallel via batched bmm to improve MFU and reduce Python overhead.
@torch.compile()
def block_causal_lact_swiglu_csdm_vec_old(
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
    # If nothing survives (should not happen when dh>0), fallback to a zero output.
    if E == 0:
        return v.new_zeros(v.shape)

    # Precompute norms for post-update renorm (channel-wise)
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    if momentum is not None:
        dw1_m = torch.zeros_like(w1)
        dw0_m = torch.zeros_like(w0)
        dw2_m = torch.zeros_like(w2)

    qT = q.transpose(1, 2)  # [B, dk, l]
    vT = v.transpose(1, 2)  # [B, dv, l]
    out = torch.zeros_like(vT)  # [B, dv, l]

    seq_len = k.shape[1]
    e_index = 0

    # Pre-slice weight views once (avoid Python loops in matmul)
    # Build lists of views; later we'll stack for batched bmm.
    w1_chunks = []
    for ei in range(E):
        st = int(slice_starts[ei].item())
        sz = int(slice_sizes[ei].item())
        w1_chunks.append(w1[:, :, st:st+sz])  # [B, dv, sz]
    # Concatenate along a new slice dim: [B, E, dv, sz_e] (ragged sizes -> pack separately per ei)
    # Because sizes vary by at most 1, we still need per-ei packing. We'll process with a loop only for assignment,
    # while matmuls are done in a single batched bmm over B*E by padding to max_sz and masking.
    max_sz = int(slice_sizes.max().item())
    # Pack W1 into [B,E,dv,max_sz] with zero padding at tail
    W1p = w1.new_zeros((B, E, dv, max_sz))
    for ei in range(E):
        sz = int(slice_sizes[ei].item())
        W1p[:, ei, :, :sz] = w1_chunks[ei]

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
        hidden = sw * hid_before              # [B, dh, l]

        # Pack hidden slices to [B,E,max_sz,l]
        Hpack = hidden.new_zeros((B, E, max_sz, hidden.shape[-1]))
        for ei in range(E):
            st = int(slice_starts[ei].item()); sz = int(slice_sizes[ei].item())
            Hpack[:, ei, :sz, :] = hidden[:, st:st+sz, :]

        # Batched apply for all slices: (B*E, dv, max_sz) @ (B*E, max_sz, l) -> (B*E, dv, l)
        W1_be = W1p.reshape(B*E, dv, max_sz)
        H_be  = Hpack.reshape(B*E, max_sz, hidden.shape[-1])
        y_be  = torch.bmm(W1_be, H_be)                      # [B*E, dv, l]
        y_all = y_be.view(B, E, dv, -1)                     # [B,E,dv,l]

        # Weight by gates and sum slices: gi: [B,l,E] -> [B,E,1,l]
        gi_t = gi.transpose(1, 2).unsqueeze(2)
        y = (y_all * gi_t).sum(dim=1)                       # [B,dv,l]
        out[:, :, s_index:e_index] = y

        # === Updates (vectorized) ===
        # dhidden_e = (W1_e^T @ vi) for all slices
        Vexp = vi.unsqueeze(1).expand(B, E, dv, vi.shape[-1])       # [B,E,dv,l] (no copy)
        V_be = Vexp.reshape(B*E, dv, vi.shape[-1])
        dH_be = torch.bmm(W1_be.transpose(1,2), V_be)               # [B*E, max_sz, l]
        dH = dH_be.view(B, E, max_sz, -1)                           # [B,E,max_sz,l]

        # dgate_before per slice: silu' * upstream
        # upstream per slice: dgate_e = dH * hid_before_slice
        Hidpack = Hpack  # hidden includes silu already; for dgate we need hid_before only
        HBpack = hidden.new_zeros((B, E, max_sz, hidden.shape[-1]))
        for ei in range(E):
            st = int(slice_starts[ei].item()); sz = int(slice_sizes[ei].item())
            HBpack[:, ei, :sz, :] = hid_before[:, st:st+sz, :]
        dgate_e = dH * HBpack
        GBpack = hidden.new_zeros((B, E, max_sz, hidden.shape[-1]))
        for ei in range(E):
            st = int(slice_starts[ei].item()); sz = int(slice_sizes[ei].item())
            GBpack[:, ei, :sz, :] = gate_before[:, st:st+sz, :]
        dgate_bef = silu_backprop(dgate_e, GBpack)                   # [B,E,max_sz,l]

        # LR * gates per slice
        Gexp = gi.unsqueeze(2)                                       # [B,l,1,E]
        Gexp = Gexp.permute(0,3,2,1)                                 # [B,E,1,l]
        lr1e = (lr1i.unsqueeze(1) * Gexp)                            # [B,E,1,l]
        lr0e = (lr0i.unsqueeze(1) * Gexp)
        lr2e = (lr2i.unsqueeze(1) * Gexp)

        # Compute dw1 for all slices: (B*E, dv, l) @ (B*E, l, max_sz)
        H_t = Hpack.permute(0,1,3,2)                                 # [B,E,l,max_sz]
        H_t_be = (H_t * lr1e.permute(0,1,3,2)).reshape(B*E, -1, max_sz)  # [B*E,l,max_sz]
        dw1_be = torch.bmm(V_be, H_t_be)                             # [B*E, dv, max_sz]
        dw1_all = dw1_be.view(B, E, dv, max_sz)

        # Compute dw0: (B*E, max_sz, l) @ (B*E, l, dk)
        Ki = ki  # [B,l,dk]
        Ki_be = (Ki.unsqueeze(1) * lr0e.permute(0,1,3,2)).reshape(B*E, -1, Ki.shape[-1])  # [B*E,l,dk]
        dw0_be = torch.bmm(dgate_bef.reshape(B*E, max_sz, -1), Ki_be)                     # [B*E, max_sz, dk]
        dw0_all = dw0_be.view(B, E, max_sz, dk)

        # Compute dw2: (B*E, max_sz, l) @ (B*E, l, dk)
        swpack = hidden.new_zeros((B, E, max_sz, hidden.shape[-1]))
        for ei in range(E):
            st = int(slice_starts[ei].item()); sz = int(slice_sizes[ei].item())
            swpack[:, ei, :sz, :] = sw[:, st:st+sz, :]
        Ki_be2 = (Ki.unsqueeze(1) * lr2e.permute(0,1,3,2)).reshape(B*E, -1, Ki.shape[-1])
        dw2_be = torch.bmm((dH * swpack).reshape(B*E, max_sz, -1), Ki_be2)
        dw2_all = dw2_be.view(B, E, max_sz, dk)

        # Momentum (optional)
        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :].mean(dim=1, keepdim=True)  # [B,1,1]
            m_be = m_i.repeat_interleave(E, dim=0).view(B, E, 1, 1)
            # Apply to packed grads
            dw1_all = dw1_all + dw1_m.new_zeros(dw1_all.shape) * m_be  # keep simple; or maintain packed momentum
            dw0_all = dw0_all + dw0_m.new_zeros(dw0_all.shape) * m_be
            dw2_all = dw2_all + dw2_m.new_zeros(dw2_all.shape) * m_be

        # Optional muon orthogonalization
        if use_muon:
            dw1_all = zeropower_via_newtonschulz5(dw1_all.reshape(B*E, dv, max_sz)).view_as(dw1_all)
            dw0_all = zeropower_via_newtonschulz5(dw0_all.reshape(B*E, max_sz, dk)).view_as(dw0_all)
            dw2_all = zeropower_via_newtonschulz5(dw2_all.reshape(B*E, max_sz, dk)).view_as(dw2_all)

        # Scatter-add packed grads back to dense shapes
        dw1 = w1.new_zeros((B, dv, dh))
        dw0 = w0.new_zeros((B, dh, dk))
        dw2 = w2.new_zeros((B, dh, dk))
        offset = 0
        for ei in range(E):
            st = int(slice_starts[ei].item()); sz = int(slice_sizes[ei].item())
            dw1[:, :, st:st+sz] += dw1_all[:, ei, :, :sz]
            dw0[:, st:st+sz, :] += dw0_all[:, ei, :sz, :]
            dw2[:, st:st+sz, :] += dw2_all[:, ei, :sz, :]

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

        # Pack slices and batched apply
        Hpack = hidden.new_zeros((B, E, max_sz, hidden.shape[-1]))
        for ei in range(E):
            st = int(slice_starts[ei].item()); sz = int(slice_sizes[ei].item())
            Hpack[:, ei, :sz, :] = hidden[:, st:st+sz, :]
        y_be  = torch.bmm(W1p.reshape(B*E, dv, max_sz), Hpack.reshape(B*E, max_sz, -1))
        y_all = y_be.view(B, E, dv, -1)
        gi_t = gi.transpose(1, 2).unsqueeze(2)
        y = (y_all * gi_t).sum(dim=1)
        out[:, :, s_index:e_index] = y

    return out.transpose(1, 2)
