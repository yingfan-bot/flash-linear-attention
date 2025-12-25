"""
Tests for chunked prefill consistency with non-chunked prefill in LaCT model.

The key fixes verified by these tests:
1. should_update_last = is_chunked_prefill and not is_last_chunk
   (Last chunk should NOT update weights, matching non-chunked behavior)
2. Momentum state continuity between chunks via cache
   (momentum_state parameter passed to block_causal_lact_swiglu)

Usage:
    cd /home/fanying/flash-linear-attention
    CUDA_VISIBLE_DEVICES=1 python tests/test_chunked_prefill.py
"""

import sys
import torch

sys.path.insert(0, '/home/fanying/flash-linear-attention')

from custom_models.lact_model.modeling_lact import LaCTForCausalLM
from custom_models.lact_model.ttt_operation import block_causal_lact_swiglu

MODEL_PATH = '/data/fanying/nh4-swa2048-rope-760M-64K-4B-65536/batch1.seqlen65536.bs1.warmup1024.update1.steps61440.lr1e-3.cosine.8gpu'


def test_momentum_state_continuity():
    """Test that passing momentum_state produces consistent results."""
    print("\n=== Test: Momentum State Continuity ===")
    b, l, d = 1, 4096, 64
    chunk_size = 2048
    torch.manual_seed(42)
    
    w0 = torch.randn(b, d, d, device='cuda', dtype=torch.bfloat16)
    w1 = torch.randn(b, d, d, device='cuda', dtype=torch.bfloat16)
    w2 = torch.randn(b, d, d, device='cuda', dtype=torch.bfloat16)
    q = torch.randn(b, l, d, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(b, l, d, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(b, l, d, device='cuda', dtype=torch.bfloat16)
    lr0 = torch.randn(b, l, 1, device='cuda', dtype=torch.float32) * 0.01
    lr1 = torch.randn(b, l, 1, device='cuda', dtype=torch.float32) * 0.01
    lr2 = torch.randn(b, l, 1, device='cuda', dtype=torch.float32) * 0.01
    momentum = torch.randn(b, l, 1, device='cuda', dtype=torch.bfloat16).sigmoid()
    
    # Non-chunked: single call with all tokens
    out_nc, (w0_nc, w1_nc, w2_nc), m_nc = block_causal_lact_swiglu(
        w0.clone(), w1.clone(), w2.clone(), q.clone(), k.clone(), v.clone(),
        lr0.clone(), lr1.clone(), lr2.clone(),
        chunk_size=chunk_size, use_muon=True, momentum=momentum.clone(),
        return_final_state=True, update_last_chunk=True, momentum_state=None
    )
    
    # Chunked: two calls with momentum_state passing
    result1 = block_causal_lact_swiglu(
        w0.clone(), w1.clone(), w2.clone(),
        q[:, :2048, :].clone(), k[:, :2048, :].clone(), v[:, :2048, :].clone(),
        lr0[:, :2048, :].clone(), lr1[:, :2048, :].clone(), lr2[:, :2048, :].clone(),
        chunk_size=chunk_size, use_muon=True, momentum=momentum[:, :2048, :].clone(),
        return_final_state=True, update_last_chunk=True, momentum_state=None
    )
    out1, (w0_1, w1_1, w2_1), m_state1 = result1
    
    result2 = block_causal_lact_swiglu(
        w0_1, w1_1, w2_1,
        q[:, 2048:, :].clone(), k[:, 2048:, :].clone(), v[:, 2048:, :].clone(),
        lr0[:, 2048:, :].clone(), lr1[:, 2048:, :].clone(), lr2[:, 2048:, :].clone(),
        chunk_size=chunk_size, use_muon=True, momentum=momentum[:, 2048:, :].clone(),
        return_final_state=True, update_last_chunk=True, momentum_state=m_state1
    )
    out2, (w0_ch, w1_ch, w2_ch), m_state2 = result2
    
    # Check outputs
    out1_diff = (out_nc[:, :2048, :].float() - out1.float()).abs().max().item()
    out2_diff = (out_nc[:, 2048:, :].float() - out2.float()).abs().max().item()
    w0_diff = (w0_nc.float() - w0_ch.float()).abs().max().item()
    
    print(f"  Output[0:2048] diff: {out1_diff:.10f}")
    print(f"  Output[2048:4096] diff: {out2_diff:.10f}")
    print(f"  w0 final diff: {w0_diff:.10f}")
    
    assert out1_diff < 1e-5, f"First chunk output mismatch: {out1_diff}"
    assert out2_diff < 1e-5, f"Second chunk output mismatch: {out2_diff}"
    assert w0_diff < 0.1, f"w0 diff too large: {w0_diff}"
    print("  PASSED!")


def test_without_momentum_has_larger_diff():
    """Test that NOT passing momentum_state produces larger weight differences."""
    print("\n=== Test: Without Momentum Has Larger Diff ===")
    b, l, d = 1, 4096, 64
    chunk_size = 2048
    torch.manual_seed(42)
    
    w0 = torch.randn(b, d, d, device='cuda', dtype=torch.bfloat16)
    w1 = torch.randn(b, d, d, device='cuda', dtype=torch.bfloat16)
    w2 = torch.randn(b, d, d, device='cuda', dtype=torch.bfloat16)
    q = torch.randn(b, l, d, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(b, l, d, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(b, l, d, device='cuda', dtype=torch.bfloat16)
    lr0 = torch.randn(b, l, 1, device='cuda', dtype=torch.float32) * 0.01
    lr1 = torch.randn(b, l, 1, device='cuda', dtype=torch.float32) * 0.01
    lr2 = torch.randn(b, l, 1, device='cuda', dtype=torch.float32) * 0.01
    momentum = torch.randn(b, l, 1, device='cuda', dtype=torch.bfloat16).sigmoid()
    
    # Non-chunked
    _, (w0_nc, _, _), _ = block_causal_lact_swiglu(
        w0.clone(), w1.clone(), w2.clone(), q.clone(), k.clone(), v.clone(),
        lr0.clone(), lr1.clone(), lr2.clone(),
        chunk_size=chunk_size, use_muon=True, momentum=momentum.clone(),
        return_final_state=True, update_last_chunk=True, momentum_state=None
    )
    
    # Chunked WITHOUT momentum_state passing
    _, (w0_1, w1_1, w2_1), _ = block_causal_lact_swiglu(
        w0.clone(), w1.clone(), w2.clone(),
        q[:, :2048, :].clone(), k[:, :2048, :].clone(), v[:, :2048, :].clone(),
        lr0[:, :2048, :].clone(), lr1[:, :2048, :].clone(), lr2[:, :2048, :].clone(),
        chunk_size=chunk_size, use_muon=True, momentum=momentum[:, :2048, :].clone(),
        return_final_state=True, update_last_chunk=True, momentum_state=None
    )
    _, (w0_no_m, _, _), _ = block_causal_lact_swiglu(
        w0_1, w1_1, w2_1,
        q[:, 2048:, :].clone(), k[:, 2048:, :].clone(), v[:, 2048:, :].clone(),
        lr0[:, 2048:, :].clone(), lr1[:, 2048:, :].clone(), lr2[:, 2048:, :].clone(),
        chunk_size=chunk_size, use_muon=True, momentum=momentum[:, 2048:, :].clone(),
        return_final_state=True, update_last_chunk=True, momentum_state=None  # NOT passing
    )
    
    w0_diff_no_m = (w0_nc.float() - w0_no_m.float()).abs().max().item()
    print(f"  w0 diff WITHOUT momentum passing: {w0_diff_no_m:.10f}")
    assert w0_diff_no_m > 0.3, f"Expected larger diff without momentum, got {w0_diff_no_m}"
    print("  PASSED!")


def test_model_prefill_prediction_match():
    """Test that chunked and non-chunked prefill produce same final prediction."""
    print("\n=== Test: Model Prefill Prediction Match ===")
    seq_len = 8192
    torch.manual_seed(42)
    
    model = LaCTForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to('cuda')
    model.eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device='cuda')
    
    with torch.no_grad():
        out_nc = model(input_ids.clone(), use_cache=True, force_chunked_prefill=False)
        pred_nc = out_nc.logits[:, -1].argmax(dim=-1).item()
    
    del model
    torch.cuda.empty_cache()
    model = LaCTForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to('cuda')
    model.eval()
    
    with torch.no_grad():
        out_ch = model(input_ids.clone(), use_cache=True, force_chunked_prefill=True)
        pred_ch = out_ch.logits[:, -1].argmax(dim=-1).item()
    
    print(f"  Non-chunked prediction: {pred_nc}")
    print(f"  Chunked prediction: {pred_ch}")
    assert pred_nc == pred_ch, f"Predictions differ: nc={pred_nc}, ch={pred_ch}"
    print("  PASSED!")
    
    del model
    torch.cuda.empty_cache()


def test_model_weight_diff():
    """Test that weight differences between chunked and non-chunked are small."""
    print("\n=== Test: Model Weight Differences ===")
    seq_len = 8192
    torch.manual_seed(42)
    
    model = LaCTForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to('cuda')
    model.eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device='cuda')
    
    with torch.no_grad():
        out_nc = model(input_ids.clone(), use_cache=True, force_chunked_prefill=False)
        cache_nc = out_nc.past_key_values
    
    del model
    torch.cuda.empty_cache()
    model = LaCTForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to('cuda')
    model.eval()
    
    with torch.no_grad():
        out_ch = model(input_ids.clone(), use_cache=True, force_chunked_prefill=True)
        cache_ch = out_ch.past_key_values
    
    all_passed = True
    for layer_idx in [0, 11, 23]:
        w0_diff = (cache_nc[layer_idx][2].float() - cache_ch[layer_idx][2].float()).abs().max().item()
        status = "OK" if w0_diff < 0.2 else "FAIL"
        print(f"  Layer {layer_idx}: w0 diff = {w0_diff:.6f} [{status}]")
        if w0_diff >= 0.2:
            all_passed = False
    
    assert all_passed, "Some layer weight diffs too large"
    print("  PASSED!")
    
    del model
    torch.cuda.empty_cache()


def test_model_decode_consistency():
    """Test that decoding from chunked/non-chunked caches produces same tokens."""
    print("\n=== Test: Model Decode Consistency ===")
    seq_len = 8192
    torch.manual_seed(42)
    
    model = LaCTForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to('cuda')
    model.eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device='cuda')
    
    with torch.no_grad():
        out_nc = model(input_ids.clone(), use_cache=True, force_chunked_prefill=False)
        pred_nc = out_nc.logits[:, -1].argmax(dim=-1).item()
        cache_nc = out_nc.past_key_values
    
    del model
    torch.cuda.empty_cache()
    model = LaCTForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).to('cuda')
    model.eval()
    
    with torch.no_grad():
        out_ch = model(input_ids.clone(), use_cache=True, force_chunked_prefill=True)
        pred_ch = out_ch.logits[:, -1].argmax(dim=-1).item()
        cache_ch = out_ch.past_key_values
    
    # Decode 10 tokens and check they match
    match_count = 0
    nc_tokens, ch_tokens = [], []
    for i in range(10):
        with torch.no_grad():
            next_nc = model(torch.tensor([[pred_nc]], device='cuda'),
                          use_cache=True, past_key_values=cache_nc)
            pred_nc = next_nc.logits[:, -1].argmax(dim=-1).item()
            cache_nc = next_nc.past_key_values
            nc_tokens.append(pred_nc)
            
            next_ch = model(torch.tensor([[pred_ch]], device='cuda'),
                          use_cache=True, past_key_values=cache_ch)
            pred_ch = next_ch.logits[:, -1].argmax(dim=-1).item()
            cache_ch = next_ch.past_key_values
            ch_tokens.append(pred_ch)
        
        if pred_nc == pred_ch:
            match_count += 1
    
    print(f"  Non-chunked tokens: {nc_tokens}")
    print(f"  Chunked tokens:     {ch_tokens}")
    print(f"  Match rate: {match_count}/10")
    assert match_count >= 8, f"Decode match rate too low: {match_count}/10"
    print("  PASSED!")
    
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    print("=" * 60)
    print("Chunked Prefill Consistency Tests")
    print("=" * 60)
    
    test_momentum_state_continuity()
    test_without_momentum_has_larger_diff()
    test_model_prefill_prediction_match()
    test_model_weight_diff()
    test_model_decode_consistency()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
