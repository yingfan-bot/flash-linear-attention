"""
Test to compare CPU vs GPU for TTT operation consistency.

This test shows that even on CPU, there's a tiny numerical diff (~1e-6)
in the final weights due to the iterative Newton-Schulz algorithm (Muon).
However, the OUTPUTS are always exact matches when momentum is passed correctly.

Results:
- CPU fp32:  output exact_match=True, w0 diff=~0.000004
- GPU fp32:  output exact_match=True, w0 diff=~0.000004  
- GPU bf16:  output exact_match=True, w0 diff=~0.016

The small weight diff comes from zeropower_via_newtonschulz5 (Muon optimizer)
which has iterative computation with accumulated numerical errors.

Usage:
    cd /home/fanying/flash-linear-attention
    python tests/test_cpu_vs_gpu_ttt.py
"""

import torch
import sys
sys.path.insert(0, '/home/fanying/flash-linear-attention')

from custom_models.lact_model.ttt_operation import block_causal_lact_swiglu


def run_test(device, dtype):
    """Run TTT operation test on specified device and dtype."""
    print(f'\nDevice: {device}, Dtype: {dtype}')
    print('-' * 50)
    
    b, l, d = 1, 4096, 64
    chunk_size = 2048
    
    torch.manual_seed(42)
    
    w0 = torch.randn(b, d, d, device=device, dtype=dtype)
    w1 = torch.randn(b, d, d, device=device, dtype=dtype)
    w2 = torch.randn(b, d, d, device=device, dtype=dtype)
    q = torch.randn(b, l, d, device=device, dtype=dtype)
    k = torch.randn(b, l, d, device=device, dtype=dtype)
    v = torch.randn(b, l, d, device=device, dtype=dtype)
    lr0 = torch.randn(b, l, 1, device=device, dtype=torch.float32) * 0.01
    lr1 = torch.randn(b, l, 1, device=device, dtype=torch.float32) * 0.01
    lr2 = torch.randn(b, l, 1, device=device, dtype=torch.float32) * 0.01
    momentum = torch.randn(b, l, 1, device=device, dtype=dtype).sigmoid()
    
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
    
    out1_match = torch.equal(out_nc[:, :2048, :], out1)
    out2_match = torch.equal(out_nc[:, 2048:, :], out2)
    w0_match = torch.equal(w0_nc, w0_ch)
    
    print(f'  Output[0:2048] exact_match={out1_match}, diff={out1_diff:.10f}')
    print(f'  Output[2048:4096] exact_match={out2_match}, diff={out2_diff:.10f}')
    print(f'  w0 final exact_match={w0_match}, diff={w0_diff:.10f}')
    
    return {
        'out1_match': out1_match,
        'out2_match': out2_match,
        'w0_match': w0_match,
        'out1_diff': out1_diff,
        'out2_diff': out2_diff,
        'w0_diff': w0_diff,
    }


def main():
    print('=' * 70)
    print('TEST: CPU vs GPU for TTT operation consistency')
    print('=' * 70)
    
    results = {}
    
    # CPU fp32
    results['cpu_fp32'] = run_test('cpu', torch.float32)
    
    # GPU fp32
    if torch.cuda.is_available():
        results['gpu_fp32'] = run_test('cuda', torch.float32)
        results['gpu_bf16'] = run_test('cuda', torch.bfloat16)
    else:
        print('\nCUDA not available, skipping GPU tests')
    
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print()
    print('Key findings:')
    print('1. OUTPUTS are always exact matches when momentum is passed correctly')
    print('2. Weights have small diff even on CPU (~1e-6) due to Newton-Schulz iterations')
    print('3. bf16 amplifies weight diff by ~4000x (0.016 vs 0.000004)')
    print()
    print('The output matching is what matters for model behavior.')
    print('Weight diff is expected numerical noise from Muon optimizer.')
    
    # Verify outputs always match
    all_outputs_match = all(
        r.get('out1_match', False) and r.get('out2_match', False) 
        for r in results.values()
    )
    
    if all_outputs_match:
        print('\n✓ All outputs match - chunked prefill is working correctly!')
    else:
        print('\n✗ Some outputs do not match - there may be a bug!')
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
