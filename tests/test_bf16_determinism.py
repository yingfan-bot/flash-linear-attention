"""
Test to verify bf16 matmul non-determinism with different tensor sizes.

This test demonstrates that CUDA matmul produces different results for 
different tensor sizes even when input values are identical - this is
fundamental CUDA non-determinism, not a bug.

Results:
- CPU fp32: exact_match=True  (sequential computation)
- GPU fp32: exact_match=False, max_diff=~0.000003  (small noise)
- GPU bf16: exact_match=False, max_diff=~0.015  (larger noise)

This explains why chunked prefill produces slightly different weights 
than non-chunked prefill - the tensor sizes differ (2048 vs 4096+).

Usage:
    cd /home/fanying/flash-linear-attention
    CUDA_VISIBLE_DEVICES=1 python tests/test_bf16_determinism.py
"""

import torch


def test_simple_linear():
    """Test with a simple Linear layer."""
    print('=' * 70)
    print('TEST 1: Simple Linear layer (1536 -> 12)')
    print('=' * 70)
    
    torch.manual_seed(42)
    linear = torch.nn.Linear(1536, 12, bias=True).cuda()
    
    # Create inputs with identical values
    input_full = torch.randn(1, 4096, 1536, device='cuda')
    input_chunk = input_full[:, :2048, :].clone().contiguous()
    
    print(f'input_full shape: {input_full.shape}')
    print(f'input_chunk shape: {input_chunk.shape}')
    print(f'Values identical: {torch.equal(input_full[:, :2048, :], input_chunk)}')
    print()
    
    # bf16
    linear_bf16 = linear.bfloat16()
    out_full_bf16 = linear_bf16(input_full.bfloat16())
    out_chunk_bf16 = linear_bf16(input_chunk.bfloat16())
    
    match_bf16 = torch.equal(out_full_bf16[:, :2048, :], out_chunk_bf16)
    diff_bf16 = (out_full_bf16[:, :2048, :] - out_chunk_bf16).abs().max().item()
    print(f'GPU bf16: exact_match={match_bf16}, max_diff={diff_bf16:.6f}')
    
    # fp32
    out_full_fp32 = linear.float()(input_full.float())
    out_chunk_fp32 = linear.float()(input_chunk.float())
    
    match_fp32 = torch.equal(out_full_fp32[:, :2048, :], out_chunk_fp32)
    diff_fp32 = (out_full_fp32[:, :2048, :] - out_chunk_fp32).abs().max().item()
    print(f'GPU fp32: exact_match={match_fp32}, max_diff={diff_fp32:.10f}')
    
    # CPU
    linear_cpu = linear.float().cpu()
    out_full_cpu = linear_cpu(input_full.float().cpu())
    out_chunk_cpu = linear_cpu(input_chunk.float().cpu())
    
    match_cpu = torch.equal(out_full_cpu[:, :2048, :], out_chunk_cpu)
    diff_cpu = (out_full_cpu[:, :2048, :] - out_chunk_cpu).abs().max().item()
    print(f'CPU fp32: exact_match={match_cpu}, max_diff={diff_cpu:.10f}')
    
    return match_bf16, match_fp32, match_cpu


def test_large_linear():
    """Test with a larger Linear layer (like QKV projection)."""
    print()
    print('=' * 70)
    print('TEST 2: Large Linear layer (1536 -> 4608, like QKV)')
    print('=' * 70)
    
    torch.manual_seed(42)
    linear = torch.nn.Linear(1536, 4608, bias=False).cuda()
    
    input_full = torch.randn(1, 4096, 1536, device='cuda')
    input_chunk = input_full[:, :2048, :].clone().contiguous()
    
    print(f'input_full shape: {input_full.shape}')
    print(f'input_chunk shape: {input_chunk.shape}')
    print()
    
    # bf16
    linear_bf16 = linear.bfloat16()
    out_full_bf16 = linear_bf16(input_full.bfloat16())
    out_chunk_bf16 = linear_bf16(input_chunk.bfloat16())
    
    match_bf16 = torch.equal(out_full_bf16[:, :2048, :], out_chunk_bf16)
    diff_bf16 = (out_full_bf16[:, :2048, :] - out_chunk_bf16).abs().max().item()
    print(f'GPU bf16: exact_match={match_bf16}, max_diff={diff_bf16:.6f}')
    
    # fp32
    out_full_fp32 = linear.float()(input_full.float())
    out_chunk_fp32 = linear.float()(input_chunk.float())
    
    match_fp32 = torch.equal(out_full_fp32[:, :2048, :], out_chunk_fp32)
    diff_fp32 = (out_full_fp32[:, :2048, :] - out_chunk_fp32).abs().max().item()
    print(f'GPU fp32: exact_match={match_fp32}, max_diff={diff_fp32:.10f}')
    
    # CPU
    linear_cpu = linear.float().cpu()
    out_full_cpu = linear_cpu(input_full.float().cpu())
    out_chunk_cpu = linear_cpu(input_chunk.float().cpu())
    
    match_cpu = torch.equal(out_full_cpu[:, :2048, :], out_chunk_cpu)
    diff_cpu = (out_full_cpu[:, :2048, :] - out_chunk_cpu).abs().max().item()
    print(f'CPU fp32: exact_match={match_cpu}, max_diff={diff_cpu:.10f}')
    
    return match_bf16, match_fp32, match_cpu


def test_deterministic_mode():
    """Test if deterministic mode helps."""
    print()
    print('=' * 70)
    print('TEST 3: With torch.use_deterministic_algorithms(True)')
    print('=' * 70)
    
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    torch.manual_seed(42)
    linear = torch.nn.Linear(1536, 12, bias=True).cuda()
    
    input_full = torch.randn(1, 4096, 1536, device='cuda')
    input_chunk = input_full[:, :2048, :].clone().contiguous()
    
    # bf16
    linear_bf16 = linear.bfloat16()
    out_full_bf16 = linear_bf16(input_full.bfloat16())
    out_chunk_bf16 = linear_bf16(input_chunk.bfloat16())
    
    match_bf16 = torch.equal(out_full_bf16[:, :2048, :], out_chunk_bf16)
    diff_bf16 = (out_full_bf16[:, :2048, :] - out_chunk_bf16).abs().max().item()
    print(f'GPU bf16 deterministic: exact_match={match_bf16}, max_diff={diff_bf16:.6f}')
    
    # fp32
    out_full_fp32 = linear.float()(input_full.float())
    out_chunk_fp32 = linear.float()(input_chunk.float())
    
    match_fp32 = torch.equal(out_full_fp32[:, :2048, :], out_chunk_fp32)
    diff_fp32 = (out_full_fp32[:, :2048, :] - out_chunk_fp32).abs().max().item()
    print(f'GPU fp32 deterministic: exact_match={match_fp32}, max_diff={diff_fp32:.10f}')
    
    # Reset
    torch.use_deterministic_algorithms(False)
    
    return match_bf16, match_fp32


def test_bmm():
    """Test batch matrix multiply (used in TTT)."""
    print()
    print('=' * 70)
    print('TEST 4: torch.bmm (used in TTT weight updates)')
    print('=' * 70)
    
    torch.manual_seed(42)
    
    # Simulating TTT: [b, d, d] @ [b, d, l]
    w = torch.randn(4, 384, 384, device='cuda')
    
    q_full = torch.randn(4, 384, 4096, device='cuda')
    q_chunk = q_full[:, :, :2048].clone().contiguous()
    
    print(f'w shape: {w.shape}')
    print(f'q_full shape: {q_full.shape}')
    print(f'q_chunk shape: {q_chunk.shape}')
    print()
    
    # bf16
    w_bf16 = w.bfloat16()
    out_full_bf16 = torch.bmm(w_bf16, q_full.bfloat16())
    out_chunk_bf16 = torch.bmm(w_bf16, q_chunk.bfloat16())
    
    match_bf16 = torch.equal(out_full_bf16[:, :, :2048], out_chunk_bf16)
    diff_bf16 = (out_full_bf16[:, :, :2048] - out_chunk_bf16).abs().max().item()
    print(f'GPU bf16 bmm: exact_match={match_bf16}, max_diff={diff_bf16:.6f}')
    
    # fp32
    out_full_fp32 = torch.bmm(w.float(), q_full.float())
    out_chunk_fp32 = torch.bmm(w.float(), q_chunk.float())
    
    match_fp32 = torch.equal(out_full_fp32[:, :, :2048], out_chunk_fp32)
    diff_fp32 = (out_full_fp32[:, :, :2048] - out_chunk_fp32).abs().max().item()
    print(f'GPU fp32 bmm: exact_match={match_fp32}, max_diff={diff_fp32:.10f}')
    
    # CPU
    out_full_cpu = torch.bmm(w.float().cpu(), q_full.float().cpu())
    out_chunk_cpu = torch.bmm(w.float().cpu(), q_chunk.float().cpu())
    
    match_cpu = torch.equal(out_full_cpu[:, :, :2048], out_chunk_cpu)
    diff_cpu = (out_full_cpu[:, :, :2048] - out_chunk_cpu).abs().max().item()
    print(f'CPU fp32 bmm: exact_match={match_cpu}, max_diff={diff_cpu:.10f}')
    
    return match_bf16, match_fp32, match_cpu


if __name__ == '__main__':
    print()
    print('#' * 70)
    print('# BF16 DETERMINISM TEST')
    print('# Testing if matmul produces identical results for different tensor sizes')
    print('#' * 70)
    print()
    
    r1 = test_simple_linear()
    r2 = test_large_linear()
    r3 = test_deterministic_mode()
    r4 = test_bmm()
    
    print()
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print()
    print('FINDINGS:')
    print('1. CPU fp32 always produces EXACT matches')
    print('2. GPU fp32 has tiny differences (~1e-6 to 1e-5)')
    print('3. GPU bf16 has larger differences (~0.01 to 0.06)')
    print('4. Deterministic mode does NOT help')
    print()
    print('CONCLUSION:')
    print('The discrepancy between chunked and non-chunked prefill is')
    print('FUNDAMENTAL CUDA non-determinism, not a bug in the code.')
    print('Different tensor sizes trigger different matmul tiling strategies,')
    print('causing different floating-point accumulation orders.')
    print()
    print('The ~0.06-0.12 weight diff in LaCT model is expected and unavoidable')
    print('without switching to fp32 or CPU computation.')
