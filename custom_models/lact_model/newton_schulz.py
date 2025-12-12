import torch


def newton_schulz_differentiable(
    X: torch.Tensor,
    norm_eps: float = 1e-7,
    shift_eps: float = 1e-3,  # for backward pass stability
    high_precision: bool = False,  # use float64
) -> torch.Tensor:
    """
    Newton-Schulz orthonormalization with custom backward pass.
    https://kexue.fm/archives/11025
    """
    assert X.ndim >= 2, "Input must be at least 2D"
    return NewtonSchulzFunction.apply(X, norm_eps, shift_eps, high_precision)


class NewtonSchulzFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        M: torch.Tensor,
        norm_eps: float,
        shift_eps: float,
        high_precision: bool,
    ):
        O = _newton_schulz_forward(M, norm_eps=norm_eps, high_precision=high_precision)
        ctx.save_for_backward(O, M)
        ctx.norm_eps = norm_eps
        ctx.shift_eps = shift_eps
        ctx.high_precision = high_precision
        return O

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, dO: torch.Tensor):
        O, M = ctx.saved_tensors
        norm_eps = ctx.norm_eps
        shift_eps = ctx.shift_eps
        high_precision = ctx.high_precision
        dM = _newton_schulz_backward(
            dO,
            O=O,
            M=M,
            norm_eps=norm_eps,
            shift_eps=shift_eps,
            high_precision=high_precision,
        )
        return dM, None, None, None


NS_CONSTS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]


@torch.compile(fullgraph=True)
def _newton_schulz_forward(
    M: torch.Tensor, norm_eps: float = 1e-7, high_precision: bool = False
):
    """
    Orthonormalize a matrix using Newton-Schulz iteration.
    """
    if high_precision:
        dtype = torch.float64
        ns_consts = NS_CONSTS + [(2.0, -1.5, 0.5)] * 10
    else:
        dtype = torch.bfloat16
        ns_consts = NS_CONSTS

    X = M.to(dtype=dtype)
    if M.size(-2) > M.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + norm_eps)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A.mT)
        X = a * X + B @ X

    if M.size(-2) > M.size(-1):
        X = X.mT
    X = X.to(dtype=M.dtype)
    return X


@torch.compile(fullgraph=True)
def _newton_schulz_backward(
    dO: torch.Tensor,
    O: torch.Tensor,
    M: torch.Tensor,
    norm_eps: float = 1e-7,
    shift_eps: float = 1e-3,
    high_precision: bool = False,
):
    """
    Backward pass for Newton-Schulz orthonormalization.
    https://kexue.fm/archives/11025
    https://arxiv.org/pdf/2201.08663

    M = input to forward pass
    O = newton_schulz(M) computed in forward pass
    dO = gradient of loss w.r.t. O
    Return dM = gradient of loss w.r.t. M
    """
    original_shape = dO.shape
    original_dtype = dO.dtype

    if high_precision:
        mul_dtype = torch.float64
        norm_dtype = torch.float64
        ns_consts = NS_CONSTS + [(2.0, -1.5, 0.5)] * 10
    else:
        mul_dtype = torch.bfloat16
        norm_dtype = torch.float32
        ns_consts = NS_CONSTS

    # Flatten all batch dimensions
    m, n = original_shape[-2:]
    dO = dO.reshape(-1, m, n)
    M = M.reshape(-1, m, n).to(mul_dtype)
    O = O.reshape(-1, m, n).to(mul_dtype)

    A = M @ O.mT  # shape (batch, m, m)
    B = O.mT @ M  # shape (batch, n, n)

    # Normalize A, B, dO
    # norm(A) should equal norm(B)
    A = A.to(norm_dtype)
    B = B.to(norm_dtype)
    dO = dO.to(norm_dtype)
    norm = A.norm(dim=(-2, -1), keepdim=True)
    A = A / (norm + norm_eps)
    B = B / (norm + norm_eps)
    dO = dO / (norm + norm_eps)

    # Add shift for numerical stability
    IA = torch.eye(A.size(-1), device=A.device, dtype=A.dtype).unsqueeze(0)
    IB = torch.eye(B.size(-1), device=B.device, dtype=B.dtype).unsqueeze(0)
    A = A + shift_eps * IA
    B = B + shift_eps * IB

    A = A.to(mul_dtype)
    B = B.to(mul_dtype)
    dO = dO.to(mul_dtype)
    Z = torch.zeros_like(dO.mT)  # shape (batch, n, m)

    # Solve the Sylvester equation AX + XB = dO
    H = torch.cat(
        [torch.cat([A, -dO], dim=-1), torch.cat([Z, -B], dim=-1)],
        dim=-2,
    )  # shape (batch, m+n, m+n)

    # Newton-Schulz iteration to compute matrix sign function of H
    # This is DIFFERENT from the forward pass NS because there is no transposing
    for a, b, c in ns_consts:
        A = H @ H
        B = b * A + c * (A @ A)
        H = a * H + B @ H

    # Answer is -1/2 times top-right block of the result
    X = -0.5 * H[:, :m, -n:]

    # Solve for dM
    dM = X - (O @ X.mT @ O)
    dM = dM.to(dtype=original_dtype).reshape(original_shape)
    return dM


if __name__ == "__main__":
    from torch.autograd.gradcheck import gradcheck
    from triton.testing import do_bench

    d = 1024
    norm_eps = 1e-7
    M1 = torch.randn(4, d, d, dtype=torch.bfloat16, device="cuda")
    M2 = M1.clone()
    G = torch.randn_like(M1)

    # Backward using custom function
    M1.requires_grad_(True)
    M1.retain_grad()
    O1 = newton_schulz_differentiable(M1, norm_eps=norm_eps)
    O1.backward(gradient=G)
    print("M1 grad:", M1.grad)

    # Backward using torch autograd on forward function
    M2.requires_grad_(True)
    M2.retain_grad()
    O2 = _newton_schulz_forward(M2, norm_eps=norm_eps)
    O2.backward(gradient=G)
    print("M2 grad:", M2.grad)

    # Compare gradients
    error = M1.grad - M2.grad
    error_rms = error.square().mean().sqrt()
    print("Difference:", error)
    print("RMS error:", error_rms)
    print("Relative RMS error:", error_rms / (M2.grad.square().mean().sqrt()))
    print("Max error:", error.abs().max())
    print()

    # Benchmark
    print("Benchmarking NS with custom backward function...")
    M1.grad = None
    bench = do_bench(
        lambda: newton_schulz_differentiable(M1).backward(gradient=G),
        grad_to_none=M1,
    )
    print(bench)

    print("Benchmarking NS with torch autograd...")
    M2.grad = None
    bench = do_bench(
        lambda: _newton_schulz_forward(M2).backward(gradient=G),
        grad_to_none=M2,
    )
    print(bench)

    print()

    # Run grad check in high precision
    print("Running gradcheck in float64...")
    M = torch.randn(4, 16, 16, device="cpu", dtype=torch.float64, requires_grad=True)
    result = gradcheck(
        lambda X: newton_schulz_differentiable(
            X, norm_eps=1e-7, shift_eps=1e-7, high_precision=True
        ),
        M,
    )
    print("gradcheck passed:", result)
