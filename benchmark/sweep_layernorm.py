import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing

SIZES = [
    (512,  1024),
    (1024, 1024),
    (2048, 1024),
    (4096, 1024),
    (8192, 1024),
    (8192, 2048),
    (8192, 4096),
]

@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in [4, 8, 16, 32]],
    key=['N'],
)
@triton.jit
def layernorm_fwd_kernel(
    X_ptr, W_ptr, B_ptr,
    Y_ptr, Mean_ptr, Rstd_ptr,
    row_stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_ptr += row * row_stride
    Y_ptr += row * row_stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x    = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc   = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Mean_ptr + row, mean)
    tl.store(Rstd_ptr + row, rstd)
    w = tl.load(W_ptr + cols, mask=mask)
    b = tl.load(B_ptr + cols, mask=mask)
    tl.store(Y_ptr + cols, xc * rstd * w + b, mask=mask)


def triton_layernorm(x, weight, bias, eps=1e-5):
    batch, n = x.shape
    y    = torch.empty_like(x)
    mean = torch.empty(batch, device=x.device, dtype=x.dtype)
    rstd = torch.empty(batch, device=x.device, dtype=x.dtype)
    layernorm_fwd_kernel[(batch,)](
        x, weight, bias, y, mean, rstd,
        x.stride(0), n, eps,
        BLOCK_SIZE=triton.next_power_of_2(n),
    )
    return y


def bench(fn, warmup=25, reps=100):
    ms, _, _ = triton.testing.do_bench(fn, warmup=warmup, rep=reps, quantiles=[0.5, 0.05, 0.95])
    return ms * 1000  # µs


if __name__ == '__main__':
    print("size,pytorch_us,compiled_us,triton_us")
    for N, C in SIZES:
        x = torch.randn(N, C, device='cuda', dtype=torch.float32)
        w = torch.ones(C,     device='cuda', dtype=torch.float32)
        b = torch.zeros(C,    device='cuda', dtype=torch.float32)

        compiled_ln = torch.compile(lambda x, w, b: F.layer_norm(x, (C,), w, b))
        _ = compiled_ln(x, w, b)
        torch.cuda.synchronize()

        pt  = bench(lambda: F.layer_norm(x, (C,), w, b))
        cmp = bench(lambda: compiled_ln(x, w, b))
        tri = bench(lambda: triton_layernorm(x, w, b))
        print(f"{N}x{C},{pt:.1f},{cmp:.1f},{tri:.1f}")
