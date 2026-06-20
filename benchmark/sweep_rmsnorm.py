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
def rmsnorm_fwd_kernel(
    X_ptr, G_ptr,
    Y_ptr, Rstd_ptr,
    row_stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row   = tl.program_id(0)
    X_ptr += row * row_stride
    Y_ptr += row * row_stride
    cols  = tl.arange(0, BLOCK_SIZE)
    mask  = cols < N
    x     = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rms   = tl.sum(x * x, axis=0) / N
    rstd  = 1.0 / tl.sqrt(rms + eps)
    tl.store(Rstd_ptr + row, rstd)
    g = tl.load(G_ptr + cols, mask=mask)
    tl.store(Y_ptr + cols, x * rstd * g, mask=mask)


def triton_rmsnorm(x, gamma, eps=1e-5):
    batch, n = x.shape
    y    = torch.empty_like(x)
    rstd = torch.empty(batch, device=x.device, dtype=x.dtype)
    rmsnorm_fwd_kernel[(batch,)](
        x, gamma, y, rstd,
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
        x     = torch.randn(N, C, device='cuda', dtype=torch.float32)
        gamma = torch.ones(C,    device='cuda', dtype=torch.float32)

        compiled_rms = torch.compile(lambda x, g: F.rms_norm(x, (C,), g))
        _ = compiled_rms(x, gamma)
        torch.cuda.synchronize()

        pt  = bench(lambda: F.rms_norm(x, (C,), gamma))
        cmp = bench(lambda: compiled_rms(x, gamma))
        tri = bench(lambda: triton_rmsnorm(x, gamma))
        print(f"{N}x{C},{pt:.1f},{cmp:.1f},{tri:.1f}")
