"""Two fused C++ kernels for LeWi's gate, one CPU thread, float32 (built on first use with torch.utils.cpp_extension).

``gate_forward(W, recv, send, R2) -> (g, W_eff)``
    For every group of a stack: the scores ``s_ij = R2 - ||recv_i - send_j||^2`` from the embeddings, the gate
    ``g = sigmoid(s)`` and the effective weight ``W_eff = W * g`` where ``s > 0`` and 0 elsewhere, in one pass.

``gate_backward_adam(gW_eff, W, g, recv, send, m, v, step_size, bc2_sqrt, b1, b2, eps) -> (grad_recv, grad_send, gn2)``
    Per element: the weight gradient ``M * gW_eff * g`` and the score gradient ``M * gW_eff * W * g * (1 - g)`` with
    ``M = 1[g > 1/2]``; Adam on ``W`` in the same pass (``torch.optim.Adam``'s single-tensor arithmetic, in place on
    ``W``, ``m`` and ``v``); then the score gradient is reduced into the embedding gradients. ``gn2`` is the squared
    norm of the weight gradient per group. Nothing of size ``W`` is written out between the two phases.

The kernels compute the same mathematics as the autograd form of :class:`csfrbench.stack.Stack` (float rounding
aside) with two guards that whole-budget runs need: the score is floored at -80 before the exponential (the gate of a
pair that far apart is 1.8e-35, as good as 0, and the exponential stays finite), and a second moment below the
smallest normal float is taken as 0 before the square root. :func:`ops` returns ``None`` where the extension cannot be
built (no C++ compiler, or no ``ninja``); the stack then uses its PyTorch form.
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
import sysconfig
import warnings

_SOURCE = r"""
#include <torch/extension.h>
#include <cmath>
#include <vector>

std::vector<torch::Tensor> gate_forward(torch::Tensor W, torch::Tensor recv, torch::Tensor send, double R2) {
    TORCH_CHECK(W.is_contiguous() && recv.is_contiguous() && send.is_contiguous(), "contiguous tensors expected");
    TORCH_CHECK(W.scalar_type() == torch::kFloat32, "float32 expected");
    const int64_t G = W.size(0), P = W.size(1), Q = W.size(2), R = recv.size(2);
    TORCH_CHECK(R <= 8 && P <= 128 && Q <= 128, "layers of at most 128 units and embeddings of at most 8 dimensions");
    auto g = torch::empty_like(W);
    auto We = torch::empty_like(W);
    const float* wp = W.data_ptr<float>(); const float* rc = recv.data_ptr<float>(); const float* sd = send.data_ptr<float>();
    float* gp = g.data_ptr<float>(); float* ep = We.data_ptr<float>();
    const float fR2 = (float)R2;
    float sT[8 * 128], sn[128], srow[128];
    for (int64_t gi = 0; gi < G; ++gi) {
        const float* sdg = sd + gi * Q * R;
        for (int64_t j = 0; j < Q; ++j) { float a = 0.f; for (int64_t k = 0; k < R; ++k) { const float x = sdg[j * R + k]; sT[k * Q + j] = x; a += x * x; } sn[j] = a; }
        for (int64_t i = 0; i < P; ++i) {
            const float* ri = rc + gi * P * R + i * R;
            float rn = 0.f; for (int64_t k = 0; k < R; ++k) rn += ri[k] * ri[k];
            for (int64_t j = 0; j < Q; ++j) srow[j] = fR2 - rn - sn[j];
            for (int64_t k = 0; k < R; ++k) { const float rk = 2.0f * ri[k]; const float* st = sT + k * Q; for (int64_t j = 0; j < Q; ++j) srow[j] += rk * st[j]; }
            const int64_t base = gi * P * Q + i * Q;
            for (int64_t j = 0; j < Q; ++j) {
                const float sij = srow[j];
                const float z = sij > -80.0f ? sij : -80.0f;
                const float gij = 1.0f / (1.0f + expf(-z));
                gp[base + j] = gij;
                ep[base + j] = sij > 0.0f ? wp[base + j] * gij : 0.0f;
            }
        }
    }
    return {g, We};
}

std::vector<torch::Tensor> gate_backward_adam(torch::Tensor gWe, torch::Tensor W, torch::Tensor g, torch::Tensor recv, torch::Tensor send,
                                              torch::Tensor m, torch::Tensor v, torch::Tensor step_size, double bc2_sqrt, double b1, double b2, double eps) {
    TORCH_CHECK(gWe.is_contiguous() && W.is_contiguous() && g.is_contiguous() && recv.is_contiguous() && send.is_contiguous()
                && m.is_contiguous() && v.is_contiguous() && step_size.is_contiguous(), "contiguous tensors expected");
    const int64_t G = W.size(0), P = W.size(1), Q = W.size(2), R = recv.size(2);
    TORCH_CHECK(R <= 8 && P <= 128 && Q <= 128, "layers of at most 128 units and embeddings of at most 8 dimensions");
    auto grad_recv = torch::empty_like(recv);
    auto grad_send = torch::empty_like(send);
    auto gn2 = torch::empty({G}, W.options());
    const float* gwe = gWe.data_ptr<float>(); float* w = W.data_ptr<float>(); const float* gg = g.data_ptr<float>();
    const float* rc = recv.data_ptr<float>(); const float* sd = send.data_ptr<float>();
    float* mp = m.data_ptr<float>(); float* vp = v.data_ptr<float>(); const float* ss = step_size.data_ptr<float>();
    float* grc = grad_recv.data_ptr<float>(); float* gsd = grad_send.data_ptr<float>(); float* gn = gn2.data_ptr<float>();
    const float one_b1 = 1.0f - (float)b1, fb2 = (float)b2, one_b2 = 1.0f - (float)b2, inv_bc2 = 1.0f / (float)bc2_sqrt, feps = (float)eps;
    float gs[128 * 128], rs[128], cs[128], accr[128 * 8], accs[128 * 8], sT[8 * 128];
    for (int64_t gi = 0; gi < G; ++gi) {
        const int64_t base = gi * P * Q, rb = gi * P * R, sb = gi * Q * R;
        const float step = ss[gi];
        float gsum = 0.f;
        for (int64_t i = 0; i < P; ++i) {
            const int64_t o = base + i * Q;
            float* gsi = gs + i * Q;
            for (int64_t j = 0; j < Q; ++j) {
                const float gij = gg[o + j];
                const bool open = gij > 0.5f;
                const float gw = open ? gwe[o + j] * gij : 0.0f;
                gsi[j] = open ? gwe[o + j] * w[o + j] * gij * (1.0f - gij) : 0.0f;
                float mm = mp[o + j]; mm = mm + one_b1 * (gw - mm); mp[o + j] = mm;
                float vv = vp[o + j] * fb2 + one_b2 * gw * gw; vp[o + j] = vv;
                const float vs = vv >= 1.17549435e-38f ? sqrtf(vv) : 0.0f;
                w[o + j] = w[o + j] - step * (mm / (vs * inv_bc2 + feps));
                gsum += gw * gw;
            }
        }
        gn[gi] = gsum;
        for (int64_t j = 0; j < Q; ++j) { cs[j] = 0.f; for (int64_t k = 0; k < R; ++k) { sT[k * Q + j] = sd[sb + j * R + k]; accs[j * R + k] = 0.f; } }
        for (int64_t i = 0; i < P; ++i) {
            const float* gsi = gs + i * Q;
            float r = 0.f; for (int64_t j = 0; j < Q; ++j) { r += gsi[j]; cs[j] += gsi[j]; }
            rs[i] = r;
            for (int64_t k = 0; k < R; ++k) { const float* st = sT + k * Q; float a = 0.f; for (int64_t j = 0; j < Q; ++j) a += gsi[j] * st[j]; accr[i * R + k] = a; }
        }
        for (int64_t k = 0; k < R; ++k) {
            float col[128]; for (int64_t j = 0; j < Q; ++j) col[j] = 0.f;
            for (int64_t i = 0; i < P; ++i) { const float rik = rc[rb + i * R + k]; const float* gsi = gs + i * Q; for (int64_t j = 0; j < Q; ++j) col[j] += gsi[j] * rik; }
            for (int64_t j = 0; j < Q; ++j) accs[j * R + k] = col[j];
        }
        for (int64_t i = 0; i < P; ++i) for (int64_t k = 0; k < R; ++k) grc[rb + i * R + k] = -2.0f * (rs[i] * rc[rb + i * R + k] - accr[i * R + k]);
        for (int64_t j = 0; j < Q; ++j) for (int64_t k = 0; k < R; ++k) gsd[sb + j * R + k] = -2.0f * (cs[j] * sd[sb + j * R + k] - accs[j * R + k]);
    }
    return {grad_recv, grad_send, gn2};
}
"""

_MODULE = None
_TRIED = False


def _flags() -> tuple[str, list[str]]:
    """Compiler flags: vectorized exponentials and roots, exact divisions (AVX2 on x86-64 when the processor has it)."""
    base = ["-O3", "-ffast-math", "-fno-reciprocal-math"]
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        try:
            import torch
            capability = torch.backends.cpu.get_cpu_capability()
        except Exception:  # noqa: BLE001
            capability = "DEFAULT"
        if capability in ("AVX2", "AVX512"):
            return "x86v3", base + ["-march=x86-64-v3"]
        return "x86", base
    return machine or "generic", base


def _find_ninja() -> None:
    """Put the ``ninja`` binary of the ``ninja`` wheel on ``PATH``, also when the environment is not activated."""
    if shutil.which("ninja"):
        return
    places = [os.path.dirname(sys.executable), sysconfig.get_path("scripts")]
    try:
        import ninja
        places += [getattr(ninja, "BIN_DIR", "") or "",
                   os.path.join(os.path.dirname(os.path.dirname(ninja.__file__)), "bin")]
    except ImportError:
        pass
    for place in places:
        if place and os.path.isfile(os.path.join(place, "ninja")):
            os.environ["PATH"] = place + os.pathsep + os.environ.get("PATH", "")
            return


def ops():
    """The compiled kernels, or ``None`` when they cannot be built here (the stack then uses PyTorch)."""
    global _MODULE, _TRIED
    if _TRIED:
        return _MODULE
    _TRIED = True
    if os.environ.get("CSFRBENCH_NO_KERNELS"):
        return None
    try:
        from torch.utils.cpp_extension import load_inline
        _find_ninja()
        tag, flags = _flags()
        _MODULE = load_inline(name=f"csfrbench_lewi_gate_v1_{tag}", cpp_sources=_SOURCE,
                              functions=["gate_forward", "gate_backward_adam"], extra_cflags=flags, verbose=False)
    except Exception as exc:  # noqa: BLE001  no compiler, no ninja, or a platform without these flags
        warnings.warn(f"the fused LeWi kernels could not be built ({type(exc).__name__}: {exc}); "
                      "the stack uses its PyTorch form, which computes the same step more slowly", stacklevel=2)
        _MODULE = None
    return _MODULE
