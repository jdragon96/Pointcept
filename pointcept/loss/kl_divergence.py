import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- loss
def kl_divergence(p_logits, q_logits, dim=1, temperature=1.0, mask=None, reduction="mean"):
    """KL(P || Q) = sum_c p_c * (log p_c - log q_c), with P, Q = softmax(logits / T) along `dim`.

    P is the target (teacher), Q the prediction pulled toward it. Any shape works --
    (N, C), (B, C, N), (B, C, H, W): every position off `dim` is one distribution.
    `mask` has the logits shape minus `dim` and keeps valid positions (e.g. drops padded points).
    reduction: "mean" over positions (== F.kl_div "batchmean" for 2-D input), "sum" or "none".
    Logits must be finite (a -inf logit gives 0 * inf = nan).
    """
    log_p = F.log_softmax(p_logits / temperature, dim)
    log_q = F.log_softmax(q_logits / temperature, dim)
    kl = (log_p.exp() * (log_p - log_q)).sum(dim)
    if mask is not None:
        kl = kl[mask]
    if reduction == "none":
        return kl
    if reduction == "sum":
        return kl.sum()
    return kl.mean() if kl.numel() else kl.sum() * 0


class KLDivergenceLoss(nn.Module):
    """Distillation loss T^2 * KL(teacher || student) (Hinton et al. 2015).

    The T^2 factor cancels the 1/T^2 that softening puts on the gradient, so the loss
    weight does not have to be retuned when T changes. The teacher is detached.
    """

    def __init__(self, temperature=1.0, dim=1, reduction="mean"):
        super().__init__()
        self.temperature, self.dim, self.reduction = temperature, dim, reduction

    def forward(self, student_logits, teacher_logits, mask=None):
        kl = kl_divergence(teacher_logits.detach(), student_logits, self.dim,
                           self.temperature, mask, self.reduction)
        return self.temperature ** 2 * kl


# ----------------------------------------------------------------- checks
def gaussian_kl(mu1, s1, mu2, s2):
    """Closed form KL(N(mu1, s1^2) || N(mu2, s2^2))."""
    return math.log(s2 / s1) + (s1 ** 2 + (mu1 - mu2) ** 2) / (2 * s2 ** 2) - 0.5


def gaussian_log_pdf(x, mu, s):
    return -0.5 * ((x - mu) / s) ** 2 - math.log(s * math.sqrt(2 * math.pi))


def self_check():
    torch.manual_seed(0)
    # (N, C) against torch's own kl_div
    p, q = torch.randn(512, 17), torch.randn(512, 17)
    ref = F.kl_div(q.log_softmax(1), p.log_softmax(1), reduction="batchmean", log_target=True)
    ours = kl_divergence(p, q)
    print(f"(N, C)       ours={ours:.6f}  F.kl_div={ref:.6f}")
    assert torch.allclose(ours, ref, atol=1e-6)

    # (B, C, N) with a padding mask
    p, q = torch.randn(2, 17, 300), torch.randn(2, 17, 300)
    valid = torch.zeros(2, 300, dtype=torch.bool)
    valid[0, :300], valid[1, :180] = True, True
    flat = lambda t: t.permute(0, 2, 1)[valid]                        # -> (n_valid, C)
    ref = F.kl_div(flat(q).log_softmax(1), flat(p).log_softmax(1),
                   reduction="batchmean", log_target=True)
    ours = kl_divergence(p, q, mask=valid)
    print(f"(B, C, N)    ours={ours:.6f}  F.kl_div={ref:.6f}  (masked)")
    assert torch.allclose(ours, ref, atol=1e-6)

    # identical distributions -> 0, and KL >= 0 everywhere
    assert kl_divergence(p, p).abs() < 1e-6
    assert (kl_divergence(p, q, reduction="none") >= -1e-6).all()

    # discretised Gaussians against the closed form (softmax of log-pdf = normalised density)
    x = torch.linspace(-15, 15, 20001, dtype=torch.float64)
    lp, lq = gaussian_log_pdf(x, 0, 1), gaussian_log_pdf(x, 1, 2)
    print(f"Gaussian     KL(P||Q)={kl_divergence(lp, lq, dim=0):.6f}  "
          f"closed={gaussian_kl(0, 1, 1, 2):.6f}")
    print(f"             KL(Q||P)={kl_divergence(lq, lp, dim=0):.6f}  "
          f"closed={gaussian_kl(1, 2, 0, 1):.6f}")
    assert abs(kl_divergence(lp, lq, dim=0) - gaussian_kl(0, 1, 1, 2)) < 1e-6
    print("self-check ok")


# ----------------------------------------------------------------- plot
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def _legend(ax, **kw):
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, **kw)


def plot_kl_divergence(save_path=None, show=False):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    mu_p, s_p, mu_q, s_q = 0.0, 1.0, 1.0, 2.0
    x = torch.linspace(-15, 15, 20001, dtype=torch.float64)
    lp, lq = gaussian_log_pdf(x, mu_p, s_p), gaussian_log_pdf(x, mu_q, s_q)
    p, q = lp.exp(), lq.exp()
    kl_pq = kl_divergence(lp, lq, dim=0).item()
    kl_qp = kl_divergence(lq, lp, dim=0).item()
    view = (x > -5) & (x < 8)
    xv = x[view].numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), facecolor=SURFACE)
    fig.suptitle("KL divergence: asymmetry and distillation temperature", x=0.06, ha="left",
                 color=INK, fontsize=14, fontweight="bold")

    # (a) the two distributions
    ax = axes[0, 0]
    _style(ax, "Two distributions", "x", "density")
    ax.plot(xv, p[view], color=BLUE, lw=2, label=f"P = N({mu_p:g}, {s_p:g}²)")
    ax.plot(xv, q[view], color=ORANGE, lw=2, label=f"Q = N({mu_q:g}, {s_q:g}²)")
    ax.annotate("P", (0.0, p.max().item()), xytext=(8, -4), textcoords="offset points",
                color=INK, fontsize=10, fontweight="bold")
    ax.annotate("Q", (3.4, math.exp(gaussian_log_pdf(3.4, mu_q, s_q))), xytext=(4, 4),
                textcoords="offset points", color=INK, fontsize=10, fontweight="bold")
    ax.set_ylim(bottom=0)
    _legend(ax, loc="upper right")

    # (b) where each direction pays: the integrand, area under it = KL
    ax = axes[0, 1]
    _style(ax, "Pointwise contribution (area = KL)", "x", "integrand")
    f_pq, f_qp = (p * (lp - lq))[view], (q * (lq - lp))[view]
    ax.axhline(0, color=AXIS, lw=1)
    ax.fill_between(xv, 0, f_pq, color=BLUE, alpha=0.18, lw=0)
    ax.fill_between(xv, 0, f_qp, color=ORANGE, alpha=0.18, lw=0)
    ax.plot(xv, f_pq, color=BLUE, lw=2, label=f"p·log(p/q)   KL(P‖Q) = {kl_pq:.3f}")
    ax.plot(xv, f_qp, color=ORANGE, lw=2, label=f"q·log(q/p)   KL(Q‖P) = {kl_qp:.3f}")
    ax.text(0.98, 0.04, "negative pieces exist,\nthe total is always ≥ 0", transform=ax.transAxes,
            ha="right", va="bottom", color=MUTED, fontsize=8)
    _legend(ax, loc="upper left")

    # (c) sweep the mean of Q: forward vs reverse, our implementation vs closed form
    ax = axes[1, 0]
    _style(ax, "Shifting Q's mean", "μ_Q − μ_P", "KL (nats)")
    shifts = torch.linspace(-4, 4, 161, dtype=torch.float64)
    fwd, rev = [], []
    for d in shifts.tolist():
        lq_d = gaussian_log_pdf(x, d, s_q)
        fwd.append(kl_divergence(lp, lq_d, dim=0).item())
        rev.append(kl_divergence(lq_d, lp, dim=0).item())
    ax.plot(shifts, fwd, color=BLUE, lw=2, label="KL(P‖Q)  forward")
    ax.plot(shifts, rev, color=ORANGE, lw=2, label="KL(Q‖P)  reverse")
    marks = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    ax.plot(marks, [gaussian_kl(0, s_p, d, s_q) for d in marks], "o", ms=6, mfc=SURFACE,
            mec=INK2, mew=1.2, ls="none", label="closed form")
    ax.plot(marks, [gaussian_kl(d, s_q, 0, s_p) for d in marks], "o", ms=6, mfc=SURFACE,
            mec=INK2, mew=1.2, ls="none")
    i = 30                                                             # shift = -2.5
    for name, ys in (("reverse", rev), ("forward", fwd)):
        ax.annotate(name, (shifts[i], ys[i]), xytext=(10, 4), textcoords="offset points",
                    ha="left", va="bottom", color=INK, fontsize=9)
    ax.set_ylim(bottom=0)
    _legend(ax, loc="upper center")

    # (d) distillation: why the loss is scaled by T^2
    ax = axes[1, 1]
    _style(ax, "Distillation: gradient size vs temperature", "temperature T",
           "‖∂loss/∂student logits‖")
    torch.manual_seed(0)
    teacher = torch.randn(4096, 17) * 3
    student0 = teacher + torch.randn(4096, 17) * 2
    temps = torch.logspace(-0.3, 1.3, 25).tolist()
    g_scaled, g_plain = [], []
    for t in temps:
        for scaled, out in ((True, g_scaled), (False, g_plain)):
            s = student0.clone().requires_grad_(True)
            loss = KLDivergenceLoss(t)(s, teacher)
            if not scaled:
                loss = loss / t ** 2
            loss.backward()
            out.append(s.grad.norm().item())
    ax.plot(temps, g_scaled, color=BLUE, lw=2, label="T²·KL  (KLDivergenceLoss)")
    ax.plot(temps, g_plain, color=ORANGE, lw=2, label="KL only")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.annotate("T²·KL", (temps[-1], g_scaled[-1]), xytext=(-4, 6), textcoords="offset points",
                ha="right", va="bottom", color=INK, fontsize=9)
    ax.annotate("KL only  ∝ 1/T²", (temps[17], g_plain[17]), xytext=(8, 4),
                textcoords="offset points", ha="left", va="bottom", color=INK, fontsize=9)
    _legend(ax, loc="lower left")

    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.5, w_pad=2.5)
    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=SURFACE)
        print(f"saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_suffix(".png")))
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    self_check()
    plot_kl_divergence(a.out, a.show)
