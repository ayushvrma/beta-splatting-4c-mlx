"""Diagnostic: measure peak memory of one training step at multiple resolutions.

Useful to figure out what dominates the peak so we can size chunk_size / tile
size correctly for higher-res runs. Not a unit test — a one-off probe.

Usage:
    python -m mlx_impl.tests.probe_memory --ply out/lego_v5.ply \
        --resolutions 100 128 150 200 --chunk-sizes 256 128 64
"""

from __future__ import annotations
import argparse
import time
import numpy as np
import mlx.core as mx

from mlx_impl.beta_model import BetaModel
from mlx_impl.dataset import load_nerf_synthetic
from mlx_impl.densification import apply_position_noise, regularization_loss
from mlx_impl.losses import ssim
from mlx_impl.train import render_view


def one_step(model: BetaModel, cam, background: mx.array, chunk_size: int) -> float:
    """Run one forward+backward and return peak MB observed during it."""
    def loss_fn(params):
        model.set_parameters(params)
        image, _, _ = render_view(model, cam, background, chunk_size=chunk_size)
        gt = cam.image
        l1 = mx.abs(image - gt).mean()
        img_nchw = mx.transpose(image[None], (0, 3, 1, 2))
        gt_nchw = mx.transpose(gt[None], (0, 3, 1, 2))
        s = ssim(img_nchw, gt_nchw)
        return 0.8 * l1 + 0.2 * (1.0 - s)

    grad_fn = mx.value_and_grad(loss_fn)
    mx.reset_peak_memory()
    t0 = time.time()
    loss, grads = grad_fn(model.parameters())
    # Force materialization so the peak reflects the actual work.
    _ = float(loss)
    for g in grads.values():
        if g.size > 0:
            _ = float(g.sum())
    elapsed = time.time() - t0
    peak_mb = mx.get_peak_memory() / 1e6
    return peak_mb, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ply", required=True)
    p.add_argument("--scene", default="lego")
    p.add_argument("--resolutions", type=int, nargs="+",
                   default=[100, 128, 150, 200],
                   help="Image sizes to sweep (square).")
    p.add_argument("--chunk-sizes", type=int, nargs="+",
                   default=[256, 128, 64])
    p.add_argument("--sh-degree", type=int, default=0)
    p.add_argument("--sb-number", type=int, default=2)
    args = p.parse_args()

    model = BetaModel(sh_degree=args.sh_degree, sb_number=args.sb_number,
                      color_channels=3)
    model.load_ply(args.ply)
    n_prim = int(model._xyz.shape[0])
    print(f"[model] {n_prim} primitives")
    bg = mx.array([1.0, 1.0, 1.0])

    print("")
    print(f"{'resolution':>10}  {'chunk':>6}  {'peak_MB':>10}  {'step_s':>7}")
    print("-" * 42)
    for res in args.resolutions:
        # Recompute downscale: NeRF-synthetic base resolution is 800.
        downscale = max(1, 800 // res)
        cams = load_nerf_synthetic(args.scene, split="train",
                                   downscale=downscale, white_bg=True,
                                   max_cameras=1)
        cam = cams[0]
        actual_res = cam.width
        for chunk in args.chunk_sizes:
            try:
                peak_mb, elapsed = one_step(model, cam, bg, chunk_size=chunk)
                print(f"{actual_res:>10}  {chunk:>6}  {peak_mb:>10.0f}  {elapsed:>7.2f}")
            except Exception as e:
                print(f"{actual_res:>10}  {chunk:>6}  {'FAIL':>10}  {str(e)[:30]}")


if __name__ == "__main__":
    main()
