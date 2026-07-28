# project_journal.md

Definitive one-page status. Longer dated log in `journal.md`; deep design docs elsewhere in `docs/`. Update this when a phase changes state.

## Why

Local Deformable Beta Splatting (DBS) on Apple Silicon, extended to 4-channel RGBA, with inference optimized for real-time. Motivations: kill Colab dependency; give alpha the same view-dependent freedom SB gives RGB; ship a viable renderer, not just paper numbers.

## Status (2026-07-28)

**MLX work migrated to `paper-implenentations/beta-splatting-mlx/`** (nested checkout at `./beta-splatting-mlx/`, gitignored). New pushes go to `https://github.com/ayushvrma/paper-implenentations`. This repo retains the CUDA-side fork; the MLX subtree is duplicated here for now (deferred deletion — running processes still import it).

Branch: `feat/rgba-inference`, pushed to `origin/feat/rgba-inference`. **Track A works end-to-end and converges to good quality.** MLX-DBS on lego at 100×100 hits **PSNR 32.42 dB on training views** and **PSNR 30.77 dB / SSIM 0.9709 on the 200-view held-out test split** (`lego_v5.ply`, 3827 primitives, 1000 densify + 8000 refine total). Real 3D on-device, no CUDA. Total wall time from cold start to v5: ~33 min on M4 Pro.

## Done

- **Docs + memory bootstrapped.** `docs/mlx_port_plan.md` is the source-of-truth plan; `docs/rgba_extension.md` locks Design A2 (alpha = 4th SB intensity, multiplicative with geometric opacity). Memory tracks user profile + Apple Silicon preference + project state.
- **Apple Silicon landscape verified.** msplat, gsplat-mps, splat-apple, MetalSplatter, RadianceKit exist and were evaluated. None cover DBS's Beta kernel / SB color / MCMC ops — port must be written.
- **MLX scaffold (`mlx_impl/`, 684 lines) runs on M4 Pro + Metal.** `BetaModel` state + activations + PLY I/O, pure-MLX SB and SH forward, MLX SSIM/L1/PSNR — all channel-agnostic from line 1.
- **Phase-1 parity tests pass on-device.** SB max_abs 5e-8 vs numpy ref (C=3 and C=4). SH bit-exact across degrees 0–3, C ∈ {3, 4}. Losses bit-exact. PLY save/load roundtrip < 1e-6.
- **Real 3D on-device via msplat (baseline, not DBS).** Trained lego in 31 s, 7000 iters, 51,907 splats, PSNR 25.09 / SSIM 0.9174 → `lego_ns_7k.ply`. Confirms M4 Pro can carry DBS-scale loads and gives a numerical reference point.
- **MCMC densification ported to MLX.** `relocate_gs`, `add_new_gs`, `_sample_alives`, `_update_params`, position-noise term, opacity/scale regularizers. Custom `MutableAdam` optimizer whose state grows/shrinks with the parameter tensors. 8/8 standalone tests pass, including the (1-new_op)^(ratio+1) = 1-old_op invariance. This is what makes DBS actually grow toward `cap_max` instead of pruning-only like msplat.
- **Geometry pipeline in MLX.** `build_covariance_3d`, `world_to_cam`, EWA `persp_proj`, `add_blur_and_invert`, `project()`, plus `beta_alpha` (bounded-support Beta kernel per-pixel eval). 5/5 tests pass.
- **Track A soft rasterizer.** Pure MLX, chunked front-to-back compositing, differentiable via autograd — no custom vjp. 5/5 tests pass including gradient flow through all inputs.
- **Dataset loader + training loop.** NeRF-synthetic → MLX cameras with OpenGL→OpenCV axis flip handled. Full training loop wires model + render + loss + reg + Adam + MCMC densify.
- **First working end-to-end DBS training on Apple Silicon.** 1000 iters on lego at 100×100: loss 0.77 → 0.03, primitives 3000 → 5650, **PSNR 27.34 dB on training views, 4 min wall time, 21 GB peak memory.** Rendered output visually recognizable — see `out/lego_1k_view5.png`.
- **Checkpointed rasterizer + periodic saves + `--resume`.** `mx.checkpoint` in the rasterizer drops active memory during training from 1.6 GB → 15 MB. Periodic PLY saves via `--save-every` rescue mid-run OOMs. `--resume` reloads a PLY for continued training. Rendered `out/lego_v4_final_view5.png`.
- **Resume + refinement-only strategy → PSNR 30.34.** After densify-heavy runs kept OOMing at ~5,650 primitives, switched to: densify to 3,827 primitives, save, then resume and refine 3,000 more iters with no growth. Peak memory plateaus at 15 GB (safe). Final: **mean PSNR 30.34 dB on 20 training views, best single-view 31.25 dB.**
- **Held-out test-set eval landed.** New `mlx_impl/eval/held_out.py` renders every camera in `transforms_test.json`, reports per-view + summary PSNR/SSIM, dumps JSON, and saves worst-K side-by-sides. For 4c models: also reports RGB-only PSNR and alpha-MSE. Results (lego, 200 views @ 100×100): `lego_v4.ply` **28.57 / 0.9553**, `lego_v5.ply` **30.77 / 0.9709**, `lego_1k.ply` **25.98 / 0.9294**.
- **Extended refinement run (v5) → PSNR 30.77 on test.** Resumed from v4, ran 5000 more iters at 100×100 (no densification). Total refinement now 8000 iters. Trajectory: iter2000 → 29.99, iter3000 → 30.31, iter5000 → 30.77. Diminishing returns after iter 3000. Wall time 17.6 min. **NaN loss appeared around iter 3800** but the model kept improving — params stayed finite. Root cause not investigated yet; a numerical guard in SSIM or the beta activation is the likely fix.
- **4c RGBA plumbing wired + smoke-tested.** `--color-channels {3,4}` in `train.py` and `eval/held_out.py`. Dataset returns HWC-4 raw RGBA for 4c (no bg-composite), HWC-3 composited for 3c. `mlx_impl/tests/test_rgba.py` covers shape/forward/backward through L1 + SSIM at C=4. Passes 3/3.
- **Repo split.** MLX work moved to `paper-implenentations/beta-splatting-mlx/`. Old repo keeps the CUDA fork.

## Doing

Nothing in flight — v5 is the current best, waiting to run 4c training on lego as the next experiment.

## Next (in order)

1. **First 4-channel training run on lego.** Plumbing is wired + smoke-tested. Same recipe as v4/v5 (1k densify + 3k refine, 100×100), then held-out eval reports per-channel breakdown. Success = RGB PSNR within 0.3 dB of the 3c baseline (30.77) and low alpha-MSE.
2. **Investigate the NaN loss at iter ~3800.** Cheap diagnostic — most likely SSIM division-by-zero on a converged region or the beta activation saturating. Add a numerical guard.
3. **Pixel-tile the Track A rasterizer.** 200×200 blew peak to 60 GB (vs 15 GB at 100×100) — non-linear memory scaling with H×W. Blocks all higher-res runs. Design: process H×W grid in tiles, checkpoint each tile's backward. Also profile to figure out what actually dominates the peak.
4. **LPIPS on the held-out matrix.** PSNR/SSIM done; LPIPS via lpipsPyTorch is torch-based — pin PyTorch for eval-only, or reimplement LPIPS-Alex in MLX.
5. **Track B: Metal-shader rasterizer + hand-written vjp (~5–7 weeks).** Custom Metal via `mx.fast.metal_kernel` + `@mx.custom_function`. Track A stays as reference implementation. This is what unlocks paper-scale training on-device.
6. **Inference optimizations.** Ranked list in `docs/inference_optimization.md`. Top: frustum culling audit, occlusion-ε tuning, PLAS-sorted PLYs, lobe pruning.

## Rejected / parked

- **Fork msplat.** Would inherit its 3DGS-vanilla design and permanent merge burden. Also license-adjacent friction.
- **Upstream DBS to msplat.** Personal research repo, unlikely to absorb the scope change.
- **Design A1 (alpha replaces geometric opacity).** Breaks DBS's opacity-only MCMC invariance proof.
- **Design C (per-primitive scalar alpha).** No view dependence — reduces to a second opacity.
- **CUDA/Colab path as primary.** Pivoted 2026-07-10; kept as fallback in `implementation_plan.md`.

## References

- Full plan: `docs/mlx_port_plan.md`
- Long log: `docs/journal.md`
- 4c design: `docs/rgba_extension.md`
- Method notes: `docs/method_dbs.md`
- Apple Silicon catalog: `docs/apple_silicon_options.md`
- Inference plan: `docs/inference_optimization.md`
