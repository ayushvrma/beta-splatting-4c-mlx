# beta-splatting-mlx

MLX/Metal port of **Deformable Beta Splatting (DBS)** for Apple Silicon, with a research direction extending color to 4-channel RGBA (see `docs/rgba_extension.md`).

## What this is

- Pure-MLX implementation of DBS training + inference on Apple Silicon (M-series). No CUDA, no PyTorch, no Colab.
- Reference "Track A" soft rasterizer (differentiable via MLX autograd, correctness-first).
- MCMC densification, MutableAdam, checkpointed rasterizer, PLY I/O compatible with the CUDA reference implementation.
- Research fork extending DBS color to 4 channels (RGBA), with alpha treated as a view-dependent SB intensity.

## Status (2026-07-28)

Track A trains end-to-end on lego at 100×100:

| PLY | Primitives | Total iters | Train PSNR | **Test PSNR** | **Test SSIM** |
|---|---|---|---|---|---|
| `artifacts/lego_v4.ply` | 3827 | 4000 (1k densify + 3k refine) | 30.34 | **28.57** | 0.9553 |

Real 3D reconstruction, on-device, no cloud. 15 min end-to-end training on M4 Pro / 24 GB.

Higher resolutions (200×200+) need pixel-tiling in the rasterizer — currently blows the 24 GB memory budget without it. That work is next.

## Layout

```
beta-splatting-mlx/
├── mlx_impl/            # the port itself
│   ├── beta_model.py    # BetaModel state + activations + PLY I/O
│   ├── color/           # SB + SH forwards
│   ├── geometry/        # projection, EWA, Beta kernel eval
│   ├── rasterizer/      # Track A pure-MLX soft rasterizer
│   ├── densification.py # MCMC relocate/add_new + noise term
│   ├── optimizer.py     # MutableAdam (state grows/shrinks with params)
│   ├── dataset.py       # NeRF-synthetic loader (OpenGL→OpenCV axis flip)
│   ├── losses.py        # L1, SSIM, PSNR — all channel-agnostic
│   ├── train.py         # training loop with --resume, --save-every, checkpointed rast
│   ├── eval/            # held-out test-set eval on transforms_test.json
│   └── tests/           # standalone unit tests (per-module)
├── docs/                # research documentation
│   ├── project_journal.md   # one-page status — start here
│   ├── journal.md           # dated long log
│   ├── mlx_port_plan.md     # source-of-truth port plan (phases)
│   ├── rgba_extension.md    # 4-channel research direction (Design A2)
│   ├── method_dbs.md        # DBS method notes
│   ├── apple_silicon_options.md
│   └── inference_optimization.md
├── artifacts/           # reference trained model + one sample render
├── eval/                # per-view PSNR/SSIM JSON summaries
└── requirements.txt
```

## Getting started

macOS 14+, Apple Silicon, ~24 GB unified memory recommended.

```bash
# 1. Xcode Command Line Tools for Metal (~1 GB, one-time)
xcode-select --install

# 2. Fresh venv (Python 3.11 or 3.12)
python3.12 -m venv .venv-mlx
source .venv-mlx/bin/activate

# 3. Deps
pip install -r requirements.txt

# 4. Metal sanity check
python -c "import mlx.core as mx; print('metal:', mx.metal.is_available())"

# 5. Fetch a NeRF-synthetic scene (lego)
#    Download from https://www.matthewtancik.com/nerf and unzip so that
#    ./lego/{transforms_train.json, transforms_test.json, train/, test/}
#    all exist relative to this repo root.

# 6. Unit tests
python -m mlx_impl.tests.test_sb_forward
python -m mlx_impl.tests.test_sh_forward
python -m mlx_impl.tests.test_geometry
python -m mlx_impl.tests.test_rasterizer
python -m mlx_impl.tests.test_densification
python -m mlx_impl.tests.test_losses_and_model
```

## Train

```bash
# Densification phase (1000 iters at 100x100)
python -m mlx_impl.train --scene lego --downscale 8 \
    --iterations 1000 --cap-max 6000 \
    --save-every 500 --output out/lego.ply

# Refinement phase (3000 more iters, no growth)
python -m mlx_impl.train --scene lego --downscale 8 \
    --iterations 3000 \
    --densify-from 999999 --densify-until 999999 \
    --resume out/lego_iter1000.ply \
    --output out/lego_final.ply --eval
```

Expected: mean PSNR ~30 dB on training views, ~28.5 dB on test views, ~15 min end-to-end on M4 Pro.

## Evaluate on held-out test split

```bash
python -m mlx_impl.eval.held_out \
    --ply artifacts/lego_v4.ply \
    --scene lego --split test --downscale 8 \
    --out-json eval/my_result.json \
    --save-worst 3
```

## Design notes

- **Not a PyTorch shim.** MLX types directly (`mx.array`), no wrapper. Autograd behaves; no double-conversion.
- **Two rasterizer designs.** Track A (`rasterizer/slow.py`) is pure MLX, differentiable via autograd, correctness-first. Track B (Metal shaders + hand-written vjp) is planned for real throughput.
- **Channel-agnostic from line 1.** Model, rasterizer, and losses all handle `C ∈ {3, 4}` without preprocessor branches — the RGBA research direction (`docs/rgba_extension.md`) can be enabled by a flag once wired through the training script.
- **MCMC densification preserved.** Direct port of the DBS relocate/add_new invariance `(1 - new)^(ratio+1) = 1 - old`.
- **Checkpointed rasterizer.** `mx.checkpoint` inside the per-chunk step drops rasterizer active memory 100×, enabling refinement runs at moderate primitive counts on 24 GB.

## References

- Deformable Beta Splatting — Rong Liu et al., 2024 (CUDA reference: https://github.com/RongLiu-Leo/beta-splatting)
- MLX — Apple's array/ML framework: https://github.com/ml-explore/mlx
- 3D Gaussian Splatting — Kerbl et al., 2023

## License

Same terms as the DBS reference implementation. See `LICENSE_GAUSSIAN_SPLATTING.md` in the upstream repo.
