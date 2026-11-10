# Verification record

This file records actual checks. It does not report research measurements.

## Current machine audit: 2026-11-10 (Windows)

- Windows 11 build 26200, Intel Core i5-13420H (12 logical processors),
  16 GB RAM, Intel UHD integrated graphics, and about 352 GB free on C:.
  No NVIDIA device or `nvidia-smi` is available; CUDA and Apple MPS are
  unavailable on this machine.
- Python 3.11.0 is installed. The initial Python environment had no PyTorch,
  pytest, Diffusers, PEFT, TorchMetrics, or OpenCLIP. A new `.venv` was installed
  from `pyproject.toml`; it has PyTorch 2.6.0+cpu and Diffusers 0.35.2.
  Runtime checks confirmed `torch.cuda.is_available() == False` and MPS false.
- The fork is clean at `main` before edits. `origin` points to
  `palaksingh-git/Erasure`; `upstream` points to `dhruvvv-sh/palak_proj`.
- Repository audit confirms EEBshallow and UCE cross-attention K/V edits,
  EEBdeep LoRA self-distillation, ESD negative-guidance training, checkpointed
  orchestration, and all five metric formulas are present. EEBdeep uses ten
  templates rather than the paper's 80 and omits object retention loss as
  documented in `methodology.md`; this is not an exact paper reproduction.
- Fixed CPU model loading so a configured fp16 checkpoint variant does not
  force fp16 arithmetic on CPU. Added configuration checks to prevent stale
  checkpoint, image, and metric reuse. These checks reject old checkpoints
  without `training.json` until they are inspected; no old data are deleted.
- Python compilation and `git diff --check` passed after the edits.
  The final local test run passed: `6 passed, 1 warning` (an upstream
  NumPy deprecation warning from Accelerate).
- The SD 1.4 `fp16` variant downloaded from the specified CompVis repository
  at revision `133a221b8aa7292a167afc5127cb63fb5005638b` and loaded in
  float32 on CPU. A 50-step, seed-42, 512x512 baseline image for
  `a photo of bird` was generated and its PNG verified at
  `results/sd14_cpu_pilot/images/bird/none/clean/target/0000.png`.
  This single image is a feasibility check, not a research rate.
- The official Phan weights were extracted from the author's 979 MB archive
  mirror. `data/classifier/resnet18.pt` has 44,775,417 bytes and SHA-256
  `72d30ca70e7d54e24a26628113604c40bc872eb8dc7a44f50ec58c3a1400f1b0`.
  Strict state-dict loading passed; the baseline bird image was predicted as
  CIFAR-10 class index 2 (`bird`). This one prediction is not an accuracy rate.
- The SD 1.4 UCE clean and EEBshallow-plus-UCE checkpoints were created for
  bird in `checkpoints/sd14_cpu_uce_pilot`; the clean UCE metadata records 32
  edited projections. A 50-step, 13-image CPU generation attempt was stopped
  before its first image completed because of runtime. A separate reduced
  20-step run is configured in `configs/windows_cpu_quick.yaml`.
- In that reduced 20-step SD 1.4 run, the clean bird control generated one
  explicit-target and one trigger image. Phan's classifier predicted 1/1 bird
  for the explicit prompt and 0/1 bird for the trigger prompt. These single
  observations are stored in `results/sd14_cpu_quick/tables/`; no retention,
  FID, or CLIPScore sample was generated in this run.
- The same reduced run also completed clean-model UCE and EEBshallow followed
  by UCE, each with one explicit-target and one trigger image. All three
  conditions used seed 42, 512x512 pixels, DDIM 20 steps, and guidance 7.5.
  The measured one-image counts are:

  | Bird condition | Explicit target | Trigger | Acc_e | ASR |
  | --- | ---: | ---: | ---: | ---: |
  | No attack, no erasure | 1/1 | 0/1 | 100% | 0% |
  | No attack, UCE | 1/1 | 0/1 | 100% | 0% |
  | EEBshallow, UCE | 1/1 | 0/1 | 100% | 0% |

  These rates are descriptive observations at n=1. The UCE explicit prompt
  still classified as bird, so this pilot does not demonstrate erasure. The
  shallow trigger did not succeed on this seed. No ESD or EEBdeep condition,
  ship or dog target, retention sample, COCO FID, or CLIPScore was measured.
  Raw predictions, images, checkpoint metadata, and the comparison CSV are in
  `results/sd14_cpu_quick/` and `checkpoints/sd14_cpu_quick/`; plots are made
  only for measured metrics.
- COCO acquisition from this host failed: the official HTTPS endpoint failed
  certificate validation, and the project's HTTP endpoint stalled with zero
  bytes. `scripts/download_coco_subset.py` is provided for an accessible host.
- The six-condition research matrix remains **NOT MEASURED** in the standard
  `results/` report. With no CUDA GPU and 16 GB RAM, the two-model optimization
  stages need a suitable cloud GPU to run in practical time. No tiny-model
  output is used as a research measurement.

## Executed on 2026-11-10 (macOS arm64, 8 GB RAM)

- Python 3.12 environment installed from `pyproject.toml`.
- `python -m compileall -q src scripts tests`: passed.
- Imports of every `palak_repro` module: passed.
- `python -m pytest -q`: 3 passed.
- Hugging Face's `hf-internal-testing/tiny-stable-diffusion-pipe` loaded on CPU;
  a 64x64, two-step image was generated.
- On that tiny model, EEBshallow injection, UCE erasure, EEBdeep one-step
  optimization, ESD one-step optimization, checkpoint saving and reloading,
  and condition image generation completed. These settings are explicitly
  reduced and the model is not Stable Diffusion v1.4.
- Tiny-model UCE edit also completed on MPS with CPU linear solve.
- `palak-experiments --config configs/cpu_smoke.yaml --stage generate
  --target bird --attack none --erasure clean`: completed.
- Final offline CPU smoke generated 11 new tiny-model images in
  `results/audit_smoke/`; no network download was started.
- Final `python -m pytest -q`: 3 passed; final syntax and package import
  checks passed.
- Report stage generated six rows with `NOT MEASURED` ASR values when there
  were no evaluated conditions.
- The classifier architecture was compared against Phan's source: all 122
  state-dict tensor names and shapes match.

## Unexecuted as of this record

- Stable Diffusion v1.4 baseline and full matrix: the fp16 download was
  intentionally stopped before completion at the user's request.
- All SD 1.4 research metrics: **NOT MEASURED**.
- Phan pretrained classifier inference: its 979 MB weights archive has not
  finished downloading in this environment. The original Box URL returns 404;
  the Google Drive mirror is documented in `scripts/download_classifier.py`.
- COCO FID and CLIPScore: the COCO 2014 data have not been acquired here.

The paper's reference numbers in the comparison CSV are not results of this
project. The tiny-model output is an integration test only.
