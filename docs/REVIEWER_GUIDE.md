# Ten-minute source review: cardiac MRI segmentation

**Goal:** Help a medical-imaging ML engineer judge the *documented* experiment without requiring a GPU, checkpoint, or restricted dataset download.

**Important scope:** This project implements **one-input-channel, two-class 3D left-atrium segmentation**, not multi-organ MRI segmentation, a medical device, or a clinically validated model. Its best 0.8821 validation Dice is **author-reported** and has not been independently reproduced.

## Review path (no installation or dataset required)

| Minutes | Open | What to verify |
| --- | --- | --- |
| 0–2 | [YAML configuration](../configs/config.yaml) and [model code](../src/model.py) | One MRI channel, background/left-atrium output, 3D MONAI U-Net; check actual model parameters rather than assuming a Swin/UNETR architecture |
| 2–5 | [Dataset loader](../src/dataset.py) | Paired image/label basenames and seeded split; current code shuffles cases, but **patient identity linkage and between-case independence remain unverified** |
| 5–7 | [Training implementation](../src/train.py) | How sampling, loss, checkpoint selection and validation are performed; identify failure points for reproducibility |
| 7–9 | [Inference code](../src/inference.py) | Checkpoint dependency, specified input case and preprocessing compatibility |
| 9–10 | [Run summary](../results/training_summary.txt) and [README evidence caveats](../README.md) | Distinguish author's recorded outcome from independently rerunnable evidence; note missing checkpoint, environment lock and exact split manifest |

## Reviewer questions worth answering

1. Does the random case-level split preserve patient-level independence? Without a patient identifier or original cohort ledger, **do not assert this**.
2. Are all image/label mappings, random seeds, spatial resampling policies and train/validation transforms explicit?
3. Is the checkpoint for the claimed 0.8821 best validation score available with sufficient run metadata and dataset split provenance? If not, mark it **not independently reproducible**.
4. Would evaluation by site, scanner or independent test cohort be necessary before broader scientific claims? Yes, where permitted and feasible.
5. Are GPU and source dataset use governed by original dataset rights? Review source terms; **never upload patient scans or restricted clinical files to a public issue**.

## Concrete feedback invitation

If you can identify a real weakness, open a [methods review issue](https://github.com/aradmanamnaoon/medical-image-segmentation/issues/new?template=methods-review.yml) with an exact file, line or conceptual scenario. Use only synthetic illustrations or your own permitted data descriptions. Do not paste clinical information or proprietary research datasets. A grounded criticism or reproducibility finding is welcome even if you have not run the GPU training job.

**What this is not:** A request for reciprocal stars, an assertion of clinical validation, an independent code review report, or a promise that metrics will generalize outside MSD Task02 Heart.
