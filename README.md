# F2Mix

**Fine-grained augmentation and Feature copy-paste consistency for heterogeneous 3D CT image classification.**

F2Mix is a plug-and-play framework for binary classification of 3D CT volumes
(e.g., lung nodule malignancy prediction, distant-metastasis [M-stage]
prediction, and EGFR mutation-status prediction). It combines:

- **FGA (Fine-Grained Augmentation)** — volumetric median-filter denoising,
  unsharp masking, min-max normalization, and contrast scaling to generate a
  fine-grained augmented view of each volume, mitigating noise/texture
  heterogeneity across scanners and slice thicknesses.
- **ICP (Image-level Copy-Paste)** — an in-plane patch exchange between the
  original volume and its FGA-augmented counterpart (the depth axis is
  preserved to keep anatomical integrity).
- **FCPC (Feature Copy-Paste Consistency)** — the copy-paste is repeated at
  feature level at exactly the same spatial position, and a mutual-information
  consistency loss aligns the processed features with the originals.
- **FR (Feature Refinement)** — a primary feature is built from the two clean
  views, and the redundant component of the copy-paste features is removed by
  orthogonal projection before fusion and classification.

The full model is `L = L_CE + λ · L_F` with `λ = 0.5`.

## Repository layout

```
F2Mix/
├── main.py              # unified training / testing entry point
├── Model/
│   ├── f2mix.py         # shared F2Mix wrapper (expander, FR, classifier)
│   ├── densenet.py      # 3D DenseNet backbone          (DN.)
│   ├── resnet.py        # 3D ResNet backbone            (RN.)
│   ├── res2net.py       # 3D Res2Net backbone           (R2N.)
│   ├── vgg.py           # 3D VGG-16 backbone
│   ├── pvt.py           # 3D PVT backbone               (PVT-Tiny)
│   ├── pvt_v2.py        # 3D PVTv2 backbone
│   └── swin.py          # 3D Swin Transformer backbone
├── Utils/
│   ├── Options.py       # unified command-line options
│   ├── datasets.py      # HDF5 dataset + FGA pipeline + weak augmentation
│   ├── fcp_ops.py       # ICP / FCP operators, MI / NCC consistency losses
│   ├── metrics.py       # ACC / AUC / SEN / SPE, ROC plotting, CSV logging
│   └── common.py        # seeding, checkpointing, logging
├── scripts/
│   ├── train.sh
│   └── test.sh
└── Plot/                # visualization utilities (Grad-CAM heatmaps)
```

## Installation

```bash
pip install -r requirements.txt
```

## Data format

The code expects a single HDF5 file containing one group per split
(`train` / `val` / `test`), each with:

- `path`  – string array of NIfTI volume paths;
- `label` – integer array of binary labels (0 = negative, 1 = positive).

Volumes are resampled to `48 x 256 x 256` (configurable with `--input_size`)
with trilinear interpolation.

## Training

Any supported backbone can be trained with the same command — F2Mix is
plug-and-play:

```bash
python main.py --mode train --backbone densenet \
    --data_path your/path/to/dataset.hdf5 \
    --save_dir ./results/your_dataset
```

Or via the helper script:

```bash
bash scripts/train.sh densenet your/path/to/dataset.hdf5 0
```

Key options:

| Option | Default | Description |
|---|---|---|
| `--backbone` | `densenet` | one of `densenet, resnet, res2net, vgg, pvt, pvt_v2, swin` |
| `--batch_size` | `4` | batch size |
| `--base_lr` | `1e-5` | base learning rate (constant, no scheduler) |
| `--opt` | `SGD` | `SGD` or `AdamW` |
| `--total_epoch` | `150` | total training epochs |
| `--cp_patch` | `180,180` | in-plane copy-paste patch size |
| `--consistency_loss` | `mi` | feature consistency loss: `mi`, `mse`, or `ncc` |
| `--lambda_fcp` | `0.5` | weight of the consistency loss |
| `--weak_prob` | `0.1` | probability of each weak transformation |
| `--no_fga` | off | disable the FGA view (ablation) |
| `--resume` | – | checkpoint path to resume from |

Model selection follows the paper: the checkpoint with the highest validation
AUC is kept (`results/<dataset>/<backbone>/models/best_model.pth`).

## Testing

```bash
python main.py --mode test --backbone densenet \
    --data_path your/path/to/dataset.hdf5 \
    --test_model results/your_dataset/densenet/models/best_model.pth
```

Reports ACC, AUC, sensitivity, and specificity, and saves the ROC curve and
per-sample predictions under `results/<save_dir>/<backbone>/test/`.

## Adding a new backbone

Implement a module whose `forward(x)` maps `(B, 1, D, H, W)` to a feature map
`(B, C, d, h, w)` and expose a `pool_size` attribute (adaptive-average-pool
output size), then register it in `Model/__init__.py`:

```python
BACKBONES['my_backbone'] = MyBackbone
```

The shared F2Mix wrapper automatically derives the expander geometry and the
classification head dimensions.

## Reference

If you find this code useful, please cite the corresponding paper.
