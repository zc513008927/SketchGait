# SketchGait

This repository provides the implementation, configuration files, and supporting resources for **SketchGait** and **SketchGait++ (SketchGaitPP)** on **SUSTech1K**, based on the [OpenGait](https://github.com/ShiqiYu/OpenGait) framework.

> **Paper status:** Under review. The code, processed data, preprocessing scripts, and checkpoints are shared to facilitate reproducibility.

## Repository Contents

| Path | Description |
| --- | --- |
| [`opengait/`](opengait/) | Model implementations and the OpenGait training/evaluation framework. |
| [`configs/sketchgait/`](configs/sketchgait/) | Configurations for SketchGait and SketchGaitPP. |
| [`datasets/SUSTech1K/`](datasets/SUSTech1K/) | SUSTech1K protocol and preprocessing scripts. |
| [`output/SUSTech1K/`](output/SUSTech1K/) | Model output directories, including the checkpoint locations. |

## Data and Pretrained Checkpoints

The data and checkpoints used in our experiments are shared through **Baidu Netdisk**:

- **Download:** [sketchgait (Baidu Netdisk)](https://pan.baidu.com/s/1DillSa4rzk8-1yh4j5H2xA?pwd=5478)
- **Extraction code:** `5478`

| File | Description |
| --- | --- |
| `sils_teed_pars_merged.zip` | Preprocessed, aligned **64 × 64** three-modality SUSTech1K sequences (Silhouette, TEED, Parsing); recommended for direct training and evaluation. |
| `SUSTech1K-RAW_parsing_to_silhouette.zip` | Silhouette images extracted from the original SUSTech1K RGB data. |
| `SUSTech1K-RAW_parsing.rar` | Human parsing images extracted from the original RGB data. |
| `SUSTech1K-TEED_blackBG.zip` | TEED edge images with black backgrounds extracted from the RGB data. |
| `SketchGait-50000.pt` | Pretrained SketchGait checkpoint. |
| `SketchGaitPP-50000.pt` | Pretrained SketchGaitPP checkpoint. |

The raw-resolution modalities are provided for researchers who wish to investigate alternative preprocessing methods. For the original SUSTech1K dataset and its access terms, please refer to the [official SUSTech1K project](https://lidargait.github.io).

## Quick Start: Evaluation with the Provided Data

**1. Prepare the data.** Download and extract `sils_teed_pars_merged.zip` to a local directory. The preprocessed data contains aligned three-channel sequences at **64 × 64**, ordered as **Silhouette, TEED, Parsing**.

**2. Update the configuration.** Edit `data_cfg.dataset_root` in the relevant YAML file to point to the extracted `sils_teed_pars_merged` directory:

- [SketchGait config](configs/sketchgait/sketchgait.yaml)
- [SketchGaitPP config](configs/sketchgait/sketchgaitpp.yaml)

For example:

```yaml
data_cfg:
  dataset_name: SUSTech1K
  dataset_root: /path/to/sils_teed_pars_merged
  dataset_partition: ./datasets/SUSTech1K/SUSTech1K.json
```

**3. Place the checkpoints.** Extract or copy the downloaded checkpoint files to the corresponding directories under the repository root:

```text
output/SUSTech1K/
├── SketchGait/SketchGait/checkpoints/
│   └── SketchGait-50000.pt
└── SketchGaitpp/SketchGaitPP/checkpoints/
    └── SketchGaitPP-50000.pt
```

The provided YAML files use `evaluator_cfg.restore_hint: 50000`. Preserve the filenames and match each checkpoint to its respective configuration.

**4. Run evaluation.** From the repository root, using two GPUs:

```bash
# SketchGait
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
  opengait/main.py --cfgs ./configs/sketchgait/sketchgait.yaml --phase test

# SketchGaitPP
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
  opengait/main.py --cfgs ./configs/sketchgait/sketchgaitpp.yaml --phase test
```

Adjust `CUDA_VISIBLE_DEVICES`, `--nproc_per_node`, and the configuration settings to your hardware.

## Training

Use the same preprocessed dataset and update `data_cfg.dataset_root` as above. To train from scratch, the provided configs set `trainer_cfg.restore_hint: 0`.

```bash
# SketchGait
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
  opengait/main.py --cfgs ./configs/sketchgait/sketchgait.yaml --phase train

# SketchGaitPP
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
  opengait/main.py --cfgs ./configs/sketchgait/sketchgaitpp.yaml --phase train
```

## Optional: Preprocessing from Original-Resolution Modalities

If you prefer to reproduce or modify the preprocessing rather than use the provided `64 × 64` data:

1. Download and extract the original-resolution **Silhouette**, **Parsing**, and **TEED** archives listed above.
2. Use [`datasets/SUSTech1K/pretreatment_three_modality_alignment.py`](datasets/SUSTech1K/pretreatment_three_modality_alignment.py) to spatially align the three modalities and produce **64 × 64** sequence-level PKL files. Set the input/output paths for each modality according to your local directories.
3. Use [`datasets/SUSTech1K/sils_sketch_pars_merged`](datasets/SUSTech1K/sils_sketch_pars_merged) to concatenate the aligned modalities along the channel dimension. Update the paths at the beginning of that script before running it. Its output channel order is **Silhouette, TEED (Sketch), Parsing**.
4. Set the resulting merged-data directory as `data_cfg.dataset_root`, then run training or evaluation using the commands above.

The supplied preprocessing follows the approach used in our experiments. **We encourage further exploration of the original-resolution three-modality data**, including different resize/interpolation methods, spatial normalization strategies, and alternative preprocessing pipelines. The provided processing procedure is not intended to restrict these investigations.

## Acknowledgement

This repository builds upon [OpenGait](https://github.com/ShiqiYu/OpenGait). We also acknowledge the [SUSTech1K](https://lidargait.github.io) dataset and its contributors.

## Citation

The associated manuscript is currently under review. Citation information will be updated after publication.
