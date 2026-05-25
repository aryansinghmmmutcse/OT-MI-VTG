# OT-MI: Optimal Transport Cross-Modal Alignment with Mutual Information Query Initialization for Video Temporal Grounding

> **BMVC 2026** | [Paper](#) | [Code](https://github.com/aryansinghmmmutcse/OT-MI-VTG)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org)
[![BMVC](https://img.shields.io/badge/BMVC-2026-green.svg)](#)

## Abstract
Video Temporal Grounding (VTG) aims at accurately associating video segments with natural language descriptions. Current transformer-based methods use softmax attention for cross-modal interaction, which only matches locally without ensuring global consistency. We propose **OT-MI**, a novel framework that addresses these limitations through Optimal Transport cross-modal alignment and Mutual Information query initialization.

## Results

### QVHighlights Validation Set
| Method | R1@0.5 | R1@0.7 | mAP | HD Fair mAP | HD Fair Hit@1 |
|--------|--------|--------|-----|-------------|---------------|
| M-DETR | 53.94 | 34.84 | 32.20 | 35.69 | 55.60 |
| QD-DETR | 62.68 | 46.66 | 41.22 | 39.13 | 63.03 |
| TR-DETR | 64.66 | 48.96 | 42.62 | 39.91 | 63.42 |
| MLVTG | 65.10 | 50.50 | **44.60** | 40.50 | 65.20 |
| **OT-MI (Ours)** | **66.68** | **51.94** | 44.22 | **77.47** | **80.39** |

### Charades-STA Test Set
| Method | R1@0.5 | R1@0.7 |
|--------|--------|--------|
| MLVTG | 58.30 | **38.70** |
| **OT-MI (Ours)** | **59.75** | 38.65 |

## Installation

```bash
git clone https://github.com/aryansinghmmmutcse/OT-MI-VTG.git
cd OT-MI-VTG
pip install -r requirements.txt
```

## Data Preparation

Download features following [QD-DETR](https://github.com/wjun0830/QD-DETR):
- QVHighlights: CLIP + SlowFast features (2816 dim)
- Charades-STA: CLIP + SlowFast features (2816 dim)

Place features under `../features/` directory.

## Training

**QVHighlights:**
```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python cg_detr/train.py \
--dset_name hl --ctx_mode video_tef \
--v_feat_dirs ../features/qvhighlight/slowfast_features \
               ../features/qvhighlight/clip_features \
--v_feat_dim 2816 \
--t_feat_dir ../features/qvhighlight/clip_text_features/ \
--t_feat_dim 512 --bsz 32 --n_epoch 200 \
--results_root results_otmi --exp_id otmi
```

**Charades-STA:**
```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python cg_detr/train.py \
--dset_name charadesSTA --ctx_mode video_tef \
--v_feat_dirs ../features/charades/slowfast_features \
               ../features/charades/clip_features \
--v_feat_dim 2816 \
--t_feat_dir ../features/charades/clip_text_features/ \
--t_feat_dim 512 --bsz 32 --n_epoch 200 \
--lr 0.0002 --lw_saliency 4 --max_v_l -1 --clip_length 1 \
--results_root results_charades --exp_id otmi_charades
```

## Citation

```bibtex
@inproceedings{singh2026otmi,
  title={Optimal Transport Cross-Modal Alignment with Mutual Information 
         Query Initialization for Video Temporal Grounding},
  author={Singh, Aryan and Om, Hari},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2026}
}
```

## Acknowledgements
This work was conducted at the Department of Computer Science and Engineering, 
Indian Institute of Technology (ISM) Dhanbad, India.
