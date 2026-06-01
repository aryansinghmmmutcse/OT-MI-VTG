# TAOT-VTG

**Temporally-Aware Hierarchical Optimal Transport with 
Boundary-Sensitive Alignment for Video Temporal Grounding**

ACIVS 2026 — Okinawa, Japan

## Overview

This repository contains the official implementation of 
TAOT-VTG, a novel Video Temporal Grounding framework 
incorporating three mathematically grounded contributions:

- **TAOT Cross-Modal Attention** — entropy-regularised OT 
  with asymmetric temporal position bias
- **Multi-Scale Hierarchical OT Alignment** — frame-level 
  and segment-level transport plans combined via learned gate
- **Boundary Sharpening Loss** — entropy regularisation 
  for high-IoU moment retrieval

## Results

### QVHighlights

| Method | R1@0.5 | R1@0.7 | mAP | Fair mAP |
|--------|--------|--------|-----|----------|
| Ours   | 66.94  | 52.07  | 44.64 | 75.79 |

### Charades-STA

| Method | R1@0.5 | R1@0.7 |
|--------|--------|--------|
| Ours   | 59.88  | 39.22  |

## Environment Setup

```bash
git clone https://github.com/aryansinghmmmutcse/TAOT-VTG
cd TAOT-VTG
pip install -r requirements.txt
```

## Training

### QVHighlights
```bash
bash cg_detr/scripts/train.sh --exp_id taot_vtg
```

### Charades-STA
```bash
bash cg_detr/scripts/charades_sta/train.sh \
    --exp_id taot_vtg_charades
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{singh2026taot,
  title={Temporally-Aware Hierarchical Optimal Transport 
         with Boundary-Sensitive Alignment for 
         Video Temporal Grounding},
  author={Singh, Aryan and Om, Hari},
  booktitle={ACIVS},
  year={2026}
}
```

## Acknowledgement

This work uses the 
[Bhaskara GPU cluster](https://www.iitism.ac.in) 
at IIT (ISM) Dhanbad.
