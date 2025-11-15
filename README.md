# Safeguard Fall & ADL Detection – Hierarchical TCN

This repository contains a hierarchical Temporal Convolutional Network (TCN) model
for **fall vs ADL detection** using chest-mounted IMU signals (UMAFall dataset).

## Model

- Backbone: Dilated 1D TCN (`TemporalBlock` stack)
- Heads:
  - **Fine** head: 11 classes  
    (`walking, jogging, bending, hopping, lyingdownonabed, sittinggettinguponachair, goupstairs, godownstairs, fall_forward, fall_backward, fall_lateral`)
  - **Coarse** head: 2 classes (ADL vs FALL)
- Aggregation over multiple windows using different strategies (`logits_mean`, `probs_mean`, `majority`) with an optional **gating** on the fall probability.

## Inference

### 1. Environment

```bash
pip install -r requirements.txt
