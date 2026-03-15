# EnhPPFL

**Enhanced Privacy-Preserving Federated Learning with Adaptive Orthogonal Gaussian Sampling and Secure Aggregation**

Navneet Mishra and Prachet Bhuyan  
School of Computer Engineering, KIIT Deemed to be University, Bhubaneswar, India

---

## Overview

EnhPPFL is a federated learning framework that co-designs three mutually reinforcing mechanisms to address adaptive gradient inversion attacks, poor privacy–utility balance, and high communication overhead simultaneously:

1. **Posterior-inspired orthogonal Gaussian sampling** — noise is confined to the null space of the gradient direction, neutralising Learning-to-Invert (LTI) style attackers without extra noise budget.
2. **Adaptive layer-wise Rényi DP** — per-layer noise allocation guided by diagonal Fisher information, tracked via the moments accountant with subsampling amplification.
3. **Single-round SecAgg with adaptive top-k sparsification** — ECDH pairwise masks cancel under aggregation; error-feedback sparsification achieves approximately 73% bandwidth reduction.

Evaluated on NSL-KDD (cyber threat detection) and CIFAR-10 (image classification) under (ε = 1.0, δ = 10⁻⁵)-DP.

---

## Results summary

| Dataset | Metric | EnhPPFL | CENSOR baseline |
|---------|--------|---------|-----------------|
| NSL-KDD | F1 (weighted) | 0.919 | 0.850 |
| CIFAR-10 | Accuracy | 88.6% | 85.2% |
| CIFAR-10 | Adaptive attack success | 8% | 15% |
| Both | Communication overhead | 27% of full | 100% of full |
| Both | Computational overhead | +10.3% | +12% |

All results are under (ε = 1.0, δ = 10⁻⁵)-DP.

---

## Repository structure

```
enhppfl/
├── privacy_utils.py      Fisher information, orthogonal projection,
│                         top-k sparsification, ECDH utilities, RDP accountant
├── models.py             ResNet-18 (CIFAR-10) and 4-layer MLP (NSL-KDD)
├── client.py             Flower NumPyClient — local training + defence pipeline
├── server.py             Flower strategy — sparse aggregation + privacy tracking
├── verify_defense.py     DLG-style gradient inversion attack verification
├── analyze_results.py    Log parser and summary report generator
├── run_experiment.sh     End-to-end experiment runner
└── requirements.txt      Python dependencies
```

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0, < 2.3
- Opacus ≥ 1.4, < 1.5
- Flower (flwr) ≥ 1.6, < 2.0

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Datasets

**CIFAR-10** downloads automatically via torchvision on first run.

**NSL-KDD** must be downloaded manually from the Canadian Institute for Cybersecurity:

```
https://www.unb.ca/cic/datasets/nsl.html
```

Place `KDDTrain+.txt` and `KDDTest+.txt` in a `data/` directory. The preprocessing pipeline (one-hot encoding of `protocol_type`, `service`, and `flag`) produces 122 input features from the 41 raw columns.

---

## Running experiments

### Automated pipeline

```bash
chmod +x run_experiment.sh

# CIFAR-10 / ResNet-18
./run_experiment.sh --model resnet18 --dataset cifar10 --rounds 200

# NSL-KDD / MLP
./run_experiment.sh --model mlp --dataset nslkdd --rounds 200
```

### Manual start

**Server (Terminal 1):**

```bash
# NSL-KDD / MLP
python server.py \
    --model-type mlp \
    --total-rounds 200 \
    --privacy-epsilon 1.0 \
    --privacy-delta 1e-5 \
    --input-dim 122

# CIFAR-10 / ResNet-18
python server.py \
    --model-type resnet18 \
    --total-rounds 200 \
    --privacy-epsilon 1.0
```

**Clients (Terminals 2–11):**

```bash
python client.py \
    --client-id 0 \
    --total-clients 10 \
    --model-type mlp \
    --dataset nslkdd \
    --base-noise 2.0 \
    --lambda-sparse 0.28
```

Repeat with `--client-id 1` through `--client-id 9`.

### Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--base-noise` | 2.0 | σ_base, the base noise multiplier |
| `--base-clipping` | 1.0 | C̄, the base clipping threshold |
| `--lambda-sparse` | 0.28 | λ in k(t) = ⌈d(1 − e^{−λt/T})⌉ |
| `--privacy-epsilon` | 1.0 | Target ε for DP guarantee |
| `--total-rounds` | 200 | Number of federated rounds T |

### Defence verification

```bash
python verify_defense.py \
    --model-type resnet18 \
    --num-samples 50 \
    --noise-multiplier 2.0 \
    --save-plot
```

### Analysing results

```bash
python analyze_results.py \
    --log-dir ./logs/<experiment_name> \
    --plot
```

---

## Reproducibility

All experiments were run with a fixed random seed. The repository includes the configuration files used to produce the reported results. To reproduce:

```bash
./run_experiment.sh \
    --model resnet18 \
    --dataset cifar10 \
    --rounds 200
```

Expected output: cumulative communication reduction ≥ 73%, privacy budget ε ≤ 1.02 after 200 rounds with σ_base = 2.0 and q = 0.1.

---

## Citation

If you use this code, please cite:

```bibtex
@article{mishra2025enhppfl,
  title   = {Enhanced Privacy-Preserving Federated Learning with Adaptive
             Orthogonal {G}aussian Sampling and Secure Aggregation},
  author  = {Mishra, Navneet and Bhuyan, Prachet},
  journal = {F1000Research},
  year    = {2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
