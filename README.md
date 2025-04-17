# ML‑Enhanced Security in P2P Overlay Networks

**Adaptive, self‑healing security for decentralized Kademlia overlays written in Python.**

---

## Table of Contents

- [Key Capabilities](#key-capabilities)
- [Architecture Overview](#architecture-overview)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Simulation Internals](#simulation-internals)
- [Adaptive Defence Pipeline](#adaptive-defence-pipeline)
- [Dashboard & Observability](#dashboard--observability)
- [Baselines & Experiments](#baselines--experiments)
- [Configuration Reference](#configuration-reference)
- [Extending the Framework](#extending-the-framework)
- [Road‑map & Research Ideas](#road‑map--research-ideas)
- [Citing This Work](#citing-this-work)
- [License](#license)

---

## Key Capabilities

| Layer                   | What It Does                                                                                      | Where in Code                                    |
|------------------------|---------------------------------------------------------------------------------------------------|--------------------------------------------------|
| **Kademlia Simulator** | 1 000‑node, latency‑aware overlay with churn, value storage & iterative look‑ups (< 300 LoC).     | `Node`, `Network`, `iterative_lookup()` in `main.py` |
| **Attack Suite**       | Automatic, parameterisable DDoS, Sybil, Eclipse and Routing‑Table Poisoning generators.          | `simulate_*_attack()` functions                  |
| **ML‑based Detection** | Pre‑trained Random‑Forest (98 % acc.) classifies live traffic as benign / DDoS within < 5 ms.     | `detect_ddos_ml()` + `random_forest_p2p.pkl`    |
| **RL‑based Mitigation**| Tabular Q‑Learning agent tunes 10 security knobs every 10 s; learns optimal trade‑offs online.    | `run_rl_loop()` & `apply_action()`              |
| **Sybil / Eclipse Heuristics** | /24 subnet diversity filter + entropy checks keep malicious identities ≤ 2 % of routing tables. | `update_routing_table()`, `detect_eclipse_by_routing_diversity()` |
| **Poisoning Liveness Checks** | Active ping probes & passive lookup‑failure stats clean dead routes with < 1 % false positives. | `detect_poisoning_*`, `mitigate_routing_table_poisoning()` |
| **Web Dashboard**      | Real‑time REST JSON + single‑page Chart.js UI: success/failure doughnut, RL reward line, flagged nodes & live logs. | `templates/dashboard.html`, Flask endpoints     |
| **Baseline**           | Identical simulator without any defences for A/B evaluation.                                      | `no_mitigate.py`                                 |

> This repo is a full stack security laboratory for P2P research: from raw packets to AI counter‑measures to production‑style observability.

---

## Architecture Overview

![Refined ML-Enhanced P2P Security Flow](https://github.com/user-attachments/assets/81f8ff31-fc33-438b-8883-2c1f8f76defb)

---

## Repository Layout

```
.
├── main.py                # Full simulator *with* adaptive defences
├── no_mitigate.py         # Baseline simulator *without* any mitigation
├── templates/
│   └── dashboard.html     # Chart.js single‑page dashboard (served by Flask)
│   └── dashboard_nomit.html     # Dashboard for simulator *without* any mitigation
├── random_forest_p2p.pkl  # Pre‑trained Random‑Forest DDoS detector
├── scaler_p2p.pkl         # StandardScaler used during training
└── README.md              # (You are here)
```

> *Notebook of the original ML training pipeline is not committed to keep the repo light, but instructions to retrain are provided below.*

---

## Quick Start

### 1. Prerequisites

- Python ≥ 3.10  
- Linux / macOS / WSL (1000‑node run uses ≈ 1.2 GB RAM)  
- `gcc` & `make` (optional, for scikit‑learn wheels)

### 2. Clone & Install

```bash
git clone https://github.com/Alpi157/ML‑Enhanced‑P2P‑Security.git
cd ML‑Enhanced‑P2P‑Security
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r <(python - <<'EOF'
print('\n'.join([
    'flask', 'pandas', 'numpy', 'scikit-learn', 'joblib', 'chartjs',
]))
EOF
)
```

### 3. Run the Adaptive Simulator

```bash
python main.py
```

Access dashboard at: [http://localhost:5000](http://localhost:5000)  
Dashboard auto-refreshes.

### 4. Run the Baseline (for A/B comparison)

```bash
python no_mitigate.py
```

Open the second dashboard at the same URL in another browser tab (different process = different port).

---

## Simulation Internals

### Kademlia Overlay Parameters

| Parameter        | Value (default)          | Location                |
|------------------|---------------------------|-------------------------|
| Bit‑space        | 16 (65 536 IDs)           | `BITSPACE` const        |
| Bucket size `k`  | 7                         | `BUCKET_SIZE`           |
| Parallelism `α`  | 3                         | `ALPHA`                 |
| Max rounds       | 20                        | `MAX_ROUNDS`            |
| Base latency     | 10–50 ms random per node  | `Node.base_latency`     |

Scaling to 20‑bit space and 5 000 nodes requires only a constant change to `BITSPACE` – memory grows linearly.

---

### Attack Generators

| Attack     | Leveraged Weakness                          | Knobs                                        |
|------------|----------------------------------------------|----------------------------------------------|
| DDoS       | Unbound request queue & single‑threaded IO   | duration, #attackers, #victims, req/s       |
| Sybil      | Identity cost ≈ 0                            | #fake IDs, subnet prefix                    |
| Eclipse    | k‑bucket nearest‑ID eviction                 | #malicious peers per bucket                 |
| Poisoning  | Lazy liveness checks                         | %replacement per bucket, duration           |

Each attack yields a summary like:  
**"Routing Table Infiltration: 312/1800 entries (17.3 %) are sybil."**

---

## Adaptive Defence Pipeline

### 1. ML‑Based DDoS Detection

- **Features**: latency, per‑victim request rate, lookup failure ratio, peer diversity, base latency  
- **Model**: `scikit-learn` `RandomForestClassifier(n_estimators=200, max_depth=12)`  
- **Performance**: AUC = 0.995 on test set  
- **Speed**: 1 000 predictions ≈ 3 ms on Intel Core i7-11370H

### 2. Heuristic Guards

- **Sybil**: allow ≤ `SYBIL_MAX_PER_BUCKET` (default 2) contacts per `/24` subnet  
- **Eclipse**: raise alert if bucket entropy < `ECLIPSE_DIVERSITY_THRESHOLD` (default 0.8)  
- **Poisoning**: kill routes whose 3× pings fail > `POISONING_FAILURE_THRESHOLD` (default 0.5)

### 3. Reinforcement Learning

- **State** → ⟨`ddos_detections`, `sybil_added`, `lookup_not_found`, `avg_latency`⟩  
  Discretised into ≤ 6×6×6×11 = 2 376 cells  
- **Action space (10)** → tune thresholds, rebalance tables, force eclipse mitigation  
- **Reward** → `+success_rate×100 − latencyPenalty − ddosPenalty`  
- **Policy**: epsilon‑greedy (ε = 0.1, decay 0.998)  
- **Update interval**: 10 s  
- ✅ *Converges to > +600 cumulative reward in ~6 minutes, reducing lookup failures by 42% vs baseline*

---

## Dashboard & Observability

### Endpoints

- `/` — HTML single-page app  
- `/logs` — Last 2 000 lines (JSON)  
- `/concurrent` — Subset of lookup traces  
- `/summary` — 30-field JSON snapshot (network size, DDoS stats, reward, etc.)

### Charts & Visuals

- **Doughnut** — Lookup success vs failure (live)
- **Line** — Cumulative RL reward (last 20 intervals)
- **Tables** — Show delta since last poll for detecting spikes
- **Flag Boxes** — Live list of flagged nodes from Eclipse/Poison detection


---

## Baselines & Experiments

| Scenario                 | Metric                        | Adaptive | Baseline | Δ        |
|--------------------------|-------------------------------|----------|----------|----------|
| DDoS 50 victims, 5 attackers | Lookup success %         | 91.2     | 52.7     | +73%     |
|                          | Avg victim latency (s)        | 0.43     | 1.21     | −64%     |
| Sybil ×50                | Sybil entries after mitigation | 7.9%     | 31.5%    | −75%     |
| Eclipse 30×              | Victim bucket domination      | 19%      | 100%     | −81%     |

To reproduce:

```bash
python main.py         # adaptive
python no_mitigate.py  # baseline
```

> *Logs are deterministic if you set `PYTHONHASHSEED` and `random.seed()`*

---

## Configuration Reference

All constants are defined near the top of `main.py`. Edit and restart:

```python
BITSPACE = 16                     # Kademlia ID bit‑space
BUCKET_SIZE = 7                  # k‑bucket size
DDOS_DURATION = 30               # seconds per attack burst
SYBIL_MAX_PER_BUCKET = 2         # RL can tighten/loosen this
POISONING_FAILURE_THRESHOLD = 0.5
ECLIPSE_DIVERSITY_THRESHOLD = 0.8
```

**Environment Variables**:

- `P2P_SEED` – sets consistent seeding for repeatable runs  
- `FLASK_PORT` – defaults to 5000

---

## Extending the Framework

- **Custom ML model**: implement `detect_ddos_ml(features)` returning 0/1 (response time <10 ms preferred)
- **New Attack**: add `simulate_my_attack()` and call it from the main loop
- **RL Actions**: extend `ACTION_SPACE` and logic in `apply_action()`
- **Scaling**: increase `BITSPACE`, spawn more nodes (CPU ~ O(N²) only during bootstrap)
- **TLS / Authenticity**: wrap messages in cryptographic signatures to benchmark trade‑offs

---

## Road‑map & Research Ideas

- **Multi-Agent RL** – each node self-trains, gossip Q-values (federated RL)
- **Graph Neural Nets** – use routing structure to detect Eclipse attacks
- **Adversarial ML** – FGSM & DeepFool robustness tests on RF model
- **Real-World Deployment** – PlanetLab/Kubernetes testbed
- **IPv6 Sybil Research** – Explore /64 heuristics in large address space

---

## Citing This Work

If you use any part of this repository, please cite:

```bibtex
@misc{arman2025mlp2p,
  author       = {Alpar Arman},
  title        = {ML‑Enhanced Security in P2P Overlay Networks},
  howpublished = {GitHub},
  year         = {2025},
  url          = {https://github.com/Alpi157/ML-Enhanced-P2P-Security}
}
```

---

## License

**MIT License** – © 2025 Alpar Arman

Feel free to use, modify, and distribute – a citation or star is always appreciated!
