# FinGuard — Real-Time AML & Transaction Fraud Graph Analyzer

A full-stack fraud detection system combining unsupervised machine learning
(Isolation Forest + PyTorch Autoencoder) with graph-based structural fraud
pattern detection (cycle detection, fan-out, structuring, velocity) for
real-time transaction risk scoring.

---

## Project Structure

```
FinGuard/
├── main.py                    FastAPI application (run this to serve API)
├── train.py                   Trains and saves ML models (run this FIRST)
├── finguard_models.py         Shared Autoencoder model definition
├── generate_dataset_sample.py Synthetic dataset generator (optional, for quick testing)
├── requirements.txt           Python dependencies
├── static/
│   └── index.html             Dashboard (D3.js graph + live monitoring)
├── finguard_day1_theory.md    ML theory: Isolation Forest + Autoencoder
├── finguard_day2_theory.md    Graph algorithms + FastAPI + MongoDB theory
├── finguard_day3_theory.md    D3.js + system design theory
└── README.md                  This file
```

---

## Setup Instructions

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the dataset

**Option A — Real dataset (recommended for submission):**
Download `creditcard.csv` from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
and place it in the `FinGuard/` folder.

**Option B — Synthetic dataset (quick testing only):**
```bash
python generate_dataset_sample.py
```
This creates a 10,000-row synthetic dataset with the same structure
(V1-V28, Amount, Time, Class) so you can test the full pipeline
immediately without downloading anything. Do not use this for your
final report numbers — use the real dataset for actual results.

### 3. Train the models

```bash
python train.py --data creditcard.csv
```

This trains the Isolation Forest and Autoencoder, prints evaluation
metrics (ROC-AUC, precision, recall), and saves three files:
- `isolation_forest.pkl`
- `autoencoder.pth`
- `scaler_stats.npy`

Takes 2-5 minutes on CPU for the real 284k-row dataset.

### 4. Start MongoDB (optional)

The app works without MongoDB — it falls back to in-memory storage
automatically (data won't persist across restarts, but everything else
works identically). For persistence:

```bash
# Local MongoDB
mongod --dbpath ./data

# OR use MongoDB Atlas free tier (cloud):
# https://www.mongodb.com/cloud/atlas/register
# Then set: export MONGO_URI="mongodb+srv://..."
```

### 5. Run the API server

```bash
uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation
(Swagger UI) where you can test every endpoint directly in the browser.

### 6. Open the dashboard

Open `static/index.html` directly in your browser (double-click the file,
or use a simple HTTP server):

```bash
cd static
python -m http.server 3000
# then visit http://localhost:3000
```

The dashboard connects to the API at `http://localhost:8000` automatically.

---

## Quick Demo Flow

1. Start the API (`uvicorn main:app --reload`)
2. Open the dashboard
3. Click **"Simulate Fraud Loop"** — this submits 3 transactions forming
   a money-laundering cycle (A→B→C→A) and you'll see:
   - Risk scores spike for each transaction
   - The graph visualizes the cycle in red
   - An alert appears in the live feed
4. Try submitting a normal transaction manually — observe the low risk score

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Server + model + DB status |
| POST | `/transaction` | Score a transaction (main endpoint) |
| GET | `/graph/{account_id}` | Get transaction subgraph for an account |
| GET | `/alerts` | List recent flagged transactions |
| GET | `/stats` | Dashboard KPI summary |

Full interactive docs at `/docs` once the server is running.

---

## How It Works (Summary)

**ML Layer:** Isolation Forest isolates anomalies via random feature-space
partitioning (short isolation path = anomalous). An Autoencoder trained
exclusively on normal transactions learns to reconstruct normal patterns;
fraud causes high reconstruction error. Scores are normalized and averaged.

**Graph Layer:** Every transaction becomes a directed edge (account → account)
in a NetworkX graph. Four structural fraud patterns are detected:
- **Cycles** (money laundering loops, A→B→C→A) via Johnson's algorithm
- **Fan-out** (shell company behavior — one account paying many recipients)
- **Structuring** (multiple transactions just under the ₹50,000 reporting threshold)
- **Velocity** (10+ transactions per hour — automated fraud)

**Final Score:** `0.6 × ML_risk + 0.4 × Graph_risk`, mapped to
LOW / MEDIUM / HIGH / CRITICAL risk levels.

See the three theory `.md` files for full mathematical derivations,
architecture diagrams, and interview-style Q&A explanations of every
design decision.

---

## Tech Stack

Python · PyTorch · scikit-learn · FastAPI · MongoDB · NetworkX · D3.js · HTML/CSS/JS

---

## Author Notes

Built as a 3-day sprint project. The system runs in graceful degraded
modes if models aren't trained yet (random scores, clearly labeled
`TEST_MODE_RANDOM` in API responses) or MongoDB isn't running
(in-memory storage) — so the API and dashboard are always demoable
even mid-setup.
