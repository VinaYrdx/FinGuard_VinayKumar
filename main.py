"""
main.py — FinGuard FastAPI application (entrypoint)

Run:
    uvicorn main:app --reload --port 8000

Requires (from train.py):
    isolation_forest.pkl
    autoencoder.pth
    scaler_stats.npy

If these are missing, the API still starts in TEST MODE (random scores)
so you can demo the graph/backend without training first.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List
import networkx as nx
import numpy as np
import joblib
import torch
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from datetime import datetime, timezone
import uuid
from collections import defaultdict
import os

from finguard_models import TransactionAutoencoder

# ─────────────────────────────────────────────
# LOAD ML MODELS
# ─────────────────────────────────────────────
MODELS_LOADED = False
iso_model = None
ae_model  = None
scaler_stats = None

try:
    iso_model = joblib.load('isolation_forest.pkl')
    ae_model  = TransactionAutoencoder(input_dim=30)
    ae_model.load_state_dict(torch.load('autoencoder.pth', map_location='cpu'))
    ae_model.eval()
    scaler_stats = np.load('scaler_stats.npy', allow_pickle=True).item()
    MODELS_LOADED = True
    print("[startup] ML models loaded successfully")
except FileNotFoundError:
    print("[startup] WARNING: model files not found. Run train.py first.")
    print("[startup] API will run in TEST MODE with random risk scores.")


# ─────────────────────────────────────────────
# MONGODB CONNECTION
# ─────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_ON  = False
IN_MEMORY: List[dict] = []

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    client.server_info()
    db        = client["finguard"]
    txn_col   = db["transactions"]
    alert_col = db["alerts"]
    MONGO_ON  = True
    print("[startup] MongoDB connected")
except ServerSelectionTimeoutError:
    print("[startup] WARNING: MongoDB not reachable. Using in-memory storage.")
    print("[startup] Data will NOT persist across restarts.")


# ─────────────────────────────────────────────
# GRAPH ENGINE
# ─────────────────────────────────────────────
class FraudGraphEngine:
    """Directed transaction graph with structural fraud-pattern detectors."""

    def __init__(self):
        self.G = nx.DiGraph()
        self.velocity = defaultdict(list)

    def add_transaction(self, from_acc, to_acc, amount, txn_id, timestamp):
        self.G.add_edge(from_acc, to_acc, amount=amount,
                        txn_id=txn_id, timestamp=timestamp)
        self.velocity[from_acc].append(timestamp)

    def detect_cycles(self, account: str) -> dict:
        try:
            cycles = list(nx.simple_cycles(self.G, length_bound=6))
        except TypeError:
            # older networkx without length_bound kwarg
            cycles = [c for c in nx.simple_cycles(self.G) if len(c) <= 6]
        involved = [c for c in cycles if account in c]
        return {
            "has_cycle":   len(involved) > 0,
            "cycle_count": len(involved),
            "cycles":      [" -> ".join(c + [c[0]]) for c in involved[:3]],
        }

    def detect_fan_out(self, account: str, threshold: int = 5) -> dict:
        if account not in self.G:
            return {"fan_out_risk": False, "unique_recipients": 0, "threshold": threshold}
        unique_recip = len(set(self.G.successors(account)))
        return {
            "fan_out_risk": unique_recip >= threshold,
            "unique_recipients": unique_recip,
            "threshold": threshold,
        }

    def detect_structuring(self, account: str, threshold_amount: float = 50000.0) -> dict:
        if account not in self.G:
            return {"structuring_risk": False, "suspicious_count": 0,
                    "threshold_amount": threshold_amount}
        edges = self.G.out_edges(account, data=True)
        suspicious = [e for e in edges
                     if threshold_amount * 0.7 <= e[2].get('amount', 0) < threshold_amount]
        return {
            "structuring_risk": len(suspicious) >= 3,
            "suspicious_count": len(suspicious),
            "threshold_amount": threshold_amount,
        }

    def velocity_risk(self, account: str, window_minutes: int = 60) -> dict:
        now = datetime.now(timezone.utc)
        recent = [t for t in self.velocity.get(account, [])
                 if (now - t).total_seconds() < window_minutes * 60]
        return {
            "velocity_risk": len(recent) >= 10,
            "txn_count": len(recent),
            "window_minutes": window_minutes,
        }

    def get_account_subgraph(self, account: str, depth: int = 2) -> dict:
        if account not in self.G:
            return {"nodes": [], "edges": []}
        neighbors = {account}
        frontier  = {account}
        for _ in range(depth):
            nxt = set()
            for node in frontier:
                nxt.update(self.G.successors(node))
                nxt.update(self.G.predecessors(node))
            neighbors.update(nxt)
            frontier = nxt
        sub = self.G.subgraph(neighbors)
        return {
            "nodes": [{"id": n, "degree": sub.degree(n)} for n in sub.nodes()],
            "edges": [{"from": u, "to": v, "amount": d.get("amount", 0)}
                     for u, v, d in sub.edges(data=True)],
        }

    def compute_graph_risk_score(self, account: str) -> float:
        score = 0.0
        if self.detect_cycles(account)["has_cycle"]:           score += 0.5
        if self.detect_fan_out(account)["fan_out_risk"]:       score += 0.2
        if self.detect_structuring(account)["structuring_risk"]: score += 0.2
        if self.velocity_risk(account)["velocity_risk"]:       score += 0.1
        return min(score, 1.0)


graph_engine = FraudGraphEngine()


# ─────────────────────────────────────────────
# ML SCORING
# ─────────────────────────────────────────────
def ml_score_transaction(features: list) -> dict:
    if not MODELS_LOADED:
        risk = float(np.random.uniform(0, 1))
        return {
            "isolation_forest_risk": round(risk * 0.9, 4),
            "autoencoder_risk": round(risk, 4),
            "combined_ml_risk": round(risk, 4),
            "mode": "TEST_MODE_RANDOM",
        }

    x = np.array(features, dtype=np.float32).reshape(1, -1)

    if_raw  = float(-iso_model.decision_function(x)[0])
    if_risk = float(np.clip((if_raw + 0.5), 0, 1))

    x_t      = torch.FloatTensor(x)
    ae_error = float(ae_model.reconstruction_error(x_t).numpy()[0])
    threshold = scaler_stats.get('threshold', 1.0)
    ae_risk   = float(np.clip(ae_error / (threshold * 2 + 1e-6), 0, 1))

    combined = 0.5 * if_risk + 0.5 * ae_risk
    return {
        "isolation_forest_risk": round(if_risk, 4),
        "autoencoder_risk": round(ae_risk, 4),
        "combined_ml_risk": round(combined, 4),
        "mode": "TRAINED_MODEL",
    }


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class TransactionRequest(BaseModel):
    from_account: str = Field(..., examples=["ACC_001"])
    to_account:   str = Field(..., examples=["ACC_002"])
    amount:       float = Field(..., gt=0, examples=[5000.0])
    features:     Optional[List[float]] = Field(
        default=None, description="30 ML features (V1-V28 + Amount_scaled + Time_scaled)"
    )


class TransactionResponse(BaseModel):
    txn_id: str
    from_account: str
    to_account: str
    amount: float
    ml_risk: dict
    graph_risk: float
    graph_signals: dict
    final_risk_score: float
    risk_level: str
    flagged_as_fraud: bool
    timestamp: str


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="FinGuard API",
    description="Real-time AML and Transaction Fraud Graph Analyzer",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "running",
        "models_loaded": MODELS_LOADED,
        "mongo_connected": MONGO_ON,
        "graph_nodes": graph_engine.G.number_of_nodes(),
        "graph_edges": graph_engine.G.number_of_edges(),
    }


@app.post("/transaction", response_model=TransactionResponse)
def score_transaction(txn: TransactionRequest):
    txn_id    = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.now(timezone.utc)

    features = txn.features or ([0.0] * 29 + [txn.amount / 10000.0])
    if len(features) != 30:
        raise HTTPException(400, "features must have exactly 30 values")

    ml_scores = ml_score_transaction(features)

    graph_engine.add_transaction(txn.from_account, txn.to_account,
                                 txn.amount, txn_id, timestamp)

    graph_risk = graph_engine.compute_graph_risk_score(txn.from_account)
    graph_signals = {
        "cycles": graph_engine.detect_cycles(txn.from_account),
        "fan_out": graph_engine.detect_fan_out(txn.from_account),
        "structuring": graph_engine.detect_structuring(txn.from_account),
        "velocity": graph_engine.velocity_risk(txn.from_account),
    }

    final_score = round(min(0.6 * ml_scores["combined_ml_risk"] + 0.4 * graph_risk, 1.0), 4)
    flagged = final_score > 0.5
    risk_level = ("CRITICAL" if final_score > 0.85 else
                 "HIGH" if final_score > 0.65 else
                 "MEDIUM" if final_score > 0.40 else "LOW")

    result = {
        "txn_id": txn_id,
        "from_account": txn.from_account,
        "to_account": txn.to_account,
        "amount": txn.amount,
        "ml_risk": ml_scores,
        "graph_risk": graph_risk,
        "graph_signals": graph_signals,
        "final_risk_score": final_score,
        "risk_level": risk_level,
        "flagged_as_fraud": flagged,
        "timestamp": timestamp.isoformat(),
    }

    if MONGO_ON:
        txn_col.insert_one({**result, "_id": txn_id})
        if flagged:
            alert_col.insert_one({**result, "_id": txn_id})
    else:
        IN_MEMORY.append(result)

    return result


@app.get("/graph/{account_id}")
def get_graph(account_id: str, depth: int = 2):
    subgraph = graph_engine.get_account_subgraph(account_id, depth)
    if not subgraph["nodes"]:
        raise HTTPException(404, f"Account {account_id} not in graph")

    for node in subgraph["nodes"]:
        node["risk"] = round(graph_engine.compute_graph_risk_score(node["id"]), 3)

    return {
        "account": account_id,
        "depth": depth,
        "subgraph": subgraph,
        "summary": {
            "total_nodes": len(subgraph["nodes"]),
            "total_edges": len(subgraph["edges"]),
            "account_risk": graph_engine.compute_graph_risk_score(account_id),
            "cycle_detected": graph_engine.detect_cycles(account_id)["has_cycle"],
        },
    }


@app.get("/alerts")
def get_alerts(limit: int = 50):
    if MONGO_ON:
        alerts = list(alert_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
    else:
        alerts = list(reversed([r for r in IN_MEMORY if r.get("flagged_as_fraud")]))[:limit]
    return {"alerts": alerts, "count": len(alerts)}


@app.get("/stats")
def get_stats():
    if MONGO_ON:
        total   = txn_col.count_documents({})
        flagged = alert_col.count_documents({})
        high    = alert_col.count_documents({"risk_level": {"$in": ["HIGH", "CRITICAL"]}})
    else:
        total   = len(IN_MEMORY)
        flagged = sum(1 for r in IN_MEMORY if r.get("flagged_as_fraud"))
        high    = sum(1 for r in IN_MEMORY if r.get("risk_level") in ["HIGH", "CRITICAL"])

    cycles_detected = sum(
        1 for n in graph_engine.G.nodes() if graph_engine.detect_cycles(n)["has_cycle"]
    )

    return {
        "total_transactions": total,
        "flagged_count": flagged,
        "high_risk_count": high,
        "flag_rate": round(flagged / max(total, 1) * 100, 2),
        "graph_nodes": graph_engine.G.number_of_nodes(),
        "graph_edges": graph_engine.G.number_of_edges(),
        "cycles_detected": cycles_detected,
    }


# Serve the dashboard HTML directly from FastAPI (optional convenience)
if os.path.exists("static"):
    app.mount("/dashboard", StaticFiles(directory="static", html=True), name="dashboard")
