"""
train.py
Run this FIRST to train and save both ML models.

Usage:
    python train.py --data creditcard.csv

Output:
    isolation_forest.pkl   (scikit-learn model)
    autoencoder.pth        (PyTorch state dict)
    scaler_stats.npy       (mean/std/threshold for inference)
"""

import argparse
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from finguard_models import TransactionAutoencoder
except ModuleNotFoundError as e:
    sys.exit(
        f"\nMissing dependency: {e.name}\n"
        f"Run: pip install -r requirements.txt\n"
        f"(PyTorch is required to train the Autoencoder.)\n"
    )


def load_and_preprocess(path: str):
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} transactions | Fraud: {df['Class'].sum()} "
          f"({df['Class'].mean()*100:.3f}%)")

    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled']   = scaler.fit_transform(df[['Time']])
    df = df.drop(['Amount', 'Time'], axis=1)

    features = [c for c in df.columns if c != 'Class']
    X = df[features].values.astype(np.float32)
    y = df['Class'].values
    return X, y


def train_isolation_forest(X, y, contamination=0.002):
    print("\n[1/2] Training Isolation Forest...")
    iso = IsolationForest(n_estimators=100, contamination=contamination,
                          random_state=42, n_jobs=-1)
    iso.fit(X)
    scores = -iso.decision_function(X)
    preds  = (iso.predict(X) == -1).astype(int)
    auc    = roc_auc_score(y, scores)
    print(f"     ROC-AUC: {auc:.4f}")
    print(classification_report(y, preds, target_names=['Normal', 'Fraud'],
                                zero_division=0))
    return iso


def train_autoencoder(X, y, epochs=50, lr=1e-3, batch_size=256):
    print("[2/2] Training Autoencoder (normal-only)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_normal = X[y == 0]
    X_train, X_val = train_test_split(X_normal, test_size=0.1, random_state=42)

    feat_mean = X_train.mean(axis=0)
    feat_std  = X_train.std(axis=0) + 1e-8

    train_t = torch.FloatTensor(X_train).to(device)
    val_t   = torch.FloatTensor(X_val).to(device)
    loader  = DataLoader(TensorDataset(train_t, train_t),
                         batch_size=batch_size, shuffle=True)

    model   = TransactionAutoencoder(input_dim=X.shape[1]).to(device)
    optim   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for X_batch, _ in loader:
            optim.zero_grad()
            recon = model(X_batch)
            loss  = loss_fn(recon, X_batch)
            loss.backward()
            optim.step()
            batch_losses.append(loss.item())
        train_loss = float(np.mean(batch_losses))

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(val_t), val_t).item()
            print(f"     Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    model.eval()
    X_all_t   = torch.FloatTensor(X).to(device)
    ae_errors = model.reconstruction_error(X_all_t).cpu().numpy()

    normal_errors = ae_errors[y == 0]
    threshold     = float(np.percentile(normal_errors, 95))
    ae_preds      = (ae_errors > threshold).astype(int)
    auc           = roc_auc_score(y, ae_errors)
    print(f"     ROC-AUC: {auc:.4f} | Threshold: {threshold:.4f}")
    print(classification_report(y, ae_preds, target_names=['Normal', 'Fraud'],
                                zero_division=0))

    return model, threshold, feat_mean, feat_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='creditcard.csv', help='Path to creditcard.csv')
    args = parser.parse_args()

    X, y = load_and_preprocess(args.data)

    iso = train_isolation_forest(X, y)
    joblib.dump(iso, 'isolation_forest.pkl')
    print("     Saved: isolation_forest.pkl")

    model, threshold, feat_mean, feat_std = train_autoencoder(X, y)
    torch.save(model.state_dict(), 'autoencoder.pth')
    np.save('scaler_stats.npy',
           {'mean': feat_mean, 'std': feat_std, 'threshold': threshold},
           allow_pickle=True)
    print("     Saved: autoencoder.pth, scaler_stats.npy")

    print("\nTraining complete. Run: uvicorn main:app --reload")


if __name__ == '__main__':
    main()
