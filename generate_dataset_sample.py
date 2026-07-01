"""
generate_dataset_sample.py

If you don't want to download the full 144MB Kaggle dataset,
this generates a realistic SYNTHETIC version with the same structure
(284 features V1-V28, Amount, Time, Class) so you can test the
full pipeline end-to-end immediately.

For your actual project submission, use the REAL Kaggle dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Usage:
    python generate_dataset_sample.py
    -> creates creditcard.csv (synthetic, 10000 rows)
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N_NORMAL = 9800
N_FRAUD  = 200

# Normal transactions: tight multivariate cluster
normal = np.random.randn(N_NORMAL, 28) * 1.0
normal_amount = np.abs(np.random.normal(80, 60, N_NORMAL))
normal_time   = np.sort(np.random.uniform(0, 172800, N_NORMAL))  # 2 days in seconds

# Fraud: shifted distribution + occasional large amounts
fraud = np.random.randn(N_FRAUD, 28) * 2.5 + 1.5
fraud_amount = np.abs(np.random.normal(450, 350, N_FRAUD))
fraud_time   = np.sort(np.random.uniform(0, 172800, N_FRAUD))

X      = np.vstack([normal, fraud])
amount = np.concatenate([normal_amount, fraud_amount])
time   = np.concatenate([normal_time, fraud_time])
labels = np.array([0]*N_NORMAL + [1]*N_FRAUD)

# Shuffle
idx = np.random.permutation(len(X))
X, amount, time, labels = X[idx], amount[idx], time[idx], labels[idx]

cols = [f'V{i}' for i in range(1, 29)]
df = pd.DataFrame(X, columns=cols)
df['Time']   = time
df['Amount'] = amount
df['Class']  = labels

df.to_csv('creditcard.csv', index=False)
print(f"Created creditcard.csv with {len(df):,} rows ({labels.sum()} fraud, "
      f"{labels.mean()*100:.2f}%)")
print("NOTE: this is SYNTHETIC data for testing. Use the real Kaggle dataset")
print("for your actual project submission and report.")
