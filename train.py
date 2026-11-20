"""
train.py -- STARTER. Clean the data, fit baselines, train the network, save it for the server.

    python train.py --team 7 --hidden 32,32
Writes model.pt and config.json into a new folder under model_runs/.
Copy the chosen run's two files into model/ to serve it with app/main.py.

Look for TODO. The training loop is given; the decisions are yours.
"""
import argparse
import json
import os
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import torch

from app.model import FEATURES, BatteryNet

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--hidden", default="32,32")
ap.add_argument("--epochs", type=int, default=1500)
a = ap.parse_args()
hidden = tuple(int(h) for h in a.hidden.split(","))

df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")
# TODO Mission 1: find the sensor glitches (plot every column first!) and write a rule that
# marks them. Say in your report how many you found and why each rule is safe.
bad = (
    (df["battery_v"] <= 0)
    | (df["temp_c"] < -20)
    | (df["wifi_rssi_dbm"] >= 0)
)
print(f"{len(df)} rows, {bad.sum()} glitch rows dropped")
df = df[~bad]

rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
X, y = tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32)
Xv, yv = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)
# Reuse training statistics for validation and inference.
mu = X.mean(axis=0)
sd = X.std(axis=0)
sd[sd == 0] = 1

mae = lambda p, t: float(np.mean(np.abs(p - t)))
print(f"baseline (predict the mean): {mae(np.full_like(yv, y.mean()), yv):.2f} days")

X_train_linear = np.column_stack((X, np.ones(len(X))))
X_val_linear = np.column_stack((Xv, np.ones(len(Xv))))
coefficients, _, _, _ = np.linalg.lstsq(X_train_linear, y, rcond=None)
linear_predictions = X_val_linear @ coefficients
print(f"baseline (linear regression): {mae(linear_predictions, yv):.2f} days")

torch.manual_seed(0)
net = BatteryNet(hidden)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
Xvt = torch.tensor((Xv - mu) / sd)
for ep in range(a.epochs):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        b = perm[i:i + 128]
        loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 300 == 0 or ep == a.epochs - 1:
        with torch.no_grad():
            print(f"  epoch {ep:4d}  train MAE {mae(net(Xt).numpy(), y):.2f}  "
                  f"val MAE {mae(net(Xvt).numpy(), yv):.2f} days")
with torch.no_grad():
    # Mission 2 overfitting check: error on rows the network learned from vs rows it never saw.
    train_mae = mae(net(Xt).numpy(), y)
    val_mae = mae(net(Xvt).numpy(), yv)
print(f"network {hidden}:            train {train_mae:.2f} days, val {val_mae:.2f} days "
      f"(gap {val_mae - train_mae:+.2f})")

os.makedirs("model_runs", exist_ok=True)
widths = ",".join(str(width) for width in hidden)
run_dir = mkdtemp(prefix=f"team_{a.team:02d}_hidden_{widths}_", dir="model_runs")
torch.save(net.state_dict(), os.path.join(run_dir, "model.pt"))
with open(os.path.join(run_dir, "config.json"), "w") as config_file:
    json.dump(dict(team=a.team, hidden=list(hidden), mu=mu.tolist(), sd=sd.tolist(),
                   features=FEATURES, train_mae=round(train_mae, 2), val_mae=round(val_mae, 2)),
              config_file, indent=2)
print(f"Saved model and config to {run_dir}")
