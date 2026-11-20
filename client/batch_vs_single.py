"""
client/batch_vs_single.py -- Mission 3: time N single /predict calls against ONE /predict_batch call
with the same N readings, and check that both give the same answers.

    python client/batch_vs_single.py --url http://localhost:8000 --n 400
    python client/batch_vs_single.py --url https://<your-codespace>-8000.app.github.dev --n 400
Needs: pip install requests numpy
"""
import argparse
import time

import numpy as np
import requests


def reading(r):
    # Same random readings as client/bench.py, so results are comparable.
    return {"battery_v": round(r.uniform(3.4, 4.2), 3), "temp_c": round(r.uniform(10, 45), 1),
            "rings_per_day": int(r.integers(0, 30)), "wifi_rssi_dbm": int(r.integers(-88, -45)),
            "age_months": int(r.integers(0, 37))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    url = a.url.rstrip("/")
    r = np.random.default_rng(2)         # fixed seed: the same readings every run
    rows = [reading(r) for _ in range(a.n)]

    s = requests.Session()               # keep-alive: reuse one connection, like bench.py
    s.get(url + "/health", timeout=30)   # warm-up, so neither method pays the first-connection cost

    # 1) N separate requests, one reading each
    t0 = time.perf_counter()
    single, single_server_ms = [], 0.0
    for row in rows:
        resp = s.post(url + "/predict", json=row, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        single.append(body["days_left"])
        single_server_ms += body["timings_ms"]["server_total"]
    single_ms = (time.perf_counter() - t0) * 1000

    # 2) ONE request carrying all N readings
    t0 = time.perf_counter()
    resp = s.post(url + "/predict_batch", json={"rows": rows}, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    batch_ms = (time.perf_counter() - t0) * 1000
    batch = body["days_left"]

    print(f"{a.n} single /predict calls : {single_ms:9.1f} ms total  "
          f"({single_ms / a.n:.2f} ms per reading, server busy {single_server_ms:.1f} ms of it)")
    print(f"1 /predict_batch of {a.n}  : {batch_ms:9.1f} ms total  "
          f"({batch_ms / a.n:.3f} ms per reading, server busy {body['timings_ms']['server_total']:.1f} ms of it)")
    print(f"batch server stages     : {body['timings_ms']}")
    print(f"speed-up                : {single_ms / batch_ms:.1f}x")
    diff = max(abs(x - y) for x, y in zip(single, batch))
    print(f"same answers?           : {'yes' if diff == 0 else 'NO'} (largest difference {diff} days)")


if __name__ == "__main__":
    main()
