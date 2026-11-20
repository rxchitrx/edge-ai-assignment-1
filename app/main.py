"""
app/main.py -- STARTER battery API. Look for TODO.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

GET  /health          -> {"status": "ok", "team": 7, "params": 1281, "val_mae": 3.4}
GET  /model_info      -> the per-layer table: params, bytes, MACs, and whether it fits the doorbell
POST /predict         one reading  -> {"days_left": 52.1, "timings_ms": {...}}
POST /predict_batch   {"rows": [reading, ...]} -> {"days_left": [...], "timings_ms": {...}}
A reading that is out of range gets HTTP 422 with the reason, never a guess.
"""
import json
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.model import FEATURES, BatteryNet

cfg = json.load(open("model/config.json"))
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load("model/model.pt", weights_only=True))
net.eval()
torch.set_num_threads(1)
MU, SD = np.array(cfg["mu"], np.float32), np.array(cfg["sd"], np.float32)
N_PARAMS = sum(p.numel() for p in net.parameters())


class Reading(BaseModel):
    # Limits come from what real hardware can report, not from our data's min/max, so a real
    # reading just outside the training data still gets an answer while a glitch gets HTTP 422.
    # ge = "greater or equal", le = "less or equal", lt = "strictly less than".
    model_config = ConfigDict(allow_inf_nan=False)  # NaN / Infinity are never real readings

    # Single Li-ion cell: dead/cut-off around 2.5-3.0 V, full charge 4.2 V (4.35 V for HV cells).
    # Glitch seen in our data: 0.0 V (disconnected ADC).
    battery_v: float = Field(ge=2.5, le=4.5)
    # Industrial electronics are rated -40..85 C, which covers any outdoor doorbell.
    # Glitch seen in our data: -127 C (DS18B20 "sensor not responding" code).
    temp_c: float = Field(ge=-40, le=85)
    # A count, so never negative. 1000 rings a day is already far beyond a busy front door.
    rings_per_day: float = Field(ge=0, le=1000)
    # Wi-Fi RSSI is always negative: about -120 dBm is below any receiver's sensitivity, and even
    # right next to the router it stays below about -10 dBm. Glitch seen in our data: 0 dBm.
    wifi_rssi_dbm: float = Field(ge=-120, le=-10)
    # Age can't be negative; 240 months (20 years) is far beyond a doorbell's life.
    age_months: float = Field(ge=0, le=240)


class Batch(BaseModel):
    rows: list[Reading] = Field(min_length=1, max_length=10000)


app = FastAPI(title="Doorbell battery API")
ms = lambda t0: round((time.perf_counter() - t0) * 1000, 3)


# ESP32-S3 has about 300 KB of SRAM free for our program. 1 KB = 1024 bytes (memory sizes are
# powers of two), so the budget is 300 * 1024 = 307,200 bytes.
SRAM_BUDGET_BYTES = 300 * 1024


def describe(network):
    """Per-layer table for ANY BatteryNet, read from the network itself (used by /model_info
    and by the Mission 4 width table, so both always count the same way)."""
    layers = []

    for layer in network.net:
        if not isinstance(layer, torch.nn.Linear):
            continue

        inputs = layer.in_features
        outputs = layer.out_features
        macs = inputs * outputs
        params = macs + outputs

        layers.append({
            "name": f"linear{len(layers) + 1}",
            "in": inputs,
            "out": outputs,
            "params": params,
            "bytes_f32": params * 4,
            "macs": macs,
        })

    total_params = sum(layer["params"] for layer in layers)
    total_bytes_f32 = sum(layer["bytes_f32"] for layer in layers)  # 4 bytes per parameter
    total_bytes_int8 = total_params                                  # 1 byte per parameter
    return {
        "layers": layers,
        "total_params": total_params,
        "total_bytes_f32": total_bytes_f32,
        "total_bytes_int8": total_bytes_int8,
        "total_macs": sum(layer["macs"] for layer in layers),
        "fits_300kb_sram_f32": total_bytes_f32 <= SRAM_BUDGET_BYTES,
        "fits_300kb_sram_int8": total_bytes_int8 <= SRAM_BUDGET_BYTES,
    }


@app.get("/model_info")
def model_info():
    return describe(net)


@app.get("/health")
def health():
    return {"status": "ok", "team": cfg["team"], "params": N_PARAMS, "val_mae": cfg["val_mae"]}


async def run(request: Request, schema):
    t_all = time.perf_counter(); t = {}
    t0 = time.perf_counter()
    try:
        data = schema.model_validate_json(await request.body())
    except ValidationError as e:
        # include_input=False: don't echo the bad value back. A NaN input can't be written as JSON,
        # so echoing it would crash this 422 reply into a 500.
        detail = json.loads(e.json(include_input=False, include_url=False))
        return JSONResponse(status_code=422, content={"error": "bad reading", "detail": detail})
    t["parse"] = ms(t0)
    rows = data.rows if isinstance(data, Batch) else [data]
    # Each stage gets its own stopwatch, started right before and read right after it.
    t0 = time.perf_counter()
    x = (np.array([[getattr(r, f) for f in FEATURES] for r in rows], np.float32) - MU) / SD
    t["normalise"] = ms(t0)

    t0 = time.perf_counter()
    with torch.no_grad():
        y = net(torch.from_numpy(x)).numpy()
    t["inference"] = ms(t0)

    t0 = time.perf_counter()
    days = [round(float(v), 2) for v in np.maximum(y, 0)]
    body = {"days_left": days if isinstance(data, Batch) else days[0]}
    t["build"] = ms(t0)
    t["server_total"] = ms(t_all)
    body["timings_ms"] = t
    return body


@app.post("/predict")
async def predict(request: Request):
    return await run(request, Reading)


@app.post("/predict_batch")
async def predict_batch(request: Request):
    return await run(request, Batch)
