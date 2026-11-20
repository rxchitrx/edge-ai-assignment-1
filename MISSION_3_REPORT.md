# Mission 3 Report: Serve It, and Time Every Stage
**Team 16 (Team MH)** · Model served: 5 → 16 → 16 → 1 (385 params, val MAE 4.97 days)

Every item follows the loop: **predict with a number → measure → explain the gap.**

---

## 0. Before Mission 3: the LLM's prediction
We asked an LLM, before measuring: *"In a FastAPI server running a tiny PyTorch model, which stage
of a web request will be the slowest?"*

> LLM answer: Network latency is likely the bottleneck, not neural network inference.

## 1. Our predictions (written before measuring)


| Quantity | Our prediction |
|---|---|
| Model inference time for one reading |1 ms |
| Laptop → Codespace round trip | 100-300 ms |
| Slowest stage of a request |Network Communication|

---

## 2. Input ranges: glitches get HTTP 422
Code: `class Reading` in `app/main.py`. The limits come from **what real hardware can report**, not
from our data's min and max, so a real reading slightly outside our data still gets an answer.

| Field | Range accepted | Why | Glitch in our data |
|---|---|---|---|
| `battery_v` | 2.5 to 4.5 V | Li-ion cut-off is about 2.5–3.0 V; full charge is 4.2 V (4.35 V for HV cells) | 0.0 V |
| `temp_c` | -40 to 85 °C | Industrial electronics rating, covers any outdoor doorbell | -127 °C (DS18B20 error code) |
| `rings_per_day` | 0 to 1000 | A count can't be negative; 1000/day is far beyond any front door | none |
| `wifi_rssi_dbm` | -120 to -10 dBm | RSSI is always negative; -120 is below any receiver's sensitivity, about -10 is right next to the router | 0 dBm |
| `age_months` | 0 to 240 | Can't be negative; 20 years is far beyond a doorbell's life | none |
| any field | must be a finite number | `NaN` / `Infinity` are never real readings (`allow_inf_nan=False`) | — |

### Test results (local server)
| Reading sent | Expected | Got |
|---|---|---|
| Normal reading | 200 | **200**, 139.72 days |
| `battery_v: 0` | 422 | **422** |
| `temp_c: -127` | 422 | **422** |
| `wifi_rssi_dbm: 0` | 422 | **422** |
| `rings_per_day: -3` | 422 | **422** |
| `age_months: -1` | 422 | **422** |
| `age_months` missing | 422 | **422** |
| `temp_c: "hot"` | 422 | **422** |
| `temp_c: NaN` / `Infinity` | 422 | **422** (was 500 before the fix below) |
| Heatwave `temp_c: 49` (real, outside our data) | 200 | **200** |
| `battery_v: 3.30` (real, below our data) | 200 | **200** |
| Batch with one glitch row | 422 | **422** (the whole batch is refused) |

Repeatable evidence: `python client/test_glitches.py --url http://127.0.0.1:8000`
```
OK   normal reading                           expected 200, got 200  days_left=139.72
OK   battery_v 0 V (glitch in our data)       expected 422, got 422  ['battery_v'] Input should be greater than or equal to 2.5
OK   temp_c -127 C (glitch in our data)       expected 422, got 422  ['temp_c'] Input should be greater than or equal to -40
OK   wifi_rssi_dbm 0 (glitch in our data)     expected 422, got 422  ['wifi_rssi_dbm'] Input should be less than or equal to -10
OK   rings_per_day -3                         expected 422, got 422  ['rings_per_day'] Input should be greater than or equal to 0
OK   age_months -1                            expected 422, got 422  ['age_months'] Input should be greater than or equal to 0
OK   age_months missing                       expected 422, got 422  ['age_months'] Field required
OK   temp_c is text                           expected 422, got 422  ['temp_c'] Input should be a valid number, unable to parse string as a number
OK   temp_c NaN                               expected 422, got 422  ['temp_c'] Input should be a finite number
OK   heatwave 49 C (real, outside our data)   expected 200, got 200  days_left=57.46
OK   battery 3.30 V (real, below our data)    expected 200, got 200  days_left=0.0
OK   bench.py extremes                        expected 200, got 200  days_left=2.83

12/12 as expected
```
Optional: screenshots of three 422 replies from `/docs`.

**Bug found while testing:** `NaN` was first answered with **500 Internal Server Error**, not 422.
The rule had worked: Pydantic refused NaN. But the 422 reply copied the bad input back
into its error message, and `NaN` is not valid JSON, so building the reply crashed.
Fix: `e.json(include_input=False)`. The reply still says which field was wrong and why.

---

## 3. Stage timing
Each stage in `run()` (`app/main.py`) has its own stopwatch (`time.perf_counter()`), the same way
`parse` was timed in the starter:

| Stage | What it does |
|---|---|
| `parse` | Read the request body and validate it against `Reading` (ranges included) |
| `normalise` | Turn readings into a NumPy array and apply `(x - MU) / SD` |
| `inference` | Run the network with `torch.no_grad()` |
| `build` | Clip to ≥ 0, round, build the reply dictionary |
| `server_total` | Everything above, from start to finish |

---

## 4. Measurements

### 4a. Local run (laptop → server on the same laptop), `--n 200`
Run on Windows against `http://127.0.0.1:8000`. Files: `bench_20261120_222425.csv` (keep-alive)
and `bench_20261120_222429.csv` (no keep-alive).

| Stage (p50 / p95 / p99 ms) | keep-alive | `--no-keepalive` |
|---|---|---|
| **first request** | 4.8 | 21.4 |
| round_trip | 2.93 / 4.19 / 5.35 | 14.83 / 29.72 / 31.13 |
| parse | 0.042 / 0.087 / 0.141 | 0.069 / 0.122 / 0.230 |
| normalise | 0.035 / 0.058 / 0.074 | 0.058 / 0.082 / 0.127 |
| inference | 0.233 / 0.411 / 0.552 | 0.383 / 0.696 / 0.935 |
| build | 0.018 / 0.034 / 0.045 | 0.024 / 0.043 / 0.053 |
| server_total | 0.355 / 0.606 / 0.800 | 0.565 / 0.995 / 1.286 |
| network_and_framework | 2.56 / 3.68 / 4.86 | 14.33 / 29.17 / 30.46 |

**`localhost` vs `127.0.0.1` (surprise found while measuring):** with `--url http://localhost:8000`,
the first request took **2073 ms**, and `--no-keepalive` took about 2 s for *every* request.
With `127.0.0.1` the first request took **4.8 ms**. The cause is on the Windows client side, not
the server: `localhost` is tried as IPv6 (`::1`) first, uvicorn only listens on IPv4, and Windows
waits about 2 s before falling back to IPv4. With keep-alive this cost is paid only once, on the
first connection; without keep-alive it's paid on every request.

### 4b. Laptop → Codespace (the real one)
Server in GitHub Codespaces (branch `rishi`), port 8000 Public. Measured on 21 Nov 2026 from the laptop
over home Wi-Fi. Files: `bench_20261121_001335.csv` (keep-alive), `bench_20261121_001607.csv` (no keep-alive).
```
python client/bench.py --url https://<codespace>-8000.app.github.dev --n 200
python client/bench.py --url https://<codespace>-8000.app.github.dev --n 200 --no-keepalive
python client/bench.py --url http://localhost:8000 --n 200      # inside the Codespace terminal
```

| Stage (p50 / p95 / p99 ms) | Laptop → Codespace | Laptop → Codespace, `--no-keepalive` | Inside Codespace (localhost) |
|---|---|---|---|
| **first request** | **584.8** | **681.4** | **6.5** |
| round_trip | 40.87 / 310.61 / 700.50 | 594.24 / 1689.21 / 2547.55 | 2.13 / 4.12 / 5.20 |
| parse | 0.051 / 0.066 / 0.084 | 0.047 / 0.071 / 0.078 | 0.027 / 0.179 / 0.435 |
| normalise | 0.046 / 0.067 / 0.084 | 0.047 / 0.076 / 0.084 | 0.022 / 0.033 / 0.056 |
| inference | 0.230 / 0.321 / 0.371 | 0.228 / 0.303 / 0.363 | 0.130 / 0.205 / 0.364 |
| build | 0.021 / 0.036 / 0.060 | 0.020 / 0.028 / 0.053 | 0.016 / 0.018 / 0.030 |
| server_total | 0.376 / 0.490 / 0.591 | 0.373 / 0.485 / 0.563 | 0.234 / 0.505 / 1.030 |
| network_and_framework | 40.40 / 310.24 / 700.00 | 593.83 / 1688.71 / 2547.17 | 1.84 / 3.72 / 4.83 |

The inside-Codespace run is saved in the Codespace as `bench_20261120_185806.csv` (Codespace clock is UTC).

**Live checks on the public address (same day):** `client/test_glitches.py` → 12/12 as expected,
and our imitation of the grading checks → Health, model_info, Broken readings, Stage timing and
Batch all PASS.

### 4c. 400 single `/predict` calls vs one `/predict_batch` of 400
Script: `client/batch_vs_single.py` (same random readings as `bench.py`, fixed seed, keep-alive).

| | Local (laptop → laptop) | **Laptop → Codespace** |
|---|---|---|
| 400 single calls | 1,102 ms (2.75 ms per reading) | **22,127 ms** (55.3 ms per reading; server busy only 167 ms of it) |
| 1 batch of 400 | 7.4 ms | **84.0 ms** (0.21 ms per reading; server busy 13.9 ms) |
| Speed-up | about 150× | **about 263×** |
| Same answers? | Yes (0.0 days difference) | **Yes** (0.0 days difference) |
| Batch server stages | parse 1.35 · normalise 0.56 · inference 0.35 · build 0.23 ms | parse 12.11 · normalise 1.24 · inference 0.30 · build 0.19 ms |

---

## 5. Where does the time go?
Typical request, laptop → Codespace with keep-alive, p50: **round trip 40.87 ms**

```
network + framework  ██████████████████████████████████████████████████  40.40 ms  (98.9%)
inference            ▏                                                     0.23 ms  ( 0.6%)
parse                ▏                                                     0.05 ms  ( 0.1%)
normalise            ▏                                                     0.05 ms  ( 0.1%)
build                ▏                                                     0.02 ms  ( 0.05%)
                     └─ all server work together (server_total) = 0.38 ms, under 1% of the round trip
```
(p50s are taken per column, so they don't add up exactly.)

The neural network is **about 1/180th** of a typical request. Almost all the time is spent getting the
request from the laptop to GitHub's servers, through GitHub's port-forwarding proxy into the
Codespace, through uvicorn/FastAPI, and back.

**Was the LLM right?** **Yes.** It said network latency, not inference, would be the bottleneck, and
network + framework is 98.9% of the p50 round trip. If it had been asked for numbers, it would also
have needed to know the tail is much worse than the typical case: p95 is 311 ms and p99 is 700 ms,
while `server_total` stays under 0.6 ms even at p99. All of that variation comes from the
network, not our code.

**Our predictions vs measurements:**
| Quantity | We predicted | We measured | Gap and why |
|---|---|---|---|
| Inference, one reading | 1 ms | **0.23 ms** (p50) | 4× faster than we guessed: 352 multiply-adds is tiny for a CPU. Most of the 0.23 ms is PyTorch's per-call overhead, not arithmetic. |
| Laptop → Codespace round trip | 100–300 ms | **41 ms** p50, 311 ms p95, 585 ms first request | The typical request is faster than we guessed, because keep-alive reuses one connection. Our range describes the slow tail and the first request, which pay extra (sections 6.1 and 6.2). |
| Slowest stage | Network communication | Network + framework, **98.9%** | ✅ Correct |

## 6. Explanations

**6.1 Why is `--no-keepalive` so much slower? (41 → 594 ms at p50, about 15×)**
With keep-alive, `bench.py` opens one HTTPS connection and reuses it for all 200 requests. With
`--no-keepalive`, every request opens a brand-new connection: a TCP handshake, then a **TLS
handshake** to agree on encryption keys. Each handshake is one or more extra trips across the
internet, and GitHub's proxy also has to set up a fresh route to our Codespace. That adds about
**550 ms per request**, more than 1,000× what the server itself spends (0.37 ms). Locally the same
change only cost +12 ms (2.9 → 14.8 ms), because there is no TLS and no distance.
**Lesson:** for a doorbell talking to a server, keep the connection open, or the handshakes cost
far more time (and radio power) than the prediction.

**6.2 Why is the first request slower? (585 ms vs 41 ms typical)**
It pays the same connection setup (TCP + TLS + proxy route) that every `--no-keepalive` request pays.
The two numbers match: the first keep-alive request (585 ms) is close to a typical no-keepalive
request (594 ms). The server isn't the cause: `server_total` is under 1 ms on every request.

**6.3 Why is localhost inside the Codespace faster than laptop → Codespace?**
**2.1 ms vs 40.9 ms at p50 (about 19× faster), and a 6.5 ms first request vs 585 ms.** Inside the
Codespace the request never leaves the machine: no internet distance, no TLS (plain `http://`), and
no GitHub port-forwarding proxy. What's left (1.84 ms of network + framework) is just Python's
`requests` library and uvicorn/FastAPI handling HTTP on one computer. That's close to our laptop-only
run (2.9 ms round trip, 2.56 ms network + framework).

The server's own work is **the same order in every column**, well under 1 ms (p50 `server_total`
0.23–0.38 ms). So the 39 ms gap between "laptop → Codespace" and "inside Codespace" is network, not
our code. The inside run is even a little faster (0.23 vs 0.38 ms). A likely reason: here requests
arrive back-to-back every ~2 ms, so the CPU stays busy and its caches stay warm, while over the
internet the server sits idle about 40 ms between requests. Its p99 (1.03 ms) is a little higher
because the benchmark client and the server share the same small Codespace CPU.

**6.4 Why is one batch 263× faster than 400 singles? (22.1 s vs 84 ms)**
400 singles = **400 trips** over the internet, each paying about 41–55 ms of network + framework time.
22,127 ms ÷ 400 = 55 ms per reading, close to our 4b round trip (the average is pulled above the
41 ms p50 by the slow tail). One batch = **1 trip**: the network cost is paid once. Inside the server,
inference for 400 rows took **0.30 ms**, about the same as for 1 row (0.23 ms), because the network
does all 400 as one matrix multiplication. The biggest batch stage is actually `parse` (12 ms), because
the range checks run on every one of the 400 readings. The speed-up is bigger over the internet
(263×) than locally (150×) because each saved trip is worth more.
