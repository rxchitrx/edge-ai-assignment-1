# Mission 2 Report: A Small Network, Opened Up
**Team 16 (Team MH)** · Data: `data/team_16_train.csv` (1,976 rows after dropping 24 glitches;
1,581 train / 395 validation, seed 0)

---

## 1. Parameters, bytes and MACs by hand
**Rule for a `Linear(a, b)` layer:** a×b weights + b biases = parameters · a×b multiply-accumulates (MACs)
per prediction · float32 = 4 bytes per parameter, int8 = 1 byte. ReLU layers have no parameters.

### 5 → 32 → 32 → 1 (the starter network)
| Layer | in → out | Parameters | float32 bytes | MACs |
|---|---|---|---|---|
| linear1 | 5 → 32 | 5×32 + 32 = **192** | 768 | 160 |
| linear2 | 32 → 32 | 32×32 + 32 = **1,056** | 4,224 | 1,024 |
| linear3 | 32 → 1 | 32×1 + 1 = **33** | 132 | 32 |
| **Total** | | **1,281** | **5,124** | **1,216** |

### 5 → 16 → 16 → 1 (our served model)
| Layer | in → out | Parameters | float32 bytes | MACs |
|---|---|---|---|---|
| linear1 | 5 → 16 | 5×16 + 16 = **96** | 384 | 80 |
| linear2 | 16 → 16 | 16×16 + 16 = **272** | 1,088 | 256 |
| linear3 | 16 → 1 | 16×1 + 1 = **17** | 68 | 16 |
| **Total** | | **385** | **1,540** | **352** |

**Check:** `GET /model_info` computes this table from the network itself (`describe(net)` in
`app/main.py`) and returns exactly these numbers: 385 params, 1,540 bytes, 352 MACs. `/health` also
reports `params: 385`. The 32,32 totals match the README's example (1,281 / 5,124 / 1,216).

✏️ Each of us should redo one table on paper and check it against the above.

---

## 2. Normalisation
**What:** each input becomes `(x - mu) / sd`, where `mu` and `sd` are the mean and standard deviation
of that column computed from the **training rows only** (`train.py`). They're saved in
`model/config.json`, and the server applies exactly the same transformation to every request.

**Why training rows only:** the validation rows stand in for data the model has never seen. If
`mu`/`sd` came from all rows, information from the validation set would leak into training, and the
validation error would look better than what we'd get on truly new readings (the company's 400).

**With vs. without (same 5 → 32 → 32 → 1, 1500 epochs, seed 0):**
| | Validation MAE |
|---|---|
| Without normalisation (starter) | 9.59 days |
| With normalisation | **4.91 days** |

**Why it helps:** the raw inputs live on very different scales. `battery_v` moves within about 1 V
(sd 0.24), while `wifi_rssi_dbm` spans tens of dBm (sd 12.6). With one learning rate for every weight,
the large-scale inputs dominate the early gradients, and the network barely learns from `battery_v`,
which is the input that matters most. After normalisation, every input has a mean of 0 and a
spread of about 1, so they all learn at a similar pace.

---

## 3. Width sweep (1 and 2 hidden layers)
All with normalisation, 1500 epochs, seed 0. Runs are saved in `model_runs/`; the plot is
`graphs/model_tradeoff_plot.png`.

| Hidden layers | Parameters | Val MAE (days) | | Hidden layers | Parameters | Val MAE (days) |
|---|---|---|---|---|---|---|
| 4 | 29 | 13.06 | | 4,4 | 49 | 8.95 |
| 8 | 57 | 15.42 | | 8,8 | 129 | 6.02 |
| 16 | 113 | 10.52 | | **16,16** | **385** | **4.97** |
| 32 | 225 | 9.96 | | 32,32 | 1,281 | 4.91 |
| 64 | 449 | 15.42 | | 64,64 | 4,609 | 4.65 |

**What it shows:**
- **Two hidden layers beat one at every size**, even when the one-layer net has more parameters
  (64 → 449 params, 15.42 days vs 8,8 → 129 params, 6.02 days). Stacking two ReLU layers lets the
  network build the curved shapes we saw in Mission 1 (battery S-curve, 1/rings) far more cheaply.
- **Two layers: diminishing returns after 16,16.** 32,32 has 3.3× the parameters for 0.06 days better;
  64,64 has 12× the parameters for 0.32 days better.
- **Open question for our Skeptic:** one-layer widths 8 and 64 both scored exactly 15.42. Bigger one-layer
  nets should usually do no worse. ✏️ Rerun `python train.py --team 16 --hidden 64` and check whether the
  number repeats (and whether the val MAE is still falling at epoch 1499, i.e. undertrained).

---

## 4. Overfitting check (chosen size 16,16)
`train.py` now prints training MAE next to validation MAE at every checkpoint, and saves `train_mae` in
the run's `config.json`. Retraining 16,16 (`model_runs/team_16_hidden_16,16_n56s4ceh/`) gave:

| Epoch | Train MAE | Val MAE |
|---|---|---|
| 0 | 74.09 | 72.03 |
| 300 | 5.71 | 5.58 |
| 600 | 5.14 | 5.49 |
| 900 | 4.93 | 5.34 |
| 1200 | 4.67 | 5.13 |
| 1499 | **4.40** | **4.97** |

We also measured the **served** model (`model/model.pt`) on the same split: **train 4.40, val 4.97**.

**Is it overfitting? No, not meaningfully.** The gap is only +0.58 days (about 13%), and validation MAE
is **still falling** at the last epoch. A model that overfits shows training error dropping while
validation error stops improving or rises; ours hasn't reached that point. With 385 parameters and
1,581 training rows (about 4 rows per parameter), the network doesn't have the capacity to memorise.

---

## 5. Final model and expected error on the company's 400 readings
**Choice: 5 → 16 → 16 → 1** (385 parameters, 1,540 bytes float32, 352 MACs).
- Within 0.3 days of the best model we tried (64,64), at one-twelfth of the size.
- Tiny enough to fit the doorbell's SRAM many times over (Mission 4): `fits_300kb_sram_f32: true`.
- Not overfitting (section 4).

**Expected MAE on the 400 unseen readings: about 5.0 to 5.5 days.**
Reasoning: our validation MAE (4.97 days) is measured on 395 rows the network never trained on, so it's
the best estimate we have. We expect slightly worse, not better, because (a) we picked 16,16 partly by
looking at validation scores, which makes 4.97 a little optimistic; (b) 395 rows is a small sample, so
the true error could easily be a few tenths of a day either side; (c) the company's readings may include
values near the edges of our ranges, where we have little training data.
✏️ Team: agree on this number and keep it; it's what we'll compare against when marks come back.
