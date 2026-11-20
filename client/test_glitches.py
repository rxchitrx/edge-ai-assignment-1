"""
client/test_glitches.py -- Mission 3: send broken and real-but-unusual readings to /predict and
check each one gets the expected HTTP code (422 = refused, 200 = answered).

    python client/test_glitches.py --url http://127.0.0.1:8000
    python client/test_glitches.py --url https://<your-codespace>-8000.app.github.dev
Needs: pip install requests
"""
import argparse
import json

import requests

GOOD = {"battery_v": 3.86, "temp_c": 28.0, "rings_per_day": 5, "wifi_rssi_dbm": -66, "age_months": 22}

# (description, body as raw JSON text, expected HTTP code)
CASES = [
    ("normal reading", GOOD, 200),
    ("battery_v 0 V (glitch in our data)", {**GOOD, "battery_v": 0.0}, 422),
    ("temp_c -127 C (glitch in our data)", {**GOOD, "temp_c": -127.0}, 422),
    ("wifi_rssi_dbm 0 (glitch in our data)", {**GOOD, "wifi_rssi_dbm": 0}, 422),
    ("rings_per_day -3", {**GOOD, "rings_per_day": -3}, 422),
    ("age_months -1", {**GOOD, "age_months": -1}, 422),
    ("age_months missing", {k: v for k, v in GOOD.items() if k != "age_months"}, 422),
    ("temp_c is text", {**GOOD, "temp_c": "hot"}, 422),
    ("temp_c NaN", {**GOOD, "temp_c": float("nan")}, 422),
    ("heatwave 49 C (real, outside our data)", {**GOOD, "temp_c": 49.0}, 200),
    ("battery 3.30 V (real, below our data)", {**GOOD, "battery_v": 3.30}, 200),
    ("bench.py extremes", {"battery_v": 3.4, "temp_c": 10.0, "rings_per_day": 29,
                           "wifi_rssi_dbm": -88, "age_months": 36}, 200),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    url = ap.parse_args().url.rstrip("/") + "/predict"
    passed = 0
    for name, body, want in CASES:
        # json.dumps writes float("nan") as NaN, the way a buggy sensor client might send it
        resp = requests.post(url, data=json.dumps(body), timeout=30,
                             headers={"Content-Type": "application/json"})
        got = resp.status_code
        if got == 200:
            why = f"days_left={resp.json()['days_left']}"
        else:
            d = resp.json().get("detail", [{}])[0]
            why = f"{d.get('loc')} {d.get('msg')}"
        mark = "OK  " if got == want else "FAIL"
        passed += got == want
        print(f"{mark} {name:40s} expected {want}, got {got}  {why}")
    print(f"\n{passed}/{len(CASES)} as expected")


if __name__ == "__main__":
    main()
