"""
hi_poller.py
------------
Run this OUTSIDE Fusion 360 (double-click or: python hi_poller.py)
Polls your Flask server every 1.5s and writes the latest HI to hi_log.json.
"""

import urllib.request
import json
import time
import os
from datetime import datetime

SERVER_URL    = "http://127.0.0.1:5000/current"
OUTPUT_JSON   = r"C:\Ritesh\Acads\3-2\DT Project\hi_log.json"
POLL_INTERVAL = 1.5

print(f"Polling {SERVER_URL} every {POLL_INTERVAL}s ...")
print(f"Writing to: {OUTPUT_JSON}")
print("Press Ctrl+C to stop.\n")

while True:
    try:
        with urllib.request.urlopen(SERVER_URL, timeout=2) as resp:
            data = json.loads(resp.read().decode())

        if "error" not in data:
            output = {
                "health_indicator_pca": data.get("health_indicator_pca"),
                "day_index":            data.get("day_index"),
                "rms":                  data.get("rms"),
                "kurtosis":             data.get("kurtosis"),
                "crest_factor":         data.get("crest_factor"),
                "timestamp":            datetime.now().isoformat()
            }

            tmp_path = OUTPUT_JSON + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(output, f, indent=2)
            os.replace(tmp_path, OUTPUT_JSON)

            print(f"Day {output['day_index']:>3} | HI: {output['health_indicator_pca']:.4f}", end="\r")

    except Exception as e:
        print(f"\n[Waiting for server...] {e}", end="\r")

    time.sleep(POLL_INTERVAL)