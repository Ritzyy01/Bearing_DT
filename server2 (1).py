from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import os, glob, json, time, threading, traceback

app = Flask(__name__)
CORS(app) # Allows web dashboard to fetch data

# ── Configuration ─────────────────────────────────────────────────────────────
BEARING_DATA_DIR = r"C:\Ritesh\Acads\3-2\DT Project\BearingData"
FS               = 97656   # Hz (from assignment spec)
MAX_FFT_HZ       = 10000   # Focus on the 10kHz range where Spectral Kurtosis is highest
TOTAL_DAYS       = 50
STREAM_DELAY_S   = 2.0     # Delay between days to simulate real-time

# ── Thread-Safe Shared State ──────────────────────────────────────────────────
state_lock = threading.Lock()
_current_state = {}
_is_streaming = False

# ── Signal Processing & Math Helpers ──────────────────────────────────────────

def extract_rpm(mat):
    """Extract shaft RPM from cumulative tach angle signal (PPR=2)."""
    try:
        tach_key = next(k for k in mat if not k.startswith("_") and "tach" in k.lower())
        tach     = mat[tach_key].flatten().astype(np.float64)
        fs_tach  = len(tach) / 6.0          # ~426.5 Hz over 6 s recording
        omega    = np.median(np.diff(tach)) * fs_tach   # rad/s
        rpm      = (omega / (2 * np.pi)) * 60.0
        return round(float(rpm), 2) if rpm > 0 else None
    except Exception:
        return None

def pick_vibration_key(mat):
    """Dynamically find the vibration array in the MATLAB dictionary."""
    keys = [k for k in mat if not k.startswith("_")]
    for k in keys:
        if "vibration" in k.lower() or k.lower().endswith(("_de_time", "_time")):
            return k
    raise ValueError("No valid vibration signal found in .mat file")

def compute_health_metrics(raw):
    """Extract time-domain features as defined in the project scope."""
    rms   = float(np.sqrt(np.mean(raw ** 2)))
    kurt  = float(np.mean((raw - raw.mean()) ** 4) / (np.std(raw) ** 4 + 1e-12))-3
    crest = float(np.max(np.abs(raw)) / (rms + 1e-12))

    # Normalize Kurtosis (Healthy ≈ 3, Failing >> 3) to a 100-0 Health Score
    health = ((float(np.clip(100.0 * (1.0 - (kurt) / 2.56), 0.0, 100.0))))

    return {
        "rms": round(rms, 6),
        "kurtosis": round(kurt, 4),
        "crest_factor": round(crest, 4),
        "health_score": round(health, 2),
    }

def process_mat(filepath, day_index, all_health_scores):
    """Process a single day's vibration data."""
    mat = sio.loadmat(filepath)
    key = pick_vibration_key(mat)
    raw = mat[key].flatten().astype(np.float64)
    raw -= raw.mean()

    N = len(raw)
    
    # Frequency Domain (FFT)
    windowed  = raw * np.hanning(N)
    fft_vals  = np.fft.rfft(windowed)
    freqs     = np.fft.rfftfreq(N, d=1.0 / FS)
    amplitude = (2.0 / N) * np.abs(fft_vals)

    mask    = freqs <= MAX_FFT_HZ
    freqs_c = freqs[mask]
    amp_c   = amplitude[mask]

    peak_idx, _ = signal.find_peaks(amp_c, height=np.max(amp_c) * 0.1)
    top   = sorted(peak_idx, key=lambda i: amp_c[i], reverse=True)[:5]
    peaks = [{"freq": round(float(freqs_c[i]), 2), "amp": round(float(amp_c[i]), 6)} for i in top]

    # Health & Features
    health = compute_health_metrics(raw)

    # RUL Prediction (Linear fit over recent history)
    rul_days = None
    if len(all_health_scores) >= 3:
        n = min(10, len(all_health_scores))
        scores = np.array(all_health_scores[-n:])
        days = np.arange(len(scores))
        slope, intercept = np.polyfit(days, scores, 1)
        if slope < 0:
            days_to_zero = -intercept / slope - days[-1]
            rul_days = max(0.0, float(days_to_zero))
        else:
            rul_days = float(TOTAL_DAYS - day_index)

    # RPM & Theoretical Fault Frequencies (Inner Race Fault focus)
    rpm = extract_rpm(mat)
    fault_freqs = {}
    if rpm:
        Nb, Bd, Pd = 9, 7.94, 38.50 # Bearing kinematics parameters
        fr = rpm / 60.0
        ratio = Bd / Pd 
        fault_freqs = {
            "bpfi": round((Nb / 2) * fr * (1 + ratio), 3), # Inner Race Fault Freq
            "bpfo": round((Nb / 2) * fr * (1 - ratio), 3),
        }

    # Downsample array for web transmission
    def ds(arr, n=1000):
        step = max(1, len(arr) // n)
        return arr[::step].tolist()

    return {
        "day_index": day_index + 1,
        "day_total": TOTAL_DAYS,
        "filename": os.path.basename(filepath),
        "raw_signal_sample": ds(raw, 500), # Send smaller payload to prevent lag
        "freqs": ds(freqs_c, 500),
        "amplitude": ds(amp_c, 500),
        "peaks": peaks,
        **health,
        "rul_days": round(rul_days, 1) if rul_days is not None else None,
        "rpm": rpm,
        "fault_freqs": fault_freqs,
    }

# ── Simulation Engine ─────────────────────────────────────────────────────────

def stream_worker():
    """Background thread to process data and update the shared state."""
    global _current_state, _is_streaming
    files = sorted(glob.glob(os.path.join(BEARING_DATA_DIR, "*.mat")))
    health_scores = []
    
    for idx, filepath in enumerate(files):
        if not _is_streaming: break # Stop if interrupted
        
        try:
            result = process_mat(filepath, idx, health_scores)
            health_scores.append(result["health_score"])
            #result["health_score"]=(sum(health_scores) / (len(health_scores) + 1e-12)-88.0)*10/1.2
            with state_lock:
                _current_state = result
                
        except Exception as e:
            traceback.print_exc()
            with state_lock:
                _current_state = {"error": str(e), "day_index": idx + 1}
                
        time.sleep(STREAM_DELAY_S)
        
    _is_streaming = False
# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/start-simulation", methods=["POST"])
def start_simulation():
    """Triggers the background data stream."""
    global _is_streaming
    if _is_streaming:
        return jsonify({"message": "Simulation is already running."}), 400
    
    _is_streaming = True
    thread = threading.Thread(target=stream_worker)
    thread.daemon = True
    thread.start()
    return jsonify({"message": "Simulation started successfully."})

@app.route("/stop-simulation", methods=["POST"])
def stop_simulation():
    """Stops the data stream."""
    global _is_streaming
    _is_streaming = False
    return jsonify({"message": "Simulation stopped."})

@app.route("/current")
def current():
    """Fusion 360 and Web Dashboard hit this endpoint to get real-time state."""
    with state_lock:
        if not _current_state:
            return jsonify({"error": "No data. Call /start-simulation first"}), 404
        return jsonify(_current_state)

@app.route("/debug")
def debug():
    files = sorted(glob.glob(os.path.join(BEARING_DATA_DIR, "*.mat")))
    return jsonify({
        "status": "ok",
        "streaming": _is_streaming,
        "files_found": len(files),
        "directory": BEARING_DATA_DIR
    })
@app.route("/")
def index():
    """Serves the index.html frontend dashboard."""
    # This assumes index.html is in the exact same folder as server.py
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")
if __name__ == "__main__":
    print(f"Starting Digital Twin API on port 5000...")
    print(f"Looking for data in: {BEARING_DATA_DIR}")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)