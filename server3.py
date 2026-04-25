from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import numpy as np
import scipy.io as sio
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os, glob, time, threading, traceback

app = Flask(__name__)
CORS(app)

# ── Configuration ─────────────────────────────────────────────────────────────
BEARING_DATA_DIR = r"C:\Ritesh\Acads\3-2\DT Project\BearingData"
FS               = 97656   
MAX_FFT_HZ       = 10000   # Limit FFT to 10kHz (where inner race faults show best)
TOTAL_DAYS       = 50
TRAIN_DAYS       = 20      
STREAM_DELAY_S   = 1.5     # Sped up slightly for a better viewing experience

# ── Thread-Safe Shared State ──────────────────────────────────────────────────
state_lock = threading.Lock()
_current_state = {}
_is_streaming = False

# ── Feature Engineering & PCA Engine ──────────────────────────────────────────

class PCADegradationModel:
    def __init__(self, data_dir, train_days):
        self.data_dir = data_dir
        self.train_days = train_days
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=1)
        
        self.selected_features_idx = []
        self.raw_feature_history = [] 
        self.baseline_hi = 0.0        
        self.max_hi = 1.0           # Discovered during pre-scan
        self.ema_hi = 1.0           # Used for smoothing the final output
        self.is_fitted = False

    def pick_vibration_key(self, mat):
        keys = [k for k in mat if not k.startswith("_")]
        for k in keys:
            if "vibration" in k.lower() or k.lower().endswith(("_de_time", "_time")):
                return k
        return keys[0]

    def extract_time_features(self, raw):
        raw = raw - np.mean(raw)
        rms = np.sqrt(np.mean(raw**2))
        kurtosis = stats.kurtosis(raw, fisher=False)
        crest_factor = np.max(np.abs(raw)) / (rms + 1e-12)
        return np.array([rms, kurtosis, crest_factor])

    def compute_fft(self, raw):
        """Computes the Fast Fourier Transform of the signal."""
        N = len(raw)
        windowed = raw * np.hanning(N) # Apply Hanning window to reduce spectral leakage
        fft_vals = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(N, d=1.0/FS)
        amp = (2.0 / N) * np.abs(fft_vals)
        
        # Filter to our maximum frequency of interest
        mask = freqs <= MAX_FFT_HZ
        return freqs[mask], amp[mask]

    def calculate_monotonicity(self, feature_array):
        diffs = np.diff(feature_array)
        pos_diffs = np.sum(diffs > 0)
        neg_diffs = np.sum(diffs < 0)
        return abs(pos_diffs - neg_diffs) / len(diffs)

    def fit(self):
        print(f"--- Building PCA Model on first {self.train_days} days ---")
        files = sorted(glob.glob(os.path.join(self.data_dir, "*.mat")))
        if len(files) < self.train_days:
            raise ValueError(f"Need at least {self.train_days} files to train.")

        # Phase 1: Train PCA
        train_raw_features = []
        for filepath in files[:self.train_days]:
            mat = sio.loadmat(filepath)
            v = mat[self.pick_vibration_key(mat)].flatten().astype(np.float64)
            train_raw_features.append(self.extract_time_features(v))
            
        train_raw_features = np.array(train_raw_features)
        smoothed_features = np.zeros_like(train_raw_features)
        for i in range(self.train_days):
            start_idx = max(0, i - 4)
            smoothed_features[i] = np.mean(train_raw_features[start_idx:i+1], axis=0)

        feature_names = ["RMS", "Kurtosis", "CrestFactor"]
        monotonicities = []
        for col in range(smoothed_features.shape[1]):
            score = self.calculate_monotonicity(smoothed_features[:, col])
            monotonicities.append(score)

        self.selected_features_idx = [i for i, score in enumerate(monotonicities) if score > 0.3]
        if len(self.selected_features_idx) == 0:
            self.selected_features_idx = list(range(len(feature_names)))

        selected_train_data = smoothed_features[:, self.selected_features_idx]
        normalized_train = self.scaler.fit_transform(selected_train_data)
        self.pca.fit(normalized_train)
        self.baseline_hi = self.pca.transform(normalized_train[0].reshape(1, -1))[0][0]
        
        # Phase 2: Pre-scan all 50 days to find Absolute Max HI for scaling
        print("--- Scanning all 50 days to establish 1-to-0 scale ---")
        all_raw_hi = []
        scan_feature_history = []
        for filepath in files:
            mat = sio.loadmat(filepath)
            v = mat[self.pick_vibration_key(mat)].flatten().astype(np.float64)
            feats = self.extract_time_features(v)
            scan_feature_history.append(feats)
            
            # Smooth
            window = scan_feature_history[-5:]
            smoothed = np.mean(window, axis=0)
            
            # PCA Transform
            sel = smoothed[self.selected_features_idx]
            norm = self.scaler.transform(sel.reshape(1, -1))
            pca_val = self.pca.transform(norm)[0][0]
            
            raw_hi = abs(pca_val - self.baseline_hi)
            all_raw_hi.append(raw_hi)
            
        self.max_hi = max(all_raw_hi)
        self.is_fitted = True
        print(f"--- Model Built! Max PCA variance found: {self.max_hi:.2f} ---")

    def transform_daily(self, raw_signal):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transforming.")

        daily_features = self.extract_time_features(raw_signal)
        self.raw_feature_history.append(daily_features)
        
        history_window = self.raw_feature_history[-5:]
        smoothed_daily = np.mean(history_window, axis=0)
        
        selected_daily = smoothed_daily[self.selected_features_idx]
        normalized_daily = self.scaler.transform(selected_daily.reshape(1, -1))
        pca_val = self.pca.transform(normalized_daily)[0][0]
        
        # Calculate raw degradation
        raw_hi = abs(pca_val - self.baseline_hi)
        
        # Normalize to 1 (Healthy) -> 0 (Failed)
        scaled_hi = 1.0 - (raw_hi / self.max_hi)
        scaled_hi = max(0.0, min(1.0, scaled_hi)) # Clamp between 0 and 1
        
        # Apply Exponential Moving Average (EMA) for visual smoothing
        # Alpha of 0.2 means 20% new data, 80% history (very smooth)
        alpha = 0.2
        self.ema_hi = (alpha * scaled_hi) + ((1 - alpha) * self.ema_hi)
        
        return self.ema_hi, daily_features


degradation_model = PCADegradationModel(BEARING_DATA_DIR, TRAIN_DAYS)

# ── Simulation Engine ─────────────────────────────────────────────────────────

def stream_worker():
    global _current_state, _is_streaming
    files = sorted(glob.glob(os.path.join(BEARING_DATA_DIR, "*.mat")))
    
    # Reset states for a fresh simulation run
    degradation_model.raw_feature_history = []
    degradation_model.ema_hi = 1.0
    
    for idx, filepath in enumerate(files):
        if not _is_streaming: break
        
        try:
            mat = sio.loadmat(filepath)
            v = mat[degradation_model.pick_vibration_key(mat)].flatten().astype(np.float64)
            
            # Get HI, features, and FFT
            hi_val, daily_features = degradation_model.transform_daily(v)
            freqs, amps = degradation_model.compute_fft(v)
            
            # Downsample FFT for the UI (sending ~500 points instead of 50,000)
            ds = max(1, len(freqs)//500)
            
            result = {
                "day_index": idx + 1,
                "day_total": TOTAL_DAYS,
                "filename": os.path.basename(filepath),
                "health_indicator_pca": round(float(hi_val), 4),
                "rms": round(float(daily_features[0]), 4),
                "kurtosis": round(float(daily_features[1]), 4),
                "crest_factor": round(float(daily_features[2]), 4),
                "fft_freqs": freqs[::ds].tolist(),
                "fft_amps": amps[::ds].tolist()
            }
            
            with state_lock:
                _current_state = result
                
        except Exception as e:
            traceback.print_exc()
            with state_lock:
                _current_state = {"error": str(e), "day_index": idx + 1}
                
        time.sleep(STREAM_DELAY_S)
        
    _is_streaming = False

# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")

@app.route("/start-simulation", methods=["POST"])
def start_simulation():
    global _is_streaming
    if _is_streaming: return jsonify({"message": "Running."}), 400
    if not degradation_model.is_fitted: return jsonify({"error": "PCA not fitted."}), 500
        
    _is_streaming = True
    thread = threading.Thread(target=stream_worker)
    thread.daemon = True
    thread.start()
    return jsonify({"message": "Started."})

@app.route("/stop-simulation", methods=["POST"])
def stop_simulation():
    global _is_streaming
    _is_streaming = False
    return jsonify({"message": "Stopped."})

@app.route("/current")
def current():
    with state_lock:
        if not _current_state: return jsonify({"error": "No data."}), 404
        return jsonify(_current_state)

if __name__ == "__main__":
    print(f"Starting Digital Twin Backend...")
    try:
        degradation_model.fit()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        exit(1)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)