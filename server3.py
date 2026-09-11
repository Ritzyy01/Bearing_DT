from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import numpy as np
import scipy.io as sio
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os, glob, time, threading, traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

    # ── Plot Styling ───────────────────────────────────────────────────────────
    PLOT_STYLE = {
        'figure.facecolor': '#0f172a', 'axes.facecolor': '#1e293b',
        'axes.edgecolor': '#334155',   'axes.labelcolor': '#cbd5e1',
        'axes.titlecolor': '#f1f5f9',  'xtick.color': '#94a3b8',
        'ytick.color': '#94a3b8',      'text.color': '#f1f5f9',
        'grid.color': '#334155',       'grid.alpha': 0.6,
        'legend.facecolor': '#1e293b', 'legend.edgecolor': '#475569',
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    }

    def _phase_shade(self, ax):
        ax.axvspan(1,  20, alpha=0.08, color='#10b981')
        ax.axvspan(20, 40, alpha=0.08, color='#fbbf24')
        ax.axvspan(40, 50, alpha=0.08, color='#e11d48')
        ax.axvline(20, color='#fbbf24', lw=1, ls='--', alpha=0.5)
        ax.axvline(40, color='#e11d48', lw=1, ls='--', alpha=0.5)

    def generate_plots(self, all_signals, all_features, smoothed, ema_hi, scaled_hi, out_dir=None):
        """Generate 7 publication-quality figures from real bearing data. Called after fit()."""
        if out_dir is None:
            out_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(out_dir, exist_ok=True)
        plt.rcParams.update(self.PLOT_STYLE)

        days = np.arange(1, TOTAL_DAYS + 1)
        BPFI, BPFO, BSF = 162.0, 107.0, 68.0  # Fault frequencies for 624ZZ-class bearing

        print("--- Generating plots ---")

        # ── Fig 1: Feature Evolution ──────────────────────────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle('Feature Evolution over 50-Day Bearing Life', fontsize=15, fontweight='bold', y=0.98)
        feat_labels = ['RMS Vibration (g)', 'Kurtosis', 'Crest Factor']
        colors  = ['#38bdf8', '#a78bfa', '#34d399']
        raw_col = ['#1e6a8a', '#5b44a0', '#1a7a58']
        for ax, fi, label, color, rc in zip(axes, range(3), feat_labels, colors, raw_col):
            self._phase_shade(ax)
            ax.plot(days, all_features[:, fi], lw=1, alpha=0.35, color=rc, label='Raw')
            ax.plot(days, smoothed[:, fi], lw=2.5, color=color, label='5-day MA')
            ax.set_ylabel(label); ax.grid(True, alpha=0.4); ax.legend(loc='upper left', fontsize=9)
        axes[-1].set_xlabel('Day')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(out_dir, 'fig1_feature_evolution.png'), dpi=180, bbox_inches='tight')
        plt.close(); print("  ✓ fig1_feature_evolution.png")

        # ── Fig 2: PCA Health Indicator ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle('PCA-Fused Health Indicator (EMA Smoothed) – Degradation Trend', fontsize=14, fontweight='bold')
        ax.axvspan(1, 20, alpha=0.08, color='#10b981')
        ax.axvspan(20, 40, alpha=0.08, color='#fbbf24')
        ax.axvspan(40, 50, alpha=0.08, color='#e11d48')
        ax.axvline(20, color='#fbbf24', lw=1.2, ls='--', alpha=0.7, label='Fault Onset')
        ax.axvline(40, color='#e11d48', lw=1.2, ls='--', alpha=0.7, label='Rapid Degradation')
        ax.plot(days, np.array(scaled_hi), lw=1.5, alpha=0.4, color='#6366f1', label='Scaled HI (raw)')
        ax.plot(days, ema_hi, lw=3.5, color='#10b981', label='EMA-Smoothed HI')
        for lo, hi_lim, col in [(0.70, 1.05, '#10b981'), (0.30, 0.70, '#fbbf24'), (0, 0.30, '#e11d48')]:
            mask = (ema_hi >= lo) & (ema_hi <= hi_lim)
            ax.fill_between(days, 0, ema_hi, where=mask, alpha=0.15, color=col)
        ax.axhline(0.70, color='#fbbf24', lw=1, ls=':', alpha=0.8)
        ax.axhline(0.30, color='#e11d48', lw=1, ls=':', alpha=0.8)
        ax.text(51, 0.85, 'Healthy', color='#10b981', va='center', fontsize=9)
        ax.text(51, 0.50, 'Warning', color='#fbbf24', va='center', fontsize=9)
        ax.text(51, 0.15, 'Critical', color='#e11d48', va='center', fontsize=9)
        ax.set_xlabel('Day'); ax.set_ylabel('Health Indicator (1.0 = Healthy → 0.0 = Failed)')
        ax.set_ylim(0, 1.1); ax.set_xlim(1, 55); ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fig2_health_indicator.png'), dpi=180, bbox_inches='tight')
        plt.close(); print("  ✓ fig2_health_indicator.png")

        # ── Fig 3: FFT Comparison at 4 snapshots ─────────────────────────────
        snapshot_days   = [1, 20, 35, 50]
        snapshot_colors = ['#10b981', '#38bdf8', '#fbbf24', '#e11d48']
        snapshot_labels = ['Day 1 – Healthy', 'Day 20 – Fault Onset', 'Day 35 – Early Degradation', 'Day 50 – Failure']
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle('Frequency Spectrum (FFT) at Key Lifecycle Stages', fontsize=14, fontweight='bold')
        axes = axes.flatten()
        for ax, day, color, label in zip(axes, snapshot_days, snapshot_colors, snapshot_labels):
            freqs, amps = self.compute_fft(all_signals[day - 1])
            ax.plot(freqs, amps, lw=1.2, color=color, alpha=0.9)
            ax.fill_between(freqs, 0, amps, alpha=0.15, color=color)
            for fname, fval, fcol in [('BPFI', BPFI, '#f97316'), ('BPFO', BPFO, '#c084fc'), ('BSF', BSF, '#22d3ee')]:
                for h in range(1, 5):
                    fx = fval * h
                    if fx <= MAX_FFT_HZ:
                        ax.axvline(fx, color=fcol, lw=0.8, ls='--', alpha=0.6)
                        if h == 1:
                            ylim = ax.get_ylim()
                            ax.text(fx + 30, ylim[1] * 0.88, fname, color=fcol, fontsize=7, rotation=90, va='top')
            ax.set_title(label, fontsize=11, color=color)
            ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Amplitude')
            ax.set_xlim(0, MAX_FFT_HZ); ax.grid(True, alpha=0.35)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, 'fig3_fft_comparison.png'), dpi=180, bbox_inches='tight')
        plt.close(); print("  ✓ fig3_fft_comparison.png")

        # ── Fig 4: Time Domain Waveforms ──────────────────────────────────────
        fig, axes = plt.subplots(2, 1, figsize=(13, 7))
        fig.suptitle('Vibration Time-Domain Signal: Healthy vs Failure State', fontsize=14, fontweight='bold')
        for ax, day, color, label in zip(axes, [1, 50], ['#10b981', '#e11d48'],
                                          ['Day 1 – Healthy Baseline', 'Day 50 – Complete Failure']):
            sig = all_signals[day - 1]
            n_show = min(4096, len(sig))
            t_ms = np.linspace(0, n_show / FS * 1000, n_show)
            ax.plot(t_ms, sig[:n_show], lw=0.8, color=color, alpha=0.85)
            ax.set_ylabel('Amplitude (g)'); ax.set_title(label, fontsize=11, color=color)
            ax.grid(True, alpha=0.35)
            rms_v  = np.sqrt(np.mean(sig**2))
            kurt_v = stats.kurtosis(sig, fisher=False)
            ax.text(0.98, 0.92, f'RMS={rms_v:.3f}  Kurtosis={kurt_v:.2f}',
                    transform=ax.transAxes, ha='right', va='top', color='#f1f5f9', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e293b', edgecolor='#475569'))
        axes[-1].set_xlabel('Time (ms)')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, 'fig4_time_domain.png'), dpi=180, bbox_inches='tight')
        plt.close(); print("  ✓ fig4_time_domain.png")

        # ── Fig 5: Monotonicity + PC1 scatter ────────────────────────────────
        def calc_mono(arr):
            d = np.diff(arr)
            return abs(np.sum(d > 0) - np.sum(d < 0)) / len(d)
        mono_scores = [calc_mono(smoothed[:, i]) for i in range(3)]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle('PCA Model Analysis', fontsize=14, fontweight='bold')
        bar_colors = ['#38bdf8', '#a78bfa', '#34d399']
        bars = axes[0].bar(['RMS', 'Kurtosis', 'Crest Factor'], mono_scores,
                           color=bar_colors, edgecolor='#334155', linewidth=1.2)
        axes[0].axhline(0.3, color='#fbbf24', lw=1.8, ls='--', label='Selection Threshold (0.3)')
        axes[0].set_title('Feature Monotonicity Scores'); axes[0].set_ylabel('Monotonicity Score')
        axes[0].set_ylim(0, 1.05); axes[0].legend(); axes[0].grid(True, alpha=0.4, axis='y')
        for bar, score in zip(bars, mono_scores):
            axes[0].text(bar.get_x() + bar.get_width()/2, score + 0.02,
                         f'{score:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        all_norm = self.scaler.transform(smoothed[:, self.selected_features_idx])
        all_pc1  = self.pca.transform(all_norm).flatten()
        sc = axes[1].scatter(days, all_pc1, c=days, cmap='RdYlGn_r', s=60,
                             edgecolors='#334155', linewidths=0.5, zorder=3)
        axes[1].axvline(20, color='#fbbf24', lw=1.2, ls='--', alpha=0.7, label='Fault Onset')
        axes[1].set_title('PCA Score (PC1) Over Time'); axes[1].set_xlabel('Day')
        axes[1].set_ylabel('PC1 Score'); axes[1].legend(); axes[1].grid(True, alpha=0.4)
        cbar = plt.colorbar(sc, ax=axes[1]); cbar.set_label('Day')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(out_dir, 'fig5_pca_analysis.png'), dpi=180, bbox_inches='tight')
        plt.close(); print("  ✓ fig5_pca_analysis.png")

        # ── Fig 6: Architecture Diagram ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis('off')
        fig.suptitle('Digital Twin System Architecture & Data Flow', fontsize=14, fontweight='bold')
        boxes = [
            (1.1,  2.5, '.mat\nBearing Data\n(50 Days)',         '#1d4ed8', '#bfdbfe'),
            (3.8,  2.5, 'Feature\nExtraction\nRMS · Kurt · CF',  '#065f46', '#a7f3d0'),
            (6.5,  2.5, 'PCA\nDegradation\nModel',               '#4c1d95', '#ddd6fe'),
            (9.2,  2.5, 'Flask REST\nAPI Server\n/current',       '#7c2d12', '#fed7aa'),
            (11.9, 2.5, 'Browser\nDashboard\n+ Fusion 360',       '#1e3a5f', '#bae6fd'),
        ]
        for (x, y, text, ec, fc) in boxes:
            rect = plt.Rectangle((x-1.0, y-1.1), 2.0, 2.2, lw=2,
                                  edgecolor=ec, facecolor=fc+'33', zorder=2)
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center', fontsize=9,
                    fontweight='bold', color='#f1f5f9', zorder=3, multialignment='center')
        for i in range(len(boxes)-1):
            ax.annotate('', xy=(boxes[i+1][0]-1.0, 2.5), xytext=(boxes[i][0]+1.0, 2.5),
                        arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=2.0))
        for i, lbl in enumerate(['97.6 kHz\nvibration', 'Smoothed\n5-day MA', 'EMA HI\n+ FFT', 'NDJSON\nSSE / poll']):
            xm = (boxes[i][0] + boxes[i+1][0]) / 2
            ax.text(xm, 3.85, lbl, ha='center', va='bottom', fontsize=8, color='#94a3b8')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(os.path.join(out_dir, 'fig6_architecture.png'), dpi=180, bbox_inches='tight')
        plt.close(); print("  ✓ fig6_architecture.png")

        # ── Fig 7: Correlation Heatmap ────────────────────────────────────────
        corr = np.corrcoef(smoothed.T)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.suptitle('Feature Correlation Matrix', fontsize=13, fontweight='bold')
        im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(['RMS', 'Kurtosis', 'Crest Factor'])
        ax.set_yticklabels(['RMS', 'Kurtosis', 'Crest Factor'])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=13,
                        color='black' if abs(corr[i,j]) < 0.7 else 'white', fontweight='bold')
        plt.colorbar(im, ax=ax, label='Pearson r')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'fig7_correlation_heatmap.png'), dpi=180, bbox_inches='tight')
        plt.close(); print("  ✓ fig7_correlation_heatmap.png")

        print(f"--- All plots saved to: {out_dir} ---")

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
        # Also collect all signals, features, and smoothed values for plotting
        print("--- Scanning all 50 days to establish 1-to-0 scale ---")
        all_raw_hi = []
        scan_feature_history = []
        all_signals    = []  # raw vibration per day
        all_features   = []  # raw feature vectors per day
        all_smoothed   = []  # smoothed feature vectors per day

        for filepath in files:
            mat = sio.loadmat(filepath)
            v = mat[self.pick_vibration_key(mat)].flatten().astype(np.float64)
            all_signals.append(v)

            feats = self.extract_time_features(v)
            all_features.append(feats)
            scan_feature_history.append(feats)
            
            window   = scan_feature_history[-5:]
            smoothed = np.mean(window, axis=0)
            all_smoothed.append(smoothed)
            
            sel     = smoothed[self.selected_features_idx]
            norm    = self.scaler.transform(sel.reshape(1, -1))
            pca_val = self.pca.transform(norm)[0][0]
            
            raw_hi = abs(pca_val - self.baseline_hi)
            all_raw_hi.append(raw_hi)
            
        self.max_hi = max(all_raw_hi)
        self.is_fitted = True
        print(f"--- Model Built! Max PCA variance found: {self.max_hi:.2f} ---")

        # Build full scaled HI + EMA for plotting
        scaled_hi = [max(0.0, min(1.0, 1.0 - v / self.max_hi)) for v in all_raw_hi]
        ema_plot  = [scaled_hi[0]]
        for v in scaled_hi[1:]:
            ema_plot.append(0.2 * v + 0.8 * ema_plot[-1])
        ema_plot = np.array(ema_plot)

        # Generate all plots now using the real data
        self.generate_plots(
            all_signals   = all_signals,
            all_features  = np.array(all_features),
            smoothed      = np.array(all_smoothed),
            ema_hi        = ema_plot,
            scaled_hi     = scaled_hi,
            out_dir       = os.path.dirname(os.path.abspath(__file__))
        )

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