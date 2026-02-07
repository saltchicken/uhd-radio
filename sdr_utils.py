import signal
import numpy as np
import sys



class SignalHandler:
    """
    ‼️ Extracted signal handling logic.
    Replaces the global RUNNING variable and handler function in all scripts.
    """
    def __init__(self):
        self.running = True
        signal.signal(signal.SIGINT, self.handler)
    
    def handler(self, signum, frame):
        print("\n--> Signal caught. Shutting down...")
        self.running = False

def generate_chirp_probe(length):
    """
    ‼️ Extracted chirp generation.
    Used by channel_sounding, csi_analysis, and object_detection.
    """
    t = np.arange(length)
    k = 1.0 
    chirp = np.exp(1j * np.pi * k * (t**2 / length))
    window = np.hanning(length)
    return (chirp * window).astype(np.complex64) * 0.7

def calculate_csi_metrics(cir_window, sample_rate):
    """
    ‼️ Extracted CSI metric calculation.
    Unifies logic from csi_analysis and object_detection.
    Returns dictionary with all metrics including linear CFR for detection.
    """
    pdp = np.abs(cir_window)**2
    thresh = np.max(pdp) * 0.1
    valid_indices = np.where(pdp > thresh)[0]
    
    if len(valid_indices) < 2: 
        rms_delay = 0.0
        coherence_bw = sample_rate 
    else:
        first_path = valid_indices[0]
        pdp_clean = pdp[valid_indices]
        delays_sec = (valid_indices - first_path) / sample_rate
        total_power = np.sum(pdp_clean)
        mean_delay = np.sum(pdp_clean * delays_sec) / total_power
        sq_delay_error = (delays_sec - mean_delay)**2
        rms_delay = np.sqrt(np.sum(pdp_clean * sq_delay_error) / total_power)
        
        if rms_delay > 1e-12:
            coherence_bw = 1.0 / (5.0 * rms_delay)
        else:
            coherence_bw = sample_rate

    cfr_complex = np.fft.fftshift(np.fft.fft(cir_window))
    cfr_mag_linear = np.abs(cfr_complex)
    cfr_mag_db = 20 * np.log10(cfr_mag_linear + 1e-12)
    
    return {
        "rms_delay_us": rms_delay * 1e6, 
        "coherence_bw_khz": coherence_bw / 1e3, 
        "cfr_db": cfr_mag_db,
        "cfr_linear": cfr_mag_linear,
        "pdp": pdp 
    }

def ascii_sparkline(data, width=40):
    """
    ‼️ Extracted sparkline visualizer.
    """
    if len(data) == 0: return ""
    chunk_size = max(1, len(data) // width)
    reduced = [np.max(data[i:i+chunk_size]) for i in range(0, len(data), chunk_size)]
    reduced = reduced[:width]
    chars = " _.-=oO#"
    m = np.max(reduced)
    if m == 0: return "_" * width
    line = ""
    for val in reduced:
        idx = int((val / m) * (len(chars) - 1))
        line += chars[idx]
    return line

def ascii_bar_chart(data, width=40):
    """
    ‼️ Extracted bar chart visualizer (unicode blocks).
    """
    if len(data) == 0: return ""
    d_min, d_max = np.min(data), np.max(data)
    if d_max == d_min: norm_data = np.zeros_like(data)
    else: norm_data = (data - d_min) / (d_max - d_min)
    chunk = len(norm_data) // width
    if chunk < 1: chunk = 1
    resampled = [np.mean(norm_data[i:i+chunk]) for i in range(0, len(norm_data), chunk)][:width]
    chars = "  ▂▃▄▅▆▇█"
    line = ""
    for val in resampled:
        idx = int(val * (len(chars) - 1))
        line += chars[idx]
    return line

def ascii_compass(angle_deg):
    """
    ‼️ Extracted compass visualizer.
    """
    width = 50
    center_idx = width // 2
    norm = (angle_deg + 90) / 180
    pos = int(norm * (width - 1))
    chars = ['-'] * width
    chars[center_idx] = '|'
    pos = max(0, min(width-1, pos))
    chars[pos] = 'O'
    return "".join(chars)