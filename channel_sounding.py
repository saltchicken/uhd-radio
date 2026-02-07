import uhd
import numpy as np
import threading
import time
import signal
import sys

# ==========================================
# ‼️ CONFIGURATION
# ==========================================
FREQ = 915e6
RATE = 1e6
GAIN = 60          # High gain for loopback/antenna gap
CHIRP_LEN = 256    # Length of the sounding pulse
GAP_LEN = 2000     # Silence between pulses (to let multipath settle)
THRESHOLD = 0.05   # Detection threshold

RUNNING = True

# ‼️ ROBUST STREAM MODE SETUP
# We define MODE_NAME explicitly to ensure logic downstream works correctly.
try:
    STREAM_MODE_START = uhd.types.StreamMode.start_continuous
    STREAM_MODE_STOP = uhd.types.StreamMode.stop_continuous
    MODE_NAME = "Native Continuous"
except AttributeError:
    STREAM_MODE_START = uhd.types.StreamMode.num_done
    STREAM_MODE_STOP = uhd.types.StreamMode.num_done
    MODE_NAME = "Manual Burst (Fallback)"

def handler(signum, frame):
    global RUNNING
    print("\n--> Signal caught. Shutting down...")
    RUNNING = False
signal.signal(signal.SIGINT, handler)

# ==========================================
# ‼️ SIGNAL GENERATION (The Probe)
# ==========================================

def generate_chirp_probe(length):
    """
    ‼️ Generates a Linear Frequency Modulated (LFM) Chirp.
    This signal sweeps frequency, providing excellent auto-correlation 
    properties (Pulse Compression) for channel sounding.
    """
    t = np.arange(length)
    # Linear chirp: exp(j * pi * (k/N) * t^2)
    # Sweeps from DC to Bandwidth/2
    k = 1.0 
    chirp = np.exp(1j * np.pi * k * (t**2 / length))
    
    # Apply windowing to reduce spectral leakage (side lobes)
    window = np.hanning(length)
    return (chirp * window).astype(np.complex64) * 0.7

# Pre-calculate the probe and its conjugate for correlation
PROBE_TX = generate_chirp_probe(CHIRP_LEN)
# ‼️ REFERENCE: Time-reversed conjugate is needed for convolution-based correlation
PROBE_REF = np.conj(PROBE_TX[::-1]) 

# ==========================================
# ‼️ ANALYSIS ENGINE (CIS)
# ==========================================

def analyze_channel_response(rx_chunk, sample_rate=RATE):
    """
    ‼️ Performs Cross-Correlation to extract Channel Impulse Response (CIR).
    """
    # 1. Correlate RX signal with the known Probe
    # 'valid' mode returns only the part where signals fully overlap
    correlation = np.correlate(rx_chunk, PROBE_TX, mode='valid')
    
    # 2. Calculate Magnitude (Power Delay Profile)
    # We normalize by the length to keep values sane
    mag = np.abs(correlation) / CHIRP_LEN
    
    peak_idx = np.argmax(mag)
    peak_val = mag[peak_idx]
    
    # 3. Calculate Noise Floor (exclude the peak region)
    # We take a slice away from the peak to estimate noise
    noise_region = np.concatenate([mag[:max(0, peak_idx-20)], mag[min(len(mag), peak_idx+20):]])
    if len(noise_region) > 0:
        noise_floor = np.mean(noise_region)
    else:
        noise_floor = 0.0001
        
    # 4. Calculate SNR
    snr_linear = peak_val / (noise_floor + 1e-9)
    snr_db = 10 * np.log10(snr_linear)
    
    return {
        "peak_val": peak_val,
        "peak_idx": peak_idx,
        "snr_db": snr_db,
        "noise_floor": noise_floor,
        "profile": mag # The full impulse response vector
    }

def ascii_sparkline(data, width=40):
    """
    ‼️ Renders a text-based plot of the channel impulse response.
    """
    if len(data) == 0: return ""
    
    # Downsample data to fit in 'width' characters
    chunk_size = max(1, len(data) // width)
    # Max pooling downsample to preserve peaks
    reduced = [np.max(data[i:i+chunk_size]) for i in range(0, len(data), chunk_size)]
    reduced = reduced[:width] # trim safety
    
    chars = " _.-=oO#"
    m = np.max(reduced)
    if m == 0: return "_" * width
    
    line = ""
    for val in reduced:
        idx = int((val / m) * (len(chars) - 1))
        line += chars[idx]
    return line

# ==========================================
# THREADS
# ==========================================

def tx_daemon(usrp):
    print("   [TX] Sounding Daemon Active.")
    
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    tx_streamer = usrp.get_tx_stream(st_args)
    
    # ‼️ Construct Frame: Silence -> Chirp -> Silence
    padding = np.zeros(GAP_LEN, dtype=np.complex64)
    frame = np.concatenate([padding, PROBE_TX, padding])
    
    md = uhd.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst = True
    
    while RUNNING:
        try:
            md.has_time_spec = False
            tx_streamer.send(frame.reshape(1, -1), md)
            time.sleep(1.0) # Ping once per second
        except Exception:
            pass

def rx_analysis_loop(usrp):
    print(f"   [RX] CIS Analysis Loop Active ({MODE_NAME}).")
    
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    rx_streamer = usrp.get_rx_stream(st_args)
    
    # Buffer needs to be large enough to catch the chirp + multipath delay
    buff_len = 10000 
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    cmd = uhd.types.StreamCMD(STREAM_MODE_START)
    cmd.stream_now = True
    
    # ‼️ FIX: Explicitly check MODE_NAME variable
    if MODE_NAME == "Manual Burst (Fallback)":
        cmd.num_samps = buff_len
    
    rx_streamer.issue_stream_cmd(cmd)
    
    while RUNNING:
        # ‼️ FIX: Re-issue command only if we are in manual mode
        if MODE_NAME == "Manual Burst (Fallback)":
            rx_streamer.issue_stream_cmd(cmd)
            
        samps = rx_streamer.recv(recv_buffer, metadata, 0.1)
        
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
             if metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                continue
             continue

        if samps > 0:
            data = recv_buffer[0][:samps]
            
            # Simple energy detection to trigger analysis
            if np.max(np.abs(data)) > THRESHOLD:
                
                # ‼️ Run Analysis
                results = analyze_channel_response(data)
                
                # Only print valid locks
                if results['snr_db'] > 10: # >10dB SNR check
                    # Get a slice of the profile around the peak for visualization
                    center = results['peak_idx']
                    # View window: 50 samples left, 50 right
                    start_view = max(0, center - 40)
                    end_view = min(len(results['profile']), center + 40)
                    view_data = results['profile'][start_view:end_view]
                    
                    graph = ascii_sparkline(view_data)
                    
                    print(f"   [CIS] SNR: {results['snr_db']:.1f}dB | Peak: {results['peak_val']:.3f}")
                    print(f"         Impulse: [{graph}]")

    # Cleanup
    stop_cmd = uhd.types.StreamCMD(STREAM_MODE_STOP)
    rx_streamer.issue_stream_cmd(stop_cmd)

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("--> Initializing Channel Sounder...")
    try:
        usrp = uhd.usrp.MultiUSRP("type=b200")
    except RuntimeError:
        print("‼️ No USRP found.")
        sys.exit(1)
        
    usrp.set_rx_rate(RATE, 0)
    usrp.set_tx_rate(RATE, 0)
    usrp.set_rx_freq(uhd.types.TuneRequest(FREQ), 0)
    usrp.set_tx_freq(uhd.types.TuneRequest(FREQ), 0)
    usrp.set_rx_gain(GAIN, 0)
    usrp.set_tx_gain(GAIN, 0)
    usrp.set_tx_antenna("TX/RX", 0)
    usrp.set_rx_antenna("RX2", 0)
    
    time.sleep(1.0)
    
    t_tx = threading.Thread(target=tx_daemon, args=(usrp,))
    t_tx.daemon = True
    t_tx.start()
    
    try:
        rx_analysis_loop(usrp)
    except KeyboardInterrupt:
        pass
