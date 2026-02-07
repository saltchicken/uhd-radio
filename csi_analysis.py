import uhd
import numpy as np
import threading
import time
import signal
import sys
from usrp_driver import B210UnifiedDriver

FREQ = 5.8e9
RATE = 20e6
GAIN = 60           
CHIRP_LEN = 256     
GAP_LEN = 2000      
THRESHOLD = 0.05    

RUNNING = True


STREAM_MODE_START = uhd.types.StreamMode.start_cont
STREAM_MODE_STOP = uhd.types.StreamMode.stop_cont
MODE_NAME = "Native Continuous"

def handler(signum, frame):
    global RUNNING
    print("\n--> Signal caught. Shutting down...")
    RUNNING = False
signal.signal(signal.SIGINT, handler)

def generate_chirp_probe(length):
    t = np.arange(length)
    k = 1.0 
    chirp = np.exp(1j * np.pi * k * (t**2 / length))
    window = np.hanning(length)
    return (chirp * window).astype(np.complex64) * 0.7

PROBE_TX = generate_chirp_probe(CHIRP_LEN)

def calculate_csi_metrics(cir_window, sample_rate):
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
    cfr_mag_db = 20 * np.log10(np.abs(cfr_complex) + 1e-12)
    
    return {
        "rms_delay_us": rms_delay * 1e6, 
        "coherence_bw_khz": coherence_bw / 1e3, 
        "cfr_db": cfr_mag_db,
        "pdp": pdp 
    }

def process_rx_packet(rx_chunk):
    correlation = np.correlate(rx_chunk, PROBE_TX, mode='valid')
    mag = np.abs(correlation)
    peak_idx = np.argmax(mag)
    peak_val = mag[peak_idx]
    noise_floor = np.mean(mag[:max(0, peak_idx-20)]) if peak_idx > 20 else 0.0001
    snr_db = 10 * np.log10(peak_val / (noise_floor + 1e-9))
    
    if snr_db > 10:
        start = max(0, peak_idx - 10)
        end = min(len(correlation), peak_idx + 50)
        cir_window = correlation[start:end]
        
        metrics = calculate_csi_metrics(cir_window, RATE)
        metrics['snr_db'] = snr_db
        metrics['peak_val'] = peak_val
        return metrics
        
    return None

def ascii_bar_chart(data, height=5, width=40):
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

def tx_daemon(usrp, driver): 
    print("   [TX] Sounding Daemon Active.")
    tx_streamer = driver.get_tx_streamer()
    padding = np.zeros(GAP_LEN, dtype=np.complex64)
    frame = np.concatenate([padding, PROBE_TX, padding])
    md = uhd.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst = True
    
    while RUNNING:
        try:
            md.has_time_spec = False
            tx_streamer.send(frame.reshape(1, -1), md)
            time.sleep(0.5) 
        except Exception:
            pass

def rx_analysis_loop(usrp, driver): 
    print(f"   [RX] CSI Analysis Active ({MODE_NAME}).")
    rx_streamer = driver.get_rx_streamer()
    
    buff_len = 10000 
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    cmd = uhd.types.StreamCMD(STREAM_MODE_START)
    cmd.stream_now = True
    

    rx_streamer.issue_stream_cmd(cmd)
    
    while RUNNING:

            
        samps = rx_streamer.recv(recv_buffer, metadata, 0.1)
        
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
             if metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                 continue
             continue

        if samps > 0:
            data = recv_buffer[0][:samps]
            if np.max(np.abs(data)) > THRESHOLD:
                result = process_rx_packet(data)
                
                if result:
                    print("-" * 50)
                    print(f"‼️ CSI CAPTURE | SNR: {result['snr_db']:.1f} dB")
                    print(f"   RMS Delay Spread:     {result['rms_delay_us']:.3f} us")
                    print(f"   Coherence Bandwidth: {result['coherence_bw_khz']:.1f} kHz")
                    print(f"   CIR (Time):  [{ascii_bar_chart(result['pdp'])}]")
                    print(f"   CFR (Freq):  [{ascii_bar_chart(result['cfr_db'])}]")

    stop_cmd = uhd.types.StreamCMD(STREAM_MODE_STOP)
    rx_streamer.issue_stream_cmd(stop_cmd)

if __name__ == "__main__":
    print("--> Initializing CSI Analyzer...")
    driver = B210UnifiedDriver(FREQ, RATE, GAIN)
    usrp = driver.initialize()
    
    t_tx = threading.Thread(target=tx_daemon, args=(usrp, driver))
    t_tx.daemon = True
    t_tx.start()
    
    try:
        rx_analysis_loop(usrp, driver)
    except KeyboardInterrupt:
        pass
