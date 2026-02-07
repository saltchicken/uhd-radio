import uhd
import numpy as np
import threading
import time
import sys
from usrp_driver import B210UnifiedDriver 
import sdr_utils

FREQ = 915e6
RATE = 1e6
GAIN = 60            
CHIRP_LEN = 256     
GAP_LEN = 2000        
THRESHOLD = 0.05    

CALIBRATION_FRAMES = 40       
DETECTION_THRESHOLD = 2.5
CSI_WIN_SIZE = 64             


sig_handler = sdr_utils.SignalHandler()

STREAM_MODE_START = uhd.types.StreamMode.start_cont
STREAM_MODE_STOP = uhd.types.StreamMode.stop_cont
MODE_NAME = "Native Continuous"


PROBE_TX = sdr_utils.generate_chirp_probe(CHIRP_LEN)



def process_rx_packet(rx_chunk):
    correlation = np.correlate(rx_chunk, PROBE_TX, mode='valid')
    mag = np.abs(correlation)
    peak_idx = np.argmax(mag)
    peak_val = mag[peak_idx]
    noise_floor = np.mean(mag[:max(0, peak_idx-20)]) if peak_idx > 20 else 0.0001
    snr_db = 10 * np.log10(peak_val / (noise_floor + 1e-9))
    
    if snr_db > 10:
        PRE_CURSOR = 10
        start_idx = peak_idx - PRE_CURSOR
        end_idx = start_idx + CSI_WIN_SIZE
        cir_window = np.zeros(CSI_WIN_SIZE, dtype=np.complex64)
        src_start = max(0, start_idx)
        src_end = min(len(correlation), end_idx)
        dst_start = src_start - start_idx
        dst_end = dst_start + (src_end - src_start)
        
        if src_end > src_start:
             cir_window[dst_start:dst_end] = correlation[src_start:src_end]

        if np.sum(np.abs(cir_window)) < 1e-6:
            return None
        

        metrics = sdr_utils.calculate_csi_metrics(cir_window, RATE)
        metrics['snr_db'] = snr_db
        metrics['peak_val'] = peak_val
        return metrics
        
    return None



def tx_daemon(usrp, driver): 
    print("   [TX] Sounding Daemon Active.")
    tx_streamer = driver.get_tx_streamer()
    padding = np.zeros(GAP_LEN, dtype=np.complex64)
    frame = np.concatenate([padding, PROBE_TX, padding])
    md = uhd.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst = True
    

    while sig_handler.running:
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

    baseline_cfr = None
    cal_frames = []
    frame_count = 0
    
    print("\n   [DETECTION] 🟡 CALIBRATING... Keep area static.")
    

    while sig_handler.running:
        samps = rx_streamer.recv(recv_buffer, metadata, 0.1)
        
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
             continue

        if samps > 0:
            data = recv_buffer[0][:samps]
            if np.max(np.abs(data)) > THRESHOLD:
                result = process_rx_packet(data)
                
                if result:
                    current_cfr = result['cfr_db']
                    if frame_count < CALIBRATION_FRAMES:
                        cal_frames.append(current_cfr)
                        frame_count += 1
                        sys.stdout.write(f"\r   [DETECTION] Calibrating: {frame_count}/{CALIBRATION_FRAMES}")
                        sys.stdout.flush()
                        
                        if frame_count == CALIBRATION_FRAMES:
                            baseline_cfr = np.mean(np.array(cal_frames), axis=0)
                            print(f"\n   [DETECTION] 🟢 Calibration Complete.")
                            print(f"   [DETECTION] 📏 Using Fixed Threshold: {DETECTION_THRESHOLD:.2f}")
                            
                    else:
                        diff_vector = np.abs(current_cfr - baseline_cfr)
                        anomaly_score = np.mean(diff_vector)
                        is_detected = anomaly_score > DETECTION_THRESHOLD
                        status_icon = "🔴 OBJECT DETECTED" if is_detected else "🟢 Clear"
                        
                        print("-" * 60)
                        print(f"‼️ STATUS: {status_icon}")
                        print(f"   Anomaly Score: {anomaly_score:.2f} (Thresh: {DETECTION_THRESHOLD:.2f})")

                        print(f"   Baseline: [{sdr_utils.ascii_bar_chart(baseline_cfr)}]")
                        print(f"   Current:  [{sdr_utils.ascii_bar_chart(current_cfr)}]")
                        print(f"   Delta:    [{sdr_utils.ascii_bar_chart(diff_vector)}]")

    stop_cmd = uhd.types.StreamCMD(STREAM_MODE_STOP)
    rx_streamer.issue_stream_cmd(stop_cmd)

if __name__ == "__main__":
    print("--> Initializing CSI Analyzer (Object Detection Mode)...")
    driver = B210UnifiedDriver(FREQ, RATE, GAIN)
    usrp = driver.initialize()
    
    t_tx = threading.Thread(target=tx_daemon, args=(usrp, driver))
    t_tx.daemon = True
    t_tx.start()
    
    try:
        rx_analysis_loop(usrp, driver)
    except KeyboardInterrupt:
        pass