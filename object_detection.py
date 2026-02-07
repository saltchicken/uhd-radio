import uhd
import numpy as np
import sys

from usrp_driver import B210UnifiedDriver, PeriodicTransmitter 
import sdr_utils


args = sdr_utils.get_standard_args("Object Detection", default_freq=915e6)

CHIRP_LEN = 256      
GAP_LEN = 2000        
THRESHOLD = 0.05    

CALIBRATION_FRAMES = 40        
DETECTION_THRESHOLD = 2.5
CSI_WIN_SIZE = 64             

sig_handler = sdr_utils.SignalHandler()

PROBE_TX = sdr_utils.generate_chirp_probe(CHIRP_LEN)
# Prepare TX Frame for PeriodicTransmitter
padding = np.zeros(GAP_LEN, dtype=np.complex64)
TX_FRAME = np.concatenate([padding, PROBE_TX, padding])


def process_rx_packet_refactored(rx_chunk):

    res = sdr_utils.correlate_and_detect(rx_chunk, PROBE_TX)
    
    if res['snr_db'] > 10:
        peak_idx = res['peak_idx']
        # Extract CSI Window
        PRE_CURSOR = 10
        start_idx = peak_idx - PRE_CURSOR
        end_idx = start_idx + CSI_WIN_SIZE
        cir_window = np.zeros(CSI_WIN_SIZE, dtype=np.complex64)
        
        # Safe array slicing with bounds checking
        src_start = max(0, start_idx)
        src_end = min(len(res['correlation']), end_idx)
        dst_start = src_start - start_idx
        dst_end = dst_start + (src_end - src_start)
        
        if src_end > src_start:
             cir_window[dst_start:dst_end] = res['correlation'][src_start:src_end]

        if np.sum(np.abs(cir_window)) < 1e-6:
            return None
        
        metrics = sdr_utils.calculate_csi_metrics(cir_window, args.rate)
        metrics['snr_db'] = res['snr_db']
        metrics['peak_val'] = res['peak_val']
        return metrics
        
    return None


def rx_analysis_loop(usrp, driver): 
    print(f"   [RX] Object Detection Active ({driver.MODE_NAME}).")
    rx_streamer = driver.get_rx_streamer()
    
    buff_len = 10000 
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    cmd = uhd.types.StreamCMD(driver.STREAM_MODE_START)
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

                result = process_rx_packet_refactored(data)
                
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


    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))

if __name__ == "__main__":
    print("--> Initializing Object Detector...")
    driver = B210UnifiedDriver(args.freq, args.rate, args.gain)
    usrp = driver.initialize()
    

    tx_thread = PeriodicTransmitter(driver, sig_handler, TX_FRAME, interval=0.5)
    tx_thread.start()
    
    try:
        rx_analysis_loop(usrp, driver)
    except KeyboardInterrupt:
        pass