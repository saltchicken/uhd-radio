import uhd
import numpy as np
import sys
import time
import threading

from sdr_lib.usrp_driver import B210UnifiedDriver
from sdr_lib import sdr_utils

# ‼️ CHANGED: Default frequency set to 2.412e9 (WiFi Ch 1) and higher gain
args = sdr_utils.get_standard_args("MIMO Object Detection", default_freq=2.412e9, default_gain=75)

# Configuration
CHIRP_LEN = 256       
GAP_LEN = 2000        
THRESHOLD = 0.05    

CALIBRATION_FRAMES = 40        
DETECTION_THRESHOLD = 2.0 # ‼️ Adjusted for MIMO sensitivity
CSI_WIN_SIZE = 64             

sig_handler = sdr_utils.SignalHandler()

# Generate Probe
PROBE_TX = sdr_utils.generate_chirp_probe(CHIRP_LEN)

# ‼️ CHANGED: Create a MIMO Frame (2 Channels). 
# We transmit on Ch0, silence on Ch1 (or duplicate if desired).
# USRP expects (2, N) array for send() when 2 channels are active.
padding = np.zeros(GAP_LEN, dtype=np.complex64)
single_channel_frame = np.concatenate([padding, PROBE_TX, padding])

# Stack to create (2, N). sending on Ch0 only.
TX_FRAME_MIMO = np.vstack([single_channel_frame, np.zeros_like(single_channel_frame)])


class MIMOPeriodicTransmitter(threading.Thread):
    """
    ‼️ NEW: Custom thread to handle 2-channel transmission structure.
    The standard PeriodicTransmitter assumes 1 channel.
    """
    def __init__(self, driver, sig_handler, frame_data, interval=0.5):
        super().__init__()
        self.driver = driver
        self.handler = sig_handler
        self.frame = frame_data
        self.interval = interval
        self.daemon = True

    def run(self):
        print(f"   [TX] MIMO Background Transmitter Active (Every {self.interval}s)")
        tx_streamer = self.driver.get_tx_streamer()
        
        md = uhd.types.TXMetadata()
        md.start_of_burst = True
        md.end_of_burst = True
        
        while self.handler.running:
            try:
                md.has_time_spec = False
                # send() expects (num_channels, num_samps)
                tx_streamer.send(self.frame, md)
                time.sleep(self.interval)
            except Exception:
                pass


def extract_csi_for_channel(rx_data, peak_idx, sample_rate):
    """
    Helper to slice and compute CSI metrics for a specific channel data array.
    """
    # Extract CSI Window
    PRE_CURSOR = 10
    start_idx = peak_idx - PRE_CURSOR
    end_idx = start_idx + CSI_WIN_SIZE
    cir_window = np.zeros(CSI_WIN_SIZE, dtype=np.complex64)
    
    # Safe array slicing
    src_start = max(0, start_idx)
    src_end = min(len(rx_data), end_idx)
    dst_start = src_start - start_idx
    dst_end = dst_start + (src_end - src_start)
    
    if src_end > src_start:
         cir_window[dst_start:dst_end] = rx_data[src_start:src_end]

    if np.sum(np.abs(cir_window)) < 1e-6:
        return None
    
    metrics = sdr_utils.calculate_csi_metrics(cir_window, sample_rate)
    return metrics


def rx_mimo_analysis_loop(usrp, driver): 
    print(f"   [RX] MIMO Object Detection Active ({driver.MODE_NAME}).")
    rx_streamer = driver.get_rx_streamer()
    
    # ‼️ CHANGED: Buffer for 2 channels
    buff_len = 10000 
    recv_buffer = np.zeros((2, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    cmd = uhd.types.StreamCMD(driver.STREAM_MODE_START)
    
    # ‼️ CHANGED: Fixed RuntimeError for MIMO alignment
    # 'stream_now=True' fails on multi-channel because alignment cannot be guaranteed.
    # We must schedule the start time slightly in the future.
    cmd.stream_now = False
    cmd.time_spec = usrp.get_time_now() + uhd.types.TimeSpec(0.05)
    
    rx_streamer.issue_stream_cmd(cmd)

    # ‼️ CHANGED: Separate baselines for Ch0 and Ch1
    baseline_cfr_ch0 = None
    baseline_cfr_ch1 = None
    
    cal_frames_ch0 = []
    cal_frames_ch1 = []
    
    frame_count = 0
    
    print("\n   [MIMO DETECT] 🟡 CALIBRATING... Keep area static.")
    
    while sig_handler.running:
        samps = rx_streamer.recv(recv_buffer, metadata, 0.1)
        
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
             continue

        if samps > 0:
            ch0_data = recv_buffer[0][:samps]
            ch1_data = recv_buffer[1][:samps]
            
            # Detect on Ch0 (Primary)
            # We assume if Ch0 hears it, Ch1 likely does too (co-located)
            if np.max(np.abs(ch0_data)) > THRESHOLD:

                # 1. Correlate Ch0 to find sync point
                res_ch0 = sdr_utils.correlate_and_detect(ch0_data, PROBE_TX)
                
                if res_ch0['snr_db'] > 10:
                    peak_idx = res_ch0['peak_idx']
                    
                    # 2. Extract metrics for BOTH channels using Ch0's timing
                    metrics_ch0 = extract_csi_for_channel(ch0_data, peak_idx, args.rate)
                    metrics_ch1 = extract_csi_for_channel(ch1_data, peak_idx, args.rate)
                    
                    if metrics_ch0 and metrics_ch1:
                        
                        cfr0 = metrics_ch0['cfr_db']
                        cfr1 = metrics_ch1['cfr_db']

                        # --- Calibration Phase ---
                        if frame_count < CALIBRATION_FRAMES:
                            cal_frames_ch0.append(cfr0)
                            cal_frames_ch1.append(cfr1)
                            frame_count += 1
                            sys.stdout.write(f"\r   [MIMO DETECT] Calibrating: {frame_count}/{CALIBRATION_FRAMES}")
                            sys.stdout.flush()
                            
                            if frame_count == CALIBRATION_FRAMES:
                                baseline_cfr_ch0 = np.mean(np.array(cal_frames_ch0), axis=0)
                                baseline_cfr_ch1 = np.mean(np.array(cal_frames_ch1), axis=0)
                                print(f"\n   [MIMO DETECT] 🟢 Calibration Complete.")
                                print(f"   [MIMO DETECT] 📏 Threshold: {DETECTION_THRESHOLD:.2f}")
                        
                        # --- Detection Phase ---
                        else:
                            # ‼️ CHANGED: Calculate anomaly on both antennas
                            diff0 = np.abs(cfr0 - baseline_cfr_ch0)
                            diff1 = np.abs(cfr1 - baseline_cfr_ch1)
                            
                            score0 = np.mean(diff0)
                            score1 = np.mean(diff1)
                            
                            # ‼️ CHANGED: Combined score (Average of both spatial paths)
                            combined_score = (score0 + score1) / 2
                            
                            is_detected = combined_score > DETECTION_THRESHOLD
                            status_icon = "🔴 OBJECT DETECTED" if is_detected else "🟢 Clear"
                            
                            print("-" * 60)
                            print(f"‼️ MIMO STATUS: {status_icon}")
                            print(f"   Combined Score: {combined_score:.2f} (Thresh: {DETECTION_THRESHOLD})")
                            print(f"   Antenna A Score: {score0:.2f} | Antenna B Score: {score1:.2f}")
                            print(f"   Ch0 Delta: [{sdr_utils.ascii_bar_chart(diff0)}]")
                            print(f"   Ch1 Delta: [{sdr_utils.ascii_bar_chart(diff1)}]")

    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))

if __name__ == "__main__":
    print("--> Initializing MIMO Object Detector...")
    
    # ‼️ CHANGED: Initialize 2 channels
    driver = B210UnifiedDriver(args.freq, args.rate, args.gain, num_channels=2)
    usrp = driver.initialize()
    
    # ‼️ CHANGED: Use local MIMO transmitter class
    tx_thread = MIMOPeriodicTransmitter(driver, sig_handler, TX_FRAME_MIMO, interval=0.5)
    tx_thread.start()
    
    try:
        rx_mimo_analysis_loop(usrp, driver)
    except KeyboardInterrupt:
        pass
