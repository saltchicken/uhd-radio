import uhd
import numpy as np
import sys
import time

from sdr_lib.usrp_driver import B210UnifiedDriver 
from sdr_lib import sdr_utils

# ‼️ CHANGED: Default to WiFi Channel 1 (2.412 GHz) and higher gain
args = sdr_utils.get_standard_args("WiFi Direction Finder", default_freq=2.412e9, default_gain=75)

# Physical Configuration

# ‼️ CHANGED: For 2.4 GHz, spacing must be much smaller (~6.25cm).
# If you leave them at 35cm, you will get "Grating Lobes" (fake peaks every few degrees).
ANTENNA_SPACING = 0.0625  
SCAN_SPEED = 2.0        
# ‼️ NEW: WiFi is bursty, so we use a threshold to ignore noise
PEAK_THRESHOLD = -40.0

sig_handler = sdr_utils.SignalHandler()

def run_beamformer(usrp, driver):
    print(f"--> WiFi Beamformer Active on {args.freq/1e9:.3f} GHz")
    print(f"--> Required Spacing: {ANTENNA_SPACING*100:.2f} cm (Critical for 2.4GHz)")
    
    # Check for spacing physics
    wavelength = 3e8 / args.freq
    if ANTENNA_SPACING > (wavelength / 1.5):
        print(f"\n⚠️  CRITICAL WARNING: Spacing ({ANTENNA_SPACING*100:.1f}cm) is too wide for WiFi!")
        print(f"   For 2.4GHz, antennas must be ~6.2cm apart.")
        print("   Current setup will produce 'Grating Lobes' (ghost signals).\n")
    
    rx_streamer = driver.get_rx_streamer()
    
    buff_len = 4096
    recv_buffer = np.zeros((2, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    

    cmd = uhd.types.StreamCMD(driver.STREAM_MODE_START)
    cmd.stream_now = False
    cmd.time_spec = usrp.get_time_now() + uhd.types.TimeSpec(0.05)
    
    rx_streamer.issue_stream_cmd(cmd)


    calibrated = False
    calibration_offset = 0.0
    
    current_angle = -60.0
    scan_direction = 1
    
    # ‼️ RESTORED: Sweep history for peak searching
    sweep_history = [] 

    last_update_time = time.time()
    
    print("\n[SETUP] ⚠️  B210 Phase Ambiguity Detected.")
    print("[SETUP] Place a constant source (phone hotspot) at 0 deg and press Ctrl+C to Calibrate.")
    print("[SETUP] WiFi beacons are intermittent, so calibration might take a few tries.\n")
    
    try:
        while sig_handler.running:

            samps = rx_streamer.recv(recv_buffer, metadata, 0.1)
            
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                continue
                
            if samps > 0:
                ch0 = recv_buffer[0][:samps]
                ch1 = recv_buffer[1][:samps]
                
                # 1. Measurement - Max Hold for WiFi bursts
                pwr_ch0 = np.max(np.abs(ch0)**2)
                pwr_db_omn = 10 * np.log10(pwr_ch0 + 1e-12)


                if not calibrated and pwr_db_omn > -35:
                    correlation = np.mean(ch1 * np.conj(ch0))
                    calibration_offset = np.angle(correlation)
                    calibrated = True
                    print(f"\n[CAL] ✅ LOCKED! Hardware Offset: {np.degrees(calibration_offset):.1f} deg")
                    print("[CAL] Starting WiFi Sweep...\n")

                # 2. Compute Steering Phase
                steer_phase = sdr_utils.calculate_steering_phase(current_angle, args.freq, ANTENNA_SPACING)
                
                # 3. Apply Beamforming Weights
                beam_signal = sdr_utils.apply_beamforming(ch0, ch1, steer_phase, calibration_offset)
                
                # 4. Measure Beamformed Power
                pwr_beam = np.max(np.abs(beam_signal)**2)
                pwr_db_beam = 10 * np.log10(pwr_beam + 1e-12)
                
                # 5. Calculate "Array Gain"
                gain_db = pwr_db_beam - pwr_db_omn
                
                # ‼️ Record data for DoA calculation
                if calibrated:
                    sweep_history.append((current_angle, pwr_db_beam))

                if time.time() - last_update_time > 0.05:
                    
                    bar_len = 30
                    angle_norm = (current_angle + 90) / 180
                    angle_pos = int(angle_norm * bar_len)
                    angle_pos = max(0, min(bar_len-1, angle_pos))
                    
                    visual = ["."] * bar_len
                    visual[angle_pos] = "O" 
                    visual_str = "".join(visual)
                    
                    status = "CALIBRATING..." if not calibrated else "SCANNING"
                    
                    sys.stdout.write(
                        f"\r[{status}] Angle: {current_angle:5.1f}° [{visual_str}] "
                        f"Sig: {pwr_db_beam:3.0f}dB | "
                        f"Gain: {gain_db:+4.1f}dB"
                    )
                    sys.stdout.flush()

                    # 7. Sweep Logic
                    if calibrated:
                        current_angle += (SCAN_SPEED * scan_direction)
                        
                        # Check bounds
                        hit_upper_limit = (scan_direction == 1 and current_angle >= 60)
                        hit_lower_limit = (scan_direction == -1 and current_angle <= -60)

                        if hit_upper_limit or hit_lower_limit:
                            if sweep_history:
                                # Find tuple with max power
                                best_angle, peak_pwr = max(sweep_history, key=lambda x: x[1])
                                
                                is_edge_artifact = abs(best_angle) >= 58.0
                                
                                if peak_pwr > PEAK_THRESHOLD:
                                    if is_edge_artifact:
                                         print(f"\n⚠️  [EDGE] Ignored peak at {best_angle:.1f}° (Side lobe/Error)")
                                    else:
                                         print(f"\n‼️  [ROUTER FOUND] Direction: {best_angle:.1f}° (Strength: {peak_pwr:.1f} dB)")
                                
                                sweep_history = [] 

                        # Reverse direction
                        if current_angle > 60: scan_direction = -1
                        if current_angle < -60: scan_direction = 1
                    
                    last_update_time = time.time()

    except KeyboardInterrupt:
        pass
        
    print("\n--> Stopping Beamformer...")
    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))

if __name__ == "__main__":
    driver = B210UnifiedDriver(args.freq, args.rate, args.gain, num_channels=2)
    usrp = driver.initialize()
    run_beamformer(usrp, driver)
