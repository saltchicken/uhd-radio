import uhd
import numpy as np
import sys
import time

from sdr_lib.usrp_driver import B210UnifiedDriver 
from sdr_lib import sdr_utils


args = sdr_utils.get_standard_args("Digital Beamformer", default_freq=433.92e6, default_gain=60)

# Physical Configuration

# You MUST use cables to space antennas ~35cm apart for this to work.
# If you use the B210 ports directly (0.05m), the beam will be uselessly wide.
ANTENNA_SPACING = 0.35  
SCAN_SPEED = 2.0        

sig_handler = sdr_utils.SignalHandler()

def run_beamformer(usrp, driver):
    print(f"--> Beamforming Array Active on {args.freq/1e6:.3f} MHz")
    print(f"--> Spacing: {ANTENNA_SPACING*100:.1f} cm")
    
    # Check for spacing physics
    wavelength = 3e8 / args.freq
    if ANTENNA_SPACING < (wavelength / 8):
        print(f"⚠️  WARNING: Spacing ({ANTENNA_SPACING}m) is very small for this wavelength ({wavelength:.2f}m).")
        print("   Directionality will be extremely poor. Use extension cables!")
    
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
    

    last_update_time = time.time()
    
    print("\n[SETUP] ⚠️  B210 Phase Ambiguity Detected.")
    print("[SETUP] Place source at 0 deg (Boresight) and press Ctrl+C once to Calibrate.")
    print("[SETUP] Or wait for auto-sweep...\n")
    
    try:
        while sig_handler.running:

            samps = rx_streamer.recv(recv_buffer, metadata, 0.1)
            
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                continue
                
            if samps > 0:
                ch0 = recv_buffer[0][:samps]
                ch1 = recv_buffer[1][:samps]
                
                # 1. Measurement
                pwr_ch0 = np.mean(np.abs(ch0)**2)
                pwr_db_omn = 10 * np.log10(pwr_ch0 + 1e-12)


                if not calibrated and pwr_db_omn > -30:
                    correlation = np.mean(ch1 * np.conj(ch0))
                    calibration_offset = np.angle(correlation)
                    calibrated = True
                    print(f"\n[CAL] ✅ LOCKED! Hardware Offset: {np.degrees(calibration_offset):.1f} deg")
                    print("[CAL] Starting Beam Sweep...\n")

                # 2. Compute Steering Phase
                steer_phase = sdr_utils.calculate_steering_phase(current_angle, args.freq, ANTENNA_SPACING)
                
                # 3. Apply Beamforming Weights
                beam_signal = sdr_utils.apply_beamforming(ch0, ch1, steer_phase, calibration_offset)
                
                # 4. Measure Beamformed Power
                pwr_beam = np.mean(np.abs(beam_signal)**2)
                pwr_db_beam = 10 * np.log10(pwr_beam + 1e-12)
                
                # 5. Calculate "Array Gain"
                gain_db = pwr_db_beam - pwr_db_omn
                

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
                        f"Omni: {pwr_db_omn:3.0f}dB | Beam: {pwr_db_beam:3.0f}dB | "
                        f"Gain: {gain_db:+4.1f}dB"
                    )
                    sys.stdout.flush()

                    # 7. Sweep Logic
                    if calibrated:
                        current_angle += (SCAN_SPEED * scan_direction)
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