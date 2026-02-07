import uhd
import numpy as np
import sys
import time

from sdr_lib.usrp_driver import B210UnifiedDriver
from sdr_lib import sdr_utils


args = sdr_utils.get_standard_args("Wideband Spectrum Scanner", default_freq=430e6, default_rate=2e6)


START_FREQ = 420e6
STOP_FREQ = 440e6
STEP_SIZE = 1e6  # 1 MHz hops


# Calculate how many "bins" or steps we have in our scan range
NUM_STEPS = int(round((STOP_FREQ - START_FREQ) / STEP_SIZE))
BANDWIDTH_VIEW = NUM_STEPS 

DWELL_TIME = 0.1 
DISPLAY_THRESHOLD = -15.0 

sig_handler = sdr_utils.SignalHandler()

def run_scanner(usrp, driver):
    print(f"   [SCAN] Starting Wideband Scan: {START_FREQ/1e6:.1f} MHz -> {STOP_FREQ/1e6:.1f} MHz")
    print(f"   [SCAN] Squelch Threshold: {DISPLAY_THRESHOLD} dB")
    print(f"   [SCAN] Steps: {NUM_STEPS} | Width: {BANDWIDTH_VIEW} chars")
    
    rx_streamer = driver.get_rx_streamer()
    
    # Buffer Setup
    buff_len = 4096 
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    # Start streaming once
    cmd = uhd.types.StreamCMD(driver.STREAM_MODE_START)
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)


    # Initialize with a low value (-100 dB)
    scan_power_levels = [-120.0] * NUM_STEPS

    current_freq = START_FREQ
    
    # Header for the dashboard
    print(f"{'FREQ':<10} | {'PWR':<6} | {'BAND HISTORY map':<{BANDWIDTH_VIEW}} | {'STATUS'}")
    print("-" * (40 + BANDWIDTH_VIEW))

    while sig_handler.running:
        
        # 1. Hop Frequency
        driver.tune_frequency(current_freq)
        
        # 2. Flush Buffer
        rx_streamer.recv(recv_buffer, metadata, 0.1) 
        
        # 3. Continuous Listening (Peak Hold for this step)
        start_dwell = time.time()
        max_power_db = -120.0

        while time.time() - start_dwell < DWELL_TIME:
            samps = rx_streamer.recv(recv_buffer, metadata, 0.05)
            
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                continue

            if samps > 0:
                data = recv_buffer[0][:samps]
                
                # Instantaneous power
                power = np.mean(np.abs(data)**2)
                power_db = 10 * np.log10(power + 1e-12)
                
                if power_db > max_power_db:
                    max_power_db = power_db


        freq_idx = int(round((current_freq - START_FREQ) / STEP_SIZE))
        
        # Safety check for index bounds
        if 0 <= freq_idx < NUM_STEPS:
            scan_power_levels[freq_idx] = max_power_db


        # This shows the "State" of the entire band at once
        band_visual = ""
        for i, p in enumerate(scan_power_levels):
            # Is this the frequency we are currently listening to?
            is_current = (i == freq_idx)
            
            if is_current:
                # Cursor logic
                if p > DISPLAY_THRESHOLD: band_visual += "█" # Active + Current
                else: band_visual += "▽" # Scanning here
            else:
                # History logic
                if p > DISPLAY_THRESHOLD: band_visual += "|" # Was active previously
                else: band_visual += "_" # Was quiet previously

        indicator = " "
        if max_power_db > -35: indicator = "STRONG" 
        elif max_power_db > DISPLAY_THRESHOLD: indicator = "ACTIVE"
        else: indicator = "Scanning..."

        # Print the dashboard line
        sys.stdout.write(f"\r{current_freq/1e6:7.1f}MHz | {max_power_db:5.0f} | [{band_visual}] {indicator:<10}")
        sys.stdout.flush()

        # 6. Advance Frequency
        current_freq += STEP_SIZE
        
        # Loop back to start if we hit the end
        if current_freq >= STOP_FREQ:
            current_freq = START_FREQ

    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))

if __name__ == "__main__":
    print("--> Initializing Spectrum Scanner...")
    driver = B210UnifiedDriver(START_FREQ, args.rate, args.gain)
    usrp = driver.initialize()
    
    try:
        run_scanner(usrp, driver)
    except KeyboardInterrupt:
        pass
