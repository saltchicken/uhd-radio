import uhd
import numpy as np
import sys
import time

from sdr_lib.usrp_driver import B210UnifiedDriver
from sdr_lib import sdr_utils

# ‼️ Scanner-specific configuration: Default center to 433.92 MHz
args = sdr_utils.get_standard_args("Wideband Spectrum Scanner", default_freq=433.92e6, default_rate=2e6)

# ‼️ Narrowed Scan Range (433 MHz - 435 MHz) for faster loop times
START_FREQ = 420e6
STOP_FREQ = 440e6
STEP_SIZE = 1e6  # 1 MHz hops
BANDWIDTH_VIEW = 40  
DWELL_TIME = 0.1 # ‼️ Listen for 0.1s per step

sig_handler = sdr_utils.SignalHandler()

def run_scanner(usrp, driver):
    print(f"   [SCAN] Starting Wideband Scan: {START_FREQ/1e6:.1f} MHz -> {STOP_FREQ/1e6:.1f} MHz")
    rx_streamer = driver.get_rx_streamer()
    
    # Buffer Setup
    buff_len = 4096 
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    # Start streaming once
    cmd = uhd.types.StreamCMD(driver.STREAM_MODE_START)
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)

    current_freq = START_FREQ
    
    # Header for the dashboard
    print(f"{'FREQ (MHz)':<12} | {'MAX PWR':<10} | {'ACTIVITY'}")
    print("-" * 50)

    while sig_handler.running:
        
        # 1. Hop Frequency
        driver.tune_frequency(current_freq)
        
        # 2. Flush Buffer
        rx_streamer.recv(recv_buffer, metadata, 0.1) 
        
        # ‼️ 3. Continuous Listening (Peak Hold)
        # Instead of sleeping, we loop and collect data for the entire DWELL_TIME
        start_dwell = time.time()
        max_power_db = -120.0
        best_data_chunk = None

        while time.time() - start_dwell < DWELL_TIME:
            samps = rx_streamer.recv(recv_buffer, metadata, 0.05)
            
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                continue

            if samps > 0:
                data = recv_buffer[0][:samps]
                
                # Instantaneous power of this chunk
                power = np.mean(np.abs(data)**2)
                power_db = 10 * np.log10(power + 1e-12)
                
                # Keep the strongest signal seen during this dwell
                if power_db > max_power_db:
                    max_power_db = power_db
                    best_data_chunk = data

        # 4. Visualization
        if best_data_chunk is not None:
            # Use the data chunk that had the highest power for the visual
            bar_chart = sdr_utils.ascii_bar_chart(np.abs(best_data_chunk[:100]), width=20) 
            
            indicator = " "
            if max_power_db > -35: indicator = "█ STRONG" 
            elif max_power_db > -55: indicator = "▄ Active"
            elif max_power_db > -70: indicator = "  Weak  "
            else: indicator = "  ...   "

            sys.stdout.write(f"\r{current_freq/1e6:9.1f} MHz | {max_power_db:6.1f} dB | [{bar_chart}] {indicator}")
            sys.stdout.flush()

        # 5. Advance Frequency
        current_freq += STEP_SIZE
        if current_freq > STOP_FREQ:
            current_freq = START_FREQ
            print() # Newline for the next sweep
            
        # ‼️ No sleep here! The sleep happens in the data collection loop.

    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))

if __name__ == "__main__":
    print("--> Initializing Spectrum Scanner...")
    driver = B210UnifiedDriver(START_FREQ, args.rate, args.gain)
    usrp = driver.initialize()
    
    try:
        run_scanner(usrp, driver)
    except KeyboardInterrupt:
        pass
