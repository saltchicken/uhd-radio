import uhd
import numpy as np
from datetime import datetime

from sdr_lib.usrp_driver import B210UnifiedDriver
from sdr_lib import sdr_utils


args = sdr_utils.get_standard_args("ASCII Waterfall Plotter", default_freq=430e6, default_rate=2e6, default_gain=40)

# Configuration
START_FREQ = 420e6
STOP_FREQ = 440e6
STEP_SIZE = 500e3
DWELL_TIME = 0.02

NUM_STEPS = int(round((STOP_FREQ - START_FREQ) / STEP_SIZE))


# Previous MAX was -40, which is easily hit by noise at high gain.
# Shifted range up to handle stronger signals/noise floor.
MIN_DB = -80.0
MAX_DB = -10.0

sig_handler = sdr_utils.SignalHandler()

def run_waterfall(usrp, driver):
    print(f"   [WATERFALL] Range: {START_FREQ/1e6:.1f} - {STOP_FREQ/1e6:.1f} MHz")
    print(f"   [WATERFALL] Resolution: {STEP_SIZE/1e3:.0f} kHz | Cols: {NUM_STEPS}")
    print(f"   [WATERFALL] Dyn Range: {MIN_DB} dBm to {MAX_DB} dBm")
    
    rx_streamer = driver.get_rx_streamer()
    
    # Large buffer to prevent overflow during tuning
    buff_len = 65536 
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    cmd = uhd.types.StreamCMD(driver.STREAM_MODE_START)
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)

    row_data = np.zeros(NUM_STEPS)
    line_counter = 0

    # Print Header
    header = f"{'TIME':<10} | {START_FREQ/1e6:.1f} MHz" + " " * (NUM_STEPS - 20) + f"{STOP_FREQ/1e6:.1f} MHz | {'RANGE (dB)':<10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    while sig_handler.running:
        
        # 1. Sweep across the band
        for i in range(NUM_STEPS):
            if not sig_handler.running: break

            freq = START_FREQ + (i * STEP_SIZE)
            driver.tune_frequency(freq)
            
            # Robust Flush: Drain the buffer
            for _ in range(20): 
                if not sig_handler.running: break
                samps = rx_streamer.recv(recv_buffer, metadata, 0.0) 
                if samps == 0: break
            
            # Measure
            samps = rx_streamer.recv(recv_buffer, metadata, DWELL_TIME + 0.1)
            
            if samps > 0 and metadata.error_code == uhd.types.RXMetadataErrorCode.none:
                data = recv_buffer[0][:samps]
                power = np.mean(np.abs(data)**2)
                row_data[i] = 10 * np.log10(power + 1e-12)
            else:
                row_data[i] = -120.0

        # 2. Render Row
        if sig_handler.running:
            timestamp = datetime.now().strftime("%H:%M:%S")
            density_line = sdr_utils.ascii_density_map(row_data, MIN_DB, MAX_DB)
            

            # This helps you see if you are hitting the floor (-80) or ceiling (-10)
            row_min = np.min(row_data)
            row_max = np.max(row_data)
            
            print(f"{timestamp} | {density_line} | {row_min:3.0f}..{row_max:3.0f}")
            
            # 3. Periodically reprint header
            line_counter += 1
            if line_counter >= 20:
                print("-" * len(header))
                print(header)
                print("-" * len(header))
                line_counter = 0

    print("\n--> Stopping Stream...")
    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))

if __name__ == "__main__":
    driver = B210UnifiedDriver(START_FREQ, args.rate, args.gain)
    usrp = driver.initialize()
    
    try:
        run_waterfall(usrp, driver)
    except KeyboardInterrupt:
        pass