import uhd
import numpy as np
import sys
import time
import collections
from datetime import datetime

from sdr_lib.usrp_driver import B210UnifiedDriver
from sdr_lib import sdr_utils


args = sdr_utils.get_standard_args("Wideband Waterfall", default_freq=915e6, default_rate=20e6, default_gain=40)


UPDATE_RATE = 30.0 # Target Frames Per Second
FFT_SIZE = 2048    # FFT Resolution
VIEW_WIDTH = 100   # ASCII Character Width
WATERFALL_HEIGHT = 20

# Visualization Dynamic Range (dB)
MIN_DB = -30.0
MAX_DB = -10.0

sig_handler = sdr_utils.SignalHandler()

def resize_spectrum(fft_data, target_width):
    """
    ‼️ Resizes high-res FFT data to fit ASCII width.
    Uses 'Max Hold' downsampling so narrow signals don't disappear.
    """
    source_len = len(fft_data)
    chunk_size = source_len // target_width
    if chunk_size < 1: return fft_data[:target_width]
    
    resized = np.zeros(target_width)
    for i in range(target_width):
        start = i * chunk_size
        end = start + chunk_size
        # Use max to preserve signal peaks in the bin
        resized[i] = np.max(fft_data[start:end])
    return resized

def run_waterfall(usrp, driver):
    # Calculate Frequency Edges for display
    bw_hz = args.rate
    start_freq = args.freq - (bw_hz / 2)
    stop_freq = args.freq + (bw_hz / 2)

    print(f"   [WATERFALL] Center: {args.freq/1e6:.1f} MHz | BW: {bw_hz/1e6:.1f} MHz")
    print(f"   [WATERFALL] Span: {start_freq/1e6:.1f} - {stop_freq/1e6:.1f} MHz")
    print(f"   [WATERFALL] FFT: {FFT_SIZE} bins -> {VIEW_WIDTH} chars")
    
    rx_streamer = driver.get_rx_streamer()
    
    # Buffer Setup
    # We need at least FFT_SIZE, but allow some overhead
    buff_len = FFT_SIZE * 4 
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    # Ensure Analog Bandwidth is open (if hardware supports it)
    try:
        usrp.set_rx_bandwidth(args.rate, 0)
    except:
        pass

    cmd = uhd.types.StreamCMD(driver.STREAM_MODE_START)
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)


    history_buffer = collections.deque(maxlen=WATERFALL_HEIGHT)
    for _ in range(WATERFALL_HEIGHT):
        history_buffer.append("") 

    # Header
    header = f"{'TIME':<10} | {start_freq/1e6:.1f} MHz" + " " * (VIEW_WIDTH - 20) + f"{stop_freq/1e6:.1f} MHz | {'RANGE (dB)':<10}"
    header_border = "-" * len(header)
    
    print("\n" * (WATERFALL_HEIGHT + 4))

    # Pre-compute Window
    window = np.hanning(FFT_SIZE)
    frame_interval = 1.0 / UPDATE_RATE

    while sig_handler.running:
        loop_start = time.time()


        # Since we sleep to control FPS, the buffer fills with "old" data.
        # We flush it to ensure the FFT represents 'now'.
        while True:
            samps = rx_streamer.recv(recv_buffer, metadata, 0.0)
            if samps == 0: break
        

        # We request exactly FFT_SIZE samples.
        samps = rx_streamer.recv(recv_buffer, metadata, 0.1)
        
        if samps >= FFT_SIZE:
            raw_data = recv_buffer[0][:FFT_SIZE]
            

            # Windowing -> FFT -> Shift (Center DC) -> Mag -> Log
            fft_result = np.fft.fft(raw_data * window)
            fft_shifted = np.fft.fftshift(fft_result)
            psd_db = 10 * np.log10(np.abs(fft_shifted)**2 + 1e-12)
            
            # Normalize/Calibrate (Rough offset to match dBm somewhat)
            psd_db -= 20 


            display_row = resize_spectrum(psd_db, VIEW_WIDTH)

            # 5. Render
            timestamp = datetime.now().strftime("%H:%M:%S")
            density_line = sdr_utils.ascii_density_map(display_row, MIN_DB, MAX_DB)
            
            row_min = np.min(display_row)
            row_max = np.max(display_row)
            
            new_line = f"{timestamp} | {density_line} | {row_min:3.0f}..{row_max:3.0f}"
            history_buffer.append(new_line)
            
            # Redraw
            sys.stdout.write(f"\033[{WATERFALL_HEIGHT + 3}A")
            print(header_border)
            print(header)
            print(header_border)
            for line in history_buffer:
                print(f"{line}\033[K")
            sys.stdout.flush()

        # 6. FPS Control
        elapsed = time.time() - loop_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("\n--> Stopping Stream...")
    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))

if __name__ == "__main__":
    driver = B210UnifiedDriver(args.freq, args.rate, args.gain)
    usrp = driver.initialize()
    
    try:
        run_waterfall(usrp, driver)
    except KeyboardInterrupt:
        pass
