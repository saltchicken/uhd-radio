import uhd
import numpy as np
import threading
import time
import sys

from sdr_lib.usrp_driver import B210UnifiedDriver
from sdr_lib import sdr_utils


args = sdr_utils.get_standard_args("Doppler CW Radar", default_freq=2.45e9, default_rate=200e3, default_gain=70)

sig_handler = sdr_utils.SignalHandler()

# Radar Config
FFT_SIZE = 2048
SPEED_OF_LIGHT = 3e8
WAVELENGTH = SPEED_OF_LIGHT / args.freq
MAX_DISPLAY_SPEED = 3.0 # m/s (approx walking speed)

# Determine frequency resolution:
# Resolution = Sample_Rate / FFT_Size
# At 200kHz / 2048 = ~97 Hz per bin.
# At 2.45GHz, 1 m/s = 16 Hz doppler. 
# We need oversampling or fine interpolation, or just higher frequency.
# If we used 5.8GHz, 1 m/s = 38 Hz. Better.

class ContinuousWaveTransmitter(threading.Thread):
    """
    ‼️ Custom thread for this app.
    Unlike PeriodicTransmitter, this pushes a continuous stream 
    of a pure sine wave (CW) to create the radar illuminator.
    """
    def __init__(self, driver):
        super().__init__()
        self.driver = driver
        self.daemon = True
        self.running = True

    def run(self):
        print(f"   [TX] CW Illuminator Active on {args.freq/1e9:.3f} GHz")
        tx_streamer = self.driver.get_tx_streamer()
        
        # Create a buffer of 1s worth of Tone
        # We use a slight offset in digital domain if we wanted to avoid DC, 
        # but for CW radar, 0Hz (DC) is fine as the carrier.
        num_samps = int(args.rate) 
        t = np.arange(num_samps) / args.rate
        # Create a smooth tone
        tone = 0.8 * np.exp(1j * 2 * np.pi * 0 * t) # 0 Hz relative to LO
        
        frame = tone.astype(np.complex64)
        
        md = uhd.types.TXMetadata()
        md.start_of_burst = True
        md.end_of_burst = False
        
        while self.running and sig_handler.running:
            try:
                tx_streamer.send(frame, md)
                md.start_of_burst = False
            except Exception:
                break
        
        # End burst
        md.end_of_burst = True
        tx_streamer.send(np.zeros(100, dtype=np.complex64), md)


def run_radar_loop(usrp, driver):
    rx_streamer = driver.get_rx_streamer()
    
    # Buffer slightly larger than FFT size to allow for windowing/overlap if needed
    buff_len = FFT_SIZE
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    cmd = uhd.types.StreamCMD(driver.STREAM_MODE_START)
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)

    # Window function to reduce spectral leakage
    window = np.blackman(FFT_SIZE)
    
    print(f"   [RX] Doppler Analysis | Resolution: {args.rate/FFT_SIZE:.1f} Hz/bin")
    print(f"   [RX] Wavelength: {WAVELENGTH*100:.1f} cm")
    print("-" * 70)

    avg_spectrum = None
    alpha = 0.1 # Averaging factor for background subtraction

    while sig_handler.running:
        samps = rx_streamer.recv(recv_buffer, metadata, 0.1)
        
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            continue

        if samps == buff_len:
            data = recv_buffer[0] * window
            
            # FFT Processing
            spectrum = np.fft.fftshift(np.fft.fft(data))
            mag = np.abs(spectrum)
            

            # The direct path from TX->RX (leakage) is massive and static (at DC).
            # We track the average background and subtract it to see moving items.
            if avg_spectrum is None:
                avg_spectrum = mag
            else:
                avg_spectrum = (1 - alpha) * avg_spectrum + alpha * mag
            
            # Dynamic subtraction: Look for changes *relative* to the static background
            diff_spectrum = mag - avg_spectrum
            
            # Zero out the center DC bin (static leakage is too strong)
            center = FFT_SIZE // 2
            dc_width = 4 
            diff_spectrum[center-dc_width : center+dc_width] = 0
            
            # Find strongest Doppler peak
            peak_idx = np.argmax(diff_spectrum)
            peak_val = diff_spectrum[peak_idx]
            
            # Threshold to avoid noise
            if peak_val > 5.0: # Arbitrary magnitude threshold
                
                # Calculate frequency shift
                # Bins map from -rate/2 to +rate/2
                bin_freqs = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1/args.rate))
                doppler_shift_hz = bin_freqs[peak_idx]
                
                # Calculate Velocity: v = (fd * lambda) / 2
                # Note: Factor of 2 is for round-trip (monostatic radar)
                velocity_ms = (doppler_shift_hz * WAVELENGTH) / 2
                
                # Visuals
                gauge = sdr_utils.ascii_dual_gauge(velocity_ms, MAX_DISPLAY_SPEED)
                
                direction = "STATIC"
                if velocity_ms > 0.1: direction = "APPROACHING"
                elif velocity_ms < -0.1: direction = "RECEDING   "
                
                sys.stdout.write(f"\r   [RADAR] {velocity_ms:+5.2f} m/s | {doppler_shift_hz:+6.1f} Hz | [{gauge}] {direction}")
                sys.stdout.flush()
            else:
                # Decay the display if nothing detected
                sys.stdout.write(f"\r   [RADAR]  0.00 m/s |    0.0 Hz | [{sdr_utils.ascii_dual_gauge(0, 1)}] SCANNING...  ")
                sys.stdout.flush()

    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))

if __name__ == "__main__":
    print("--> Initializing Doppler Radar...")
    
    # Initialize USRP
    driver = B210UnifiedDriver(args.freq, args.rate, args.gain)
    usrp = driver.initialize()
    
    # Start the Continuous Wave Transmitter
    tx_thread = ContinuousWaveTransmitter(driver)
    tx_thread.start()
    
    # Start the Receiver Loop
    try:
        run_radar_loop(usrp, driver)
    except KeyboardInterrupt:
        pass