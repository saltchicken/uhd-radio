import uhd
import numpy as np
import sys
import time
from usrp_driver import B210UnifiedDriver 
import sdr_utils

FREQ = 433.92e6
RATE = 1e6
GAIN = 60

ANTENNA_SPACING_METERS = 0.163
CALIBRATION_PHASE_OFFSET = 0.0
SQUELCH = 0.005


sig_handler = sdr_utils.SignalHandler()

def calculate_aoa(ch0, ch1):
    correlation_vector = ch1 * np.conj(ch0)
    avg_correlation = np.mean(correlation_vector)
    raw_phase = np.angle(avg_correlation)
    phase_diff = (raw_phase - CALIBRATION_PHASE_OFFSET + np.pi) % (2 * np.pi) - np.pi
    wavelength = 3e8 / FREQ
    arg = (phase_diff * wavelength) / (2 * np.pi * ANTENNA_SPACING_METERS)
    arg = max(-1.0, min(1.0, arg))
    theta_deg = np.degrees(np.arcsin(arg))
    return theta_deg, phase_diff, raw_phase



def run_mimo_loop(usrp, driver):
    streamer = driver.get_rx_streamer()
    
    buff_len = 4096 
    recv_buffer = np.zeros((2, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()


    stream_cmd = uhd.types.StreamCMD(driver.STREAM_MODE_START)
    burst_duration = buff_len / RATE
    next_time = usrp.get_time_now() + uhd.types.TimeSpec(0.05)
    
    stream_cmd.stream_now = False
    stream_cmd.time_spec = next_time
    streamer.issue_stream_cmd(stream_cmd)


    print(f"\n--> MIMO Stream Active on {FREQ/1e6} MHz ({driver.MODE_NAME})")
    print(f"--> Antenna Spacing: {ANTENNA_SPACING_METERS*100:.1f} cm")
    print("--> Waiting for signal threshold...\n")

    last_print = 0


    while sig_handler.running:
        samps = streamer.recv(recv_buffer, metadata, burst_duration + 0.1)

        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            continue

        if samps > 0:
            ch0_data = recv_buffer[0][:samps]
            ch1_data = recv_buffer[1][:samps]

            power = np.mean(np.abs(ch0_data)**2)
            
            if power > SQUELCH:
                angle, phase, raw_phase = calculate_aoa(ch0_data, ch1_data)
                rssi_db = 10 * np.log10(power + 1e-12)

                if time.time() - last_print > 0.1:

                    compass = sdr_utils.ascii_compass(angle)
                    sys.stdout.write(f"\r[MIMO] RSSI: {rssi_db:3.0f}dB | AoA: {angle:5.1f}° | RawPh: {raw_phase:5.2f} | [{compass}]")
                    sys.stdout.flush()
                    last_print = time.time()

    print("\n--> Stopping Stream...")

    streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))

if __name__ == "__main__":
    driver = B210UnifiedDriver(FREQ, RATE, GAIN, num_channels=2)
    u = driver.initialize()
    run_mimo_loop(u, driver)