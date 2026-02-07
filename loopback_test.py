import uhd
import numpy as np
import time

from sdr_lib.usrp_driver import B210UnifiedDriver, PeriodicTransmitter 
from sdr_lib import sdr_utils


args = sdr_utils.get_standard_args("Loopback Test", default_gain=50)

sig_handler = sdr_utils.SignalHandler()


def generate_tone_frame(rate, duration=0.05, freq=50e3):
    num_samps = int(rate * duration)
    t = np.arange(num_samps) / rate
    tone = 0.7 * np.exp(1j * 2 * np.pi * freq * t)
    return tone.astype(np.complex64)

TX_FRAME = generate_tone_frame(args.rate)

def run_robust_rx(usrp, driver):
    print(f"   [RX Main] Starting Loop ({driver.MODE_NAME})...")
    rx_streamer = driver.get_rx_streamer()
    
    buff_len = int(args.rate * 0.05) 
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    def issue_start_cmd():
        cmd = uhd.types.StreamCMD(driver.STREAM_MODE_START)
        cmd.stream_now = True
        rx_streamer.issue_stream_cmd(cmd)

    issue_start_cmd()
    
    pkts_received = 0
    silence_counter = 0
    debug_timer = time.time()
    
    while sig_handler.running:
        samps = rx_streamer.recv(recv_buffer, metadata, 0.1)
        
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            err = metadata.error_code
            if err == uhd.types.RXMetadataErrorCode.overflow:
                continue
            elif err == uhd.types.RXMetadataErrorCode.timeout:
                silence_counter += 1
                if silence_counter > 20: 
                    print("   [RX Watchdog] ⚠️ Radio silent (Timeout). Restarting stream...")
                    issue_start_cmd()
                    silence_counter = 0
                continue
            else:
                print(f"   [RX Main] Error: {err}")
                continue

        if samps > 0:
            silence_counter = 0
            data_chunk = recv_buffer[0][:samps]
            magnitudes = np.abs(data_chunk)
            peak = np.max(magnitudes)
            avg = np.mean(magnitudes)

            if peak > 0.01: 
                pkts_received += 1
                bar_len = int(peak * 40)
                if bar_len > 40: bar_len = 40
                bar = "#" * bar_len
                print(f"   [RX] Pkt #{pkts_received} | Amp: {peak:.4f} | {bar}")
            
            if time.time() - debug_timer > 1.0:
                print(f"   [RX Status] Noise Floor: {avg:.6f} | listening...")
                debug_timer = time.time()

    print("   [RX Main] Cleaning up...")
    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))

if __name__ == "__main__":
    print("--> Initializing B210 Loopback...")
    driver = B210UnifiedDriver(args.freq, args.rate, args.gain)
    usrp = driver.initialize()
    

    tx_thread = PeriodicTransmitter(driver, sig_handler, TX_FRAME, interval=1.0)
    tx_thread.start()
    
    try:
        run_robust_rx(usrp, driver)
    except KeyboardInterrupt:
        pass
