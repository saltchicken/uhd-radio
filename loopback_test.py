import uhd
import numpy as np
import threading
import time
import sys
import signal
from usrp_driver import B210UnifiedDriver 

# ==========================================
# ==========================================
RX_RATE = 1e6
TX_RATE = 1e6
FREQ = 915e6
GAIN = 50 
RUNNING = True


STREAM_MODE_START = uhd.types.StreamMode.start_cont
STREAM_MODE_STOP = uhd.types.StreamMode.stop_cont
MODE_NAME = "Native Continuous"

def handler(signum, frame):
    global RUNNING
    print("\n--> Signal caught. Shutting down...")
    RUNNING = False
signal.signal(signal.SIGINT, handler)

def tx_daemon(usrp, driver):
    """
    Runs in background. Wakes up once per second to send a pulse.
    """
    print("   [TX Daemon] Launched.")
    tx_streamer = driver.get_tx_streamer()

    num_samps = int(TX_RATE * 0.05) 
    t = np.arange(num_samps) / TX_RATE
    tone = 0.7 * np.exp(1j * 2 * np.pi * 50e3 * t)
    tone = tone.astype(np.complex64)
    
    md = uhd.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst = True
    
    while RUNNING:
        try:
            md.has_time_spec = False
            tx_streamer.send(tone.reshape(1, -1), md)
            time.sleep(1.0)
        except Exception as e:
            print(f"   [TX Daemon] Error: {e}")
            time.sleep(1.0)
            
    print("   [TX Daemon] Exiting.")

def run_robust_rx(usrp, driver):
    print(f"   [RX Main] Starting Loop ({MODE_NAME})...")
    rx_streamer = driver.get_rx_streamer()
    
    buff_len = int(RX_RATE * 0.05) 
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    def issue_start_cmd():
        cmd = uhd.types.StreamCMD(STREAM_MODE_START)
        cmd.stream_now = True

        rx_streamer.issue_stream_cmd(cmd)

    issue_start_cmd()
    
    pkts_received = 0
    silence_counter = 0
    debug_timer = time.time()
    
    while RUNNING:

            
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
    stop_cmd = uhd.types.StreamCMD(STREAM_MODE_STOP)
    rx_streamer.issue_stream_cmd(stop_cmd)

if __name__ == "__main__":
    print("--> Initializing B210 Loopback...")
    driver = B210UnifiedDriver(FREQ, RX_RATE, GAIN)
    usrp = driver.initialize()
    
    tx_t = threading.Thread(target=tx_daemon, args=(usrp, driver))
    tx_t.daemon = True 
    tx_t.start()
    
    try:
        run_robust_rx(usrp, driver)
    except KeyboardInterrupt:
        pass
