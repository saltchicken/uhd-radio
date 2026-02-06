import uhd
import numpy as np
import threading
import time
import sys
import signal

# ==========================================
# ‼️ REDESIGN: Constants & Globals
# ==========================================
RX_RATE = 1e6
TX_RATE = 1e6
FREQ = 915e6
# ‼️ CHANGED: Increased Gain significantly to ensure signal detection
GAIN = 50 
RUNNING = True

# ‼️ Dynamic StreamMode Resolution (Robust)
try:
    STREAM_MODE_START = uhd.types.StreamMode.start_continuous
    STREAM_MODE_STOP = uhd.types.StreamMode.stop_continuous
    MODE_NAME = "Native Continuous"
except AttributeError:
    # Fallback for versions where start_continuous is missing/renamed
    STREAM_MODE_START = uhd.types.StreamMode.num_done
    STREAM_MODE_STOP = uhd.types.StreamMode.num_done
    MODE_NAME = "Manual Burst (Fallback)"

def handler(signum, frame):
    global RUNNING
    print("\n--> Signal caught. Shutting down...")
    RUNNING = False

signal.signal(signal.SIGINT, handler)

# ==========================================
# ‼️ NEW: Dedicated TX Daemon
# ==========================================
def tx_daemon(usrp):
    """
    Runs in background. Wakes up once per second to send a pulse.
    """
    print("   [TX Daemon] Launched.")
    
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    tx_streamer = usrp.get_tx_stream(st_args)
    
    # ‼️ Create a distinct Tone Pulse (Longer: 50ms)
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
            
            # ‼️ DEBUG: Print when we fire
            # print("   [TX Daemon] --> Pulse Sent") 
            time.sleep(1.0)
        except Exception as e:
            print(f"   [TX Daemon] Error: {e}")
            time.sleep(1.0)
            
    print("   [TX Daemon] Exiting.")

# ==========================================
# ‼️ NEW: Main RX Loop (Priority)
# ==========================================
def run_robust_rx(usrp):
    """
    Main thread strictly handles RX.
    Includes 'Watchdog' logic to restart stream if it dies.
    """
    print(f"   [RX Main] Starting Loop ({MODE_NAME})...")
    
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    rx_streamer = usrp.get_rx_stream(st_args)
    
    # Buffer Setup
    buff_len = int(RX_RATE * 0.05) 
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    def issue_start_cmd():
        cmd = uhd.types.StreamCMD(STREAM_MODE_START)
        cmd.stream_now = True
        if MODE_NAME == "Manual Burst (Fallback)":
            cmd.num_samps = buff_len 
        rx_streamer.issue_stream_cmd(cmd)

    issue_start_cmd()
    
    pkts_received = 0
    silence_counter = 0
    debug_timer = time.time()
    
    while RUNNING:
        if MODE_NAME == "Manual Burst (Fallback)":
            issue_start_cmd()
            
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
            # Calculate raw amplitude
            magnitudes = np.abs(data_chunk)
            peak = np.max(magnitudes)
            avg = np.mean(magnitudes)
            
            # ‼️ CHANGED: Lower threshold to 0.01 to catch weak signals
            if peak > 0.01: 
                pkts_received += 1
                bar_len = int(peak * 40)
                if bar_len > 40: bar_len = 40
                bar = "#" * bar_len
                print(f"   [RX] Pkt #{pkts_received} | Amp: {peak:.4f} | {bar}")
            
            # ‼️ NEW: Periodic Noise Floor Report (Every 1s)
            # This proves the RX is actually working, even if no packets are found.
            if time.time() - debug_timer > 1.0:
                print(f"   [RX Status] Noise Floor: {avg:.6f} | listening...")
                debug_timer = time.time()

    print("   [RX Main] Cleaning up...")
    stop_cmd = uhd.types.StreamCMD(STREAM_MODE_STOP)
    rx_streamer.issue_stream_cmd(stop_cmd)


if __name__ == "__main__":
    print("--> Initializing B210...")
    
    try:
        usrp = uhd.usrp.MultiUSRP("type=b200")
    except RuntimeError:
        print("‼️ Device not found.")
        sys.exit(1)
        
    # Config
    usrp.set_rx_rate(RX_RATE, 0)
    usrp.set_tx_rate(TX_RATE, 0)
    usrp.set_rx_freq(uhd.types.TuneRequest(FREQ), 0)
    usrp.set_tx_freq(uhd.types.TuneRequest(FREQ), 0)
    usrp.set_rx_gain(GAIN, 0)
    usrp.set_tx_gain(GAIN, 0)
    
    # ‼️ Antenna Config
    usrp.set_tx_antenna("TX/RX", 0)
    usrp.set_rx_antenna("RX2", 0)
    
    print("--> Waiting for LO Lock...")
    time.sleep(1.0)
    
    tx_t = threading.Thread(target=tx_daemon, args=(usrp,))
    tx_t.daemon = True 
    tx_t.start()
    
    try:
        run_robust_rx(usrp)
    except KeyboardInterrupt:
        pass
