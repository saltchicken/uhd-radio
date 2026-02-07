import uhd
import numpy as np
import threading
import time
import sys
import signal

# ==========================================

# ==========================================
RX_RATE = 1e6           
TX_RATE = 1e6
FREQ = 915e6
GAIN = 50               
SPS = 100               
MESSAGE = "Hello World" 

RUNNING = True

try:
    STREAM_MODE_START = uhd.types.StreamMode.start_continuous
    STREAM_MODE_STOP = uhd.types.StreamMode.stop_continuous
    MODE_NAME = "Native Continuous"
except AttributeError:
    STREAM_MODE_START = uhd.types.StreamMode.num_done
    STREAM_MODE_STOP = uhd.types.StreamMode.num_done
    MODE_NAME = "Manual Burst (Fallback)"

def handler(signum, frame):
    global RUNNING
    print("\n--> Signal caught. Shutting down...")
    RUNNING = False
signal.signal(signal.SIGINT, handler)

# ==========================================

# ==========================================

def text_to_bits(text):
    bits = []
    for char in text:
        val = ord(char)
        for i in range(7, -1, -1):
            bits.append((val >> i) & 1)
    return bits

def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte_chunk = bits[i:i+8]
        if len(byte_chunk) < 8:
            break
        val = 0
        for bit in byte_chunk:
            val = (val << 1) | bit
        chars.append(chr(val))
    return "".join(chars)

def modulate_dbpsk(text):

    # We create a new string where the first char is the length count
    # e.g., "\x0B" + "Hello World" (where \x0B is 11)
    full_payload = chr(len(text)) + text
    bits = text_to_bits(full_payload)
    
    samples = []
    current_phase = 0.0
    

    # 1. Pilot: 50 symbols of '0' (Constant Phase) for locking
    for _ in range(50):
        symbol = np.exp(1j * current_phase)
        samples.append(symbol)

    # 2. Sync Marker: A single '1' (Phase Flip) to mark start of text
    current_phase += np.pi
    samples.append(np.exp(1j * current_phase))
        
    # 3. Payload
    for bit in bits:
        if bit == 1:
            current_phase += np.pi 
        symbol = np.exp(1j * current_phase)
        samples.append(symbol)
        
    symbols = np.array(samples, dtype=np.complex64)
    tx_signal = np.repeat(symbols, SPS)
    return tx_signal * 0.7

def demodulate_dbpsk_robust(rx_chunk):
    """
    ‼️ ROBUST DEMODULATOR (NEW)
    Tries multiple timing offsets to find the best lock.
    """
    best_text = ""
    best_score = -1


    # We try offsets 0, 10, 20... 90 to find the center of the symbol
    for offset in range(0, SPS, 10):
        
        # 1. Downsample at this offset
        raw_symbols = rx_chunk[offset::SPS]
        if len(raw_symbols) < 60: continue

        # 2. Differential Detection
        # Calculate phase change between adjacent symbols
        # diff[i] = symbol[i] * conj(symbol[i-1])
        diffs = raw_symbols[1:] * np.conj(raw_symbols[:-1])
        phase_diffs = np.angle(diffs)
        
        # Map to bits: abs(angle) > 90deg is a '1', else '0'
        detected_bits = (np.abs(phase_diffs) > (np.pi / 2)).astype(int)
        

        # We expect ~50 zeros at the start. 
        # Let's count how many zeros are in the first 48 bits.
        preamble_zeros = np.sum(detected_bits[:48] == 0)
        
        # If this offset didn't find the preamble, it's garbage alignment. Skip.
        if preamble_zeros < 40:
            continue
            

        try:
            # Look in a small window where we expect the Sync Bit
            sync_window = detected_bits[45:60]
            # Find index of first '1' relative to the window
            sync_rel_idx = np.where(sync_window == 1)[0][0]
            start_of_data = 45 + sync_rel_idx + 1
        except IndexError:
            continue
            

        # The first 8 bits after sync are now the Length Byte
        len_bits = detected_bits[start_of_data : start_of_data + 8]
        if len(len_bits) < 8: continue
        
        # Convert bits to int
        msg_len = 0
        for bit in len_bits:
            msg_len = (msg_len << 1) | bit
            
        # Sanity Check: If length is huge or 0, this offset is probably wrong
        if msg_len == 0 or msg_len > 100:
            continue


        payload_start = start_of_data + 8
        payload_end = payload_start + (msg_len * 8)
        
        # Ensure we have enough bits
        if len(detected_bits) < payload_end:
            continue
            
        payload_bits = detected_bits[payload_start : payload_end]
        text = bits_to_text(payload_bits)
        
        # Filter non-printable chars
        clean_text = "".join([c for c in text if 32 <= ord(c) <= 126])
        
        # Score this result based on length
        if len(clean_text) > best_score:
            best_score = len(clean_text)
            best_text = clean_text

    return best_text

# ==========================================
# WORKER THREADS
# ==========================================

def tx_daemon(usrp):
    print("   [TX Daemon] Modulating message...")
    tx_data = modulate_dbpsk(MESSAGE)
    
    # Silence padding
    padding = np.zeros(2000, dtype=np.complex64)
    tx_buffer = np.concatenate([padding, tx_data, padding]) # Pad both sides
    
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    tx_streamer = usrp.get_tx_stream(st_args)
    
    md = uhd.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst = True
    md.has_time_spec = False
    
    print(f"   [TX Daemon] Packet size: {len(tx_buffer)} samples")
    print("   [TX Daemon] Broadcasting every 2.0s...")
    
    while RUNNING:
        try:
            tx_streamer.send(tx_buffer.reshape(1, -1), md)
            time.sleep(2.0) 
        except Exception:
            pass

def rx_thread(usrp):
    print(f"   [RX Main] Listening for Data...")
    
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    rx_streamer = usrp.get_rx_stream(st_args)
    
    buff_len = 50000 
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    cmd = uhd.types.StreamCMD(STREAM_MODE_START)
    cmd.stream_now = True
    if MODE_NAME == "Manual Burst (Fallback)":
        cmd.num_samps = buff_len
    rx_streamer.issue_stream_cmd(cmd)
    
    while RUNNING:
        if MODE_NAME == "Manual Burst (Fallback)":
            rx_streamer.issue_stream_cmd(cmd)
            
        samps = rx_streamer.recv(recv_buffer, metadata, 0.1)
        
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            if metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                continue
            if MODE_NAME != "Manual Burst (Fallback)":
                rx_streamer.issue_stream_cmd(cmd)
            continue
            
        if samps > 0:
            data = recv_buffer[0][:samps]
            
            # Trigger
            magnitudes = np.abs(data)
            threshold = 0.05 
            
            high_signal_indices = np.where(magnitudes > threshold)[0]
            
            if len(high_signal_indices) > 2000:
                start_idx = high_signal_indices[0]
                
                # Ensure we have a good chunk
                if start_idx + 20000 < samps:
                    packet_chunk = data[start_idx : start_idx + 20000]
                    

                    msg = demodulate_dbpsk_robust(packet_chunk)
                    
                    if len(msg) > 0:
                         print(f"   [RX] 📬 RECEIVED: '{msg}'")
                         
    stop_cmd = uhd.types.StreamCMD(STREAM_MODE_STOP)
    rx_streamer.issue_stream_cmd(stop_cmd)

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("--> Initializing Data Modem...")
    usrp = uhd.usrp.MultiUSRP("type=b200")
    
    usrp.set_rx_rate(RX_RATE, 0)
    usrp.set_tx_rate(TX_RATE, 0)
    usrp.set_rx_freq(uhd.types.TuneRequest(FREQ), 0)
    usrp.set_tx_freq(uhd.types.TuneRequest(FREQ), 0)
    usrp.set_rx_gain(GAIN, 0)
    usrp.set_tx_gain(GAIN, 0)
    usrp.set_tx_antenna("TX/RX", 0)
    usrp.set_rx_antenna("RX2", 0)
    
    time.sleep(1.0)
    
    t = threading.Thread(target=tx_daemon, args=(usrp,))
    t.daemon = True
    t.start()
    
    try:
        rx_thread(usrp)
    except KeyboardInterrupt:
        pass