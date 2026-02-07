import uhd
import numpy as np
import time
import threading

from usrp_driver import B210UnifiedDriver, PeriodicTransmitter
import sdr_utils


args = sdr_utils.get_standard_args("DBPSK Data Modem", default_gain=50)

SPS = 100
MESSAGE = "Hello World" 

sig_handler = sdr_utils.SignalHandler()

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
    full_payload = chr(len(text)) + text
    bits = text_to_bits(full_payload)
    samples = []
    current_phase = 0.0

    # Sync sequence
    for _ in range(50):
        symbol = np.exp(1j * current_phase)
        samples.append(symbol)

    # Start delimiter
    current_phase += np.pi
    samples.append(np.exp(1j * current_phase))
        
    for bit in bits:
        if bit == 1:
            current_phase += np.pi 
        symbol = np.exp(1j * current_phase)
        samples.append(symbol)
        
    symbols = np.array(samples, dtype=np.complex64)
    tx_signal = np.repeat(symbols, SPS)
    return tx_signal * 0.7

def demodulate_dbpsk_robust(rx_chunk):
    best_text = ""
    best_score = -1

    for offset in range(0, SPS, 10):
        raw_symbols = rx_chunk[offset::SPS]
        if len(raw_symbols) < 60: continue

        diffs = raw_symbols[1:] * np.conj(raw_symbols[:-1])
        phase_diffs = np.angle(diffs)
        detected_bits = (np.abs(phase_diffs) > (np.pi / 2)).astype(int)
        
        preamble_zeros = np.sum(detected_bits[:48] == 0)
        if preamble_zeros < 40: continue

        try:
            sync_window = detected_bits[45:60]
            sync_rel_idx = np.where(sync_window == 1)[0][0]
            start_of_data = 45 + sync_rel_idx + 1
        except IndexError:
            continue
            
        len_bits = detected_bits[start_of_data : start_of_data + 8]
        if len(len_bits) < 8: continue
        
        msg_len = 0
        for bit in len_bits:
            msg_len = (msg_len << 1) | bit
            
        if msg_len == 0 or msg_len > 100: continue

        payload_start = start_of_data + 8
        payload_end = payload_start + (msg_len * 8)
        
        if len(detected_bits) < payload_end: continue
            
        payload_bits = detected_bits[payload_start : payload_end]
        text = bits_to_text(payload_bits)
        clean_text = "".join([c for c in text if 32 <= ord(c) <= 126])
        
        if len(clean_text) > best_score:
            best_score = len(clean_text)
            best_text = clean_text

    return best_text

def rx_thread(usrp, driver):
    print(f"   [RX Main] Listening for Data...")
    rx_streamer = driver.get_rx_streamer()
    
    buff_len = 50000 
    recv_buffer = np.zeros((1, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    
    cmd = uhd.types.StreamCMD(driver.STREAM_MODE_START)
    cmd.stream_now = True
    rx_streamer.issue_stream_cmd(cmd)
    
    while sig_handler.running:
        samps = rx_streamer.recv(recv_buffer, metadata, 0.1)
        
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            if metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                continue
            continue
            
        if samps > 0:
            data = recv_buffer[0][:samps]
            magnitudes = np.abs(data)
            threshold = 0.05 
            
            high_signal_indices = np.where(magnitudes > threshold)[0]
            
            if len(high_signal_indices) > 2000:
                start_idx = high_signal_indices[0]
                if start_idx + 20000 < samps:
                    packet_chunk = data[start_idx : start_idx + 20000]
                    msg = demodulate_dbpsk_robust(packet_chunk)
                    if len(msg) > 0:
                          print(f"   [RX] 📬 RECEIVED: '{msg}'")
                          
    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))

if __name__ == "__main__":
    print("--> Initializing Data Modem...")
    driver = B210UnifiedDriver(args.freq, args.rate, args.gain)
    usrp = driver.initialize()
    

    # Replaces the specific tx_daemon logic which was just a loop anyway
    print(f"   [TX] Modulating '{MESSAGE}'...")
    tx_data = modulate_dbpsk(MESSAGE)
    padding = np.zeros(2000, dtype=np.complex64)
    TX_FRAME = np.concatenate([padding, tx_data, padding])

    tx_thread = PeriodicTransmitter(driver, sig_handler, TX_FRAME, interval=2.0)
    tx_thread.start()
    
    try:
        rx_thread(usrp, driver)
    except KeyboardInterrupt:
        pass