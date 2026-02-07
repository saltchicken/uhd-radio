import uhd
import numpy as np
import threading
import time
from usrp_driver import B210UnifiedDriver
import sdr_utils

FREQ = 5.8e9
RATE = 20e6
GAIN = 60            
CHIRP_LEN = 256     
GAP_LEN = 2000      
THRESHOLD = 0.05    


sig_handler = sdr_utils.SignalHandler()


PROBE_TX = sdr_utils.generate_chirp_probe(CHIRP_LEN)



def process_rx_packet(rx_chunk):
    correlation = np.correlate(rx_chunk, PROBE_TX, mode='valid')
    mag = np.abs(correlation)
    peak_idx = np.argmax(mag)
    peak_val = mag[peak_idx]
    noise_floor = np.mean(mag[:max(0, peak_idx-20)]) if peak_idx > 20 else 0.0001
    snr_db = 10 * np.log10(peak_val / (noise_floor + 1e-9))
    
    if snr_db > 10:
        start = max(0, peak_idx - 10)
        end = min(len(correlation), peak_idx + 50)
        cir_window = correlation[start:end]
        

        metrics = sdr_utils.calculate_csi_metrics(cir_window, RATE)
        metrics['snr_db'] = snr_db
        metrics['peak_val'] = peak_val
        return metrics
        
    return None



def tx_daemon(usrp, driver): 
    print("   [TX] Sounding Daemon Active.")
    tx_streamer = driver.get_tx_streamer()
    padding = np.zeros(GAP_LEN, dtype=np.complex64)
    frame = np.concatenate([padding, PROBE_TX, padding])
    md = uhd.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst = True
    

    while sig_handler.running:
        try:
            md.has_time_spec = False
            tx_streamer.send(frame.reshape(1, -1), md)
            time.sleep(0.5) 
        except Exception:
            pass

def rx_analysis_loop(usrp, driver): 

    print(f"   [RX] CSI Analysis Active ({driver.MODE_NAME}).")
    rx_streamer = driver.get_rx_streamer()
    
    buff_len = 10000 
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
            if np.max(np.abs(data)) > THRESHOLD:
                result = process_rx_packet(data)
                
                if result:
                    print("-" * 50)
                    print(f"‼️ CSI CAPTURE | SNR: {result['snr_db']:.1f} dB")
                    print(f"   RMS Delay Spread:     {result['rms_delay_us']:.3f} us")
                    print(f"   Coherence Bandwidth: {result['coherence_bw_khz']:.1f} kHz")

                    print(f"   CIR (Time):  [{sdr_utils.ascii_bar_chart(result['pdp'])}]")
                    print(f"   CFR (Freq):  [{sdr_utils.ascii_bar_chart(result['cfr_db'])}]")


    stop_cmd = uhd.types.StreamCMD(driver.STREAM_MODE_STOP)
    rx_streamer.issue_stream_cmd(stop_cmd)

if __name__ == "__main__":
    print("--> Initializing CSI Analyzer...")
    driver = B210UnifiedDriver(FREQ, RATE, GAIN)
    usrp = driver.initialize()
    
    t_tx = threading.Thread(target=tx_daemon, args=(usrp, driver))
    t_tx.daemon = True
    t_tx.start()
    
    try:
        rx_analysis_loop(usrp, driver)
    except KeyboardInterrupt:
        pass