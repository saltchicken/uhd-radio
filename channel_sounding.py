import uhd
import numpy as np
import threading
import time
from usrp_driver import B210UnifiedDriver 
import sdr_utils

FREQ = 915e6
RATE = 1e6
GAIN = 60
CHIRP_LEN = 256
GAP_LEN = 2000
THRESHOLD = 0.05


sig_handler = sdr_utils.SignalHandler()



PROBE_TX = sdr_utils.generate_chirp_probe(CHIRP_LEN)
PROBE_REF = np.conj(PROBE_TX[::-1]) 

def analyze_channel_response(rx_chunk, sample_rate=RATE):
    correlation = np.correlate(rx_chunk, PROBE_TX, mode='valid')
    mag = np.abs(correlation) / CHIRP_LEN
    peak_idx = np.argmax(mag)
    peak_val = mag[peak_idx]
    
    noise_region = np.concatenate([mag[:max(0, peak_idx-20)], mag[min(len(mag), peak_idx+20):]])
    if len(noise_region) > 0:
        noise_floor = np.mean(noise_region)
    else:
        noise_floor = 0.0001
        
    snr_linear = peak_val / (noise_floor + 1e-9)
    snr_db = 10 * np.log10(snr_linear)
    
    return {
        "peak_val": peak_val,
        "peak_idx": peak_idx,
        "snr_db": snr_db,
        "noise_floor": noise_floor,
        "profile": mag
    }



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
            time.sleep(1.0) 
        except Exception:
            pass

def rx_analysis_loop(usrp, driver):

    print(f"   [RX] CIS Analysis Loop Active ({driver.MODE_NAME}).")
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
                results = analyze_channel_response(data)
                if results['snr_db'] > 10: 
                    center = results['peak_idx']
                    start_view = max(0, center - 40)
                    end_view = min(len(results['profile']), center + 40)
                    view_data = results['profile'][start_view:end_view]
                    

                    graph = sdr_utils.ascii_sparkline(view_data)
                    print(f"   [CIS] SNR: {results['snr_db']:.1f}dB | Peak: {results['peak_val']:.3f}")
                    print(f"          Impulse: [{graph}]")


    stop_cmd = uhd.types.StreamCMD(driver.STREAM_MODE_STOP)
    rx_streamer.issue_stream_cmd(stop_cmd)

if __name__ == "__main__":
    print("--> Initializing Channel Sounder...")
    driver = B210UnifiedDriver(FREQ, RATE, GAIN)
    usrp = driver.initialize()
    
    t_tx = threading.Thread(target=tx_daemon, args=(usrp, driver))
    t_tx.daemon = True
    t_tx.start()
    
    try:
        rx_analysis_loop(usrp, driver)
    except KeyboardInterrupt:
        pass