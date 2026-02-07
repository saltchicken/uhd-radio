import uhd
import numpy as np
import time
from usrp_driver import B210UnifiedDriver, PeriodicTransmitter
import sdr_utils


args = sdr_utils.get_standard_args("CSI Analyzer", default_freq=5.8e9, default_rate=20e6)
    
CHIRP_LEN = 256      
GAP_LEN = 2000       
THRESHOLD = 0.05    

sig_handler = sdr_utils.SignalHandler()

PROBE_TX = sdr_utils.generate_chirp_probe(CHIRP_LEN)
# Prepare TX Frame
padding = np.zeros(GAP_LEN, dtype=np.complex64)
TX_FRAME = np.concatenate([padding, PROBE_TX, padding])


def process_rx_packet_refactored(rx_chunk):

    res = sdr_utils.correlate_and_detect(rx_chunk, PROBE_TX)
    
    if res['snr_db'] > 10:
        peak_idx = res['peak_idx']
        start = max(0, peak_idx - 10)
        end = min(len(res['correlation']), peak_idx + 50)
        cir_window = res['correlation'][start:end]
        
        metrics = sdr_utils.calculate_csi_metrics(cir_window, args.rate)
        metrics['snr_db'] = res['snr_db']
        metrics['peak_val'] = res['peak_val']
        return metrics
        
    return None

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
             continue

        if samps > 0:
            data = recv_buffer[0][:samps]
            if np.max(np.abs(data)) > THRESHOLD:
                result = process_rx_packet_refactored(data)
                
                if result:
                    print("-" * 50)
                    print(f"‼️ CSI CAPTURE | SNR: {result['snr_db']:.1f} dB")
                    print(f"   RMS Delay Spread:     {result['rms_delay_us']:.3f} us")
                    print(f"   Coherence Bandwidth: {result['coherence_bw_khz']:.1f} kHz")
                    print(f"   CIR (Time):  [{sdr_utils.ascii_bar_chart(result['pdp'])}]")
                    print(f"   CFR (Freq):  [{sdr_utils.ascii_bar_chart(result['cfr_db'])}]")

    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))

if __name__ == "__main__":
    print("--> Initializing CSI Analyzer...")
    driver = B210UnifiedDriver(args.freq, args.rate, args.gain)
    usrp = driver.initialize()
    

    tx_thread = PeriodicTransmitter(driver, sig_handler, TX_FRAME, interval=0.5)
    tx_thread.start()
    
    try:
        rx_analysis_loop(usrp, driver)
    except KeyboardInterrupt:
        pass