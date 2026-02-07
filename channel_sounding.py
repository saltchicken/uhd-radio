import uhd
import numpy as np

from sdr_lib.usrp_driver import B210UnifiedDriver, PeriodicTransmitter 
from sdr_lib import sdr_utils


args = sdr_utils.get_standard_args("Channel Sounder", default_freq=915e6)

CHIRP_LEN = 256
GAP_LEN = 2000
THRESHOLD = 0.05

sig_handler = sdr_utils.SignalHandler()

PROBE_TX = sdr_utils.generate_chirp_probe(CHIRP_LEN)
# Prepare TX Frame
padding = np.zeros(GAP_LEN, dtype=np.complex64)
TX_FRAME = np.concatenate([padding, PROBE_TX, padding])


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
             continue

        if samps > 0:
            data = recv_buffer[0][:samps]
            if np.max(np.abs(data)) > THRESHOLD:

                res = sdr_utils.correlate_and_detect(data, PROBE_TX)
                
                if res['snr_db'] > 10: 
                    # Extract region for display
                    center = res['peak_idx']
                    start_view = max(0, center - 40)
                    end_view = min(len(res['mag']), center + 40)
                    view_data = res['mag'][start_view:end_view]
                    
                    graph = sdr_utils.ascii_sparkline(view_data)
                    print(f"   [CIS] SNR: {res['snr_db']:.1f}dB | Peak: {res['peak_val']:.3f}")
                    print(f"         Impulse: [{graph}]")

    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(driver.STREAM_MODE_STOP))

if __name__ == "__main__":
    print("--> Initializing Channel Sounder...")

    driver = B210UnifiedDriver(args.freq, args.rate, args.gain)
    usrp = driver.initialize()
    

    tx_thread = PeriodicTransmitter(driver, sig_handler, TX_FRAME, interval=1.0)
    tx_thread.start()
    
    try:
        rx_analysis_loop(usrp, driver)
    except KeyboardInterrupt:
        pass
