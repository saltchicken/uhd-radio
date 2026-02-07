import uhd
import numpy as np
import signal
import sys
import time

# ==========================================

# ==========================================
FREQ = 433.92e6
RATE = 1e6
GAIN = 60

# For 915MHz, Wavelength (lambda) is ~0.327m.
# Standard spacing is lambda/2 (~0.16m).
ANTENNA_SPACING_METERS = 0.163
CALIBRATION_PHASE_OFFSET = 0.0
SQUELCH = 0.005

RUNNING = True

STREAM_MODE_START = uhd.types.StreamMode.start_cont
STREAM_MODE_STOP = uhd.types.StreamMode.stop_cont
MODE_NAME = "Native Continuous"

def handler(signum, frame):
    global RUNNING
    print("\n--> Signal caught. Shutting down...")
    RUNNING = False
signal.signal(signal.SIGINT, handler)

# ==========================================
# HARDWARE SETUP
# ==========================================

def setup_mimo_usrp():
    """
    ‼️ Configures the USRP for 2-channel coherent reception.
    """
    print("--> Scanning for B210/MIMO Device...")
    try:
        usrp = uhd.usrp.MultiUSRP("type=b200")
    except RuntimeError:
        print("‼️ No USRP found. Ensure a B210 (or MIMO capable device) is connected.")
        sys.exit(1)

    if usrp.get_rx_num_channels() < 2:
        print(f"‼️ Error: Device only has {usrp.get_rx_num_channels()} channels. MIMO requires 2.")
        sys.exit(1)

    print(f"--> Configuring {usrp.get_rx_num_channels()} channels for MIMO...")


    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:A A:B"))

    for ch in [0, 1]:
        usrp.set_rx_rate(RATE, ch)
        treq = uhd.types.TuneRequest(FREQ)
        treq.args = uhd.types.DeviceAddr("mode_n=integer") 
        usrp.set_rx_freq(treq, ch)
        usrp.set_rx_gain(GAIN, ch)
        usrp.set_rx_antenna("RX2", ch) 

    usrp.set_time_now(uhd.types.TimeSpec(0.0))
    print("--> Waiting for LO Lock...")
    time.sleep(1.0) 
    
    return usrp

# ==========================================
# MATH & LOGIC
# ==========================================

def calculate_aoa(ch0, ch1):
    correlation_vector = ch1 * np.conj(ch0)
    avg_correlation = np.mean(correlation_vector)
    raw_phase = np.angle(avg_correlation)

    phase_diff = (raw_phase - CALIBRATION_PHASE_OFFSET + np.pi) % (2 * np.pi) - np.pi
    
    wavelength = 3e8 / FREQ
    arg = (phase_diff * wavelength) / (2 * np.pi * ANTENNA_SPACING_METERS)
    arg = max(-1.0, min(1.0, arg))

    theta_rad = np.arcsin(arg)
    theta_deg = np.degrees(theta_rad)
    
    return theta_deg, phase_diff

def ascii_compass(angle_deg):
    width = 50
    center_idx = width // 2
    norm = (angle_deg + 90) / 180
    pos = int(norm * (width - 1))
    
    chars = ['-'] * width
    chars[center_idx] = '|' 
    pos = max(0, min(width-1, pos))
    chars[pos] = 'O'        
    return "".join(chars)

# ==========================================
# MAIN LOOP
# ==========================================

def run_mimo_loop(usrp):
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0, 1] 
    streamer = usrp.get_rx_stream(st_args)

    buff_len = 4096 
    recv_buffer = np.zeros((2, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()

    stream_cmd = uhd.types.StreamCMD(STREAM_MODE_START)
    

    # To emulate continuous streaming, we must calculate exactly how long the buffer takes
    # and schedule the next chunk to start exactly when the current one ends.
    # Duration = Samples / Rate
    burst_duration = buff_len / RATE
    
    # Start schedule 0.05s in the future
    next_time = usrp.get_time_now() + uhd.types.TimeSpec(0.05)
    
    stream_cmd.stream_now = False
    stream_cmd.time_spec = next_time
    
    if "Continuous" not in MODE_NAME:
        stream_cmd.num_samps = buff_len

    # Initial Command
    streamer.issue_stream_cmd(stream_cmd)

    print(f"\n--> MIMO Stream Active on {FREQ/1e6} MHz ({MODE_NAME})")
    print(f"--> Antenna Spacing: {ANTENNA_SPACING_METERS*100:.1f} cm")
    print("--> Waiting for signal threshold...\n")

    last_print = 0

    while RUNNING:

        if MODE_NAME == "Pipelined Continuous (Emulated)":
            next_time += uhd.types.TimeSpec(burst_duration)
            stream_cmd.time_spec = next_time
            streamer.issue_stream_cmd(stream_cmd)

        # Receive with a timeout slightly longer than the burst duration
        samps = streamer.recv(recv_buffer, metadata, burst_duration + 0.1)

        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            if metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                # If we overflow, we just skip a frame, but the pipeline keeps moving
                continue
            # print(f"Metadata error: {metadata.error_code}")
            continue

        if samps > 0:
            ch0_data = recv_buffer[0][:samps]
            ch1_data = recv_buffer[1][:samps]

            power = np.mean(np.abs(ch0_data)**2)
            
            if power > SQUELCH: 
                angle, phase = calculate_aoa(ch0_data, ch1_data)
                
                if time.time() - last_print > 0.1:
                    compass = ascii_compass(angle)
                    sys.stdout.write(f"\r[MIMO DF] AoA: {angle:6.1f}° | Phase: {phase:5.2f} rad | [{compass}]")
                    sys.stdout.flush()
                    last_print = time.time()
            else:
                 if time.time() - last_print > 0.5:
                    sys.stdout.write(f"\r[MIMO DF] ... Listening (Low Signal: {power:.5f}) ... {' ' * 30}")
                    sys.stdout.flush()
                    last_print = time.time()

    print("\n--> Stopping Stream...")
    streamer.issue_stream_cmd(uhd.types.StreamCMD(STREAM_MODE_STOP))

if __name__ == "__main__":
    u = setup_mimo_usrp()
    run_mimo_loop(u)
