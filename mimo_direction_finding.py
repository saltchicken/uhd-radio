import uhd
import numpy as np
import signal
import sys
import time

# ==========================================
# ‼️ MIMO CONFIGURATION
# ==========================================
FREQ = 433.92e6
RATE = 1e6
GAIN = 60
# ‼️ Distance between antenna centers. 
# For 915MHz, Wavelength (lambda) is ~0.327m.
# Standard spacing is lambda/2 (~0.16m).
ANTENNA_SPACING_METERS = 0.163  
CALIBRATION_PHASE_OFFSET = 0.0  # Set this to "tare" the zero point if needed

RUNNING = True

# ‼️ COMPATIBILITY FIX: Handle different UHD versions
try:
    STREAM_MODE_START = uhd.types.StreamMode.start_continuous
    STREAM_MODE_STOP = uhd.types.StreamMode.stop_continuous
    MODE_NAME = "Native Continuous"
except AttributeError:
    # Fallback for older/different UHD bindings
    STREAM_MODE_START = uhd.types.StreamMode.num_done
    STREAM_MODE_STOP = uhd.types.StreamMode.num_done
    MODE_NAME = "Manual Burst (Fallback)"

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

    # ‼️ Verify we have 2 RX channels available
    if usrp.get_rx_num_channels() < 2:
        print(f"‼️ Error: Device only has {usrp.get_rx_num_channels()} channels. MIMO requires 2.")
        sys.exit(1)

    print(f"--> Configuring {usrp.get_rx_num_channels()} channels for MIMO...")

    # ‼️ Subdev Spec "A:A A:B" typically maps:
    # Channel 0 -> RF A, RX1
    # Channel 1 -> RF A, RX2
    # This forces the B210 to treat both inputs as active on the same motherboard.
    usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:A A:B"))

    # Configure parameters for BOTH channels
    for ch in [0, 1]:
        usrp.set_rx_rate(RATE, ch)
        
        # Tuning request
        treq = uhd.types.TuneRequest(FREQ)
        treq.args = uhd.types.DeviceAddr("mode_n=integer") # Force integer-N for better phase noise/coherence
        usrp.set_rx_freq(treq, ch)
        
        usrp.set_rx_gain(GAIN, ch)
        usrp.set_rx_antenna("RX2", ch) # Use RX2 port on both headers

    # ‼️ Sync time to ensure sample alignment
    usrp.set_time_now(uhd.types.TimeSpec(0.0))
    
    # Wait for LOs to lock and settle
    print("--> Waiting for LO Lock...")
    time.sleep(1.0) 
    
    return usrp

# ==========================================
# MATH & LOGIC
# ==========================================

def calculate_aoa(ch0, ch1):
    """
    ‼️ Calculates Angle of Arrival (AoA) based on Phase Difference.
    Assumes a Uniform Linear Array (ULA) configuration.
    """
    # 1. Calculate Phase Difference
    # We multiply Ch1 by the Complex Conjugate of Ch0.
    # The angle of the result is the phase difference (delta phi).
    # Averaging over the sample chunk reduces noise significantly.
    correlation_vector = ch1 * np.conj(ch0)
    avg_correlation = np.mean(correlation_vector)
    
    raw_phase = np.angle(avg_correlation)

    # Apply Calibration Offset
    # (Wraps phase back to -pi ... pi)
    phase_diff = (raw_phase - CALIBRATION_PHASE_OFFSET + np.pi) % (2 * np.pi) - np.pi

    # 2. Convert Phase to Geometric Angle
    # Formula: delta_phi = (2 * pi * d * sin(theta)) / lambda
    # Therefore: theta = arcsin( (delta_phi * lambda) / (2 * pi * d) )
    
    wavelength = 3e8 / FREQ
    
    # Calculate the argument for arcsin
    arg = (phase_diff * wavelength) / (2 * np.pi * ANTENNA_SPACING_METERS)
    
    # ‼️ Clamp argument to [-1, 1] to prevent domain errors if noise pushes it out of bounds
    arg = max(-1.0, min(1.0, arg))

    theta_rad = np.arcsin(arg)
    theta_deg = np.degrees(theta_rad)
    
    return theta_deg, phase_diff

def ascii_compass(angle_deg):
    """
    Renders a simple text-based horizon line.
    """
    # -90 (Left) ... 0 (Center) ... 90 (Right)
    width = 50
    center_idx = width // 2
    
    # Normalize angle (-90 to 90) to 0.0 to 1.0
    norm = (angle_deg + 90) / 180
    pos = int(norm * (width - 1))
    
    # Build string
    chars = ['-'] * width
    chars[center_idx] = '|' # Center marker
    
    # Clamp position just in case
    pos = max(0, min(width-1, pos))
    chars[pos] = 'O'        # Target marker
    
    return "".join(chars)

# ==========================================
# MAIN LOOP
# ==========================================

def run_mimo_loop(usrp):
    # ‼️ Requesting both channels [0, 1]
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0, 1] 
    streamer = usrp.get_rx_stream(st_args)

    buff_len = 2000
    # ‼️ Buffer shape must handle 2 channels: (2, 2000)
    recv_buffer = np.zeros((2, buff_len), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()

    # ‼️ Use compatibility constants
    stream_cmd = uhd.types.StreamCMD(STREAM_MODE_START)
    
    # ‼️ FIX: Multi-channel streams MUST be scheduled in the future to align.
    # "Stream Now" is not allowed for MIMO because it doesn't guarantee sample alignment.
    stream_cmd.stream_now = False
    stream_cmd.time_spec = usrp.get_time_now() + uhd.types.TimeSpec(0.05)

    if MODE_NAME == "Manual Burst (Fallback)":
        stream_cmd.num_samps = buff_len
    
    streamer.issue_stream_cmd(stream_cmd)

    print(f"\n--> MIMO Stream Active on {FREQ/1e6} MHz ({MODE_NAME})")
    print(f"--> Antenna Spacing: {ANTENNA_SPACING_METERS*100:.1f} cm")
    print("--> Waiting for signal threshold...\n")

    last_print = 0

    while RUNNING:
        # ‼️ Re-issue command if in fallback mode
        if MODE_NAME == "Manual Burst (Fallback)":
            # ‼️ Update schedule for next burst to avoid "late command" errors
            stream_cmd.time_spec = usrp.get_time_now() + uhd.types.TimeSpec(0.05)
            streamer.issue_stream_cmd(stream_cmd)

        samps = streamer.recv(recv_buffer, metadata, 0.1)

        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            if metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                continue
            # print(f"Metadata error: {metadata.error_code}")
            continue

        if samps > 0:
            ch0_data = recv_buffer[0][:samps]
            ch1_data = recv_buffer[1][:samps]

            # 1. Detect Signal Power (using Ch0)
            power = np.mean(np.abs(ch0_data)**2)
            
            # Only calculate direction if signal is strong enough
            if power > 0.005: 
                angle, phase = calculate_aoa(ch0_data, ch1_data)
                
                # Rate limit printing to make it readable
                if time.time() - last_print > 0.1:
                    compass = ascii_compass(angle)
                    # \r overwrites the current line
                    sys.stdout.write(f"\r[MIMO DF] AoA: {angle:6.1f}° | Phase: {phase:5.2f} rad | [{compass}]")
                    sys.stdout.flush()
                    last_print = time.time()
            else:
                 if time.time() - last_print > 0.5:
                    sys.stdout.write(f"\r[MIMO DF] ... Listening (Low Signal) ... {' ' * 40}")
                    sys.stdout.flush()
                    last_print = time.time()

    # Cleanup
    print("\n--> Stopping Stream...")
    # ‼️ Use compatibility constant for stop
    streamer.issue_stream_cmd(uhd.types.StreamCMD(STREAM_MODE_STOP))

if __name__ == "__main__":
    u = setup_mimo_usrp()
    run_mimo_loop(u)
