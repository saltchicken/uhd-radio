import uhd
import numpy as np
import sys
import time

def get_device(args="type=b200"):
    """
    Tries to connect to the USRP device.
    """
    print(f"--> Searching for device with args: '{args}'...")
    try:
        # MultiUSRP is the main class for controlling the device
        usrp = uhd.usrp.MultiUSRP(args)
        return usrp
    except RuntimeError as e:
        print(f"‼️ Error: Could not find USRP device. Make sure it is plugged in.")
        print(f"   System details: {e}")
        sys.exit(1)

def print_device_info(usrp):
    """
    Prints the 'Hello World' details of the hardware.
    """
    print("\n--> Device Connection Successful! (Hello World)")
    
    # get_pp_string() returns a 'pretty print' string of the device info
    print(f"‼️ Device Info:\n{usrp.get_pp_string()}")
    
    # Specific queries
    mboard_id = usrp.get_mboard_name(0)
    serial = usrp.get_usrp_rx_info(0).get("mboard_serial")
    print(f"--> Motherboard: {mboard_id}")
    print(f"--> Serial:      {serial}")

def test_receive(usrp, freq=915e6, rate=1e6, gain=20):
    """
    Configures the radio and pulls a small batch of samples 
    to verify the RX chain is active.
    """
    print(f"\n--> Testing RX Chain...")
    
    # 1. Configure Request
    print(f"   Tuning to {freq/1e6} MHz at {rate/1e6} Msps...")
    usrp.set_rx_rate(rate, 0)
    usrp.set_rx_freq(uhd.types.TuneRequest(freq), 0)
    usrp.set_rx_gain(gain, 0)
    

    # This ensures the hardware is stable before streaming
    print("   Waiting for LO lock...")
    max_checks = 10
    for i in range(max_checks):
        sensor_value = usrp.get_rx_sensor("lo_locked", 0).to_bool()
        if sensor_value:
            print("   ✅ LO Locked.")
            break
        time.sleep(0.1)
        if i == max_checks - 1:
            print("   ⚠️ Warning: LO verify failed (this happens on some models/bands), continuing anyway...")

    # 2. Setup Streamer
    # We want complex float output (fc32) for standard SDR processing
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    streamer = usrp.get_rx_stream(st_args)
    
    # 3. Receive Samples
    # Create a buffer to hold incoming samples
    # The B210 usually delivers packets of specific sizes, 2000 is a safe test size
    num_samps = 2000
    recv_buffer = np.zeros((1, num_samps), dtype=np.complex64)
    
    # Issue the stream command
    metadata = uhd.types.RXMetadata()
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_samps
    stream_cmd.stream_now = True
    
    print("   Requesting samples...")
    streamer.issue_stream_cmd(stream_cmd)
    
    # Receive
    samps_count = streamer.recv(recv_buffer, metadata)
    
    print(f"--> Received {samps_count} samples.")
    print(f"   First 5 samples: {recv_buffer[0][:5]}")
    
    # Simple check to see if we got non-zero data (noise floor)
    power = np.mean(np.abs(recv_buffer)**2)
    print(f"   Average Power (Noise Floor): {power:.6f}")

    if samps_count < num_samps:
         print("‼️ Warning: Received fewer samples than requested (Timeout or dropped packets).")

if __name__ == "__main__":
    # Initialize
    my_usrp = get_device("type=b200") # B210 identifies as b200 series
    
    # Run Hello World
    print_device_info(my_usrp)
    
    # Verify Hardware Functionality
    test_receive(my_usrp)