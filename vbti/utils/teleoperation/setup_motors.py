"""
Configure Feetech STS3215 motor IDs — replaces 'lerobot-setup-motors'.

Brand-new STS3215 motors ship with ID=1.  This script assigns each motor
a unique ID (1-6) one at a time, exactly as the LeRobot tutorial describes.

Run this ONCE per arm before calibration or teleoperation.

Usage:
    # Leader arm on COM7  (IDs 1–6)
    python setup_motors.py --port COM7 --arm leader

    # Follower arm on COM8  (IDs 1–6)
    python setup_motors.py --port COM8 --arm follower

Step-by-step process (prompted interactively):
    1. Connect ONLY the first motor to the controller board.
    2. The script pings ID=1 (factory default) and reassigns it.
    3. Disconnect that motor, connect the next one, press Enter.
    4. Repeat for all 6 joints.
"""

import argparse
import sys
import time

JOINT_ORDER = ["gripper", "wrist_roll", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan"]
TARGET_IDS  = [6, 5, 4, 3, 2, 1]   # assigned in reverse so pan=1, gripper=6

# STS3215 factory baud rate and the target baud rate LeRobot expects
FACTORY_BAUD = 1_000_000
TARGET_BAUD  = 1_000_000   # same — LeRobot also uses 1 Mbaud for STS3215

# SCS control table addresses
ADDR_ID       = 5
ADDR_BAUD     = 6
ADDR_TORQUE   = 40


def open_bus(port: str, baud: int):
    try:
        import scservo_sdk as scs
    except ImportError:
        print("ERROR: scservo_sdk not installed.  Run: pip install scservo-sdk")
        sys.exit(1)

    ph = scs.PortHandler(port)
    packet_handler = scs.PacketHandler(0)

    if not ph.openPort():
        print(f"ERROR: Cannot open {port}.  Is the cable connected and the port correct?")
        sys.exit(1)
    if not ph.setBaudRate(baud):
        print(f"ERROR: Cannot set baud rate {baud}.")
        ph.closePort()
        sys.exit(1)

    return ph, packet_handler, scs


def ping_id(ph, pkth, scs, motor_id: int) -> bool:
    _, comm_result, _ = pkth.ping(ph, motor_id)
    return comm_result == scs.COMM_SUCCESS


def write_byte(ph, pkth, scs, motor_id: int, address: int, value: int) -> bool:
    comm_result, _ = pkth.write1ByteTxRx(ph, motor_id, address, value)
    return comm_result == scs.COMM_SUCCESS


def set_motor_id(ph, pkth, scs, current_id: int, new_id: int) -> bool:
    """Disable torque, write new ID, verify by pinging new ID."""
    # Disable torque first (required before writing EEPROM)
    write_byte(ph, pkth, scs, current_id, ADDR_TORQUE, 0)
    time.sleep(0.05)

    # Write new ID
    ok = write_byte(ph, pkth, scs, current_id, ADDR_ID, new_id)
    if not ok:
        return False

    time.sleep(0.1)  # allow EEPROM write

    # Verify
    return ping_id(ph, pkth, scs, new_id)


def parse_args():
    p = argparse.ArgumentParser(description="Set STS3215 motor IDs for SO-101 arms")
    p.add_argument("--port", required=True,
                   help="Serial port (e.g. COM7 or COM8)")
    p.add_argument("--arm", choices=["leader", "follower"], required=True,
                   help="Which arm to configure")
    p.add_argument("--start_id", type=int, default=1,
                   help="Factory ID on each new motor (default: 1)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\nSO-101 {args.arm} arm motor setup on {args.port}")
    print("━" * 55)
    print("IMPORTANT: Connect ONLY ONE motor at a time to the controller board.")
    print("Leave the 3-pin cable connected to the motor chain, but only plug")
    print("ONE motor into the controller board at each step.\n")

    ph, pkth, scs = open_bus(args.port, FACTORY_BAUD)

    try:
        for joint_name, target_id in zip(JOINT_ORDER, TARGET_IDS):
            print(f"\n── Joint: {joint_name!r:20s}  →  ID {target_id} ──")
            input(f"   Connect ONLY the '{joint_name}' motor to the board, then press Enter…")

            # Check if the motor is already at target_id (previously configured)
            if ping_id(ph, pkth, scs, target_id):
                print(f"   Motor already at ID {target_id} — skipping.")
                continue

            # Find motor at factory default ID
            if not ping_id(ph, pkth, scs, args.start_id):
                print(f"   ERROR: No motor found at ID {args.start_id}.")
                print(f"   Check that ONLY this motor is connected and it is powered.")
                retry = input("   Retry? (y/n): ").strip().lower()
                if retry != "y":
                    continue
                if not ping_id(ph, pkth, scs, args.start_id):
                    print("   Skipping this motor.")
                    continue

            # Reassign ID
            print(f"   Found motor at ID {args.start_id} — assigning ID {target_id}…")
            if set_motor_id(ph, pkth, scs, args.start_id, target_id):
                print(f"   ✓  '{joint_name}' motor ID set to {target_id}")
            else:
                print(f"   ✗  Failed to set ID {target_id} for '{joint_name}'")

    finally:
        ph.closePort()

    print("\n━" * 55)
    print("Setup complete.  You can now chain all motors and run:")
    print(f"  python scan_motors.py --ports {args.port}")
    print("to verify all 6 IDs (1-6) are found before calibrating.\n")


if __name__ == "__main__":
    main()
