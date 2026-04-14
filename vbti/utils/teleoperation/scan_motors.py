"""
Scan serial ports for Feetech STS3215 motors.

Tries all common baud rates and motor IDs 1–20.
Use this to diagnose what is (or isn't) on the bus before running setup or calibration.

Usage:
    python scan_motors.py                     # scan COM7 and COM8
    python scan_motors.py --ports COM7 COM8
    python scan_motors.py --ports COM7 --baud 1000000
"""

import argparse
import sys

# STS3215 default baud rate is 1 000 000; but also try others just in case
BAUD_RATES = [1_000_000, 115_200, 500_000, 57_600]


def scan_port(port: str, baud_rates: list[int]) -> None:
    try:
        import scservo_sdk as scs
    except ImportError:
        print("ERROR: scservo_sdk not installed.  Run: pip install scservo-sdk")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f" Scanning {port}")
    print(f"{'='*60}")

    for baud in baud_rates:
        port_handler = scs.PortHandler(port)
        packet_handler = scs.PacketHandler(0)  # protocol 0 = Feetech

        if not port_handler.openPort():
            print(f"  [{baud:>9,}] Cannot open port — check cable / device manager")
            break

        if not port_handler.setBaudRate(baud):
            print(f"  [{baud:>9,}] Cannot set baud rate")
            port_handler.closePort()
            continue

        found = []
        for motor_id in range(1, 21):
            model_number, comm_result, _ = packet_handler.ping(port_handler, motor_id)
            if comm_result == scs.COMM_SUCCESS:
                found.append((motor_id, model_number))

        port_handler.closePort()

        if found:
            print(f"  [{baud:>9,}]  FOUND {len(found)} motor(s):")
            for mid, mnum in found:
                print(f"               ID {mid:2d}  model_number={mnum}")
        else:
            print(f"  [{baud:>9,}]  (no response)")


def parse_args():
    p = argparse.ArgumentParser(description="Scan for Feetech motors on serial ports")
    p.add_argument("--ports", nargs="+", default=["COM7", "COM8"],
                   help="Ports to scan (default: COM7 COM8)")
    p.add_argument("--baud", type=int, default=None,
                   help="Scan only this baud rate instead of all common rates")
    return p.parse_args()


def main():
    args = parse_args()
    baud_rates = [args.baud] if args.baud else BAUD_RATES

    print("Feetech motor scanner")
    print(f"Ports     : {args.ports}")
    print(f"Baud rates: {baud_rates}")
    print("Scanning IDs 1–20 on each port…")

    for port in args.ports:
        scan_port(port, baud_rates)

    print("\nDone.")
    print()
    print("Interpretation:")
    print("  No response at any baud → motors not powered, wrong port, or IDs not set yet")
    print("  All IDs = 1 at 1 000 000 → brand-new motors, run setup_motors.py next")
    print("  IDs 1-6 at 1 000 000    → motors configured, run lerobot-calibrate next")


if __name__ == "__main__":
    main()
