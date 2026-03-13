"""
Record SO-101 joint positions during teleoperation.

Connects to the SO-101 leader arm via its Feetech STS3215 serial bus
and continuously reads all 6 joint positions, saving them to a CSV file.

Usage:
    python collect_so101.py
    python collect_so101.py --port /dev/ttyUSB0 --output my_recording.csv --hz 30
"""

import argparse
import csv
import signal
import time
from pathlib import Path

from lerobot.motors.feetech import FeetechMotorsBus


# SO-101 motor IDs and models (Feetech STS3215 bus servos)
# Motor IDs match the default lerobot SO-101 leader arm configuration.
SO101_MOTORS = {
    "shoulder_pan":  (1, "sts3215"),
    "shoulder_lift": (2, "sts3215"),
    "elbow_flex":    (3, "sts3215"),
    "wrist_flex":    (4, "sts3215"),
    "wrist_roll":    (5, "sts3215"),
    "gripper":       (6, "sts3215"),
}

JOINT_NAMES = list(SO101_MOTORS.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="Record SO-101 joint positions to CSV")
    parser.add_argument("--port",   type=str,   default="/dev/ttyUSB0",
                        help="Serial port the SO-101 is connected to")
    parser.add_argument("--output", type=Path,  default=Path("recorded_positions.csv"),
                        help="Output CSV file path")
    parser.add_argument("--hz",     type=float, default=30.0,
                        help="Recording frequency in Hz (default: 30)")
    return parser.parse_args()


def main():
    args = parse_args()

    bus = FeetechMotorsBus(
        port=args.port,
        motors={name: list(cfg) for name, cfg in SO101_MOTORS.items()},
    )

    running = True

    def handle_stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    bus.connect()
    print(f"Connected to SO-101 on {args.port}")
    print(f"Recording {len(JOINT_NAMES)} joints at {args.hz} Hz")
    print(f"Output: {args.output}")
    print("Press Ctrl+C to stop.\n")

    period = 1.0 / args.hz
    frame_count = 0
    log_every = max(1, int(args.hz) * 5)  # print status every 5 seconds

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + JOINT_NAMES)

        try:
            while running:
                t_start = time.perf_counter()

                positions = bus.read("Present_Position")
                timestamp = time.time()

                # positions is a numpy array ordered by SO101_MOTORS dict order
                row = [timestamp] + [float(positions[i]) for i in range(len(JOINT_NAMES))]
                writer.writerow(row)
                frame_count += 1

                if frame_count % log_every == 0:
                    pos_str = "  ".join(
                        f"{name}: {row[i+1]:.0f}"
                        for i, name in enumerate(JOINT_NAMES)
                    )
                    print(f"Frame {frame_count:6d} | {pos_str}")

                elapsed = time.perf_counter() - t_start
                sleep_time = period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            bus.disconnect()

    print(f"\nRecording stopped. Saved {frame_count} frames to {args.output}")


if __name__ == "__main__":
    main()
