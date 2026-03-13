"""
Control RoArm V3 from SO-101 recorded positions.

Reads the CSV produced by collect_so101.py and replays the joint positions
on the RoArm V3 via its Waveshare serial JSON API.

RoArm V3 serial protocol (Waveshare ESP32 controller, 115200 baud):
  Move single joint:  {"T":106, "joint":<1-5>, "angle":<deg>, "spd":<0-1000>, "acc":<0-100>}
  Move all joints:    {"T":122, "base":<deg>, "shoulder":<deg>, "elbow":<deg>,
                        "wrist":<deg>, "hand":<deg>, "spd":<0-1000>, "acc":<0-100>}

Joint mapping (SO-101 → RoArm V3):
  shoulder_pan  → joint 1 (base rotation)
  shoulder_lift → joint 2 (shoulder)
  elbow_flex    → joint 3 (elbow)
  wrist_flex    → joint 4 (wrist)
  wrist_roll    → skipped (RoArm V3 has no wrist roll DOF)
  gripper       → joint 5 (hand/gripper)

SO-101 servo positions are in Feetech STS3215 raw steps (0-4095, centre=2048).
These are converted to degrees before being sent to the RoArm.

Usage:
    python control_roarm.py --input recorded_positions.csv
    python control_roarm.py --input recorded_positions.csv --port /dev/ttyUSB1 --speed 200
"""

import argparse
import csv
import json
import time
from pathlib import Path

import serial


# Feetech STS3215 conversion constants
SO101_STEPS_PER_REV = 4096
SO101_CENTRE        = 2048     # step value at 0 degrees


def so101_steps_to_degrees(steps: float) -> float:
    """Convert raw STS3215 servo steps to signed degrees (0 steps = -180 deg)."""
    return (steps - SO101_CENTRE) / SO101_STEPS_PER_REV * 360.0


# SO-101 joint name → RoArm V3 joint ID (1-indexed)
# wrist_roll is omitted — RoArm V3 has no equivalent DOF.
SO101_TO_ROARM_ID = {
    "shoulder_pan":  1,
    "shoulder_lift": 2,
    "elbow_flex":    3,
    "wrist_flex":    4,
    "gripper":       5,
}

# Per-joint degree offsets to align the two robots' zero positions.
# Tune these values to match your physical setup before running.
ROARM_OFFSETS = {
    1: 0.0,    # base
    2: 0.0,    # shoulder
    3: 0.0,    # elbow
    4: 0.0,    # wrist
    5: 0.0,    # gripper
}

# Per-joint scale factors (1.0 = direct mapping, -1.0 = inverted axis).
ROARM_SCALES = {
    1:  1.0,
    2:  1.0,
    3:  1.0,
    4:  1.0,
    5:  1.0,
}


def send_joint(ser: serial.Serial, joint_id: int, angle_deg: float,
               speed: int, acc: int) -> None:
    cmd = {
        "T":     106,
        "joint": joint_id,
        "angle": round(angle_deg, 2),
        "spd":   speed,
        "acc":   acc,
    }
    ser.write((json.dumps(cmd) + "\n").encode())


def parse_args():
    parser = argparse.ArgumentParser(description="Replay SO-101 positions on RoArm V3")
    parser.add_argument("--input",  type=Path, required=True,
                        help="CSV file produced by collect_so101.py")
    parser.add_argument("--port",   type=str,  default="/dev/ttyUSB1",
                        help="Serial port the RoArm V3 is connected to")
    parser.add_argument("--baud",   type=int,  default=115200,
                        help="Baud rate (default: 115200)")
    parser.add_argument("--hz",     type=float, default=30.0,
                        help="Playback frequency in Hz (default: 30, matches recording)")
    parser.add_argument("--speed",  type=int,  default=150,
                        help="RoArm joint speed 0-1000 (default: 150 — start slow)")
    parser.add_argument("--acc",    type=int,  default=10,
                        help="RoArm joint acceleration 0-100 (default: 10)")
    parser.add_argument("--delay",  type=float, default=2.0,
                        help="Seconds to wait before starting playback (default: 2)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load recording
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"ERROR: {args.input} is empty.")
        return

    print(f"Loaded {len(rows)} frames from {args.input}")

    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        print(f"Connected to RoArm V3 on {args.port} at {args.baud} baud")
        print(f"Speed: {args.speed}  Acc: {args.acc}  Hz: {args.hz}")
        print(f"Starting in {args.delay:.0f} seconds...")
        time.sleep(args.delay)
        print("Playback started.\n")

        period = 1.0 / args.hz
        log_every = max(1, int(args.hz) * 5)

        for i, row in enumerate(rows):
            t_start = time.perf_counter()

            for so101_joint, roarm_id in SO101_TO_ROARM_ID.items():
                if so101_joint not in row:
                    continue

                raw_steps = float(row[so101_joint])
                angle_deg = so101_steps_to_degrees(raw_steps)
                angle_deg = angle_deg * ROARM_SCALES[roarm_id] + ROARM_OFFSETS[roarm_id]

                send_joint(ser, roarm_id, angle_deg, speed=args.speed, acc=args.acc)

            if i % log_every == 0:
                print(f"  Frame {i+1:5d}/{len(rows)}")

            elapsed = time.perf_counter() - t_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\nPlayback complete.")


if __name__ == "__main__":
    main()
