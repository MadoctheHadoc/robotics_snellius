"""
Live teleoperation: SO-101 leader → RoArm V3 follower.

Reads SO-101 joint positions at ~30 Hz and immediately sends them to the
RoArm V3 via its Waveshare serial JSON API. Optionally saves all frames to
a CSV file for later use as training data.

Modes:
    Live teleoperation (default):
        python teleop_live.py

    Live teleoperation + record to CSV:
        python teleop_live.py --save my_episode.csv

    Replay a CSV to RoArm only (no SO-101 needed):
        python teleop_live.py --input my_episode.csv

    Dry run — read SO-101 only, do not move RoArm:
        python teleop_live.py --dry_run
"""

import argparse
import csv
import json
import signal
import time
from pathlib import Path

import serial
from lerobot.motors.feetech import FeetechMotorsBus


# ── SO-101 motor configuration ────────────────────────────────────────────────
SO101_MOTORS = {
    "shoulder_pan":  (1, "sts3215"),
    "shoulder_lift": (2, "sts3215"),
    "elbow_flex":    (3, "sts3215"),
    "wrist_flex":    (4, "sts3215"),
    "wrist_roll":    (5, "sts3215"),
    "gripper":       (6, "sts3215"),
}
JOINT_NAMES = list(SO101_MOTORS.keys())

# Feetech STS3215: 4096 steps/rev, centre at 2048 = 0 degrees
SO101_CENTRE = 2048
SO101_STEPS_PER_DEG = 4096 / 360.0


# ── Joint mapping: SO-101 → RoArm V3 ─────────────────────────────────────────
# wrist_roll is omitted — RoArm V3 has no wrist-roll DOF.
SO101_TO_ROARM_ID = {
    "shoulder_pan":  1,
    "shoulder_lift": 2,
    "elbow_flex":    3,
    "wrist_flex":    4,
    "gripper":       5,
}

# Tune these to align the two robots' zero positions before running.
# ROARM_OFFSETS: added to the converted degree value (shifts zero position).
# ROARM_SCALES:  set to -1.0 to invert an axis.
ROARM_OFFSETS = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
ROARM_SCALES  = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}


# ── Helpers ───────────────────────────────────────────────────────────────────

def so101_steps_to_degrees(steps: float) -> float:
    return (steps - SO101_CENTRE) / SO101_STEPS_PER_DEG


def send_joint(ser: serial.Serial, joint_id: int, angle_deg: float,
               speed: int, acc: int) -> None:
    cmd = {"T": 106, "joint": joint_id,
           "angle": round(angle_deg, 2), "spd": speed, "acc": acc}
    ser.write((json.dumps(cmd) + "\n").encode())


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Live SO-101 → RoArm V3 teleoperation")
    p.add_argument("--so101_port",  default="/dev/ttyUSB0",
                   help="Serial port for the SO-101 leader arm")
    p.add_argument("--roarm_port",  default="/dev/ttyUSB1",
                   help="Serial port for the RoArm V3 follower")
    p.add_argument("--roarm_baud",  type=int, default=115200,
                   help="RoArm V3 baud rate (default: 115200)")
    p.add_argument("--hz",          type=float, default=30.0,
                   help="Control loop frequency in Hz (default: 30)")
    p.add_argument("--speed",       type=int,   default=150,
                   help="RoArm joint speed 0-1000 (default: 150 — start slow)")
    p.add_argument("--acc",         type=int,   default=10,
                   help="RoArm joint acceleration 0-100 (default: 10)")
    p.add_argument("--save",        type=Path,  default=None,
                   help="Optional CSV path to record all frames as training data")
    p.add_argument("--input",       type=Path,  default=None,
                   help="Replay a recorded CSV to the RoArm only — SO-101 not needed")
    p.add_argument("--dry_run",     action="store_true",
                   help="Read SO-101 only — do not send commands to RoArm V3")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    roarm_only = args.input is not None

    # ── RoArm-only replay mode ────────────────────────────────────────────────
    if roarm_only:
        with open(args.input, newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"Input     : {args.input}  ({len(rows)} frames)")

        with serial.Serial(args.roarm_port, args.roarm_baud, timeout=0.05) as ser:
            print(f"RoArm V3  : {args.roarm_port} @ {args.roarm_baud} baud")
            print(f"Speed     : {args.speed}  Acc: {args.acc}  Hz: {args.hz}")
            print("Starting in 2 seconds...\n")
            time.sleep(2.0)

            period = 1.0 / args.hz
            log_every = max(1, int(args.hz) * 5)

            for i, row in enumerate(rows):
                t_start = time.perf_counter()

                for so101_joint, roarm_id in SO101_TO_ROARM_ID.items():
                    if so101_joint not in row:
                        continue
                    raw = float(row[so101_joint])
                    angle = so101_steps_to_degrees(raw)
                    angle = angle * ROARM_SCALES[roarm_id] + ROARM_OFFSETS[roarm_id]
                    send_joint(ser, roarm_id, angle, speed=args.speed, acc=args.acc)

                if i % log_every == 0:
                    print(f"  Frame {i+1:5d}/{len(rows)}")

                elapsed = time.perf_counter() - t_start
                sleep_time = period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        print("Playback complete.")
        return

    # ── Live teleoperation mode ───────────────────────────────────────────────
    bus = FeetechMotorsBus(
        port=args.so101_port,
        motors={name: list(cfg) for name, cfg in SO101_MOTORS.items()},
    )

    roarm_ser = None
    if not args.dry_run:
        roarm_ser = serial.Serial(args.roarm_port, args.roarm_baud, timeout=0.05)
        print(f"RoArm V3  : {args.roarm_port} @ {args.roarm_baud} baud")

    bus.connect()
    print(f"SO-101    : {args.so101_port}")
    print(f"Loop rate : {args.hz} Hz")
    if args.dry_run:
        print("DRY RUN — RoArm V3 will NOT move")
    if args.save:
        print(f"Recording : {args.save}")
    print("\nPress Ctrl+C to stop.\n")

    running = True

    def handle_stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    period = 1.0 / args.hz
    log_every = max(1, int(args.hz) * 5)
    frame_count = 0
    late_count = 0

    csv_file = None
    csv_writer = None
    if args.save:
        csv_file = open(args.save, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp"] + JOINT_NAMES)

    try:
        while running:
            t_start = time.perf_counter()

            positions = bus.read("Present_Position")
            timestamp = time.time()

            if roarm_ser is not None:
                for so101_joint, roarm_id in SO101_TO_ROARM_ID.items():
                    idx = JOINT_NAMES.index(so101_joint)
                    raw = float(positions[idx])
                    angle = so101_steps_to_degrees(raw)
                    angle = angle * ROARM_SCALES[roarm_id] + ROARM_OFFSETS[roarm_id]
                    send_joint(roarm_ser, roarm_id, angle, speed=args.speed, acc=args.acc)

            if csv_writer is not None:
                row = [timestamp] + [float(positions[i]) for i in range(len(JOINT_NAMES))]
                csv_writer.writerow(row)

            frame_count += 1

            elapsed = time.perf_counter() - t_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                late_count += 1

            if frame_count % log_every == 0:
                pos_str = "  ".join(
                    f"{n[:3]}: {float(positions[i]):.0f}"
                    for i, n in enumerate(JOINT_NAMES)
                )
                print(f"[{frame_count:6d}] {pos_str}  (late: {late_count})")

    finally:
        bus.disconnect()
        if roarm_ser is not None:
            roarm_ser.close()
        if csv_file is not None:
            csv_file.close()

    print(f"\nStopped. {frame_count} frames @ {args.hz} Hz  |  {late_count} late frames")
    if args.save and frame_count > 0:
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
