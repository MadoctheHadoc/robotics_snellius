"""
Follows this HuggingFace LeRobot SO-101 tutorial:
  https://huggingface.co/docs/lerobot/so101

And later also this tutorial on SmolVLA:
  https://huggingface.co/docs/lerobot/en/smolvla

Usage
    # Basic teleoperation (leader COM7 -> follower COM8):
    python so101_teleop.py

    # Record to CSV while teleoperating:
    python so101_teleop.py --save episode_001.csv

    # Record a LeRobot-format trainable episode with one camera:
    python so101_teleop.py --episode_dir saved_episodes/my_dataset --task "Pick up the block" --cameras 0

    # Record with multiple cameras (indices 0 and 2, named front and wrist):
    python so101_teleop.py --episode_dir saved_episodes/my_dataset --task "Pick up the block" --cameras 0 2 --camera_names front wrist

    # Use this parameter
    # (for some reason it's offset by 90 in the original thing for reasons I don't understand,possible it should be recalibrated)
    --wrist_roll_offset 90
"""

import argparse
import csv
import signal
import time
from pathlib import Path

import numpy as np

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode


# Motor Config
# RANGE_M100_100 is used here but normalization is disabled (normalize=False),
# so raw step values (0-4095) are passed directly - no calibration required.

SO101_MOTORS: dict[str, Motor] = {
    "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex":    Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex":    Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll":    Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_M100_100),
}
JOINT_NAMES = list(SO101_MOTORS.keys())

# Conversion: 4096 steps / 360 degrees → 1 degree = 11ish steps
STEPS_PER_DEG = 4096 / 360


def steps_to_deg(positions: dict) -> np.ndarray:
    """Convert raw motor steps to degrees for each joint."""
    return np.array([positions[n] / STEPS_PER_DEG for n in JOINT_NAMES], dtype=np.float32)


# CLI
def parse_args():
    p = argparse.ArgumentParser(
        description="Live SO-101 leader -> SO-101 follower teleoperation (LeRobot tutorial)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--leader_port",   default="COM7",
                   help="Serial port of the SO-101 leader arm   (default: COM7)")
    p.add_argument("--follower_port", default="COM8",
                   help="Serial port of the SO-101 follower arm (default: COM8)")
    p.add_argument("--hz",            type=float, default=30.0,
                   help="Control loop frequency in Hz (default: 30)")
    p.add_argument("--save",          type=Path,  default=None,
                   help="Optional CSV path to record frames (for training data)")
    p.add_argument("--dry_run",       action="store_true",
                   help="Read leader only — do NOT send commands to follower")
    p.add_argument("--wrist_roll_offset", type=float, default=0.0,
                   help="Offset in degrees added to wrist_roll on the follower "
                        "(e.g. 90 or -90 to correct a 90 degrees mounting difference)")

    # LeRobot episode recording
    p.add_argument("--episode_dir",   type=Path,  default=None,
                   help="Directory to store a LeRobot-format episode dataset "
                        "(creates parquet + video files trainable with SmolVLA/ACT)")
    p.add_argument("--task",          type=str,   default="teleoperation",
                   help="Natural-language task description stored in the episode "
                        "(default: 'teleoperation')")
    p.add_argument("--cameras",       type=int,   nargs="+", default=[],
                   help="One or more OpenCV camera indices for image observations "
                        "(e.g. --cameras 0 2). Omit for no camera.")
    p.add_argument("--camera_names",  type=str,   nargs="+", default=None,
                   help="Names for each camera in --cameras, used as dataset keys "
                        "(e.g. --camera_names front wrist). Defaults to cam0, cam1, …")
    p.add_argument("--cam_width",     type=int,   default=640,
                   help="Camera capture width  (default: 640)")
    p.add_argument("--cam_height",    type=int,   default=480,
                   help="Camera capture height (default: 480)")

    args = p.parse_args()

    # Validate camera names length
    if args.camera_names is not None and len(args.camera_names) != len(args.cameras):
        p.error(f"--camera_names must have the same number of entries as --cameras "
                f"(got {len(args.camera_names)} names for {len(args.cameras)} cameras)")

    # Default names: cam0, cam1, …
    if args.camera_names is None:
        args.camera_names = [f"cam{i}" for i in range(len(args.cameras))]

    return args


def build_lerobot_features(camera_names: list[str], cam_height: int, cam_width: int) -> dict:
    """
    Describe the dataset schema for LeRobotDataset.create().

    observation.state              - 6 joint positions in degrees
    action                         - 6 goal joint positions in degrees
    observation.images.<name>      - RGB camera frame per camera
    """
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        },
    }
    for name in camera_names:
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (cam_height, cam_width, 3),
            "names": ["height", "width", "channel"],
        }
    return features


# Main
def main():
    args = parse_args()

    leader = FeetechMotorsBus(port=args.leader_port,   motors=SO101_MOTORS)
    follower = None if args.dry_run else FeetechMotorsBus(
        port=args.follower_port, motors=SO101_MOTORS
    )

    offsets = {n: 0 for n in JOINT_NAMES}
    offsets["wrist_roll"] = round(args.wrist_roll_offset * STEPS_PER_DEG)

    leader.connect()
    print(f"Leader   : {args.leader_port}")

    if follower is not None:
        follower.connect()
        print(f"Follower : {args.follower_port}")
        follower.sync_write("Torque_Enable", {n: 1 for n in JOINT_NAMES}, normalize=False)
        print("Follower torque enabled")
    else:
        print("DRY RUN - follower will NOT move")

    print(f"Loop rate: {args.hz} Hz")
    if args.save:
        print(f"Recording CSV: {args.save}")

    # --- Cameras ---
    caps = {}  # name -> VideoCapture
    if args.cameras:
        import cv2
        for idx, name in zip(args.cameras, args.camera_names):
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera index {idx} (name='{name}')")
            caps[name] = cap
            print(f"Camera   : '{name}' → index {idx}  ({args.cam_width}x{args.cam_height})")

    # --- LeRobot dataset ---
    lerobot_dataset = None
    if args.episode_dir is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = build_lerobot_features(
            list(caps.keys()), args.cam_height, args.cam_width
        )

        # repo_id "/" is used for purely local datasets (no Hub upload)
        lerobot_dataset = LeRobotDataset.create(
            repo_id="/",
            fps=int(args.hz),
            features=features,
            root=args.episode_dir,
            robot_type="so101",
            use_videos=bool(caps),
            image_writer_threads=4 * len(caps) if caps else 0,
        )
        print(f"Episode  : {args.episode_dir}  task='{args.task}'")

    running = True

    def handle_stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT,  handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    period    = 1.0 / args.hz
    # Log every 5 seconds, this is what gets printed to the terminal but the loop is much faster
    log_every = max(1, int(args.hz) * 5)
    frame_count = 0
    late_count  = 0

    csv_file   = None
    csv_writer = None
    if args.save:
        csv_file = open(args.save, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp"] + JOINT_NAMES)

    try:
        while running:
            t_start = time.perf_counter()

            # Read all 6 joint positions from the leader arm (raw steps, no calibration needed)
            positions = leader.sync_read("Present_Position", normalize=False)
            timestamp = time.time()

            # Mirror raw steps to follower, applying any per-joint offsets
            goal = {n: int(positions[n]) + offsets[n] for n in JOINT_NAMES}
            if follower is not None:
                follower.sync_write("Goal_Position", goal, normalize=False)

            # Optionally record to CSV
            if csv_writer is not None:
                row = [timestamp] + [positions[n] for n in JOINT_NAMES]
                csv_writer.writerow(row)

            # Optionally record a LeRobot episode frame
            if lerobot_dataset is not None:
                import cv2
                obs_state  = steps_to_deg(positions)
                action_deg = steps_to_deg(goal)

                lerobot_frame: dict = {
                    "task":              args.task,
                    "observation.state": obs_state,
                    "action":            action_deg,
                }

                for name, cap in caps.items():
                    ret, bgr = cap.read()
                    if ret:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        if rgb.shape[:2] != (args.cam_height, args.cam_width):
                            rgb = cv2.resize(rgb, (args.cam_width, args.cam_height))
                        lerobot_frame[f"observation.images.{name}"] = rgb

                lerobot_dataset.add_frame(lerobot_frame)

            frame_count += 1

            # Timing
            elapsed    = time.perf_counter() - t_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                late_count += 1

            # Periodic status log
            if frame_count % log_every == 0:
                abbrevs = {"shoulder_pan": "span", "shoulder_lift": "slif",
                           "elbow_flex": "elbw", "wrist_flex": "wfle",
                           "wrist_roll": "wrol", "gripper": "grip"}
                pos_str = "  ".join(
                    f"{abbrevs[n]}: {positions[n]}" for n in JOINT_NAMES
                )
                print(f"[{frame_count:6d}] {pos_str}  (late: {late_count})")

    finally:
        # Disable follower torque so the arm can be moved by hand safely
        if follower is not None:
            try:
                follower.sync_write("Torque_Enable", {n: 0 for n in JOINT_NAMES}, normalize=False)
                print("Follower torque disabled")
            except Exception:
                pass
            follower.disconnect()

        leader.disconnect()

        if csv_file is not None:
            csv_file.close()

        for cap in caps.values():
            cap.release()

        # Save the LeRobot episode (encodes video, writes parquet)
        if lerobot_dataset is not None and frame_count > 0:
            print("Saving LeRobot episode …")
            lerobot_dataset.save_episode()
            print(f"Episode saved → {args.episode_dir}")

    print(f"\nStopped after {frame_count} frames @ {args.hz} Hz  ({late_count} late)")
    if args.save and frame_count > 0:
        print(f"Saved CSV to {args.save}")


if __name__ == "__main__":
    main()
