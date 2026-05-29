"""
Laptop-side client for SmolVLA remote inference.

Captures camera frames and SO-101 joint state, sends them to the inference
server running on the Snellius compute node, and prints (or executes) the
predicted actions on the local SO-101 arm.

Setup:
  1. Submit the server job:
       sbatch inference_serve_batch.sh
  2. Find the allocated node:
       squeue -u $USER          (look for the NODELIST column, e.g. gcn42)
  3. Open an SSH tunnel (keep this terminal open):
       ssh -L 5556:gcn42:5556 <user>@snellius.surf.nl
  4. Run this script:
       python vbti/utils/teleoperation/infer_smolvla_client.py

Usage:
    python infer_smolvla_client.py
    python infer_smolvla_client.py --execute              # write actions to SO-101
    python infer_smolvla_client.py --no_arm --task "pick up the block"
    python infer_smolvla_client.py --host localhost --port 5556 --hz 10

Dependencies (laptop):  pip install pyzmq opencv-python
"""

import argparse
import signal
import sys
import time

import cv2
import numpy as np
import zmq

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode


SO101_MOTORS = {
    "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex":    Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex":    Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll":    Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_M100_100),
}
JOINT_NAMES = list(SO101_MOTORS.keys())

# Feetech STS3215 valid position range
POS_MIN, POS_MAX = 0, 4095


def parse_args():
    p = argparse.ArgumentParser(description="SmolVLA inference client (laptop side)")
    p.add_argument("--host",      type=str,   default="localhost",
                   help="Inference server hostname (default: localhost, i.e. through SSH tunnel)")
    p.add_argument("--port",      type=int,   default=5556,
                   help="ZMQ port matching the server (default: 5556)")
    p.add_argument("--so101_port", type=str,  default="COM3",
                   help="Serial port the SO-101 is connected to (Windows: COM3, Linux: /dev/ttyUSB0)")
    p.add_argument("--camera",    type=int,   default=0,
                   help="OpenCV camera index (default: 0)")
    p.add_argument("--task",      type=str,   default="pick up the object",
                   help="Natural-language task description sent to the VLA")
    p.add_argument("--hz",        type=float, default=10.0,
                   help="Control loop frequency in Hz (default: 10)")
    p.add_argument("--no_arm",    action="store_true",
                   help="Skip SO-101 connection; send zero state and do not write actions")
    p.add_argument("--execute",   action="store_true",
                   help="Write predicted actions to the SO-101 arm (requires --no_arm to be off)")
    p.add_argument("--timeout_ms", type=int,  default=60000,
                   help="ZMQ receive timeout per inference request in ms (default: 60000)")
    return p.parse_args()


def open_camera(camera_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")
    return cap


def capture_jpeg(cap: cv2.VideoCapture, quality: int = 85) -> bytes:
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame from camera")
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def main():
    args = parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{args.host}:{args.port}")
    print(f"Connected to tcp://{args.host}:{args.port}")

    # RESET handshake — use a long timeout because the server may still be loading the model.
    # The ZMQ socket only binds after from_pretrained() finishes, which can take 1-2 min.
    RESET_TIMEOUT_MS = 20_000  # 1 minutes
    sock.setsockopt(zmq.RCVTIMEO, RESET_TIMEOUT_MS)
    print("Sending RESET to server (waiting up to 1 min for model to load) ...")
    sock.send_multipart([b"RESET"])
    try:
        sock.recv_multipart()
    except zmq.error.Again:
        print("\nERROR: No response from server. Check:")
        print(f"  1. SSH tunnel is open:      ssh -L {args.port}:<nodename>:{args.port} <user>@<login-node>")
        print(f"  2. Server job is running:   squeue -u $USER")
        print(f"  3. Server log shows:        'Server ready on tcp://*:{args.port}'")
        sys.exit(1)
    print("Server ready.\n")

    # Switch to the configured per-request timeout for inference
    sock.setsockopt(zmq.RCVTIMEO, args.timeout_ms)

    # Open camera
    cap = open_camera(args.camera)

    # Connect to SO-101 if needed
    bus = None
    if not args.no_arm:
        bus = FeetechMotorsBus(
            port=args.so101_port,
            motors=SO101_MOTORS,
        )
        bus.connect()
        bus.calibration = {
            name: MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=0,
                range_min=0,
                range_max=4095,
            )
            for name, m in SO101_MOTORS.items()
        }
        print(f"SO-101 connected on {args.so101_port}")

    running = True

    def handle_stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    period = 1.0 / args.hz
    step = 0
    late = 0

    print(f"Task   : \"{args.task}\"")
    print(f"Rate   : {args.hz} Hz")
    print(f"Execute: {args.execute and not args.no_arm}")
    print("Running — Ctrl+C to stop.\n")

    try:
        while running:
            t0 = time.perf_counter()

            # Read joint state
            if bus is not None:
                state = np.array(
                    [float(bus.read("Present_Position", n)) for n in JOINT_NAMES],
                    dtype=np.float32,
                )
            else:
                state = np.zeros(len(JOINT_NAMES), dtype=np.float32)

            # Capture camera frame as JPEG
            jpeg_bytes = capture_jpeg(cap)

            # Send to server
            sock.send_multipart([
                args.task.encode("utf-8"),
                state.tobytes(),
                jpeg_bytes,
            ])

            # Receive predicted action — on timeout, recreate the socket
            # (ZMQ REQ is broken after a send with no matching recv)
            try:
                parts = sock.recv_multipart()
            except zmq.error.Again:
                print("  [inference timeout — reconnecting socket]")
                sock.close()
                sock = ctx.socket(zmq.REQ)
                sock.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
                sock.connect(f"tcp://{args.host}:{args.port}")
                continue

            status = parts[0]
            if status == b"ERR":
                print(f"\n[SERVER ERROR]\n{parts[1].decode()}")
                continue
            action = np.frombuffer(parts[1], dtype=np.float32).copy()

            step += 1
            pos_str = "  ".join(f"{n[:3]}:{v:7.1f}" for n, v in zip(JOINT_NAMES, action))
            print(f"[{step:5d}] {pos_str}")

            # Write to arm
            if args.execute and bus is not None:
                goal = np.clip(action, POS_MIN, POS_MAX).astype(np.int32)
                for i, n in enumerate(JOINT_NAMES):
                    bus.write("Goal_Position", n, int(goal[i]))

            elapsed = time.perf_counter() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                late += 1

    finally:
        cap.release()
        if bus is not None:
            bus.disconnect()

    print(f"\nStopped after {step} steps ({late} late). ")


if __name__ == "__main__":
    main()
