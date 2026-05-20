"""
Run SmolVLA inference from a live camera frame and SO-101 joint state.

One-shot mode  — captures one frame, prints predicted action, exits.
Server mode    — listens on a ZMQ socket, serves inference requests from a
                 remote client (e.g. a laptop on the local network or via
                 an SSH tunnel from Snellius).

Usage (one-shot):
    python vbti/utils/teleoperation/infer_smolvla.py
    python vbti/utils/teleoperation/infer_smolvla.py --port COM3 --camera 1
    python vbti/utils/teleoperation/infer_smolvla.py --no_arm --task "grasp the cube"

Usage (server — submit via inference_serve_batch.sh on Snellius):
    python vbti/utils/teleoperation/infer_smolvla.py --serve
    python vbti/utils/teleoperation/infer_smolvla.py --serve --zmq_port 5556

Protocol (ZMQ REQ/REP, multipart):
  Client sends  → [task:utf8, state:float32×6, image:JPEG bytes]
  Server replies → [action:float32×6]
  Reset signal  → client sends [b"RESET"], server replies [b"OK"]
"""

import argparse
import socket as _socket
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


SO101_MOTORS = {
    "shoulder_pan":  (1, "sts3215"),
    "shoulder_lift": (2, "sts3215"),
    "elbow_flex":    (3, "sts3215"),
    "wrist_flex":    (4, "sts3215"),
    "wrist_roll":    (5, "sts3215"),
    "gripper":       (6, "sts3215"),
}
JOINT_NAMES = list(SO101_MOTORS.keys())

IMG_H, IMG_W = 480, 640


def parse_args():
    parser = argparse.ArgumentParser(description="SmolVLA inference: live camera + SO-101 state")
    parser.add_argument("--model_dir", type=Path, default=Path("outputs/smolvla_so101"),
                        help="Path to the saved SmolVLA model directory")
    parser.add_argument("--port",      type=str, default="/dev/ttyUSB0",
                        help="Serial port the SO-101 is connected to (one-shot mode only)")
    parser.add_argument("--camera",    type=int, default=0,
                        help="OpenCV camera index (one-shot mode only, default: 0)")
    parser.add_argument("--task",      type=str, default="pick up the object",
                        help="Natural-language task description (one-shot mode only)")
    parser.add_argument("--no_arm",    action="store_true",
                        help="Skip SO-101 and use zero state (one-shot mode only)")
    parser.add_argument("--serve",     action="store_true",
                        help="Run as a ZMQ inference server instead of one-shot")
    parser.add_argument("--zmq_port",  type=int, default=5556,
                        help="ZMQ port to bind in server mode (default: 5556)")
    return parser.parse_args()


def _build_batch(task: str, state: np.ndarray, image_rgb: np.ndarray, device) -> dict:
    frame_resized = cv2.resize(image_rgb, (IMG_W, IMG_H))
    image_tensor = torch.from_numpy(frame_resized).float().div(255.0).permute(2, 0, 1)
    return {
        "observation.state": torch.from_numpy(state).float()
            .unsqueeze(0).unsqueeze(0).to(device),         # [1, 1, 6]
        "observation.images.side": image_tensor
            .unsqueeze(0).unsqueeze(0).to(device),         # [1, 1, 3, H, W]
        "task": [task],
    }


def capture_frame(camera_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")
    try:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def serve_loop(policy, device, zmq_port: int):
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{zmq_port}")
    policy.reset()

    hostname = _socket.gethostname()
    print(f"Server ready on tcp://*:{zmq_port}")
    print(f"Hostname    : {hostname}")
    print(f"SSH tunnel  : ssh -L {zmq_port}:{hostname}:{zmq_port} <user>@snellius.surf.nl")
    print("Waiting for requests... (Ctrl+C to stop)\n")

    step = 0
    while True:
        parts = sock.recv_multipart()

        if parts[0] == b"RESET":
            policy.reset()
            sock.send_multipart([b"OK"])
            print("Policy state reset.")
            step = 0
            continue

        task = parts[0].decode("utf-8")
        state = np.frombuffer(parts[1], dtype=np.float32).copy()

        image_arr = np.frombuffer(parts[2], np.uint8)
        image_bgr = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        batch = _build_batch(task, state, image_rgb, device)

        with torch.no_grad():
            action = policy.select_action(batch)

        if action.dim() == 2:
            action = action[0]
        action_np = action.cpu().numpy().astype(np.float32)

        sock.send_multipart([action_np.tobytes()])

        step += 1
        pos_str = "  ".join(f"{n[:3]}:{v:.0f}" for n, v in zip(JOINT_NAMES, action_np))
        print(f"[{step:5d}] task=\"{task[:30]}\"  {pos_str}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading SmolVLA from {args.model_dir} ...")
    policy = SmolVLAPolicy.from_pretrained(str(args.model_dir))
    policy.eval()
    policy.to(device)
    print(f"Model on {device}\n")

    if args.serve:
        serve_loop(policy, device, args.zmq_port)
        return

    # --- One-shot mode ---
    if args.no_arm:
        state = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        print("Joint state : zeros (--no_arm)")
    else:
        bus = FeetechMotorsBus(
            port=args.port,
            motors={name: list(cfg) for name, cfg in SO101_MOTORS.items()},
        )
        bus.connect()
        try:
            positions = bus.read("Present_Position")
            state = np.array([float(positions[i]) for i in range(len(JOINT_NAMES))], dtype=np.float32)
        finally:
            bus.disconnect()
        pos_str = "  ".join(f"{n}: {v:.0f}" for n, v in zip(JOINT_NAMES, state))
        print(f"Joint state : {pos_str}")

    print(f"Capturing from camera {args.camera} ...")
    frame_rgb = capture_frame(args.camera)
    h, w = frame_rgb.shape[:2]
    print(f"Frame size  : {w}x{h}")

    batch = _build_batch(args.task, state, frame_rgb, device)

    print(f"Task        : \"{args.task}\"")
    print("Running inference ...\n")

    with torch.no_grad():
        action = policy.select_action(batch)

    if action.dim() == 2:
        action = action[0]
    action_np = action.cpu().numpy()

    print("Predicted action (joint positions):")
    for name, val in zip(JOINT_NAMES, action_np):
        print(f"  {name:<15s} {val:>10.4f}")
    print(f"\nRaw: {action_np}")


if __name__ == "__main__":
    main()
