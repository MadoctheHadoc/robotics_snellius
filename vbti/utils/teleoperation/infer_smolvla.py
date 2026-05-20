"""
Run SmolVLA inference from a live camera frame and SO-101 joint state.

Loads the finetuned model from outputs/smolvla_so101, captures a single frame
from the attached side camera, reads current joint positions from the SO-101,
and prints the predicted action chunk.

Usage:
    python vbti/utils/teleoperation/infer_smolvla.py
    python vbti/utils/teleoperation/infer_smolvla.py --port COM3 --camera 1
    python vbti/utils/teleoperation/infer_smolvla.py --no_arm --task "grasp the cube"
"""

import argparse
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

# Expected image size for the model (from config.json input_features shape)
IMG_H, IMG_W = 480, 640


def parse_args():
    parser = argparse.ArgumentParser(description="SmolVLA inference: live camera + SO-101 state")
    parser.add_argument("--model_dir", type=Path, default=Path("outputs/smolvla_so101"),
                        help="Path to the saved SmolVLA model directory")
    parser.add_argument("--port",      type=str, default="/dev/ttyUSB0",
                        help="Serial port the SO-101 is connected to")
    parser.add_argument("--camera",    type=int, default=0,
                        help="OpenCV camera index (default: 0)")
    parser.add_argument("--task",      type=str, default="pick up the object",
                        help="Natural-language task description for the VLA")
    parser.add_argument("--no_arm",    action="store_true",
                        help="Skip SO-101 connection and use a zero joint state")
    return parser.parse_args()


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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading SmolVLA from {args.model_dir} ...")
    policy = SmolVLAPolicy.from_pretrained(str(args.model_dir))
    policy.eval()
    policy.to(device)
    print(f"Model on {device}\n")

    # --- Joint state ---
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

    # --- Camera frame ---
    print(f"Capturing from camera {args.camera} ...")
    frame_rgb = capture_frame(args.camera)
    h, w = frame_rgb.shape[:2]
    print(f"Frame size  : {w}x{h}")

    # Resize to model's expected resolution and convert to float [0,1] CHW
    frame_resized = cv2.resize(frame_rgb, (IMG_W, IMG_H))
    image_tensor = torch.from_numpy(frame_resized).float().div(255.0).permute(2, 0, 1)

    # Build batch — shape conventions: [batch=1, n_obs_steps=1, ...]
    batch = {
        "observation.state": torch.from_numpy(state)
            .float().unsqueeze(0).unsqueeze(0).to(device),          # [1, 1, 6]
        "observation.images.side": image_tensor
            .unsqueeze(0).unsqueeze(0).to(device),                  # [1, 1, 3, H, W]
        "task": [args.task],
    }

    print(f"Task        : \"{args.task}\"")
    print("Running inference ...\n")

    with torch.no_grad():
        action = policy.select_action(batch)

    # select_action returns [batch, action_dim] or [action_dim]
    if action.dim() == 2:
        action = action[0]
    action_np = action.cpu().numpy()

    print("Predicted action (joint positions):")
    for name, val in zip(JOINT_NAMES, action_np):
        print(f"  {name:<15s} {val:>10.4f}")
    print(f"\nRaw: {action_np}")


if __name__ == "__main__":
    main()
