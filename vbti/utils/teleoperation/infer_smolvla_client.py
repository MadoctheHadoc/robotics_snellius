"""
Laptop-side client for SmolVLA remote inference.

Captures camera frames and SO-101 joint state, sends them to the inference
server running on the Snellius compute node, and prints (or executes) the
predicted actions on the local SO-101 arm.

Optional --debug mode opens an OpenCV window with a grid overlay and lets
you query Claude for spatial reasoning (press SPACE) without interrupting
the SmolVLA control loop.

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
    python infer_smolvla_client.py --debug --grid_def grid.json

Dependencies (laptop):  pip install pyzmq opencv-python anthropic
"""

import argparse
import base64
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

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

# Conversion between raw STS3215 ticks [0, 4095] and degrees [0°, 360°]
STEPS_PER_DEG = 4096.0 / 360.0

DEFAULT_MODEL = "claude-opus-4-8"

_AXIS_COLOR = (0, 255, 255)   # BGR yellow
_GRID_COLOR = (0, 220, 0)     # BGR green
_FROM_COLOR = (0, 255, 255)   # BGR yellow — "move from"
_TO_COLOR   = (255, 0, 255)   # BGR magenta — "move to"


# ── Grid / Claude overlay helpers ──────────────────────────────────────────────

def _draw_overlay(bgr: np.ndarray, grid: dict,
                  coord=None, move_label: str | None = None) -> np.ndarray:
    """Draw grid, axis annotations, optional coordinate rings, and move label."""
    out = bgr.copy()
    h, w = out.shape[:2]
    ox, oy   = int(grid["origin_x"]), int(grid["origin_y"])
    cw, ch   = int(grid["cell_w"]),   int(grid["cell_h"])
    cols, rows = int(grid["cols"]),   int(grid["rows"])

    for c in range(cols + 1):
        x = ox + c * cw
        if 0 <= x < w:
            cv2.line(out, (x, max(0, oy)), (x, min(h - 1, oy + rows * ch)), _GRID_COLOR, 2)
    for r in range(rows + 1):
        y = oy + r * ch
        if 0 <= y < h:
            cv2.line(out, (max(0, ox), y), (min(w - 1, ox + cols * cw), y), _GRID_COLOR, 2)

    MARGIN, TICK, FS = 15, 4, 0.28

    ax_y, end_x = oy - MARGIN, ox + cols * cw
    if ax_y > 0:
        cv2.line(out, (ox, ax_y), (end_x + 10, ax_y), _AXIS_COLOR, 1, cv2.LINE_AA)
        tip = np.array([[end_x + 10, ax_y], [end_x + 3, ax_y - 4], [end_x + 3, ax_y + 4]], np.int32)
        cv2.fillPoly(out, [tip], _AXIS_COLOR)
        for c in range(cols + 1):
            x = ox + c * cw
            if 0 <= x < w:
                cv2.line(out, (x, ax_y - TICK), (x, ax_y + TICK), _AXIS_COLOR, 1)
                if c < cols:
                    for col, tk in [((0, 0, 0), 2), (_AXIS_COLOR, 1)]:
                        cv2.putText(out, str(c), (x - 4, ax_y - TICK - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, FS, col, tk, cv2.LINE_AA)

    ax_x, end_y = ox - MARGIN, oy + rows * ch
    if ax_x > 0:
        cv2.line(out, (ax_x, oy), (ax_x, end_y + 10), _AXIS_COLOR, 1, cv2.LINE_AA)
        tip = np.array([[ax_x, end_y + 10], [ax_x - 4, end_y + 3], [ax_x + 4, end_y + 3]], np.int32)
        cv2.fillPoly(out, [tip], _AXIS_COLOR)
        for r in range(rows + 1):
            y = oy + r * ch
            if 0 <= y < h:
                cv2.line(out, (ax_x - TICK, y), (ax_x + TICK, y), _AXIS_COLOR, 1)
                if r < rows:
                    for col, tk in [((0, 0, 0), 2), (_AXIS_COLOR, 1)]:
                        cv2.putText(out, str(r), (ax_x - TICK - 13, y + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, FS, col, tk, cv2.LINE_AA)

    if coord is not None:
        fx = int(ox + coord[0] * cw);  fy = int(oy + coord[1] * ch)
        tx = int(ox + coord[2] * cw);  ty = int(oy + coord[3] * ch)
        # Dark outline ring for contrast, then coloured ring, then filled centre dot
        cv2.circle(out, (fx, fy), 17, (0, 0, 0),     3, cv2.LINE_AA)
        cv2.circle(out, (fx, fy), 14, _FROM_COLOR,    3, cv2.LINE_AA)
        cv2.circle(out, (fx, fy),  5, _FROM_COLOR,   -1)
        cv2.circle(out, (tx, ty), 17, (0, 0, 0),      3, cv2.LINE_AA)
        cv2.circle(out, (tx, ty), 14, _TO_COLOR,      3, cv2.LINE_AA)
        cv2.circle(out, (tx, ty),  5, _TO_COLOR,     -1)

    if move_label:
        label = move_label[:90]
        for col, tk in [((0, 0, 0), 2), ((255, 255, 255), 1)]:
            cv2.putText(out, label, (5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, tk, cv2.LINE_AA)

    return out


def _encode_frame(bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _build_prompt(task: str, grid: dict, prev_result: dict | None) -> str:
    cols = int(grid["cols"]) - 1
    rows = int(grid["rows"]) - 1
    prev_note = ""
    if prev_result:
        fc = prev_result.get("from_coord", [])
        tc = prev_result.get("to_coord",   [])
        mv = prev_result.get("move", "")
        prev_note = (
            f'\nThe yellow ring marks your previous "from" position {fc} and '
            f'the magenta ring marks your previous "to" position {tc} '
            f'(move: "{mv}"). Use this to judge whether your last spatial '
            f"estimate was accurate before choosing the next action.\n"
        )
    return (
        "You are analyzing a robot workspace viewed from a top-down camera.\n\n"
        "Grid coordinate system:\n"
        f"  • x-axis (horizontal): columns 0–{cols}, left to right\n"
        f"  • y-axis (vertical):   rows    0–{rows}, top to bottom\n"
        "  • Coordinates are decimal grid-cell units; 3.5 = halfway across column 3\n"
        f"{prev_note}\n"
        f"Task: {task}\n\n"
        "Carefully examine the image. Then reply with ONLY a JSON object — "
        "no text outside the JSON:\n"
        "{\n"
        '  "reasoning": "step-by-step spatial analysis",\n'
        '  "move":      "natural-language action description",\n'
        '  "from_coord": [x, y],\n'
        '  "to_coord":   [x, y]\n'
        "}"
    )


def _parse_response(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$",           "",  text.strip(), flags=re.MULTILINE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in response:\n{text[:300]}")
    return json.loads(text[start : end + 1])


def _query_claude(ai_client, model: str, frame_bgr: np.ndarray,
                  task: str, grid: dict, prev_result: dict | None) -> dict:
    import anthropic as _anthropic
    b64    = _encode_frame(frame_bgr)
    prompt = _build_prompt(task, grid, prev_result)
    with ai_client.messages.stream(
        model=model,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text",  "text": prompt},
            ],
        }],
    ) as stream:
        response = stream.get_final_message()
    raw = next((block.text for block in response.content if block.type == "text"), "")
    result = _parse_response(raw)
    result["_raw"] = raw
    return result


# ── Core helpers ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SmolVLA inference client (laptop side)")
    p.add_argument("--host",       type=str,   default="localhost",
                   help="Inference server hostname (default: localhost)")
    p.add_argument("--port",       type=int,   default=5556,
                   help="ZMQ port matching the server (default: 5556)")
    p.add_argument("--so101_port", type=str,   default="COM3",
                   help="Serial port the SO-101 is connected to (Windows: COM3, Linux: /dev/ttyUSB0)")
    p.add_argument("--camera",       type=int,   default=0,
                   help="OpenCV camera index for SmolVLA inference (default: 0)")
    p.add_argument("--debug_camera", type=int,   default=None,
                   help="Camera index for the debug window and Claude queries (default: same as --camera)")
    p.add_argument("--task",       type=str,   default="pick up the object",
                   help="Natural-language task description sent to the VLA")
    p.add_argument("--hz",         type=float, default=10.0,
                   help="Control loop frequency in Hz (default: 10)")
    p.add_argument("--no_arm",     action="store_true",
                   help="Skip SO-101 connection; send zero state and do not write actions")
    p.add_argument("--execute",    action="store_true",
                   help="Write predicted actions to the SO-101 arm")
    p.add_argument("--timeout_ms", type=int,   default=60000,
                   help="ZMQ receive timeout per inference request in ms (default: 60000)")
    # Debug / Claude overlay
    p.add_argument("--debug",      action="store_true",
                   help="Open OpenCV window; press SPACE to query Claude for spatial predictions")
    p.add_argument("--grid_def",   type=Path,  default=None,
                   help="Grid definition JSON for debug overlay (required with --debug)")
    p.add_argument("--model",      type=str,   default=DEFAULT_MODEL,
                   help=f"Claude model for debug queries (default: {DEFAULT_MODEL})")
    p.add_argument("--api_key",    type=str,   default=None,
                   help="Anthropic API key (default: $ANTHROPIC_API_KEY)")
    return p.parse_args()


def open_camera(camera_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")
    return cap


def capture_frame(cap: cv2.VideoCapture) -> np.ndarray:
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame from camera")
    return frame


def frame_to_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def main():
    args = parse_args()

    # ── ZMQ setup ──
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{args.host}:{args.port}")
    print(f"Connected to tcp://{args.host}:{args.port}")

    RESET_TIMEOUT_MS = 20_000
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
    sock.setsockopt(zmq.RCVTIMEO, args.timeout_ms)

    # ── Camera(s) ──
    cap = open_camera(args.camera)
    debug_cam_idx = args.debug_camera if args.debug_camera is not None else args.camera
    cap_debug = open_camera(debug_cam_idx) if debug_cam_idx != args.camera else cap
    if cap_debug is not cap:
        print(f"Debug camera {debug_cam_idx} opened separately from SmolVLA camera {args.camera}")

    # ── SO-101 ──
    bus = None
    if not args.no_arm:
        bus = FeetechMotorsBus(port=args.so101_port, motors=SO101_MOTORS)
        bus.connect()
        bus.calibration = {
            name: MotorCalibration(id=m.id, drive_mode=0, homing_offset=0, range_min=0, range_max=4095)
            for name, m in SO101_MOTORS.items()
        }
        print(f"SO-101 connected on {args.so101_port}")

    # ── Debug / Claude setup ──
    grid      = None
    ai_client = None
    if args.debug:
        if args.grid_def is None:
            print("WARNING: --debug works best with --grid_def; running without grid overlay.")
        else:
            with open(args.grid_def) as f:
                grid = json.load(f)
            print(f"Grid loaded: {args.grid_def}  ({int(grid['cols'])}x{int(grid['rows'])} cells)")

        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            import anthropic as _anthropic
            ai_client = _anthropic.Anthropic(api_key=api_key)
        else:
            print("WARNING: No ANTHROPIC_API_KEY — Claude queries disabled (grid overlay still shown).")

        cv2.namedWindow("SmolVLA Debug", cv2.WINDOW_NORMAL)
        hint = "SPACE=Claude query  Q=quit" if ai_client else "Q=quit  (no API key)"
        # Show placeholder immediately so the window appears before the first inference
        _placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        for _col, _tk in [((0, 0, 0), 2), ((255, 255, 255), 1)]:
            cv2.putText(_placeholder, "Waiting for SmolVLA server...",
                        (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, _col, _tk, cv2.LINE_AA)
            cv2.putText(_placeholder, hint,
                        (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _col, _tk, cv2.LINE_AA)
        cv2.imshow("SmolVLA Debug", _placeholder)
        cv2.waitKey(1)
        print(f"Debug window open — {hint}\n")

    # ── Signal handling ──
    running = True
    def handle_stop(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT,  handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    period = 1.0 / args.hz
    step   = 0
    late   = 0
    prev_claude_result: dict | None = None

    print(f"Task   : \"{args.task}\"")
    print(f"Rate   : {args.hz} Hz")
    print(f"Execute: {args.execute and not args.no_arm}")
    print("Running — Ctrl+C to stop.\n")

    try:
        while running:
            t0 = time.perf_counter()

            # ── Joint state ──
            if bus is not None:
                state = np.array(
                    [bus.read("Present_Position", n, normalize=False) / STEPS_PER_DEG
                     for n in JOINT_NAMES],
                    dtype=np.float32,
                )
            else:
                state = np.zeros(len(JOINT_NAMES), dtype=np.float32)

            # ── Capture frames ──
            bgr       = capture_frame(cap)        # SmolVLA: primary camera, plain
            bgr_debug = capture_frame(cap_debug)  # Debug/Claude: board camera

            # ── SmolVLA inference (plain frame — no overlay avoids domain shift) ──
            sock.send_multipart([args.task.encode("utf-8"), state.tobytes(), frame_to_jpeg(bgr)])

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

            raw_ticks = [str(np.clip(action[i] * STEPS_PER_DEG, 0, 4095)) for i in range(len(JOINT_NAMES))]
            print("Outputs: " + ", ".join(raw_ticks))

            if args.execute and bus is not None:
                for i, n in enumerate(JOINT_NAMES):
                    tick = int(np.clip(action[i] * STEPS_PER_DEG, 0, 4095))
                    bus.write("Goal_Position", n, tick, normalize=False)

            # ── Debug window ──
            if args.debug:
                # Annotate display frame with previous Claude prediction (if any)
                prev_coord: list | None = None
                prev_label: str | None = None
                if prev_claude_result is not None and grid is not None:
                    fc = prev_claude_result.get("from_coord")
                    tc = prev_claude_result.get("to_coord")
                    if fc and tc:
                        prev_coord = [fc[0], fc[1], tc[0], tc[1]]
                    prev_label = prev_claude_result.get("move")

                display = _draw_overlay(bgr_debug, grid, prev_coord, prev_label) if grid else bgr_debug.copy()

                # SmolVLA output bar
                smol_str = " ".join(f"{n[:3]}:{v:.0f}" for n, v in zip(JOINT_NAMES, action))
                for col, tk in [((0, 0, 0), 2), ((180, 255, 180), 1)]:
                    cv2.putText(display, f"[{step}] {smol_str}",
                                (5, display.shape[0] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, tk, cv2.LINE_AA)

                if ai_client:
                    for col, tk in [((0, 0, 0), 2), ((255, 255, 255), 1)]:
                        cv2.putText(display, "SPACE=Claude query  Q=quit",
                                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, tk, cv2.LINE_AA)

                cv2.imshow("SmolVLA Debug", display)
                key = cv2.waitKeyEx(1)

                if key in (ord('q'), ord('Q'), 27):
                    running = False

                elif key == 32 and ai_client and grid:  # SPACE → Claude query
                    # Build the frame sent to Claude: grid + previous prediction so the
                    # model can assess its own spatial accuracy round-to-round
                    frame_for_claude = _draw_overlay(bgr_debug, grid, prev_coord, prev_label)
                    print(f"\n[Claude query at step {step}]  sending to {args.model}…")
                    t_q = time.perf_counter()
                    try:
                        result = _query_claude(ai_client, args.model, frame_for_claude,
                                               args.task, grid, prev_claude_result)
                        prev_claude_result = result
                        fc = result.get("from_coord", [])
                        tc = result.get("to_coord",   [])
                        print(f"  Move      : {result.get('move', '—')}")
                        print(f"  From      : {fc}  →  To: {tc}")
                        print(f"  Reasoning : {result.get('reasoning', '—')}")
                        print(f"  API time  : {time.perf_counter() - t_q:.1f}s")

                        # Flash result frame so prediction rings are immediately visible
                        if fc and tc:
                            result_frame = _draw_overlay(bgr_debug, grid,
                                                         [fc[0], fc[1], tc[0], tc[1]],
                                                         f"→ {result.get('move', '')}")
                            reason_short = result.get("reasoning", "")[:80]
                            for col, tk in [((0, 0, 0), 2), ((255, 255, 255), 1)]:
                                cv2.putText(result_frame, reason_short, (5, 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, tk, cv2.LINE_AA)
                            cv2.imshow("SmolVLA Debug", result_frame)
                            cv2.waitKey(1)
                    except Exception as exc:
                        print(f"  Claude error: {exc}")

            elapsed    = time.perf_counter() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                late += 1

    finally:
        cap.release()
        if cap_debug is not cap:
            cap_debug.release()
        if bus is not None:
            bus.disconnect()
        cv2.destroyAllWindows()

    print(f"\nStopped after {step} steps ({late} late).")


if __name__ == "__main__":
    main()
