"""
Inference client — queries a Claude vision model for robot workspace move coordinates.

The image sent to the model always includes:
  • the grid coordinate overlay (green lines + yellow axes)
  • the previous model prediction (yellow=from ring, magenta=to ring, move text)
    so the model can judge whether its last spatial estimate was accurate

In --debug mode an OpenCV window shows each frame annotated with the freshly
predicted coordinates, letting you visually verify coordinate accuracy.

Usage:
    python inference_client.py --grid_def grid.json --camera 0 --task "Pick up the red block"
    python inference_client.py --grid_def grid.json --camera 0 --debug
    python inference_client.py --grid_def grid.json --camera 0 --model claude-opus-4-8

Requires:
    pip install anthropic opencv-python
    ANTHROPIC_API_KEY environment variable (or --api_key argument)
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import anthropic
import cv2
import numpy as np

DEFAULT_MODEL = "claude-opus-4-8"

_AXIS_COLOR = (0, 255, 255)   # BGR yellow
_GRID_COLOR = (0, 220, 0)     # BGR green
_FROM_COLOR = (0, 255, 255)   # BGR yellow — "move from"
_TO_COLOR   = (255, 0, 255)   # BGR magenta — "move to"


def _draw_overlay(bgr: np.ndarray, grid: dict,
                  coord=None, move_label: str | None = None) -> np.ndarray:
    """Draw grid, axis annotations, optional coordinate rings, and move label."""
    out = bgr.copy()
    h, w = out.shape[:2]

    ox, oy   = int(grid["origin_x"]), int(grid["origin_y"])
    cw, ch   = int(grid["cell_w"]),   int(grid["cell_h"])
    cols, rows = int(grid["cols"]),   int(grid["rows"])

    # Grid lines
    for c in range(cols + 1):
        x = ox + c * cw
        if 0 <= x < w:
            cv2.line(out, (x, max(0, oy)), (x, min(h - 1, oy + rows * ch)), _GRID_COLOR, 2)
    for r in range(rows + 1):
        y = oy + r * ch
        if 0 <= y < h:
            cv2.line(out, (max(0, ox), y), (min(w - 1, ox + cols * cw), y), _GRID_COLOR, 2)

    MARGIN, TICK, FS = 15, 4, 0.28

    # X axis
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

    # Y axis
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

    # Coordinate rings
    if coord is not None:
        fx = int(ox + coord[0] * cw);  fy = int(oy + coord[1] * ch)
        tx = int(ox + coord[2] * cw);  ty = int(oy + coord[3] * ch)
        cv2.circle(out, (fx, fy), 10, _FROM_COLOR, 2, cv2.LINE_AA)
        cv2.circle(out, (fx, fy),  3, _FROM_COLOR, -1)
        cv2.circle(out, (tx, ty), 10, _TO_COLOR,   2, cv2.LINE_AA)
        cv2.circle(out, (tx, ty),  3, _TO_COLOR,   -1)

    # Move label at bottom of frame
    if move_label:
        label = move_label[:90]
        for col, tk in [((0, 0, 0), 2), ((255, 255, 255), 1)]:
            cv2.putText(out, label, (5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, tk, cv2.LINE_AA)

    return out


def _encode_frame(bgr: np.ndarray) -> str:
    """JPEG-encode a BGR frame and return as a base64 string."""
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
    """Extract JSON from the model reply, tolerating markdown fences."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$",           "",  text.strip(), flags=re.MULTILINE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in response:\n{text[:300]}")
    return json.loads(text[start : end + 1])


def _query_model(client: anthropic.Anthropic, model: str,
                 frame_for_api: np.ndarray,
                 task: str, grid: dict,
                 prev_result: dict | None) -> dict:
    """Send the annotated frame to Claude and return the parsed move dict."""
    b64    = _encode_frame(frame_for_api)
    prompt = _build_prompt(task, grid, prev_result)

    with client.messages.stream(
        model=model,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    ) as stream:
        response = stream.get_final_message()

    raw = next(
        (block.text for block in response.content if block.type == "text"),
        "",
    )
    result = _parse_response(raw)
    result["_raw"] = raw
    return result


def main():
    p = argparse.ArgumentParser(
        description="Claude vision inference client for robot workspace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--grid_def",  type=Path,  required=True,
                   help="Grid definition JSON (from grid_editor.py)")
    p.add_argument("--camera",    type=int,   default=0,
                   help="OpenCV camera index (default: 0)")
    p.add_argument("--width",     type=int,   default=640)
    p.add_argument("--height",    type=int,   default=480)
    p.add_argument("--task",      type=str,   default="Identify and pick up the target object",
                   help="Task description sent to the model")
    p.add_argument("--model",     type=str,   default=DEFAULT_MODEL,
                   help=f"Claude model to use (default: {DEFAULT_MODEL})")
    p.add_argument("--api_key",   type=str,   default=None,
                   help="Anthropic API key (default: $ANTHROPIC_API_KEY)")
    p.add_argument("--debug",     action="store_true",
                   help="Show annotated frame in an OpenCV window after each query")
    p.add_argument("--interval",  type=float, default=0.0,
                   help="Seconds between queries in non-debug mode (0 = manual Enter)")
    args = p.parse_args()

    # Load grid definition
    with open(args.grid_def) as f:
        grid = json.load(f)
    print(f"Grid loaded: {args.grid_def}  ({int(grid['cols'])}x{int(grid['rows'])} cells)")

    # Anthropic client
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        p.error("Anthropic API key required: set ANTHROPIC_API_KEY or use --api_key")
    ai_client = anthropic.Anthropic(api_key=api_key)

    # Camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    print(f"Camera {args.camera} opened  ({args.width}x{args.height})")

    if args.debug:
        cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)
        print("\nDebug mode — controls:")
        print("  SPACE / Enter  : capture and query the model")
        print("  R              : retake without querying")
        print("  Q              : quit\n")

    prev_result: dict | None = None
    query_count = 0

    try:
        while True:
            # Capture live frame
            ret, bgr = cap.read()
            if not ret:
                print("Camera read failed, retrying…")
                time.sleep(0.1)
                continue

            # Build frame for API: include grid + previous prediction so the
            # model can evaluate how accurate its last estimate was
            prev_coord: list | None = None
            prev_label: str | None = None
            if prev_result:
                fc = prev_result.get("from_coord")
                tc = prev_result.get("to_coord")
                if fc and tc:
                    prev_coord = [fc[0], fc[1], tc[0], tc[1]]
                prev_label = prev_result.get("move")

            frame_for_api = _draw_overlay(bgr, grid, prev_coord, prev_label)

            # ── Debug: live view while waiting for trigger ──
            if args.debug:
                preview = frame_for_api.copy()
                for col, tk in [((0, 0, 0), 2), ((255, 255, 255), 1)]:
                    cv2.putText(preview, "SPACE/Enter=query  R=retake  Q=quit",
                                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, tk, cv2.LINE_AA)
                cv2.imshow("Inference", preview)
                key = cv2.waitKeyEx(30)
                if key in (ord('q'), ord('Q')):
                    break
                if key not in (32, 13):  # not space/enter — keep updating live view
                    continue

            # ── Query the model ──
            print(f"\n[Query {query_count + 1}]  sending to {args.model}…")
            t0 = time.perf_counter()
            try:
                result = _query_model(ai_client, args.model, frame_for_api,
                                      args.task, grid, prev_result)
            except Exception as exc:
                print(f"  Model error: {exc}")
                time.sleep(1.0)
                continue

            elapsed = time.perf_counter() - t0
            query_count += 1

            fc = result.get("from_coord", [])
            tc = result.get("to_coord",   [])
            print(f"  Move      : {result.get('move', '—')}")
            print(f"  From      : {fc}  →  To: {tc}")
            print(f"  Reasoning : {result.get('reasoning', '—')}")
            print(f"  API time  : {elapsed:.1f}s")

            # ── Debug: show prediction overlaid on a fresh frame ──
            should_quit = False
            if args.debug and fc and tc:
                new_coord = [fc[0], fc[1], tc[0], tc[1]]
                ret2, bgr2 = cap.read()
                base_frame = bgr2 if ret2 else bgr.copy()
                debug_frame = _draw_overlay(base_frame, grid, new_coord,
                                            f"→ {result.get('move', '')}")
                # Stamp reasoning at top
                reasoning_short = result.get("reasoning", "")[:80]
                for col, tk in [((0, 0, 0), 2), ((255, 255, 255), 1)]:
                    cv2.putText(debug_frame, reasoning_short, (5, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, tk, cv2.LINE_AA)
                # Status line
                status = f"Q{query_count}: from{fc} to{tc}  |  SPACE=next  Q=quit"
                for col, tk in [((0, 0, 0), 2), ((200, 200, 200), 1)]:
                    cv2.putText(debug_frame, status, (5, debug_frame.shape[0] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, tk, cv2.LINE_AA)

                cv2.imshow("Inference", debug_frame)
                print("  [Debug] Prediction shown — SPACE/Enter for next, Q to quit.")
                while True:
                    k = cv2.waitKeyEx(30)
                    if k in (ord('q'), ord('Q')):
                        should_quit = True
                        break
                    if k in (32, 13):
                        break

            prev_result = result
            if should_quit:
                break

            # ── Non-debug flow ──
            if not args.debug:
                if args.interval > 0:
                    time.sleep(args.interval)
                else:
                    input("  Press Enter to query again…")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDone. {query_count} queries made.")


if __name__ == "__main__":
    main()
