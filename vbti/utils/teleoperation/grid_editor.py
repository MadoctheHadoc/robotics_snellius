"""
Interactive grid definition editor for down-facing camera alignment.
Produces a JSON file consumed by so101_teleop.py --grid_def.

Usage:
    python grid_editor.py output_grid.json
    python grid_editor.py output_grid.json --cameras 0 2
    python grid_editor.py output_grid.json --cameras 0 --width 1280 --height 720

Controls (focus the OpenCV window first):
    Arrow keys  move grid origin by 5 px
    + / =       cell size +5 px (width & height together)
    -           cell size -5 px (width & height together)
    W / S       cell height +1 / -1 px
    A / D       cell width  +1 / -1 px
    [ / ]       columns -1 / +1
    , / .       rows    -1 / +1
    C           cycle to next camera
    Enter       save grid and quit
    Q           quit without saving
    H           toggle help overlay
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Windows-specific arrow key codes returned by cv2.waitKeyEx()
_ARROW_UP    = 2490368
_ARROW_DOWN  = 2621440
_ARROW_LEFT  = 2424832
_ARROW_RIGHT = 2555904

_MOVE_STEP = 5

_HELP_LINES = [
    "Arrow keys : move grid (-5/+5 px)",
    "+ / =      : cell size +5 px",
    "-          : cell size -5 px",
    "W / S      : cell height +1 / -1",
    "A / D      : cell width  +1 / -1",
    "[ / ]      : columns  -1 / +1",
    ", / .      : rows     -1 / +1",
    "C          : next camera",
    "Enter      : save & quit",
    "Q          : quit  (no save)",
    "H          : toggle this help",
]


def _draw_display(frame: np.ndarray, grid: dict, show_help: bool,
                  cam_idx: int, cameras: list) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    ox, oy   = grid["origin_x"], grid["origin_y"]
    cw, ch   = grid["cell_w"],   grid["cell_h"]
    cols, rows = grid["cols"],   grid["rows"]

    gc = (0, 220, 0)
    for c in range(cols + 1):
        x = ox + c * cw
        if 0 <= x < w:
            cv2.line(out, (x, max(0, oy)), (x, min(h - 1, oy + rows * ch)), gc, 2)
    for r in range(rows + 1):
        y = oy + r * ch
        if 0 <= y < h:
            cv2.line(out, (max(0, ox), y), (min(w - 1, ox + cols * cw), y), gc, 2)
    # Axis annotations — yellow, distinct from green grid lines
    ac     = (0, 255, 255)   # BGR yellow
    MARGIN = 15              # gap between grid border and axis line
    TICK   = 4               # half-tick length in px
    FS     = 0.28            # font scale

    # X axis: horizontal line above the grid with arrowhead →
    ax_y  = oy - MARGIN
    end_x = ox + cols * cw
    if ax_y > 0:
        cv2.line(out, (ox, ax_y), (end_x + 10, ax_y), ac, 1, cv2.LINE_AA)
        tip = np.array([[end_x + 10, ax_y],
                        [end_x + 3,  ax_y - 4],
                        [end_x + 3,  ax_y + 4]], np.int32)
        cv2.fillPoly(out, [tip], ac)
        for c in range(cols + 1):
            x = ox + c * cw
            if 0 <= x < w:
                cv2.line(out, (x, ax_y - TICK), (x, ax_y + TICK), ac, 1)
                if c < cols:
                    for color, tk in [((0, 0, 0), 2), (ac, 1)]:
                        cv2.putText(out, str(c), (x - 4, ax_y - TICK - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, FS, color, tk, cv2.LINE_AA)

    # Y axis: vertical line left of the grid with arrowhead ↓
    ax_x  = ox - MARGIN
    end_y = oy + rows * ch
    if ax_x > 0:
        cv2.line(out, (ax_x, oy), (ax_x, end_y + 10), ac, 1, cv2.LINE_AA)
        tip = np.array([[ax_x,     end_y + 10],
                        [ax_x - 4, end_y + 3],
                        [ax_x + 4, end_y + 3]], np.int32)
        cv2.fillPoly(out, [tip], ac)
        for r in range(rows + 1):
            y = oy + r * ch
            if 0 <= y < h:
                cv2.line(out, (ax_x - TICK, y), (ax_x + TICK, y), ac, 1)
                if r < rows:
                    for color, tk in [((0, 0, 0), 2), (ac, 1)]:
                        cv2.putText(out, str(r), (ax_x - TICK - 13, y + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, FS, color, tk, cv2.LINE_AA)

    # Status bar at bottom
    info = (f"Cam {cameras[cam_idx]}  |  "
            f"Origin ({ox}, {oy})  |  "
            f"Cell {cw} x {ch}  |  "
            f"Grid {cols} cols x {rows} rows  |  H = help")
    for color, thick in [((0, 0, 0), 2), ((255, 255, 255), 1)]:
        cv2.putText(out, info, (5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thick, cv2.LINE_AA)

    if show_help:
        for i, line in enumerate(_HELP_LINES):
            y_pos = 22 + i * 20
            for color, thick in [((0, 0, 0), 2), ((255, 255, 255), 1)]:
                cv2.putText(out, line, (8, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thick, cv2.LINE_AA)

    return out


def _open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    return cap


def main():
    p = argparse.ArgumentParser(
        description="Interactive grid definition editor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("output", type=Path,
                   help="Output grid definition JSON file path")
    p.add_argument("--cameras", type=int, nargs="+", default=[0],
                   help="Camera index(es) to cycle through (default: 0)")
    p.add_argument("--width",  type=int, default=640,
                   help="Camera capture width  (default: 640)")
    p.add_argument("--height", type=int, default=480,
                   help="Camera capture height (default: 480)")
    args = p.parse_args()

    # Load existing grid or start with sensible defaults
    if args.output.exists():
        with open(args.output) as f:
            grid = json.load(f)
        print(f"Loaded existing grid from {args.output}")
    else:
        grid = {
            "origin_x": 50,
            "origin_y": 50,
            "cell_w":   50,
            "cell_h":   50,
            "cols":      8,
            "rows":      6,
        }
        print("Starting with default grid (50 px cells, 8x6)")

    cam_idx = 0
    cap = _open_camera(args.cameras[cam_idx], args.width, args.height)
    show_help = True

    cv2.namedWindow("Grid Editor", cv2.WINDOW_NORMAL)
    print(f"Camera {args.cameras[cam_idx]} opened. Focus the window and use keyboard to edit.")
    print("Press Enter to save, Q to quit without saving.")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Camera hiccup — keep trying without crashing
            blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera read failed — retrying…", (10, args.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
            cv2.imshow("Grid Editor", blank)
            if cv2.waitKeyEx(200) != -1:
                pass
            continue

        display = _draw_display(frame, grid, show_help, cam_idx, args.cameras)
        cv2.imshow("Grid Editor", display)

        key = cv2.waitKeyEx(30)
        if key == -1:
            continue

        if key in (ord('q'), ord('Q')):
            print("Quit without saving.")
            break

        elif key == 13:  # Enter
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(grid, f, indent=2)
            print(f"Grid saved → {args.output}")
            break

        elif key in (ord('h'), ord('H')):
            show_help = not show_help

        elif key in (ord('c'), ord('C')):
            cap.release()
            cam_idx = (cam_idx + 1) % len(args.cameras)
            cap = _open_camera(args.cameras[cam_idx], args.width, args.height)
            print(f"Switched to camera {args.cameras[cam_idx]}")

        # Grid movement
        elif key == _ARROW_UP:
            grid["origin_y"] -= _MOVE_STEP
        elif key == _ARROW_DOWN:
            grid["origin_y"] += _MOVE_STEP
        elif key == _ARROW_LEFT:
            grid["origin_x"] -= _MOVE_STEP
        elif key == _ARROW_RIGHT:
            grid["origin_x"] += _MOVE_STEP

        # Cell size (both axes together)
        elif key in (ord('+'), ord('=')):
            grid["cell_w"] = max(5, grid["cell_w"] + 5)
            grid["cell_h"] = max(5, grid["cell_h"] + 5)
        elif key == ord('-'):
            grid["cell_w"] = max(5, grid["cell_w"] - 5)
            grid["cell_h"] = max(5, grid["cell_h"] - 5)

        # Cell height
        elif key in (ord('w'), ord('W')):
            grid["cell_h"] = max(1, grid["cell_h"] + 1)
        elif key in (ord('s'), ord('S')):
            grid["cell_h"] = max(1, grid["cell_h"] - 1)

        # Cell width
        elif key in (ord('d'), ord('D')):
            grid["cell_w"] = max(1, grid["cell_w"] + 1)
        elif key in (ord('a'), ord('A')):
            grid["cell_w"] = max(1, grid["cell_w"] - 1)

        # Column / row count
        elif key == ord('['):
            grid["cols"] = max(1, grid["cols"] - 1)
        elif key == ord(']'):
            grid["cols"] += 1
        elif key == ord(','):
            grid["rows"] = max(1, grid["rows"] - 1)
        elif key == ord('.'):
            grid["rows"] += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
