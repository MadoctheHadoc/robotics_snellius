"""
Control RoArm V3 from SO-101 recorded positions — with live keyboard control.

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
    # Replay a recording:
    python control_roarm.py --input recorded_positions.csv

    # Interactive keyboard control:
    python control_roarm.py --keyboard

    # Keyboard control on a specific port:
    python control_roarm.py --keyboard --port /dev/ttyUSB1 --speed 200

Keyboard controls (keyboard mode):
    1-5        Select active joint
    ← / →      Move selected joint by step (left = negative, right = positive)
    ↑ / ↓      Same as ← / → (alternative)
    [ / ]      Halve / double step size
    , / .      Fine decrease / increase step (1 deg)
    h          Move selected joint to 0° (home that joint)
    H          Move ALL joints to 0° (home all)
    s          Save current positions as a CSV frame (appends to --output file)
    q / Esc    Quit
"""

import argparse
import csv
import curses
import json
import time
from pathlib import Path

import serial


# ---------------------------------------------------------------------------
# Feetech STS3215 conversion constants
# ---------------------------------------------------------------------------
SO101_STEPS_PER_REV = 4096
SO101_CENTRE        = 2048     # step value at 0 degrees


def so101_steps_to_degrees(steps: float) -> float:
    """Convert raw STS3215 servo steps to signed degrees (centre = 0°)."""
    return (steps - SO101_CENTRE) / SO101_STEPS_PER_REV * 360.0


# ---------------------------------------------------------------------------
# Joint configuration
# ---------------------------------------------------------------------------

# SO-101 joint name → RoArm V3 joint ID (1-indexed)
# wrist_roll is omitted — RoArm V3 has no equivalent DOF.
SO101_TO_ROARM_ID = {
    "shoulder_pan":  1,
    "shoulder_lift": 2,
    "elbow_flex":    3,
    "wrist_flex":    4,
    "gripper":       5,
}

JOINT_NAMES = {
    1: "Base     (shoulder_pan)",
    2: "Shoulder (shoulder_lift)",
    3: "Elbow    (elbow_flex)",
    4: "Wrist    (wrist_flex)",
    5: "Gripper",
}

# Soft angle limits per joint [min_deg, max_deg].
# Adjust to match your physical robot to avoid collisions.
JOINT_LIMITS = {
    1: (-150.0, 150.0),
    2: (-120.0, 120.0),
    3: (-120.0, 120.0),
    4: (-120.0, 120.0),
    5: (-45.0,   90.0),
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

# Default step size for keyboard nudges (degrees)
DEFAULT_STEP = 5.0
MIN_STEP     = 0.5
MAX_STEP     = 45.0


# ---------------------------------------------------------------------------
# Serial helpers
# ---------------------------------------------------------------------------

def send_joint(ser: serial.Serial, joint_id: int, angle_deg: float,
               speed: int, acc: int) -> None:
    """Send a single-joint move command."""
    cmd = {
        "T":     106,
        "joint": joint_id,
        "angle": round(angle_deg, 2),
        "spd":   speed,
        "acc":   acc,
    }
    ser.write((json.dumps(cmd) + "\n").encode())


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Keyboard (curses) control mode
# ---------------------------------------------------------------------------

def keyboard_mode(args) -> None:
    """Interactive curses-based keyboard control."""

    # Current angles for all 5 joints
    angles = {jid: 0.0 for jid in range(1, 6)}
    selected_joint = 1
    step = DEFAULT_STEP
    log_messages: list[str] = []
    saved_frames = 0

    # Prepare output CSV if saving is requested
    output_writer = None
    output_file_handle = None
    if args.output:
        output_path = Path(args.output)
        file_exists = output_path.exists()
        output_file_handle = open(output_path, "a", newline="")
        fieldnames = list(SO101_TO_ROARM_ID.keys())
        output_writer = csv.DictWriter(output_file_handle, fieldnames=fieldnames)
        if not file_exists:
            output_writer.writeheader()

    def add_log(msg: str) -> None:
        log_messages.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if len(log_messages) > 12:
            log_messages.pop(0)

    def draw(stdscr, ser) -> None:
        nonlocal selected_joint, step, saved_frames

        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(50)  # 20 Hz redraw

        # Colour pairs
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN,   -1)  # selected joint
        curses.init_pair(2, curses.COLOR_CYAN,    -1)  # labels
        curses.init_pair(3, curses.COLOR_YELLOW,  -1)  # values
        curses.init_pair(4, curses.COLOR_RED,     -1)  # warnings
        curses.init_pair(5, curses.COLOR_WHITE,   -1)  # log
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # title

        BOLD = curses.A_BOLD
        C_SEL = curses.color_pair(1) | BOLD
        C_LBL = curses.color_pair(2)
        C_VAL = curses.color_pair(3) | BOLD
        C_WRN = curses.color_pair(4) | BOLD
        C_LOG = curses.color_pair(5)
        C_TTL = curses.color_pair(6) | BOLD

        add_log(f"Connected to {args.port} @ {args.baud} baud")
        add_log("Use 1-5 to select joint, ←→ to move, [ ] step size, q to quit")

        while True:
            key = stdscr.getch()

            # ---- Key handling ----
            quit_requested = False

            if key in (ord('q'), ord('Q'), 27):           # q / Esc → quit
                quit_requested = True

            elif key in (ord('1'), ord('2'), ord('3'),
                         ord('4'), ord('5')):
                selected_joint = key - ord('0')
                add_log(f"Selected joint {selected_joint}: {JOINT_NAMES[selected_joint].strip()}")

            elif key in (curses.KEY_LEFT, curses.KEY_DOWN):
                lo, hi = JOINT_LIMITS[selected_joint]
                new_angle = clamp(angles[selected_joint] - step, lo, hi)
                if new_angle != angles[selected_joint]:
                    angles[selected_joint] = new_angle
                    send_joint(ser, selected_joint, new_angle, args.speed, args.acc)
                    add_log(f"J{selected_joint} → {new_angle:+.1f}°")
                else:
                    add_log(f"J{selected_joint} at limit ({lo:.0f}° / {hi:.0f}°)")

            elif key in (curses.KEY_RIGHT, curses.KEY_UP):
                lo, hi = JOINT_LIMITS[selected_joint]
                new_angle = clamp(angles[selected_joint] + step, lo, hi)
                if new_angle != angles[selected_joint]:
                    angles[selected_joint] = new_angle
                    send_joint(ser, selected_joint, new_angle, args.speed, args.acc)
                    add_log(f"J{selected_joint} → {new_angle:+.1f}°")
                else:
                    add_log(f"J{selected_joint} at limit ({lo:.0f}° / {hi:.0f}°)")

            elif key == ord('['):
                step = max(MIN_STEP, step / 2)
                add_log(f"Step size → {step:.1f}°")

            elif key == ord(']'):
                step = min(MAX_STEP, step * 2)
                add_log(f"Step size → {step:.1f}°")

            elif key == ord(','):
                step = max(MIN_STEP, round(step - 1.0, 1))
                add_log(f"Step size → {step:.1f}°")

            elif key == ord('.'):
                step = min(MAX_STEP, round(step + 1.0, 1))
                add_log(f"Step size → {step:.1f}°")

            elif key == ord('h'):
                lo, hi = JOINT_LIMITS[selected_joint]
                angles[selected_joint] = clamp(0.0, lo, hi)
                send_joint(ser, selected_joint, angles[selected_joint], args.speed, args.acc)
                add_log(f"J{selected_joint} homed to 0°")

            elif key == ord('H'):
                for jid in range(1, 6):
                    lo, hi = JOINT_LIMITS[jid]
                    angles[jid] = clamp(0.0, lo, hi)
                    send_joint(ser, jid, angles[jid], args.speed, args.acc)
                    time.sleep(0.02)
                add_log("All joints homed to 0°")

            elif key == ord('s'):
                if output_writer:
                    # Map joint IDs back to SO-101 joint names
                    id_to_so101 = {v: k for k, v in SO101_TO_ROARM_ID.items()}
                    row = {}
                    for jid in range(1, 6):
                        so101_name = id_to_so101.get(jid)
                        if so101_name:
                            # Reverse: degrees → raw steps (approximate)
                            raw_steps = (angles[jid] / 360.0) * SO101_STEPS_PER_REV + SO101_CENTRE
                            row[so101_name] = round(raw_steps, 1)
                    output_writer.writerow(row)
                    output_file_handle.flush()
                    saved_frames += 1
                    add_log(f"Saved frame {saved_frames} → {args.output}")
                else:
                    add_log("No --output file specified; cannot save (restart with --output)")

            # ---- Drawing ----
            stdscr.erase()
            h, w = stdscr.getmaxyx()

            # Title bar
            title = " RoArm V3 — Keyboard Control "
            stdscr.addstr(0, max(0, (w - len(title)) // 2), title, C_TTL)
            stdscr.addstr(1, 0, "─" * (w - 1), C_LBL)

            # Connection info
            conn = f" Port: {args.port}  Baud: {args.baud}  Speed: {args.speed}  Acc: {args.acc} "
            stdscr.addstr(2, 1, conn, C_LBL)

            # Joint table header
            stdscr.addstr(4, 2, "  #  Joint                        Angle     Limits", C_LBL)
            stdscr.addstr(5, 2, "─" * min(54, w - 4), C_LBL)

            for jid in range(1, 6):
                lo, hi = JOINT_LIMITS[jid]
                name  = JOINT_NAMES[jid]
                angle = angles[jid]
                row   = 6 + (jid - 1)

                # Bar fill: map angle to 0-20 range
                bar_range = hi - lo if hi != lo else 1
                bar_pos = int((angle - lo) / bar_range * 20)
                bar = "█" * bar_pos + "░" * (20 - bar_pos)

                is_sel = (jid == selected_joint)
                marker = "►" if is_sel else " "
                style  = C_SEL if is_sel else C_VAL

                line = f" {marker} {jid}  {name:<28} {angle:+7.1f}°  [{lo:.0f}°,{hi:.0f}°]"
                if row < h - 1:
                    stdscr.addstr(row, 2, line, style)
                    if not is_sel and w > 80:
                        stdscr.addstr(row, 2 + len(line), f"  {bar}", C_LBL)

            # Step size display
            step_row = 12
            stdscr.addstr(step_row, 2, f" Step size: ", C_LBL)
            stdscr.addstr(step_row, 14, f"{step:.1f}°", C_VAL)
            stdscr.addstr(step_row, 20, "  [[ halve   ]] double   ,. fine", C_LBL)

            # Save counter
            if args.output:
                stdscr.addstr(step_row + 1, 2,
                              f" Saved frames: {saved_frames}  (s to save, output: {args.output})",
                              C_LBL)

            # Controls cheat-sheet
            ctrl_row = step_row + 3
            controls = [
                (" 1-5", "Select joint"),
                (" ←→ / ↑↓", "Move joint"),
                (" [ ]", "Step ÷2 / ×2"),
                (" h", "Home joint"),
                (" H", "Home all"),
                (" s", "Save frame"),
                (" q/Esc", "Quit"),
            ]
            stdscr.addstr(ctrl_row, 2, "Controls:", C_LBL)
            for ci, (key_str, desc) in enumerate(controls):
                col = 2 + ci * 15
                if col + 14 < w:
                    stdscr.addstr(ctrl_row + 1, col, key_str, C_VAL)
                    stdscr.addstr(ctrl_row + 1, col + len(key_str), f" {desc}", C_LBL)

            # Log panel
            log_top = ctrl_row + 3
            if log_top + 2 < h:
                stdscr.addstr(log_top, 2, "─" * min(60, w - 4), C_LBL)
                stdscr.addstr(log_top, 4, " Log ", C_LBL)
                for li, msg in enumerate(log_messages[-8:]):
                    log_row = log_top + 1 + li
                    if log_row < h - 1:
                        stdscr.addstr(log_row, 3, msg[:w - 4], C_LOG)

            stdscr.refresh()

            if quit_requested:
                break

    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        curses.wrapper(draw, ser)

    if output_file_handle:
        output_file_handle.close()

    print("Keyboard control ended.")


# ---------------------------------------------------------------------------
# CSV replay mode (original functionality)
# ---------------------------------------------------------------------------

def replay_mode(args) -> None:
    """Replay SO-101 recorded positions onto the RoArm V3."""

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

        period    = 1.0 / args.hz
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

            elapsed    = time.perf_counter() - t_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\nPlayback complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay SO-101 positions on RoArm V3, or control it live via keyboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input",    type=Path,
                      help="CSV file produced by collect_so101.py (replay mode)")
    mode.add_argument("--keyboard", action="store_true",
                      help="Live keyboard control mode (no CSV required)")

    parser.add_argument("--port",   type=str,   default="/dev/ttyUSB1",
                        help="Serial port the RoArm V3 is connected to (default: /dev/ttyUSB1)")
    parser.add_argument("--baud",   type=int,   default=115200,
                        help="Baud rate (default: 115200)")
    parser.add_argument("--speed",  type=int,   default=150,
                        help="RoArm joint speed 0-1000 (default: 150 — start slow!)")
    parser.add_argument("--acc",    type=int,   default=10,
                        help="RoArm joint acceleration 0-100 (default: 10)")

    # Replay-only options
    parser.add_argument("--hz",     type=float, default=30.0,
                        help="[replay] Playback frequency in Hz (default: 30)")
    parser.add_argument("--delay",  type=float, default=2.0,
                        help="[replay] Seconds to wait before playback starts (default: 2)")

    # Keyboard-only options
    parser.add_argument("--output", type=str,   default=None,
                        help="[keyboard] CSV file to append saved frames to (press s to save)")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.keyboard:
        keyboard_mode(args)
    else:
        replay_mode(args)


if __name__ == "__main__":
    main()