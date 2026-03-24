#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Optional

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageSequence


def ensure_even(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    new_h = h - (h % 2)
    new_w = w - (w % 2)
    if new_h != h or new_w != w:
        frame = frame[:new_h, :new_w]
    return frame


def read_gif_frames(gif_path: Path) -> tuple[list[np.ndarray], list[float]]:
    frames: List[np.ndarray] = []
    durations: List[float] = []
    with Image.open(gif_path) as im:
        for frame in ImageSequence.Iterator(im):
            rgba = frame.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, (255, 255, 255))
            rgb.paste(rgba, mask=rgba.split()[-1])
            arr = np.array(rgb)
            arr = ensure_even(arr)
            frames.append(arr)
            duration_ms = frame.info.get("duration", im.info.get("duration", 100))
            durations.append(max(0.02, float(duration_ms) / 1000.0))
    if not frames:
        raise RuntimeError(f"No frames found in GIF: {gif_path}")
    return frames, durations


def fps_from_durations(durations: list[float], default_fps: float) -> float:
    if not durations:
        return default_fps
    mean_dt = max(1e-3, float(np.mean(durations)))
    fps = 1.0 / mean_dt
    return max(1.0, min(60.0, fps))


def write_mp4_from_gif(gif_path: Path, out_path: Path, fps_override: Optional[float] = None) -> None:
    frames, durations = read_gif_frames(gif_path)
    fps = fps_override if fps_override and fps_override > 0 else fps_from_durations(durations, 4.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=2) as writer:
        for frame, dur in zip(frames, durations):
            repeat = 1 if fps_override else max(1, int(round(dur * fps)))
            for _ in range(repeat):
                writer.append_data(frame)


def collect_gifs(root: Path) -> List[Path]:
    return sorted(root.rglob("*.gif"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert progressive GIFs to MP4 videos.")
    parser.add_argument("--gif", type=str, default=None, help="Single GIF file to convert.")
    parser.add_argument("--input-dir", type=str, default=None, help="Root directory to recursively search for GIFs.")
    parser.add_argument("--fps", type=float, default=0.0, help="Override output FPS. 0 means infer from GIF duration.")
    parser.add_argument("--suffix", type=str, default="_video", help="Suffix added before .mp4")
    args = parser.parse_args()

    if not args.gif and not args.input_dir:
        raise SystemExit("Please provide either --gif or --input-dir")

    if args.gif:
        gif_path = Path(args.gif)
        if not gif_path.exists():
            raise FileNotFoundError(gif_path)
        out_path = gif_path.with_name(gif_path.stem + args.suffix + ".mp4")
        write_mp4_from_gif(gif_path, out_path, fps_override=args.fps if args.fps > 0 else None)
        print(f"Converted: {gif_path} -> {out_path}")
        return

    root = Path(args.input_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    gifs = collect_gifs(root)
    if not gifs:
        print(f"No GIF files found under: {root}")
        return

    for gif_path in gifs:
        out_path = gif_path.with_name(gif_path.stem + args.suffix + ".mp4")
        try:
            write_mp4_from_gif(gif_path, out_path, fps_override=args.fps if args.fps > 0 else None)
            print(f"Converted: {gif_path} -> {out_path}")
        except Exception as e:
            print(f"Failed: {gif_path} ({e})")


if __name__ == "__main__":
    main()
