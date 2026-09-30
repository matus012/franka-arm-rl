"""Phase 7 curriculum video (CPU): concatenates one close-up episode per phase (2 -> 6), each recorded by
scripts/phased_videos.py --mode closeup --num_envs 1 on its own world, with a title bar naming the phase and its world.
Input: videos/curriculum/p<k>/p<k>_closeup_env0.mp4 ; output: videos/curriculum/curriculum_phase2_to_6.mp4 (50 fps)."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "videos" / "curriculum"
TITLES = {2: "phase 2: fixed cube, fixed target", 3: "phase 3: + random cube position",
          4: "phase 4: + random cube yaw", 5: "phase 5: + random target", 6: "phase 6: + random cube mass / friction"}
try:
    FONT = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 40)
except OSError:
    FONT = ImageFont.load_default(size=40)

w = imageio.get_writer(str(D / "curriculum_phase2_to_6.mp4"), fps=50, codec="libx264", quality=8, macro_block_size=8)
for k, title in TITLES.items():
    src = D / f"p{k}" / f"p{k}_closeup_env0.mp4"
    if not src.exists():
        print("missing", src)
        continue
    r = imageio.get_reader(str(src))
    for i, fr in enumerate(r):
        if i >= 200:  # first 4 s of each episode (the whole task happens in ~2 s; then HOLD)
            break
        im = Image.fromarray(fr)
        dr = ImageDraw.Draw(im, "RGBA")
        bb = dr.textbbox((0, 0), title, font=FONT)
        x = (im.width - (bb[2] - bb[0])) // 2
        dr.rectangle([x - 16, im.height - 80, x + (bb[2] - bb[0]) + 16, im.height - 16], fill=(0, 0, 0, 170))
        dr.text((x, im.height - 74), title, font=FONT, fill=(255, 255, 255, 255))
        w.append_data(np.asarray(im))
    r.close()
w.close()
print("wrote", D / "curriculum_phase2_to_6.mp4")
