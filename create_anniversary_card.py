#!/usr/bin/env python3
"""Create a 30th anniversary photo collage card in the shape of '30'.

Photos form a continuous mosaic (no gaps, no circular cutouts) clipped
to a clear "30" silhouette.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, ImageChops

ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")
OUT_DIR = Path("/workspace")
ARTIFACTS = Path("/opt/cursor/artifacts")

W, H = 2400, 3200
COLLAGE_TOP = 160
COLLAGE_H = 2120
TEXT_TOP = 2380

BG_TOP = (255, 246, 248)
BG_BOT = (252, 236, 230)
ACCENT = (140, 55, 75)
SOFT_LINE = (210, 160, 170)


def load_photos() -> list[Image.Image]:
    paths = sorted(ASSETS.glob("*.jpg"))
    if len(paths) != 46:
        raise SystemExit(f"Expected 46 photos, found {len(paths)}")
    photos = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = ImageEnhance.Color(img).enhance(1.06)
        img = ImageEnhance.Contrast(img).enhance(1.04)
        photos.append(img)
    return photos


def make_background() -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG_TOP)
    draw = ImageDraw.Draw(canvas)
    for y in range(H):
        t = y / (H - 1)
        r = int(BG_TOP[0] * (1 - t) + BG_BOT[0] * t)
        g = int(BG_TOP[1] * (1 - t) + BG_BOT[1] * t)
        b = int(BG_TOP[2] * (1 - t) + BG_BOT[2] * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    rng = random.Random(30)
    noise = Image.new("L", (W // 2, H // 2), 128)
    nd = noise.load()
    for y in range(noise.height):
        for x in range(noise.width):
            nd[x, y] = 118 + rng.randint(0, 20)
    noise = noise.resize((W, H), Image.Resampling.BILINEAR)
    canvas = Image.composite(
        canvas,
        ImageOps.colorize(noise, (240, 220, 225), (255, 250, 252)),
        Image.new("L", (W, H), 28),
    )
    return canvas


def make_thirty_mask(width: int, height: int) -> Image.Image:
    """Solid continuous '30' silhouette from a heavy sans glyph.

    Digits are drawn separately with a clear gap, then lightly dilated so each
    digit is one unbroken filled region (no cracks for the background to show).
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not Path(font_path).exists():
        font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    font = ImageFont.truetype(font_path, size=int(height * 0.92))

    # Measure each digit so we can space them apart cleanly
    b3 = draw.textbbox((0, 0), "3", font=font)
    b0 = draw.textbbox((0, 0), "0", font=font)
    w3, h3 = b3[2] - b3[0], b3[3] - b3[1]
    w0, h0 = b0[2] - b0[0], b0[3] - b0[1]
    gap = int(width * 0.055)
    total_w = w3 + gap + w0
    digit_h = max(h3, h0)
    x0 = (width - total_w) // 2
    y0 = (height - digit_h) // 2 - int(height * 0.02)

    draw.text((x0 - b3[0], y0 - b3[1]), "3", fill=255, font=font)
    draw.text((x0 + w3 + gap - b0[0], y0 - b0[1]), "0", fill=255, font=font)

    # Thicken just enough to close antialias pinholes — keep digits separate
    for _ in range(2):
        mask = mask.filter(ImageFilter.MaxFilter(9))
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    mask = mask.point(lambda p: 255 if p > 60 else 0)
    return mask


def score_photo(photo: Image.Image) -> float:
    thumb = photo.resize((48, 48), Image.Resampling.BILINEAR)
    px = thumb.load()
    luma = [
        0.299 * px[x, y][0] + 0.587 * px[x, y][1] + 0.114 * px[x, y][2]
        for y in range(thumb.height)
        for x in range(thumb.width)
    ]
    mean = sum(luma) / len(luma)
    var = sum((v - mean) ** 2 for v in luma) / len(luma)
    return mean + 0.12 * math.sqrt(var)


Rect = tuple[int, int, int, int]  # x0, y0, x1, y1


def _rect_mask_coverage(mask_arr: np.ndarray, rect: Rect) -> float:
    x0, y0, x1, y1 = rect
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = mask_arr[y0:y1, x0:x1]
    return float(patch.mean()) / 255.0 if patch.size else 0.0


def _split_rect(rect: Rect, rng: random.Random) -> tuple[Rect, Rect] | None:
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    if w < 40 or h < 40:
        return None
    # Prefer splitting the longer axis; keep both kids decently sized
    if w >= h and w >= 70:
        cut = int(x0 + w * rng.uniform(0.32, 0.68))
        cut = min(max(cut, x0 + 28), x1 - 28)
        return (x0, y0, cut, y1), (cut, y0, x1, y1)
    if h >= 70:
        cut = int(y0 + h * rng.uniform(0.32, 0.68))
        cut = min(max(cut, y0 + 28), y1 - 28)
        return (x0, y0, x1, cut), (x0, cut, x1, y1)
    return None


def _digit_bboxes(mask_arr: np.ndarray) -> list[Rect]:
    """Bounding boxes for the '3' and '0' (two largest x-separated blobs)."""
    ys, xs = np.where(mask_arr > 0)
    if len(xs) == 0:
        return []
    # Column occupancy — find the gap between digits
    col = mask_arr.any(axis=0)
    runs: list[tuple[int, int]] = []
    start = None
    for x, on in enumerate(col):
        if on and start is None:
            start = x
        elif not on and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, mask_arr.shape[1]))
    boxes: list[Rect] = []
    for x0, x1 in runs:
        band = mask_arr[:, x0:x1]
        rows = np.where(band.any(axis=1))[0]
        if len(rows) == 0:
            continue
        boxes.append((x0, int(rows.min()), x1, int(rows.max()) + 1))
    # Keep the two widest runs (3 and 0); ignore noise
    boxes.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
    return boxes[:2]


def rectangular_tiles(mask_arr: np.ndarray, count: int, seed: int = 2030) -> list[Rect]:
    """BSP rectangular mosaic per digit — abutting tiles, zero interior gaps."""
    rng = random.Random(seed)
    boxes = _digit_bboxes(mask_arr)
    if not boxes:
        raise SystemExit("Could not locate digit regions in mask")

    # Allocate tile counts by digit area
    areas = [max(1, (b[2] - b[0]) * (b[3] - b[1])) for b in boxes]
    total_a = sum(areas)
    quotas = [max(8, round(count * a / total_a)) for a in areas]
    # Fix rounding to exact count
    while sum(quotas) > count:
        quotas[quotas.index(max(quotas))] -= 1
    while sum(quotas) < count:
        quotas[quotas.index(min(quotas))] += 1

    all_tiles: list[Rect] = []
    for box, quota in zip(boxes, quotas):
        tiles: list[Rect] = [box]
        guard = 0
        while len(tiles) < quota and guard < 4000:
            guard += 1
            scored = []
            for i, r in enumerate(tiles):
                x0, y0, x1, y1 = r
                area = (x1 - x0) * (y1 - y0)
                cov = _rect_mask_coverage(mask_arr, r)
                if area < 48 * 48 or cov < 0.1:
                    continue
                scored.append((area * (0.4 + cov), i))
            if not scored:
                break
            scored.sort(reverse=True)
            pick = scored[rng.randrange(min(4, len(scored)))][1]
            pair = _split_rect(tiles[pick], rng)
            if pair is None:
                # try next best
                ok = False
                for _, j in scored[1:6]:
                    pair = _split_rect(tiles[j], rng)
                    if pair is not None:
                        pick = j
                        ok = True
                        break
                if not ok:
                    break
            a, b = pair
            tiles.pop(pick)
            tiles.extend([a, b])

        # Prefer tiles that actually cover the digit
        tiles = [r for r in tiles if _rect_mask_coverage(mask_arr, r) >= 0.06]
        while len(tiles) < quota:
            scored = sorted(
                (
                    ((r[2] - r[0]) * (r[3] - r[1]), i)
                    for i, r in enumerate(tiles)
                    if (r[2] - r[0]) >= 56 and (r[3] - r[1]) >= 56
                ),
                reverse=True,
            )
            if not scored:
                break
            i = scored[0][1]
            pair = _split_rect(tiles[i], rng)
            if pair is None:
                break
            tiles.pop(i)
            tiles.extend(pair)
        all_tiles.extend(tiles[:quota])

    # Top up / trim to exact count across both digits
    while len(all_tiles) < count:
        scored = sorted(
            (
                ((r[2] - r[0]) * (r[3] - r[1]), i)
                for i, r in enumerate(all_tiles)
                if (r[2] - r[0]) >= 56 and (r[3] - r[1]) >= 56
            ),
            reverse=True,
        )
        if not scored:
            break
        i = scored[0][1]
        pair = _split_rect(all_tiles[i], rng)
        if pair is None:
            break
        all_tiles.pop(i)
        all_tiles.extend(pair)

    while len(all_tiles) > count:
        all_tiles.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
        a, b = all_tiles[0], all_tiles[1]
        merged = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
        all_tiles = [merged] + all_tiles[2:]

    return all_tiles[:count]


def build_mosaic(mask: Image.Image, photos: list[Image.Image]) -> Image.Image:
    """Continuous rectangular photo mosaic clipped to the 30 silhouette."""
    mask_arr = np.array(mask)
    tiles = rectangular_tiles(mask_arr, len(photos), seed=2030)
    if len(tiles) < len(photos):
        raise SystemExit(f"Only produced {len(tiles)} tiles for {len(photos)} photos")

    # Larger tiles get clearer photos
    tiles_sorted = sorted(
        tiles, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True
    )
    ranked_photos = sorted(
        range(len(photos)), key=lambda i: score_photo(photos[i]), reverse=True
    )

    rgb = Image.new("RGB", mask.size, (0, 0, 0))
    # Draw largest first, then smaller on top with 1px overlap via expanded paste
    for rect, pi in zip(tiles_sorted, ranked_photos):
        x0, y0, x1, y1 = rect
        # Overlap neighbors by 2px so JPEG/AA never reveals background hairlines
        x0e, y0e = max(0, x0 - 2), max(0, y0 - 2)
        x1e = min(mask.size[0], x1 + 2)
        y1e = min(mask.size[1], y1 + 2)
        bw, bh = x1e - x0e, y1e - y0e
        fitted = ImageOps.fit(
            photos[pi],
            (bw, bh),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.42),
        )
        rgb.paste(fitted, (x0e, y0e))

    hard = mask.point(lambda p: 255 if p > 0 else 0)
    edge = hard.filter(ImageFilter.GaussianBlur(0.7))
    out = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    out.paste(rgb, (0, 0))
    out.putalpha(edge)
    return out


def draw_message(canvas: Image.Image) -> None:
    draw = ImageDraw.Draw(canvas)
    title_path = "/usr/share/fonts/truetype/noto/NotoSerifDisplay-BoldItalic.ttf"
    body_path = "/usr/share/fonts/truetype/noto/NotoSerifDisplay-Regular.ttf"
    if not Path(title_path).exists():
        title_path = "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf"
    if not Path(body_path).exists():
        body_path = "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"

    title_font = ImageFont.truetype(title_path, 72)
    body_font = ImageFont.truetype(body_path, 44)
    sign_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf", 48
    )

    cx = W // 2
    draw.line([(cx - 220, TEXT_TOP - 40), (cx + 220, TEXT_TOP - 40)], fill=SOFT_LINE, width=2)
    d = 7
    draw.polygon(
        [
            (cx, TEXT_TOP - 40 - d),
            (cx + d, TEXT_TOP - 40),
            (cx, TEXT_TOP - 40 + d),
            (cx - d, TEXT_TOP - 40),
        ],
        fill=ACCENT,
    )

    title = "Happy Anniversary!"
    tb = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((W - (tb[2] - tb[0])) // 2, TEXT_TOP), title, font=title_font, fill=ACCENT)

    lines = [
        "If I had to do it again,",
        "I would still choose you.",
        "I love you.",
    ]
    y = TEXT_TOP + 110
    for line in lines:
        lb = draw.textbbox((0, 0), line, font=body_font)
        draw.text(((W - (lb[2] - lb[0])) // 2, y), line, font=body_font, fill=(90, 55, 65))
        y += 62

    sign = "— Jason"
    sb = draw.textbbox((0, 0), sign, font=sign_font)
    draw.text(((W - (sb[2] - sb[0])) // 2, y + 28), sign, font=sign_font, fill=ACCENT)

    foot = ImageFont.truetype(
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 28
    )
    label = "30 Years of Love"
    fb = draw.textbbox((0, 0), label, font=foot)
    draw.text(((W - (fb[2] - fb[0])) // 2, H - 90), label, font=foot, fill=(170, 120, 130))


def build_card() -> Path:
    photos = load_photos()
    canvas = make_background().convert("RGBA")

    collage_w, collage_h = W - 120, COLLAGE_H
    mask = make_thirty_mask(collage_w, collage_h)
    mosaic = build_mosaic(mask, photos)

    # Soft outer shadow only (outside the digits) so the silhouette lifts off the paper
    shadow = mask.filter(ImageFilter.GaussianBlur(18)).point(lambda p: int(p * 0.28))
    # Knock out the digit interior so shadow doesn't tint photo seams
    shadow = ImageChops.subtract(shadow, mask.point(lambda p: 255 if p > 0 else 0))
    shade = Image.new("RGBA", (collage_w, collage_h), (90, 40, 55, 0))
    shade.putalpha(shadow)

    ox = (W - collage_w) // 2
    canvas.alpha_composite(shade, (ox, COLLAGE_TOP + 6))
    canvas.alpha_composite(mosaic, (ox, COLLAGE_TOP))

    final = canvas.convert("RGB")
    draw_message(final)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_jpg = OUT_DIR / "anniversary_30_card.jpg"
    out_png = OUT_DIR / "anniversary_30_card.png"
    final.save(out_jpg, quality=95, optimize=True)
    final.save(out_png, optimize=True)
    final.save(ARTIFACTS / "anniversary_30_card.jpg", quality=95, optimize=True)
    final.save(ARTIFACTS / "anniversary_30_card.png", optimize=True)

    # QA crops
    final.crop((0, 140, 2400, 2320)).resize((900, 820), Image.Resampling.LANCZOS).save(
        ARTIFACTS / "qa_collage_region.jpg", quality=90
    )
    return out_jpg


if __name__ == "__main__":
    # numpy is required for Voronoi mosaic
    path = build_card()
    print(f"Wrote {path}")
