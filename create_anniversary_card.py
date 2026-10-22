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

# Wide print canvas matched to the "30" aspect so digits can fill the page
# (tall enough and wide enough that neither side gets cropped).
W, H = 5200, 3800
SIDE_PAD = 36
COLLAGE_TOP = 24
# Compact message band; nearly all of the page is the full "30"
TEXT_BAND = 480
COLLAGE_H = H - COLLAGE_TOP - TEXT_BAND
TEXT_TOP = COLLAGE_TOP + COLLAGE_H + 20

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
    """Solid continuous '30' scaled to fill the collage area (fully visible).

    Digits are drawn large, thickened, then uniformly scaled up so the full
    "30" uses nearly the entire width and height — nothing cropped off-page.
    """
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not Path(font_path).exists():
        font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

    # Render oversized, then scale-to-fit so both digits stay fully on-canvas
    render = Image.new("L", (width * 2, height * 2), 0)
    draw = ImageDraw.Draw(render)
    font = ImageFont.truetype(font_path, size=int(height * 1.7))

    b3 = draw.textbbox((0, 0), "3", font=font)
    b0 = draw.textbbox((0, 0), "0", font=font)
    w3, h3 = b3[2] - b3[0], b3[3] - b3[1]
    w0, h0 = b0[2] - b0[0], b0[3] - b0[1]
    gap = max(int(min(w3, w0) * 0.07), 28)
    total_w = w3 + gap + w0
    digit_h = max(h3, h0)
    x_origin = (render.width - total_w) // 2
    y_origin = (render.height - digit_h) // 2

    draw.text((x_origin - b3[0], y_origin - b3[1]), "3", fill=255, font=font)
    draw.text((x_origin + w3 + gap - b0[0], y_origin - b0[1]), "0", fill=255, font=font)

    # Heavy thicken so digit strokes are wide — each photo gets a larger tile
    for _ in range(5):
        render = render.filter(ImageFilter.MaxFilter(13))
    render = render.filter(ImageFilter.GaussianBlur(1.0))
    render = render.point(lambda p: 255 if p > 50 else 0)

    bbox = render.getbbox()
    if bbox is None:
        raise SystemExit("Failed to render 30 mask")
    cropped = render.crop(bbox)

    # Fit into collage with a slim safety pad so edges never clip
    pad = max(12, int(min(width, height) * 0.012))
    fit_w, fit_h = width - 2 * pad, height - 2 * pad
    scale = min(fit_w / cropped.width, fit_h / cropped.height)
    new_w = max(1, int(cropped.width * scale))
    new_h = max(1, int(cropped.height * scale))
    scaled = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    scaled = scaled.point(lambda p: 255 if p > 128 else 0)

    mask = Image.new("L", (width, height), 0)
    ox = (width - new_w) // 2
    oy = (height - new_h) // 2
    mask.paste(scaled, (ox, oy))
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


def place_seeds(mask_arr: np.ndarray, count: int, seed: int = 2030) -> np.ndarray:
    """Seeds inside the mask with spacing variety (large + small photo regions)."""
    rng = np.random.default_rng(seed)
    ys, xs = np.where(mask_arr > 0)
    pts: list[tuple[float, float]] = []
    min_dist = max(48.0, math.sqrt(len(xs) / (count * 2.0)))
    attempts = 0
    while len(pts) < count // 2 and attempts < 90000:
        attempts += 1
        i = int(rng.integers(0, len(xs)))
        x, y = float(xs[i]), float(ys[i])
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist**2 for px, py in pts):
            pts.append((x, y))
        if attempts % 14000 == 0:
            min_dist *= 0.9
    min_dist2 = min_dist * 0.42
    attempts = 0
    while len(pts) < count and attempts < 120000:
        attempts += 1
        i = int(rng.integers(0, len(xs)))
        x, y = float(xs[i]), float(ys[i])
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist2**2 for px, py in pts):
            pts.append((x, y))
        if attempts % 16000 == 0:
            min_dist2 *= 0.86
    while len(pts) < count:
        i = int(rng.integers(0, len(xs)))
        pts.append((float(xs[i]), float(ys[i])))
    return np.array(pts[:count], dtype=np.float64)


def voronoi_labels(mask_arr: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    h, w = mask_arr.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    ys, xs = np.where(mask_arr > 0)
    chunk = 25000
    sx, sy = seeds[:, 0], seeds[:, 1]
    for start in range(0, len(xs), chunk):
        end = min(start + chunk, len(xs))
        px = xs[start:end].astype(np.float64)[:, None]
        py = ys[start:end].astype(np.float64)[:, None]
        nearest = np.argmin((px - sx) ** 2 + (py - sy) ** 2, axis=1)
        labels[ys[start:end], xs[start:end]] = nearest
    return labels


def build_mosaic(mask: Image.Image, photos: list[Image.Image]) -> Image.Image:
    """Seamless Voronoi photo fill — every mask pixel belongs to exactly one photo."""
    mask_arr = np.array(mask)
    seeds = place_seeds(mask_arr, len(photos), seed=2030)
    labels = voronoi_labels(mask_arr, seeds)

    areas = sorted(
        ((int(np.sum(labels == i)), i) for i in range(len(photos))), reverse=True
    )
    ranked = sorted(
        range(len(photos)), key=lambda i: score_photo(photos[i]), reverse=True
    )
    photo_for = {rid: ranked[rank] for rank, (_, rid) in enumerate(areas)}

    # Accumulate with soft-edge overlap so photo joins look continuous, not gridded
    acc = np.zeros((mask.size[1], mask.size[0], 3), dtype=np.float64)
    weight = np.zeros((mask.size[1], mask.size[0]), dtype=np.float64)

    for rid in range(len(photos)):
        ys, xs = np.where(labels == rid)
        if len(xs) == 0:
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        # Expand for soft blend apron
        pad = 4
        x0e, y0e = max(0, x0 - pad), max(0, y0 - pad)
        x1e = min(mask.size[0], x1 + pad)
        y1e = min(mask.size[1], y1 + pad)
        bw, bh = x1e - x0e, y1e - y0e

        fitted = np.asarray(
            ImageOps.fit(
                photos[photo_for[rid]],
                (bw, bh),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.45),
            ),
            dtype=np.float64,
        )

        cell = (labels[y0e:y1e, x0e:x1e] == rid).astype(np.uint8) * 255
        cell_img = Image.fromarray(cell, mode="L").filter(ImageFilter.MaxFilter(7))
        # Soft falloff at cell edge for blended seams
        soft = cell_img.filter(ImageFilter.GaussianBlur(1.8))
        wts = np.asarray(soft, dtype=np.float64) / 255.0

        acc[y0e:y1e, x0e:x1e] += fitted * wts[..., None]
        weight[y0e:y1e, x0e:x1e] += wts

    weight = np.maximum(weight, 1e-6)
    rgb_arr = np.clip(acc / weight[..., None], 0, 255).astype(np.uint8)
    # Only keep pixels inside the digit (zero out exterior before alpha)
    inside = mask_arr > 0
    rgb_arr[~inside] = 0
    rgb = Image.fromarray(rgb_arr, mode="RGB")

    hard = mask.point(lambda p: 255 if p > 0 else 0)
    edge = hard.filter(ImageFilter.GaussianBlur(0.6))
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

    title_font = ImageFont.truetype(title_path, 78)
    body_font = ImageFont.truetype(body_path, 46)
    sign_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf", 50
    )

    cx = W // 2
    rule_y = TEXT_TOP - 8
    draw.line([(cx - 240, rule_y), (cx + 240, rule_y)], fill=SOFT_LINE, width=2)
    d = 7
    draw.polygon(
        [
            (cx, rule_y - d),
            (cx + d, rule_y),
            (cx, rule_y + d),
            (cx - d, rule_y),
        ],
        fill=ACCENT,
    )

    title = "Happy Anniversary!"
    tb = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((W - (tb[2] - tb[0])) // 2, TEXT_TOP + 12), title, font=title_font, fill=ACCENT)

    lines = [
        "If I had to do it again,",
        "I would still choose you.",
        "I love you.",
    ]
    y = TEXT_TOP + 110
    for line in lines:
        lb = draw.textbbox((0, 0), line, font=body_font)
        draw.text(((W - (lb[2] - lb[0])) // 2, y), line, font=body_font, fill=(90, 55, 65))
        y += 58

    sign = "— Jason"
    sb = draw.textbbox((0, 0), sign, font=sign_font)
    draw.text(((W - (sb[2] - sb[0])) // 2, y + 20), sign, font=sign_font, fill=ACCENT)

    foot = ImageFont.truetype(
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 30
    )
    label = "30 Years of Love"
    fb = draw.textbbox((0, 0), label, font=foot)
    draw.text(((W - (fb[2] - fb[0])) // 2, H - 70), label, font=foot, fill=(170, 120, 130))


def build_card() -> Path:
    photos = load_photos()
    canvas = make_background().convert("RGBA")

    collage_w, collage_h = W - 2 * SIDE_PAD, COLLAGE_H
    mask = make_thirty_mask(collage_w, collage_h)
    bbox = mask.getbbox()
    if bbox:
        # Confirm full silhouette sits inside the collage (no edge clipping)
        mx0, my0, mx1, my1 = bbox
        assert mx0 >= 2 and my0 >= 2 and mx1 <= collage_w - 2 and my1 <= collage_h - 2, (
            f"30 mask clipped: bbox={bbox} canvas=({collage_w},{collage_h})"
        )
        print(
            f"30 size {mx1 - mx0}x{my1 - my0} in collage {collage_w}x{collage_h} "
            f"({100 * (mx1 - mx0) / collage_w:.0f}% wide, {100 * (my1 - my0) / collage_h:.0f}% tall)"
        )

    mosaic = build_mosaic(mask, photos)

    # Soft outer shadow only (outside the digits) so the silhouette lifts off the paper
    shadow = mask.filter(ImageFilter.GaussianBlur(22)).point(lambda p: int(p * 0.26))
    shadow = ImageChops.subtract(shadow, mask.point(lambda p: 255 if p > 0 else 0))
    shade = Image.new("RGBA", (collage_w, collage_h), (90, 40, 55, 0))
    shade.putalpha(shadow)

    ox = SIDE_PAD
    canvas.alpha_composite(shade, (ox, COLLAGE_TOP + 8))
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

    preview = final.resize((750, int(750 * H / W)), Image.Resampling.LANCZOS)
    preview.save(ARTIFACTS / "anniversary_30_card_preview.jpg", quality=92)
    final.crop((0, 0, W, COLLAGE_TOP + COLLAGE_H + 20)).resize(
        (900, int(900 * (COLLAGE_TOP + COLLAGE_H + 20) / W)), Image.Resampling.LANCZOS
    ).save(ARTIFACTS / "qa_collage_region.jpg", quality=90)
    return out_jpg


if __name__ == "__main__":
    # numpy is required for Voronoi mosaic
    path = build_card()
    print(f"Wrote {path}")
