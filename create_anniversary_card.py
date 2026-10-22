#!/usr/bin/env python3
"""Create a 30th anniversary photo collage card in the shape of '30'."""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

ASSETS = Path("/home/ubuntu/.cursor/projects/workspace/assets")
OUT_DIR = Path("/workspace")
ARTIFACTS = Path("/opt/cursor/artifacts")

# Portrait card, print-friendly resolution
W, H = 2400, 3200
COLLAGE_TOP = 180
COLLAGE_H = 2100
TEXT_TOP = 2380

BG_TOP = (255, 246, 248)  # soft blush
BG_BOT = (252, 236, 230)  # warm rose cream
ACCENT = (140, 55, 75)  # deep rose
SOFT_LINE = (210, 160, 170)


def load_photos() -> list[Image.Image]:
    paths = sorted(ASSETS.glob("*.jpg"))
    if len(paths) != 46:
        raise SystemExit(f"Expected 46 photos, found {len(paths)}")
    photos = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        # Mild warm lift so collage feels cohesive
        img = ImageEnhance.Color(img).enhance(1.08)
        img = ImageEnhance.Contrast(img).enhance(1.05)
        photos.append(img)
    return photos


def make_background() -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG_TOP)
    draw = ImageDraw.Draw(canvas)
    for y in range(H):
        t = y / (H - 1)
        # Soft vertical wash + faint vignette-friendly blend
        r = int(BG_TOP[0] * (1 - t) + BG_BOT[0] * t)
        g = int(BG_TOP[1] * (1 - t) + BG_BOT[1] * t)
        b = int(BG_TOP[2] * (1 - t) + BG_BOT[2] * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    # Subtle paper grain
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
    """Bold filled '30' mask for packing photo circles.

    Drawn geometrically (not as a font) so stroke width stays wide enough
    for many photo medallions while the silhouette still reads clearly as 30.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Layout: two digits side by side with a small gap
    margin_x = int(width * 0.04)
    margin_y = int(height * 0.06)
    gap = int(width * 0.04)
    digit_w = (width - 2 * margin_x - gap) // 2
    digit_h = height - 2 * margin_y
    stroke = max(int(min(digit_w, digit_h) * 0.28), 90)

    def draw_three(ox: int, oy: int, dw: int, dh: int, t: int) -> None:
        # Open "3" made of two stacked arcs + a middle bar stub
        # Outer bounding box for the digit
        left, right = ox, ox + dw
        top, bottom = oy, oy + dh
        mid_y = (top + bottom) // 2
        # Upper bowl
        draw.arc(
            [left + t // 4, top, right, mid_y + t // 2],
            start=270,
            end=90,
            fill=255,
            width=t,
        )
        # Lower bowl
        draw.arc(
            [left + t // 4, mid_y - t // 2, right, bottom],
            start=270,
            end=90,
            fill=255,
            width=t,
        )
        # Soften the open left edge with short caps so packing stays dense
        draw.ellipse(
            [left + t // 4, top, left + t // 4 + t, top + t],
            fill=255,
        )
        draw.ellipse(
            [left + t // 4, bottom - t, left + t // 4 + t, bottom],
            fill=255,
        )
        draw.ellipse(
            [left + t // 3, mid_y - t // 2, left + t // 3 + t, mid_y + t // 2],
            fill=255,
        )

    # Draw 3
    from PIL import ImageChops

    three_x = margin_x
    draw_three(three_x, margin_y, digit_w, digit_h, stroke)

    # Draw 0 as a thick annulus
    zero_x = margin_x + digit_w + gap
    outer = Image.new("L", (width, height), 0)
    odraw = ImageDraw.Draw(outer)
    odraw.ellipse([zero_x, margin_y, zero_x + digit_w, margin_y + digit_h], fill=255)
    inner = Image.new("L", (width, height), 0)
    idraw = ImageDraw.Draw(inner)
    pad = stroke
    idraw.ellipse(
        [
            zero_x + pad,
            margin_y + pad,
            zero_x + digit_w - pad,
            margin_y + digit_h - pad,
        ],
        fill=255,
    )
    ring = ImageChops.subtract(outer, inner)
    mask = ImageChops.lighter(mask, ring)

    # Slight blur + rethreshold to round harsh corners for nicer packing
    mask = mask.filter(ImageFilter.MaxFilter(7))
    mask = mask.filter(ImageFilter.GaussianBlur(1.5))
    mask = mask.point(lambda p: 255 if p > 40 else 0)
    return mask


def circle_fits(
    cx: float,
    cy: float,
    r: float,
    mask: Image.Image,
    placed: list[tuple[float, float, float]],
    gap: float = 6.0,
) -> bool:
    w, h = mask.size
    if cx - r < 2 or cy - r < 2 or cx + r >= w - 2 or cy + r >= h - 2:
        return False
    # Sample points around and inside the circle against the mask
    px = mask.load()
    samples = [(cx, cy)]
    for i in range(16):
        a = 2 * math.pi * i / 16
        samples.append((cx + math.cos(a) * r * 0.92, cy + math.sin(a) * r * 0.92))
    for i in range(8):
        a = 2 * math.pi * i / 8
        samples.append((cx + math.cos(a) * r * 0.55, cy + math.sin(a) * r * 0.55))
    for sx, sy in samples:
        ix, iy = int(sx), int(sy)
        if ix < 0 or iy < 0 or ix >= w or iy >= h or px[ix, iy] < 128:
            return False
    for ox, oy, orad in placed:
        if math.hypot(cx - ox, cy - oy) < r + orad + gap:
            return False
    return True


def pack_circles(mask: Image.Image, count: int, seed: int = 30) -> list[tuple[float, float, float]]:
    """Greedy multi-pass circle packing inside the mask, varied sizes."""
    rng = random.Random(seed)
    w, h = mask.size
    # Prefer interior pixels of the glyph
    candidates = [
        (x, y)
        for y in range(0, h, 3)
        for x in range(0, w, 3)
        if mask.getpixel((x, y)) > 128
    ]
    rng.shuffle(candidates)

    # Size mix tuned so ~46 medallions fill a thick "30"
    scale = h / 2100
    radii_plan = (
        [int(r * scale) for r in (155, 145, 135, 125, 120, 115)]
        + [int(108 * scale)] * 8
        + [int(96 * scale)] * 10
        + [int(84 * scale)] * 12
        + [int(72 * scale)] * 12
        + [int(60 * scale)] * 12
    )

    placed: list[tuple[float, float, float]] = []

    for r in radii_plan:
        if len(placed) >= count:
            break
        for _ in range(1200):
            if not candidates:
                break
            cx, cy = candidates[rng.randrange(len(candidates))]
            cx += rng.uniform(-2, 2)
            cy += rng.uniform(-2, 2)
            if circle_fits(cx, cy, r, mask, placed, gap=4):
                placed.append((cx, cy, float(r)))
                break

    # Fill remaining with progressively smaller circles
    fill_r = int(58 * scale)
    while len(placed) < count and fill_r >= int(36 * scale):
        added = 0
        for _ in range(4000):
            if len(placed) >= count:
                break
            cx, cy = candidates[rng.randrange(len(candidates))]
            cx += rng.uniform(-4, 4)
            cy += rng.uniform(-4, 4)
            if circle_fits(cx, cy, fill_r, mask, placed, gap=3):
                placed.append((cx, cy, float(fill_r)))
                added += 1
                if added >= 4:
                    break
        if added == 0:
            fill_r -= 3
        elif rng.random() < 0.4:
            fill_r = max(int(36 * scale), fill_r - 2)

    return placed


def circular_crop(photo: Image.Image, diameter: int) -> Image.Image:
    """Center-weighted square crop into a circular RGBA tile."""
    side = max(diameter, 8)
    # Bias slightly upward — faces often sit in upper half of portraits
    img = ImageOps.fit(
        photo,
        (side, side),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.42),
    )
    mask = Image.new("L", (side, side), 0)
    ImageDraw.Draw(mask).ellipse((1, 1, side - 2, side - 2), fill=255)
    # Soft edge
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    out = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    out.paste(img, (0, 0))
    out.putalpha(mask)
    # Thin white ring for a polished photo-medallion look
    ring = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring)
    rd.ellipse((1, 1, side - 2, side - 2), outline=(255, 255, 255, 220), width=max(2, side // 55))
    out = Image.alpha_composite(out, ring)
    return out


def score_photo(photo: Image.Image) -> float:
    """Prefer brighter, higher-contrast images for larger slots."""
    thumb = photo.resize((64, 64), Image.Resampling.BILINEAR)
    px = thumb.load()
    luma = [
        0.299 * px[x, y][0] + 0.587 * px[x, y][1] + 0.114 * px[x, y][2]
        for y in range(thumb.height)
        for x in range(thumb.width)
    ]
    mean = sum(luma) / len(luma)
    var = sum((v - mean) ** 2 for v in luma) / len(luma)
    return mean + 0.15 * math.sqrt(var)


def draw_message(canvas: Image.Image) -> None:
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/noto/NotoSerifDisplay-BoldItalic.ttf"
        if Path("/usr/share/fonts/truetype/noto/NotoSerifDisplay-BoldItalic.ttf").exists()
        else "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
        72,
    )
    body_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/noto/NotoSerifDisplay-Regular.ttf"
        if Path("/usr/share/fonts/truetype/noto/NotoSerifDisplay-Regular.ttf").exists()
        else "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
        44,
    )
    sign_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
        48,
    )

    # Decorative rule
    cx = W // 2
    draw.line([(cx - 220, TEXT_TOP - 40), (cx + 220, TEXT_TOP - 40)], fill=SOFT_LINE, width=2)
    # Small diamond accent
    d = 7
    draw.polygon(
        [(cx, TEXT_TOP - 40 - d), (cx + d, TEXT_TOP - 40), (cx, TEXT_TOP - 40 + d), (cx - d, TEXT_TOP - 40)],
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


def build_card() -> Path:
    photos = load_photos()
    # Assign larger slots to stronger / clearer photos
    ranked = sorted(range(len(photos)), key=lambda i: score_photo(photos[i]), reverse=True)

    canvas = make_background().convert("RGBA")
    collage_w, collage_h = W - 160, COLLAGE_H
    mask = make_thirty_mask(collage_w, collage_h)

    circles: list[tuple[float, float, float]] = []
    for seed in (2030, 42, 7, 99, 1234):
        circles = pack_circles(mask, count=46, seed=seed)
        print(f"seed {seed}: packed {len(circles)} circles")
        if len(circles) >= 46:
            break
    if len(circles) < 46:
        raise SystemExit(f"Only packed {len(circles)} circles; need denser packing")

    # Sort circles largest-first; pair with ranked photos
    circles_sorted = sorted(circles, key=lambda c: c[2], reverse=True)[:46]
    # Use all photos; if we somehow have fewer circles, drop lowest-ranked
    n = min(len(circles_sorted), len(photos))
    circles_sorted = circles_sorted[:n]
    photo_order = ranked[:n]

    layer = Image.new("RGBA", (collage_w, collage_h), (0, 0, 0, 0))
    # Soft under-glow of the number for readability of the shape
    glow = mask.filter(ImageFilter.GaussianBlur(18)).point(lambda p: int(p * 0.22))
    rose_glow = Image.new("RGBA", (collage_w, collage_h), (180, 90, 110, 0))
    rose_glow.putalpha(glow)
    layer = Image.alpha_composite(layer, rose_glow)

    for (cx, cy, r), pi in zip(circles_sorted, photo_order):
        d = int(r * 2)
        tile = circular_crop(photos[pi], d)
        # Soft drop shadow
        shadow = Image.new("RGBA", (d + 12, d + 12), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shadow)
        sd.ellipse((6, 8, d + 4, d + 8), fill=(90, 40, 55, 55))
        shadow = shadow.filter(ImageFilter.GaussianBlur(4))
        layer.alpha_composite(shadow, (int(cx - r - 6), int(cy - r - 4)))
        layer.alpha_composite(tile, (int(cx - r), int(cy - r)))

    ox = (W - collage_w) // 2
    canvas.alpha_composite(layer, (ox, COLLAGE_TOP))

    # Convert for text drawing
    final = canvas.convert("RGB")
    draw_message(final)

    # Small footer years hint
    draw = ImageDraw.Draw(final)
    foot = ImageFont.truetype(
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        28,
    )
    label = "30 Years of Love"
    fb = draw.textbbox((0, 0), label, font=foot)
    draw.text(
        ((W - (fb[2] - fb[0])) // 2, H - 90),
        label,
        font=foot,
        fill=(170, 120, 130),
    )

    out = OUT_DIR / "anniversary_30_card.jpg"
    final.save(out, quality=95, optimize=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    art = ARTIFACTS / "anniversary_30_card.jpg"
    final.save(art, quality=95, optimize=True)
    # Also PNG for crisp preview
    final.save(OUT_DIR / "anniversary_30_card.png", optimize=True)
    final.save(ARTIFACTS / "anniversary_30_card.png", optimize=True)
    return out


if __name__ == "__main__":
    path = build_card()
    print(f"Wrote {path}")
