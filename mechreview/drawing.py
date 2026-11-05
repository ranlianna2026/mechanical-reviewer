"""Render a part spec as a 2D engineering drawing (top + front view, notes,
title block) and report pixel regions for each checkable feature."""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from .rules import DEEP_HOLE, DEEP_SLOT, MISSING_TOL, SHARP_CORNER, THIN_WALL, TIGHT_TOL  # noqa: E402

DPI = 120
PAGE_W, PAGE_H = 420.0, 297.0  # A3 landscape, mm
INK = "#111111"
FS = 8.5


def fmt(v, nd=1):
    return f"{v:.{nd}f}"


def rrect_pts(x0, y0, x1, y1, r, n=10):
    if r <= 0:
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    pts = []
    for cx, cy, a0, a1 in ((x1 - r, y0 + r, -90, 0), (x1 - r, y1 - r, 0, 90),
                           (x0 + r, y1 - r, 90, 180), (x0 + r, y0 + r, 180, 270)):
        for k in range(n + 1):
            a = math.radians(a0 + (a1 - a0) * k / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts.append(pts[0])
    return pts


class Sheet:
    def __init__(self):
        self.fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4), dpi=DPI)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, PAGE_W)
        self.ax.set_ylim(0, PAGE_H)
        self.ax.axis("off")
        self.px_w, self.px_h = int(round(PAGE_W / 25.4 * DPI)), int(round(PAGE_H / 25.4 * DPI))

    # -- primitives ---------------------------------------------------------
    def poly(self, pts, lw=1.1, ls="-"):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=INK, lw=lw, ls=ls, solid_capstyle="butt")

    def line(self, xs, ys, lw=0.45, ls="-"):
        self.ax.plot(xs, ys, color=INK, lw=lw, ls=ls)

    def hidden(self, pts):
        self.poly(pts, lw=0.6, ls=(0, (4, 2)))

    def circle(self, cx, cy, r, lw=1.1, ls="-", n=72):
        self.poly([(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n))
                   for k in range(n + 1)], lw=lw, ls=ls)

    def center_mark(self, cx, cy, size):
        self.line([cx - size, cx + size], [cy, cy], lw=0.35, ls=(0, (6, 1.5, 1, 1.5)))
        self.line([cx, cx], [cy - size, cy + size], lw=0.35, ls=(0, (6, 1.5, 1, 1.5)))

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", FS)
        kw.setdefault("color", INK)
        return self.ax.text(x, y, s, family="DejaVu Sans", **kw)

    def arrow(self, x0, y0, x1, y1):
        self.ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="-|>", lw=0.5, color=INK,
                                         mutation_scale=7, shrinkA=0, shrinkB=0))

    def leader(self, px, py, tx, ty, s):
        return self.ax.annotate(s, xy=(px, py), xytext=(tx, ty), fontsize=FS, color=INK,
                                ha="left", va="center", family="DejaVu Sans",
                                arrowprops=dict(arrowstyle="-|>", lw=0.5, color=INK,
                                                mutation_scale=7, shrinkA=1, shrinkB=0,
                                                relpos=(0, 0.5)))

    # -- dimensions ---------------------------------------------------------
    def hdim(self, xa, xb, y, ya, yb, s):
        for x, yf in ((xa, ya), (xb, yb)):
            sg = 1 if y > yf else -1
            self.line([x, x], [yf + sg * 1.0, y + sg * 1.5])
        if xb - xa >= 12:
            mid = (xa + xb) / 2
            self.arrow(mid, y, xa, y)
            self.arrow(mid, y, xb, y)
            return self.text(mid, y + 0.8, s, ha="center", va="bottom")
        self.line([xa, xb], [y, y])
        self.arrow(xa - 6, y, xa, y)
        self.arrow(xb + 6, y, xb, y)
        return self.text(xb + 7, y, s, ha="left", va="center")

    def vdim(self, ya, yb, x, xa, xb, s, side="left", small_text_below=False):
        for yv, xf in ((ya, xa), (yb, xb)):
            sg = 1 if x > xf else -1
            self.line([xf + sg * 1.0, x + sg * 1.5], [yv, yv])
        off, ha = (-0.8, "right") if side == "left" else (0.8, "left")
        if yb - ya >= 12:
            mid = (ya + yb) / 2
            self.arrow(x, mid, x, ya)
            self.arrow(x, mid, x, yb)
            return self.text(x + off, mid, s, ha=ha, va="center", rotation=90)
        self.line([x, x], [ya, yb])
        self.arrow(x, ya - 6, x, ya)
        self.arrow(x, yb + 6, x, yb)
        if small_text_below:  # keep clear of a chained small dimension above
            return self.text(x, ya - 7, s, ha="center", va="top", rotation=90)
        return self.text(x, yb + 7, s, ha="center", va="bottom", rotation=90)

    # -- pixel helpers ------------------------------------------------------
    def mm_box_px(self, x0, y0, x1, y1, pad=1.5):
        sx, sy = self.px_w / PAGE_W, self.px_h / PAGE_H
        xa, xb = sorted((x0, x1))
        ya, yb = sorted((y0, y1))
        return [int((xa - pad) * sx), int((PAGE_H - yb - pad) * sy),
                int(math.ceil((xb + pad) * sx)), int(math.ceil((PAGE_H - ya + pad) * sy))]

    def text_box_px(self, artist, pad=3):
        self.fig.canvas.draw()
        bb = artist.get_window_extent(self.fig.canvas.get_renderer())
        return [int(bb.x0 - pad), int(self.px_h - bb.y1 - pad),
                int(math.ceil(bb.x1 + pad)), int(math.ceil(self.px_h - bb.y0 + pad))]


def drawing_scale(part):
    # Keep the front view above the notes and title block (top edge at y=76).
    L, Wd, T_ = part["L"], part["W"], part["T"]
    return next(c for c in (2.0, 1.5, 1.0) if L * c <= 230 and (Wd + T_) * c <= 140)


def render(part, path, strict=False):
    """Draw `part` to `path` (PNG). Returns (scale, regions) where regions maps
    each issue type that can apply to this part to its pixel boxes."""
    sh = Sheet()
    L, Wd, T_ = part["L"], part["W"], part["T"]
    s = drawing_scale(part)
    ox, tv_top = 45.0, 258.0
    oy = tv_top - Wd * s
    fv_top = oy - 32
    fy = fv_top - T_ * s
    X = lambda u: ox + u * s  # noqa: E731
    Y = lambda v: oy + v * s  # noqa: E731
    Z = lambda z: fy + z * s  # noqa: E731
    right = X(L)

    # Border and view labels.
    sh.poly([(8, 8), (PAGE_W - 8, 8), (PAGE_W - 8, PAGE_H - 8), (8, PAGE_H - 8), (8, 8)], lw=1.4)
    sh.text(ox, tv_top + 26, "TOP VIEW", fontsize=9, weight="bold")
    sh.text(ox, fv_top + 6, "FRONT VIEW", fontsize=9, weight="bold")

    # ---- TOP VIEW --------------------------------------------------------
    sh.poly(rrect_pts(X(0), Y(0), X(L), Y(Wd), 0), lw=1.3)
    mh = part["mount_holes"]
    for (u, v) in ((mh["inset"], mh["inset"]), (L - mh["inset"], mh["inset"]),
                   (mh["inset"], Wd - mh["inset"]), (L - mh["inset"], Wd - mh["inset"])):
        sh.circle(X(u), Y(v), mh["D"] / 2 * s)
        sh.center_mark(X(u), Y(v), mh["D"] / 2 * s + 2)
    pk = part["pocket"]
    sh.poly(rrect_pts(X(pk["x0"]), Y(pk["y0"]), X(pk["x1"]), Y(pk["y1"]), pk["corner_r"] * s), lw=1.1)

    bh = part.get("blind_hole")
    if bh:
        sh.circle(X(bh["x"]), Y(bh["y"]), bh["D"] / 2 * s)
        sh.center_mark(X(bh["x"]), Y(bh["y"]), bh["D"] / 2 * s + 2)
    sl = part.get("slot")
    if sl:
        sx0, sx1 = sl["x"] - sl["length"] / 2, sl["x"] + sl["length"] / 2
        sy0, sy1 = sl["y"] - sl["width"] / 2, sl["y"] + sl["width"] / 2
        sh.poly(rrect_pts(X(sx0), Y(sy0), X(sx1), Y(sy1), sl["width"] / 2 * s))

    # Overall dimensions (with optional planted / hard-negative tolerances).
    l_txt = fmt(L) if part.get("tight_tol_on") != "L" else f"{L:.3f} ±0.005"
    w_txt = fmt(Wd)
    if part.get("tight_tol_on") == "W":
        w_txt = f"{Wd:.3f} ±0.005"
    elif part.get("neg_tol_W"):
        w_txt = f"{Wd:.1f} ±0.1"
    t_L = sh.hdim(X(0), X(L), Y(Wd) + 18, Y(Wd), Y(Wd), l_txt)
    t_W = sh.vdim(Y(0), Y(Wd), X(0) - 14, X(0), X(0), w_txt)
    # Mount hole location + callout.
    sh.hdim(X(0), X(mh["inset"]), Y(Wd) + 6, Y(Wd), Y(Wd - mh["inset"]), fmt(mh["inset"]))
    sh.vdim(Y(Wd - mh["inset"]), Y(Wd), X(0) - 6, X(mh["inset"]), X(0), fmt(mh["inset"]))
    sh.leader(X(L - mh["inset"]) + mh["D"] / 2 * s * 0.7, Y(Wd - mh["inset"]) + mh["D"] / 2 * s * 0.7,
              right + 26, Y(Wd) + 8, f"4X Ø{mh['D']:.1f} THRU")
    # Pocket location, wall, height, callout.
    sh.hdim(X(0), X(pk["x0"]), Y(Wd) + 12, Y(Wd), Y(pk["y1"]), fmt(pk["x0"]))
    t_wall = sh.vdim(Y(pk["y1"]), Y(Wd), right + 6, X(pk["x1"]), right, fmt(pk["wall"]), side="right")
    sh.vdim(Y(pk["y0"]), Y(pk["y1"]), right + 6, X(pk["x1"]), X(pk["x1"]), fmt(pk["height"]), side="right",
            small_text_below=True)
    pocket_lines = [f"POCKET {pk['width']:.1f} x {pk['height']:.1f}", f"{pk['depth']:.1f} DEEP"]
    if not pk.get("sharp_via_note"):
        pocket_lines.append(f"4X R{pk['corner_r']:.1f}")
    t_pocket = sh.leader(X(pk["x1"]) - 1, Y(pk["y0"]) + 1, right + 26, Y((pk["y0"] + pk["y1"]) / 2) - 4,
                         "\n".join(pocket_lines))
    # Blind hole.
    t_hole = None
    if bh:
        sh.hdim(X(0), X(bh["x"]), Y(0) - 6, Y(0), Y(bh["y"]), fmt(bh["x"]))
        sh.vdim(Y(0), Y(bh["y"]), X(0) - 6, X(0), X(bh["x"]), fmt(bh["y"]))
        t_hole = sh.leader(X(bh["x"]) + bh["D"] / 2 * s * 0.7, Y(bh["y"]) + bh["D"] / 2 * s * 0.7,
                           right + 26, Y(bh["y"]) + 3, f"Ø{bh['D']:.1f} x {bh['depth']:.1f} DEEP")
    # Slot.
    t_slot = None
    if sl:
        sh.hdim(X(0), X(sl["x"]), Y(0) - 12, Y(0), Y(sl["y"]), fmt(sl["x"]))
        sh.vdim(Y(0), Y(sl["y"]), right + 6, right, X(sx1), fmt(sl["y"]), side="right")
        t_slot = sh.leader(X(sx1) - sl["width"] / 2 * s, Y(sy0), right + 26, Y(sl["y"]) - 6,
                           f"SLOT {sl['width']:.1f} WIDE x {sl['length']:.1f} LG\n{sl['depth']:.1f} DEEP")

    # ---- FRONT VIEW (looking along +Y, hidden features dashed) ----------
    sh.poly(rrect_pts(X(0), Z(0), X(L), Z(T_), 0), lw=1.3)
    for u in (mh["inset"], L - mh["inset"]):
        for e in (-1, 1):
            sh.line([X(u + e * mh["D"] / 2)] * 2, [Z(0), Z(T_)], lw=0.6, ls=(0, (4, 2)))
    sh.hidden([(X(pk["x0"]), Z(T_)), (X(pk["x0"]), Z(T_ - pk["depth"])),
               (X(pk["x1"]), Z(T_ - pk["depth"])), (X(pk["x1"]), Z(T_))])
    if bh:
        tip = 0.3 * bh["D"]
        hx0, hx1, hb = bh["x"] - bh["D"] / 2, bh["x"] + bh["D"] / 2, T_ - bh["depth"]
        sh.hidden([(X(hx0), Z(T_)), (X(hx0), Z(hb)), (X(bh["x"]), Z(hb - tip)),
                   (X(hx1), Z(hb)), (X(hx1), Z(T_))])
    if sl:
        sh.hidden([(X(sx0), Z(T_)), (X(sx0), Z(T_ - sl["depth"])),
                   (X(sx1), Z(T_ - sl["depth"])), (X(sx1), Z(T_))])
    sh.vdim(Z(0), Z(T_), X(0) - 8, X(0), X(0), fmt(T_))
    sh.vdim(Z(T_ - pk["depth"]), Z(T_), right + 6, X(pk["x1"]), right, fmt(pk["depth"]), side="right")
    if sl:
        sh.vdim(Z(T_ - sl["depth"]), Z(T_), right + 16, X(sx1), right, fmt(sl["depth"]), side="right")

    # ---- NOTES --------------------------------------------------------------
    notes = ["NOTES:", "1. REMOVE ALL BURRS. BREAK EXTERNAL EDGES 0.2-0.5.",
             "2. ALL DIMENSIONS IN MILLIMETERS.", "3. FINISH: AS MACHINED, 1.6 Ra MAX ON POCKET FLOOR."]
    if pk.get("sharp_via_note"):
        notes.append("4. POCKET INTERNAL CORNERS TO BE SHARP (R0).")
    t_notes = [sh.text(16, 52 - i * 5.5, n, fontsize=8) for i, n in enumerate(notes)]

    # ---- TITLE BLOCK --------------------------------------------------------
    tb_x0, tb_x1, tb_y0, tb_y1 = 250.0, 412.0, 12.0, 76.0
    sh.poly(rrect_pts(tb_x0, tb_y0, tb_x1, tb_y1, 0), lw=1.2)
    rows = [f"TITLE: {part['name']}", f"PART NO: {part['number']}    REV: {part.get('rev', 'A')}",
            f"MATERIAL: {part['material']}", "GENERAL TOLERANCES (UNLESS OTHERWISE SPECIFIED):",
            "LINEAR  X.X ±0.2   X.XX ±0.1   ANGULAR ±0.5°" if part.get("general_tol", True) else "",
            f"UNITS: MM    SCALE: {s:g}:1    THIRD ANGLE PROJECTION",
            "DRAWN: SYNTHETIC EVAL SET    SHEET 1 OF 1"]
    rh = (tb_y1 - tb_y0) / len(rows)
    for i, row in enumerate(rows):
        y_line = tb_y1 - (i + 1) * rh
        if i < len(rows) - 1 and i != 3:
            sh.line([tb_x0, tb_x1], [y_line, y_line], lw=0.5)
        sh.text(tb_x0 + 3, y_line + rh / 2, row, fontsize=8, va="center",
                weight="bold" if i == 0 else "normal")

    # ---- REGIONS ------------------------------------------------------------
    pad_geom = 2.0
    regions = {
        THIN_WALL: [
            {"view": "top", "bbox": sh.mm_box_px(X(pk["x0"]), Y(pk["y1"]) - pad_geom, X(pk["x1"]), Y(Wd) + pad_geom)},
            {"view": "dimension", "bbox": sh.text_box_px(t_wall)}],
        SHARP_CORNER: [
            {"view": "top", "bbox": sh.mm_box_px(X(pk["x0"]), Y(pk["y0"]), X(pk["x1"]), Y(pk["y1"]))},
            {"view": "note", "bbox": sh.text_box_px(t_notes[-1])} if pk.get("sharp_via_note")
            else {"view": "callout", "bbox": sh.text_box_px(t_pocket)}],
        MISSING_TOL: [{"view": "title_block",
                       "bbox": sh.mm_box_px(tb_x0, tb_y1 - 5 * rh, tb_x1, tb_y1 - 3 * rh, pad=0)}],
    }
    if bh:
        tip = 0.3 * bh["D"]
        regions[DEEP_HOLE] = [
            {"view": "top", "bbox": sh.mm_box_px(X(bh["x"] - bh["D"]), Y(bh["y"] - bh["D"]),
                                                 X(bh["x"] + bh["D"]), Y(bh["y"] + bh["D"]))},
            {"view": "front", "bbox": sh.mm_box_px(X(bh["x"] - bh["D"]), Z(T_ - bh["depth"]) - tip * s,
                                                   X(bh["x"] + bh["D"]), Z(T_))},
            {"view": "callout", "bbox": sh.text_box_px(t_hole)}]
    if sl:
        regions[DEEP_SLOT] = [
            {"view": "top", "bbox": sh.mm_box_px(X(sx0), Y(sy0), X(sx1), Y(sy1))},
            {"view": "front", "bbox": sh.mm_box_px(X(sx0), Z(T_ - sl["depth"]), X(sx1), Z(T_))},
            {"view": "callout", "bbox": sh.text_box_px(t_slot)}]
    if part.get("tight_tol_on"):
        regions[TIGHT_TOL] = [{"view": "dimension",
                               "bbox": sh.text_box_px(t_L if part["tight_tol_on"] == "L" else t_W)}]

    overlaps = text_overlaps(sh)
    sh.fig.savefig(path, dpi=DPI)
    plt.close(sh.fig)
    if overlaps:
        if strict:
            raise ValueError(f"{path}: overlapping text {overlaps}")
        regions["_warnings"] = [f"overlapping text: {a!r} / {b!r}" for a, b in overlaps]
    return s, regions


def text_overlaps(sh):
    """Pairs of drawn texts whose boxes intersect (unreadable drawing)."""
    sh.fig.canvas.draw()
    rend = sh.fig.canvas.get_renderer()
    # Text.get_window_extent, not Annotation's: leader arrows may cross other text.
    items = [(t.get_text(), Text.get_window_extent(t, rend)) for t in sh.ax.texts if t.get_text().strip()]
    bad = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 1 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1:
                bad.append((items[i][0][:20], items[j][0][:20]))
    return bad


def overlay(img_path, marks, out_path):
    """Draw boxes on a rendered drawing. `marks` is [(label, regions, rgb)]."""
    im = Image.open(img_path).convert("RGB")
    dr = ImageDraw.Draw(im)
    for label, regions, rgb in marks:
        for reg in regions:
            dr.rectangle(reg["bbox"], outline=rgb, width=3)
            dr.text((reg["bbox"][0], max(0, reg["bbox"][1] - 12)), label, fill=rgb)
    im.save(out_path)
