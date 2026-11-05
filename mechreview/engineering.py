"""Quick engineering estimates that back each fix with a number.

- Pocket corners: a simplified 2D plane-stress FE model of the plate pulled
  along its length. The pocket is modeled as a cut-out (floor neglected), so
  load flows through the pocket walls -- the conservative load path that shows
  how sensitive the corners are to their radius. Slot and blind hole reduce
  thickness; mounting holes are treated as bolted and left out. Reports the
  peak-to-net-section stress ratio at the pocket corners (an indicative Kt)
  and renders a heatmap.
- Thin wall: cantilever deflection of the wall under a cutting force.
- Deep hole / narrow slot: deflection of the drill / end mill, ∝ L³ / D⁴.

These are estimates for comparing a design before and after a fix, not a
certified analysis. Assumptions are reported alongside every number.
"""

import json
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

E_MPA = {"ALUMINUM 6061-T6": 68_900, "ALUMINUM 7075-T6": 71_700,
         "STAINLESS STEEL 304": 193_000, "STEEL 1018": 205_000}
NU = 0.3
CUT_FORCE_N_PER_MM = 5.0  # radial finishing-pass force per mm of wall (≈30 N over a Ø6 cutter contact)
TOOL_E_MPA = {"carbide": 600_000, "hss": 210_000}


# ----------------------------------------------------------------- 2D FE model
def _ke():
    """Q4 plane-stress element stiffness (E=1, t=1); independent of element size."""
    D = np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]]) / (1 - NU ** 2)
    ke = np.zeros((8, 8))
    g = 1 / math.sqrt(3)
    for xi in (-g, g):
        for eta in (-g, g):
            dxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4
            deta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4
            # unit square: x = (xi+1)/2 -> d/dx = 2 d/dxi, detJ = 1/4
            dx, dy = 2 * dxi, 2 * deta
            B = np.zeros((3, 8))
            B[0, 0::2], B[1, 1::2] = dx, dy
            B[2, 0::2], B[2, 1::2] = dy, dx
            ke += B.T @ D @ B * 0.25
    return ke, D


_KE, _D = _ke()


def _inside_rrect(x, y, x0, y0, x1, y1, r):
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    if r <= 0:
        return inside
    cx = np.clip(x, x0 + r, x1 - r)
    cy = np.clip(y, y0 + r, y1 - r)
    return inside & ((x - cx) ** 2 + (y - cy) ** 2 <= r * r)


def thickness_field(part, h, ss=3):
    """Element thickness (mm) on an h-mm grid, area-averaged by ss×ss supersampling."""
    L, W, T = part["L"], part["W"], part["T"]
    nx, ny = int(round(L / h)), int(round(W / h))
    o = (np.arange(ss) + 0.5) / ss
    xs = ((np.arange(nx)[:, None] + o[None, :]) * h).ravel()
    ys = ((np.arange(ny)[:, None] + o[None, :]) * h).ravel()
    X, Y = np.meshgrid(xs, ys)  # (ny*ss, nx*ss)
    t = np.full(X.shape, float(T))
    pk = part["pocket"]
    # Pocket as a cut-out: the floor is neglected so the walls carry the load.
    t = np.where(_inside_rrect(X, Y, pk["x0"], pk["y0"], pk["x1"], pk["y1"], pk["corner_r"]), 0.0, t)
    sl = part.get("slot")
    if sl:
        t = np.where(_inside_rrect(X, Y, sl["x"] - sl["length"] / 2, sl["y"] - sl["width"] / 2,
                                   sl["x"] + sl["length"] / 2, sl["y"] + sl["width"] / 2, sl["width"] / 2),
                     np.minimum(t, T - sl["depth"]), t)
    bh = part.get("blind_hole")
    if bh:
        t = np.where((X - bh["x"]) ** 2 + (Y - bh["y"]) ** 2 <= (bh["D"] / 2) ** 2,
                     np.minimum(t, T - bh["depth"]), t)
    return t.reshape(ny, ss, nx, ss).mean(axis=(1, 3)), nx, ny


_FE_CACHE = {}


def plane_stress(part, h=None, max_elems=160_000):
    """Cached: the same geometry is solved once (quality scoring and heatmaps share it)."""
    key = json.dumps({k: part.get(k) for k in ("L", "W", "T", "pocket", "slot", "blind_hole")},
                     sort_keys=True) + f"|{h}|{max_elems}"
    if key not in _FE_CACHE:
        if len(_FE_CACHE) > 64:
            _FE_CACHE.clear()
        _FE_CACHE[key] = _plane_stress(part, h, max_elems)
    return _FE_CACHE[key]


def _plane_stress(part, h=None, max_elems=160_000):
    """Pull the plate along X (along the pocket). Returns stress-ratio field and corner peak.

    Kt is reported against the net-section nominal stress through the pocket,
    as handbooks do: sigma_net = F / ((W - pocket height) * T).
    """
    L, W, T = part["L"], part["W"], part["T"]
    h = h or max(0.2, math.sqrt(L * W / max_elems))
    t, nx, ny = thickness_field(part, h)
    t = np.maximum(t, 1e-4 * T)
    nnx = nx + 1
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    n0 = (j * nnx + i).ravel()
    nodes = np.stack([n0, n0 + 1, n0 + 1 + nnx, n0 + nnx], axis=1)
    edof = np.empty((nodes.shape[0], 8), dtype=np.int64)
    edof[:, 0::2], edof[:, 1::2] = 2 * nodes, 2 * nodes + 1
    te = t.ravel()
    rows = np.repeat(edof, 8, axis=1).ravel()
    cols = np.tile(edof, (1, 8)).ravel()
    vals = (te[:, None] * _KE.ravel()[None, :]).ravel()
    ndof = 2 * nnx * (ny + 1)
    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()

    # Load: equal and opposite 1 MPa tractions on both ends (total F = W * T), so both
    # ends contract freely; three dofs remove rigid-body motion.
    f = np.zeros(ndof)
    for col, sign in ((nx, 1.0), (0, -1.0)):
        edge_t = t[:, min(col, nx - 1)]
        for k in range(ny):
            share = sign * (W * T) * edge_t[k] / edge_t.sum() / 2
            f[2 * (k * nnx + col)] += share
            f[2 * ((k + 1) * nnx + col)] += share
    mid = (ny // 2) * nnx
    fixed = np.array([2 * mid, 2 * mid + 1, 2 * (mid + nx) + 1])
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])

    ue = u[edof]
    dndx = np.array([-1, 1, 1, -1]) / (2 * h)
    dndy = np.array([-1, -1, 1, 1]) / (2 * h)
    ex = ue[:, 0::2] @ dndx
    ey = ue[:, 1::2] @ dndy
    gxy = ue[:, 0::2] @ dndy + ue[:, 1::2] @ dndx
    c = 1 / (1 - NU ** 2)  # E = 1 cancels: displacement ∝ 1/E
    sx, sy, txy = c * (ex + NU * ey), c * (ey + NU * ex), c * (1 - NU) / 2 * gxy
    vm = np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * txy ** 2).reshape(ny, nx)
    vm[t < 0.01 * T] = 0.0  # no stress shown inside the cut-out

    pk = part["pocket"]
    nominal_net = W / (W - pk["height"])  # MPa for the 1 MPa gross load
    ratio = vm / nominal_net
    cx = (np.arange(nx) + 0.5) * h
    cy = (np.arange(ny) + 0.5) * h
    CX, CY = np.meshgrid(cx, cy)
    win = max(3.0, 5 * h)
    near = np.zeros_like(ratio, dtype=bool)
    for px in (pk["x0"], pk["x1"]):
        for py in (pk["y0"], pk["y1"]):
            near |= (np.abs(CX - px) <= win) & (np.abs(CY - py) <= win)
    k = np.argmax(np.where(near, ratio, -1))
    return {"ratio": ratio, "h": h, "nx": nx, "ny": ny, "corner_kt": float(ratio.ravel()[k]),
            "corner_peak_xy": (float(CX.ravel()[k]), float(CY.ravel()[k])),
            "max_ratio": float(ratio.max()), "sharp": pk["corner_r"] < h}


def stress_heatmap(part, fe, path, vmax=None, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .drawing import rrect_pts
    L, W = part["L"], part["W"]
    fig, ax = plt.subplots(figsize=(8, 8 * W / L * 0.9 + 1.0), dpi=110)
    im = ax.imshow(fe["ratio"], origin="lower", extent=[0, L, 0, W], cmap="inferno",
                   vmin=0, vmax=vmax or np.percentile(fe["ratio"], 99.9), interpolation="bilinear")
    pk = part["pocket"]
    xs, ys = zip(*rrect_pts(pk["x0"], pk["y0"], pk["x1"], pk["y1"], pk["corner_r"]))
    ax.plot(xs, ys, color="#7fd6ff", lw=0.9)
    sl = part.get("slot")
    if sl:
        xs, ys = zip(*rrect_pts(sl["x"] - sl["length"] / 2, sl["y"] - sl["width"] / 2,
                                sl["x"] + sl["length"] / 2, sl["y"] + sl["width"] / 2, sl["width"] / 2))
        ax.plot(xs, ys, color="#7fd6ff", lw=0.7)
    px, py = fe["corner_peak_xy"]
    ax.plot([px], [py], marker="o", mfc="none", mec="#00ff99", ms=12, mew=1.6)
    label = f"Kt ≥ {fe['corner_kt']:.1f} (sharp)" if fe["sharp"] else f"Kt ≈ {fe['corner_kt']:.1f}"
    ax.annotate(label, (px, py), xytext=(-110 if px > L / 2 else 10, -18),
                textcoords="offset points", color="#00ff99", fontsize=9, weight="bold")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("von Mises / net-section nominal stress")
    fig.text(0.01, 0.01, "Simplified 2D plane-stress estimate: plate pulled along its length, pocket floor "
             "neglected, mounting holes assumed bolted. Indicative only.", fontsize=7, color="#555")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------- beam estimates
def wall_deflection_um(part):
    """Tip deflection (µm) of the pocket's outer wall, cantilevered from the floor."""
    pk = part["pocket"]
    E = E_MPA.get(part["material"], 70_000)
    w, hgt = pk["wall"], pk["depth"]
    return 4 * CUT_FORCE_N_PER_MM * hgt ** 3 / (E * w ** 3) * 1000


def tool_deflection_um(diameter, stickout, force_n=30.0, tool="carbide"):
    """Tip deflection (µm) of a cylindrical cutter: F L³ / (3 E I), I = π D⁴ / 64."""
    I = math.pi * diameter ** 4 / 64
    return force_n * stickout ** 3 / (3 * TOOL_E_MPA[tool] * I) * 1000


def evidence(issue_type, part):
    """Headline engineering number(s) for one issue on one part."""
    from .rules import DEEP_HOLE, DEEP_SLOT, SHARP_CORNER, THIN_WALL
    if issue_type == THIN_WALL:
        return {"metric": "wall_deflection_um", "value": round(wall_deflection_um(part), 1),
                "assumption": f"{CUT_FORCE_N_PER_MM:g} N/mm radial cutting force at the top of a "
                              f"{part['pocket']['depth']} mm tall wall"}
    if issue_type == SHARP_CORNER:
        fe = plane_stress(part)
        return {"metric": "corner_kt", "value": round(fe["corner_kt"], 2),
                "assumption": "2D plane-stress FE, tension along the plate, pocket floor neglected "
                              "(load through the walls), Kt vs net-section stress"
                              + ("; R0 is singular, value is mesh-limited" if fe["sharp"] else ""), "_fe": fe}
    if issue_type == DEEP_HOLE:
        bh = part["blind_hole"]
        return {"metric": "drill_deflection_um",
                "value": round(tool_deflection_um(bh["D"], bh["depth"] + bh["D"], force_n=10, tool="hss"), 1),
                "assumption": "HSS drill, 10 N side load, stick-out = depth + D"}
    if issue_type == DEEP_SLOT:
        sl = part["slot"]
        return {"metric": "end_mill_deflection_um",
                "value": round(tool_deflection_um(sl["width"], sl["depth"] + 2, force_n=30), 1),
                "assumption": "carbide end mill Ø = slot width, 30 N radial load, stick-out = depth + 2 mm"}
    return None
