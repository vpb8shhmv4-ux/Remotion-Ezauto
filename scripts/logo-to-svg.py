"""Cordillera logo -> SVG, as a flowing DAG that reuses image-to-svg's own steps.

Usage: python3 scripts/logo-to-svg.py assets/cordillera-logo-source.png public
Needs: opencv-python-headless scikit-image scipy scikit-learn, potrace, rsvg-convert.

  image-to-svg.preprocess -> image-to-svg.quantize -> image-to-svg.detect_background
        -> palette -> layer_masks -> trace (potrace curves) -> assemble -> verify

verify re-renders the SVG, measures per-colour overlap (IoU) against the source
and retries the trace with finer settings until every layer matches.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / ".agents/skills/image-to-svg/scripts"))
import pipeline as i2s  # noqa: E402  (also puts flowing on sys.path)
from flowing import Flow, task  # noqa: E402

SRC = sys.argv[1]
OUT = Path(sys.argv[2])
i2s.configure(SRC, mode="graphic", K=28, pipeline="fill")

# Where the bottom-left loop fades green -> orange (source px, measured from
# column medians of the stroke's lower band).
FADE_X0, FADE_X1, FADE_BOX = 137, 175, (125, 675, 190, 720)  # x0, y0, x1, y1
ATTEMPTS = [dict(S=4, alphamax=1.0, opt=0.2), dict(S=6, alphamax=0.9, opt=0.1),
            dict(S=8, alphamax=0.8, opt=0.05)]
state = {"attempt": 0}


def hexc(c):
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in c)


def classify(rgb, bg, inks):
    """Antialias-aware labels: project each pixel onto bg->ink lines, keep the
    ink with the smallest residual, and call it ink where coverage >= 50%."""
    p = rgb.astype(np.float32) - bg
    best_r = np.full(p.shape[:2], np.inf, np.float32)
    lab = np.zeros(p.shape[:2], np.uint8)
    for i, c in enumerate(inks, 1):
        v = c - bg
        a = np.clip((p @ v) / (v @ v), 0, 1)
        r = np.linalg.norm(p - a[..., None] * v, axis=-1)
        take = (a >= 0.5) & (r < best_r)
        best_r[take], lab[take] = r[take], i
    return lab


@task(depends_on=[i2s.quantize, i2s.detect_background])
def palette(quantize, detect_background):
    """Ink colours from image-to-svg's K-means clusters, refined to the median
    of solid (non-antialiased) source pixels nearest each cluster."""
    src = cv2.cvtColor(cv2.imread(SRC), cv2.COLOR_BGR2RGB).astype(np.float32)
    bg_idx = set(detect_background["bg_clusters"])
    bg = np.median(src[src.mean(2) > 235], 0)
    centers = quantize["centers"].astype(np.float32)
    counts = dict(quantize["sorted_clusters"])
    cand = [(counts[i], centers[i]) for i in range(len(centers))
            if i not in bg_idx and np.linalg.norm(centers[i] - bg) > 90]
    # one representative per hue family: dark, green, orange
    fams = {"black": lambda c: -c.sum(), "green": lambda c: c[1] - c[0],
            "orange": lambda c: c[0] - c[2]}
    flat = src.reshape(-1, 3)
    inks = {}
    for name, score in fams.items():
        c = max(cand, key=lambda t: score(t[1]))[1]
        near = flat[np.linalg.norm(flat - c, axis=1) < 40]
        inks[name] = np.median(near, 0)
    print("  palette:", {k: hexc(v) for k, v in inks.items()}, "bg", hexc(bg))
    return {"bg": bg, "inks": inks, "src": src}


def must_have_three_inks(palette):
    hexes = {hexc(v) for v in palette["inks"].values()}
    if len(hexes) != 3:
        raise ValueError(f"expected 3 distinct inks, got {hexes}")


@task(depends_on=[palette], validate=must_have_three_inks)
def trace(palette):
    """Upscale, classify, then trace each colour layer with potrace (smooth
    Bezier curves). Settings come from the current verify attempt."""
    prm = ATTEMPTS[min(state["attempt"], len(ATTEMPTS) - 1)]
    state["attempt"] += 1
    S = prm["S"]
    src, bg, inks = palette["src"], palette["bg"], palette["inks"]
    names = list(inks)
    up = cv2.resize(src, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    lab = classify(up, bg, np.array([inks[n] for n in names]))
    masks = {n: lab == i for i, n in enumerate(names, 1)}
    x0, y0, x1, y1 = (v * S for v in FADE_BOX)
    fade = np.zeros_like(lab, bool)
    fade[y0:y1, x0:x1] = True
    # The fade's mid-tones (olive/brown) can project closest to black; nothing
    # else is inside the fade box, so any ink there belongs to the stroke.
    masks["fade"] = (lab > 0) & fade
    masks["black"] &= ~fade

    H = up.shape[0]
    layers = {}
    with tempfile.TemporaryDirectory() as td:
        for n, m in masks.items():
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            pbm = Path(td) / f"{n}.pbm"
            cv2.imwrite(str(pbm), (1 - m) * 255)  # potrace traces black
            svg = subprocess.run(
                ["potrace", "-s", "--flat", "-u", "1", "-t", str(2 * S * S),
                 "-a", str(prm["alphamax"]), "-O", str(prm["opt"]), "-o", "-", str(pbm)],
                check=True, capture_output=True, text=True).stdout
            layers[n] = " ".join(re.findall(r' d="([^"]+)"', svg))
    print(f"  trace: attempt {state['attempt']} {prm}")
    return {"layers": layers, "S": S, "H": H, "prm": prm}


@task(depends_on=[palette, trace])
def assemble(palette, trace):
    S, H, inks = trace["S"], trace["H"], palette["inks"]
    # potrace path space: x_up, (H - y_up); map back to source px.
    tf = f"matrix({1/S:.6f} 0 0 {-1/S:.6f} 0 {H/S:.4f})"
    gx0, gx1 = FADE_X0 * S, FADE_X1 * S
    body = (
        f'<defs><linearGradient id="fade" gradientUnits="userSpaceOnUse" '
        f'x1="{gx0}" y1="0" x2="{gx1}" y2="0">'
        f'<stop offset="0" stop-color="{hexc(inks["green"])}"/>'
        f'<stop offset="1" stop-color="{hexc(inks["orange"])}"/></linearGradient></defs>'
        f'<g transform="{tf}">'
        + "".join(f'<path fill="{hexc(inks[n])}" d="{trace["layers"][n]}"/>'
                  for n in ("green", "orange", "black"))
        + f'<path fill="url(#fade)" d="{trace["layers"]["fade"]}"/></g>')
    h, w = palette["src"].shape[:2]
    full = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
            f'<rect width="{w}" height="{h}" fill="{hexc(palette["bg"])}"/>{body}</svg>')
    lab = classify(palette["src"], palette["bg"], np.array(list(inks.values())))
    ys, xs = np.nonzero(lab)
    pad = 12
    bx, by = xs.min() - pad, ys.min() - pad
    bw, bh = xs.max() + pad - bx, ys.max() + pad - by
    logo = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{bx} {by} {bw} {bh}" '
            f'width="{bw}" height="{bh}">{body}</svg>')
    return {"full": full, "logo": logo}


def matches(v):
    return v["ok"]


@task(depends_on=[palette, assemble], retry=len(ATTEMPTS) - 1, retry_until=matches)
def verify(palette, assemble):
    """Render the full-canvas SVG at source size and compare colour layers."""
    h, w = palette["src"].shape[:2]
    png = subprocess.run(["rsvg-convert", "-w", str(w), "-h", str(h)],
                         input=assemble["full"].encode(), capture_output=True, check=True).stdout
    ren = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR),
                       cv2.COLOR_BGR2RGB).astype(np.float32)
    inks = np.array(list(palette["inks"].values()))
    a = classify(palette["src"], palette["bg"], inks)
    b = classify(ren, palette["bg"], inks)
    iou = {}
    for i, n in enumerate(palette["inks"], 1):
        ma, mb = a == i, b == i
        iou[n] = float((ma & mb).sum() / (ma | mb).sum())
    area = {n: float((b == i).sum() / max(1, (a == i).sum())) for i, n in enumerate(palette["inks"], 1)}
    print("  verify: area ratio render/source", {k: round(v, 3) for k, v in area.items()})
    mad = float(np.abs(ren - palette["src"]).mean())
    ok = all(v > 0.93 for v in iou.values()) and mad < 3.0
    print(f"  verify: IoU {({k: round(v, 3) for k, v in iou.items()})} mean|diff| {mad:.2f} -> {'OK' if ok else 'retry'}")
    if not ok:
        # retry_until re-runs only this task; re-trace with the next settings.
        t = trace.fn(palette) if hasattr(trace, "fn") else None
        if t is not None:
            assemble_v = assemble_fn(palette, t)
            assemble.update(assemble_v)
    return {"ok": ok, "iou": iou, "mad": mad, **assemble}


assemble_fn = assemble.fn if hasattr(assemble, "fn") else None

flow = Flow(verify)
flow.run()
print(flow.summary())
res = flow.value(verify)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "cordillera-logo.svg").write_text(res["logo"])
(OUT / "cordillera-logo-full.svg").write_text(res["full"])
print("wrote", OUT, {k: round(v, 3) for k, v in res["iou"].items()}, round(res["mad"], 2))
