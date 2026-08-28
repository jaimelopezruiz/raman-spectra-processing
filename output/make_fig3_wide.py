"""Widen Figure 4 (single-crystal overlay) to a ~1.585:1 landscape aspect
WITHOUT distorting any glyph.  Ticket C16.

NB the figure was typed "Figure 3" when this script was written; it is Figure 4
in the manuscript's true numbering, and the output was renamed to match in C22.
The source cache keeps its `fig_3_source_image4.png` name — it is a provenance
cache of the docx member `word/media/image4.png`, not a figure.

⚠ THIS IS RASTER SURGERY, NOT A REGENERATION. The underlying single-crystal
spectra are not in this repository (they are Alex's), so the only available
source is the manuscript's own 1000 x 752 raster, extracted from
`Paper WIP Draft - Lilly Edits Implemented.docx` at `word/media/image4.png` and
kept alongside this script as `output/fig_3_source_image4.png`. When those
spectra arrive, regenerate the figure properly and DELETE this script — a
redrawn figure supersedes anything done here.

METHOD
  1. The image splits into three zones:
       x <  X_SPLIT              y-axis title, y tick labels, y-axis spine — untouched
       y <  ROW_SPLIT, x >= X_SPLIT   plot interior — stretched horizontally by f
       y >= ROW_SPLIT            x tick labels + axis title — rebuilt, not stretched
     Only the interior is resampled, so the x axis stays linear and the data
     stay aligned with their ticks: new_x(x) = X_SPLIT + f*(x - X_SPLIT).
  2. Every element that contains a glyph is then re-composited UNSTRETCHED at
     its new position: the stretched copy is erased and the original pixels are
     pasted back, centred on new_x(original centre). Band brackets are left
     stretched (they are rules, not glyphs — agreed in the ticket).
  3. The arrows are erased by COLOUR inside their stretched footprint rather
     than by a white rectangle, so no trace pixel next to them is destroyed.
  4. The legend is repositioned (not scaled) keeping its original right margin.

Every element box below was measured from the source (connected components of
the black ink, plus colour masks for the arrows), and `verify()` re-checks two
things at run time before anything is drawn:
  * the element does not touch its padded box edge  (box would clip a glyph)
  * the padded box contains no trace-coloured pixels (whiteout would eat data)
so a wrong box fails loudly instead of quietly damaging the figure.

Run from the repo root:  .venv\\Scripts\\python.exe output\\make_fig3_wide.py
"""
import os
import sys
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(errors="replace")

OUT = "output"
DOCX = "Paper WIP Draft - Lilly Edits Implemented.docx"
MEDIA = "word/media/image4.png"
SOURCE = os.path.join("data", "figure_sources", "fig_3_source_image4.png")
TARGET = os.path.join(OUT, "fig_4_single_crystal_wide.png")

TARGET_ASPECT = 1.585      # matches the C14 survey overlay (1923 x 1213)
X_SPLIT = 125              # = the y-axis spine / the 0 cm-1 tick: nothing left of it moves
ROW_SPLIT = 669            # axis line 653-655, tick marks to 664, labels start 673

# --- element search regions (x0, x1, y0, y1), generous ----------------------
# The exact box is the tight bounding box of the ink inside each region, computed
# at run time: hand-typed boxes were clipping glyphs (the "FLO" label sits 6 px
# above the folded-LO peak spike, so a 3 px pad measured from a dilated component
# reached into the spike).
TEXT_SEARCH = [
    ("Si-Si (1)", 150, 248, 38, 76),
    ("Si-Si (2)", 252, 348, 38, 76),
    ("Si-C", 356, 446, 38, 76),
    ("C-C", 528, 612, 172, 203),        # between the legend's bottom border (169)
                                        # and the C-C bracket (y >= 205)
    ("FTO", 315, 372, 518, 545),        # stops above the folded-TO spikes (y >= 562)
    ("FLO", 400, 470, 518, 545),        # stops above the folded-LO spike (y >= 548)
    ("FTA", 138, 216, 534, 576),
    ("2nd Order SiC", 565, 735, 548, 592),
]
# The arrows are flat-colour shapes, so they are found by hue rather than by ink:
# their search region also contains trace pixels, and an ink bounding box there
# would swallow the blue spectrum running past the arrow tips. A plain RGB
# distance is no good either — grey text sits within 57 of the purple — so each
# arrow gets a channel-ordering predicate that also holds for its antialiased
# edge (a blend toward white keeps the channel ordering).
#   orange = (255, 190, 0):  r > g > b
#   purple = (108,  43, 158): b > r > g
# The thresholds are deliberately loose enough to include the antialiased edge
# (a light blend still keeps the ordering), so the erase takes the whole stretched
# arrow and the paste brings its soft edge with it — a tighter mask leaves a pale
# halo of the stretched original around the re-placed arrow. Neither predicate can
# match grey/black (r = g = b), blue or green (r <= g), or red (g = b).
# The saturation floor (r-b / b-g) is what keeps the mask off the warm and cool
# colour fringing on the black text and axes, which also satisfies the ordering.
ARROW_DEFS = [
    ("orange arrow", lambda r, g, b: (r > g + 12) & (g > b + 25) & (r - b > 60)),
    ("purple arrow", lambda r, g, b: (b > r + 8) & (r > g + 10) & (b - g > 40)),
]
ARROW_MAX_SIZE = (60, 90)   # sanity bound: an arrow bbox bigger than this means
                            # the hue mask has leaked onto something else
LEGEND = (596, 999, 0, 172)             # right border is flush with the image edge
TEXT_PAD = 3
ARROW_PAD = 2

# x tick labels: (centre_x, x0, x1) with the shared row band; centres coincide
# with the major tick marks measured on the axis (125.0 .. 834.0, step 64.45)
TICK_LABELS = [(124.5, 116, 133), (189.5, 167, 212), (253.5, 231, 276),
               (318.5, 296, 341), (382.5, 360, 405), (447.5, 419, 476),
               (512.5, 484, 541), (577.0, 549, 605), (641.0, 613, 669),
               (705.5, 677, 734), (769.0, 740, 798), (834.0, 805, 863)]
TICK_ROWS = (673, 693)
TITLE_BOX = (365, 611, 716, 748)      # "Raman shift (cm-1)"


def load_source():
    """The docx raster, cached next to this script so the figure is rebuildable
    without the manuscript."""
    if not os.path.exists(SOURCE):
        with zipfile.ZipFile(DOCX) as z:
            with open(SOURCE, "wb") as f:
                f.write(z.read(MEDIA))
        print(f"[extracted] {MEDIA} -> {SOURCE}")
    im = Image.open(SOURCE)
    flat = Image.new("RGB", im.size, (255, 255, 255))
    flat.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
    return flat


src = load_source()
W, H = src.size
a = np.asarray(src).astype(int)
r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
grey = (abs(r - g) < 45) & (abs(g - b) < 45) & (abs(r - b) < 45)
INK = a.sum(axis=2) < 480                      # any dark pixel
COLOUR = (~grey) & (a.sum(axis=2) < 720)       # a trace or an arrow, never text

W_NEW = int(round(H * TARGET_ASPECT))
F = (W_NEW - X_SPLIT) / (W - X_SPLIT)
print(f"source {W} x {H} ({W/H:.3f}:1)  ->  {W_NEW} x {H} ({W_NEW/H:.3f}:1)"
      f"   interior stretch f = {F:.4f}")


def new_x(x):
    return x if x < X_SPLIT else X_SPLIT + F * (x - X_SPLIT)


def tight_box(x0, x1, y0, y1):
    """Bounding box of the ink inside a search region."""
    sub = INK[y0:y1 + 1, x0:x1 + 1]
    ys, xs = np.where(sub)
    if not len(xs):
        raise SystemExit(f"[!] no ink in search region [{x0},{x1}]x[{y0},{y1}]")
    return x0 + int(xs.min()), x0 + int(xs.max()), y0 + int(ys.min()), y0 + int(ys.max())


def check(name, box, pad, allow_colour):
    """Fail loudly on a bad box rather than quietly damaging the figure.

    Two ways a box can be wrong: it clips the element (ink reaches the padded
    edge), or it swallows part of a trace (a whiteout there would delete data).
    """
    x0, x1, y0, y1 = box
    X0, X1 = max(0, x0 - pad), min(W - 1, x1 + pad)
    Y0, Y1 = max(0, y0 - pad), min(H - 1, y1 + pad)
    sub_ink = INK[Y0:Y1 + 1, X0:X1 + 1]
    touches = (sub_ink[0].any() or sub_ink[-1].any()
               or sub_ink[:, 0].any() or sub_ink[:, -1].any())
    n_colour = int(COLOUR[Y0:Y1 + 1, X0:X1 + 1].sum())
    note = []
    if touches:
        note.append("INK TOUCHES PADDED EDGE")
    if n_colour and not allow_colour:
        note.append(f"{n_colour} TRACE PIXELS INSIDE")
    print(f"  {name:16s} box[{x0:4d},{x1:4d}]x[{y0:4d},{y1:4d}] +{pad}px"
          f"  ink={int(sub_ink.sum()):5d} colour={n_colour:5d}"
          f"   {'  '.join(note) if note else 'ok'}")
    return note


def hue_mask(arr, pred):
    return pred(arr[:, :, 0], arr[:, :, 1], arr[:, :, 2])


print("\nElement boxes (tight bbox of the ink in each search region):")
TEXT_ELEMENTS = [(n, *tight_box(*s)) for n, *s in TEXT_SEARCH]
ARROWS = []
for name, pred in ARROW_DEFS:
    ys, xs = np.where(hue_mask(a, pred))
    ARROWS.append((name, pred, int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())))

problems = []
for name, *box in TEXT_ELEMENTS:
    problems += [(name, n) for n in check(name, box, TEXT_PAD, allow_colour=False)]
for name, pred, *box in ARROWS:
    # for an arrow the edge test must use its own hue, not "any ink"
    x0, x1, y0, y1 = box
    m = hue_mask(a, pred)
    sub = m[y0 - ARROW_PAD:y1 + ARROW_PAD + 1, x0 - ARROW_PAD:x1 + ARROW_PAD + 1]
    touching = sub[0].any() or sub[-1].any() or sub[:, 0].any() or sub[:, -1].any()
    oversize = (x1 - x0 + 1 > ARROW_MAX_SIZE[0]) or (y1 - y0 + 1 > ARROW_MAX_SIZE[1])
    note = (["ARROW HUE TOUCHES PADDED EDGE"] if touching else []) \
        + (["HUE MASK LEAKED (bbox too large)"] if oversize else [])
    print(f"  {name:16s} box[{x0:4d},{x1:4d}]x[{y0:4d},{y1:4d}] +{ARROW_PAD}px"
          f"  hue px={int(m.sum()):5d} (in box {int(sub.sum())})"
          f"   {'  '.join(note) if note else 'ok'}")
    problems += [(name, n) for n in note]
# the legend's own colour pixels are its line swatches, which sit left of x=730;
# anything coloured to the right of that would be a trace intruding into the box
lg_colour_right = int(COLOUR[LEGEND[2]:LEGEND[3] + 1, 730:LEGEND[1] + 1].sum())
problems += [("legend", n) for n in check("legend", LEGEND, 0, allow_colour=True)]
print(f"  {'legend swatch chk':16s} colour pixels right of x=730: {lg_colour_right}"
      f"   {'ok' if lg_colour_right < 50 else 'A TRACE INTRUDES INTO THE LEGEND BOX'}")
if lg_colour_right >= 50:
    problems.append(("legend", "trace intrudes"))
if problems:
    print("\n[!] refusing to build: "
          + "; ".join(f"{n}: {v}" for n, v in problems))
    raise SystemExit(1)

# --- 1. geometry -------------------------------------------------------------
canvas = Image.new("RGB", (W_NEW, H), (255, 255, 255))
# untouched left column, full height
canvas.paste(src.crop((0, 0, X_SPLIT, H)), (0, 0))
# stretched interior (everything above the tick-label band)
interior = src.crop((X_SPLIT, 0, W, ROW_SPLIT))
canvas.paste(interior.resize((W_NEW - X_SPLIT, ROW_SPLIT), Image.LANCZOS), (X_SPLIT, 0))

# --- 2. glyph-bearing elements, re-composited unstretched --------------------
def replace(x0, x1, y0, y1, pad, target_cx=None, arrow_pred=None):
    """Erase the stretched copy of an element and paste the original pixels back.

    Text and the legend sit on verified-clean white, so their stretched footprint
    is simply whited out. An arrow instead has a spectrum running past its tip, so
    only pixels still carrying the ARROW'S OWN colour are whitened and the arrow is
    pasted through its own mask — no trace pixel beside it is touched.
    """
    X0, X1 = max(0, x0 - pad), min(W - 1, x1 + pad)
    Y0, Y1 = max(0, y0 - pad), min(H - 1, y1 + pad)
    crop = src.crop((X0, Y0, X1 + 1, Y1 + 1))
    cw = X1 - X0 + 1
    sx0, sx1 = int(round(new_x(X0))), int(round(new_x(X1 + 1)))
    if arrow_pred is not None:
        block = np.asarray(canvas.crop((sx0, Y0, sx1, Y1 + 1))).astype(int)
        block[hue_mask(block, arrow_pred)] = 255
        canvas.paste(Image.fromarray(block.astype(np.uint8)), (sx0, Y0))
    else:
        canvas.paste(Image.new("RGB", (sx1 - sx0, Y1 - Y0 + 1), (255, 255, 255)), (sx0, Y0))
    cx = target_cx if target_cx is not None else new_x((X0 + X1 + 1) / 2)
    px = int(round(cx - cw / 2))
    if arrow_pred is not None:
        m = hue_mask(np.asarray(crop).astype(int), arrow_pred)
        canvas.paste(crop, (px, Y0), Image.fromarray((m * 255).astype(np.uint8), "L"))
    else:
        canvas.paste(crop, (px, Y0))
    return px, px + cw - 1


for name, x0, x1, y0, y1 in TEXT_ELEMENTS:
    replace(x0, x1, y0, y1, TEXT_PAD)
for name, pred, x0, x1, y0, y1 in ARROWS:
    replace(x0, x1, y0, y1, ARROW_PAD, arrow_pred=pred)

# legend: repositioned, not scaled, keeping its original right margin
lx0, lx1, ly0, ly1 = LEGEND
right_margin = (W - 1) - lx1
lp0, lp1 = replace(lx0, lx1, ly0, ly1, 0,
                   target_cx=(W_NEW - 1 - right_margin) - (lx1 - lx0) / 2)
print(f"\nlegend moved x[{lx0},{lx1}] -> x[{lp0},{lp1}] (right margin kept at {right_margin} px)")

# --- 3. tick-label band, rebuilt at the new tick positions -------------------
canvas.paste(Image.new("RGB", (W_NEW, H - ROW_SPLIT), (255, 255, 255)), (0, ROW_SPLIT))
ty0, ty1 = TICK_ROWS
for cx, x0, x1 in TICK_LABELS:
    crop = src.crop((x0, ty0, x1 + 1, ty1 + 1))
    canvas.paste(crop, (int(round(new_x(cx) - (x1 - x0 + 1) / 2)), ty0))
tx0, tx1, tty0, tty1 = TITLE_BOX
title = src.crop((tx0, tty0, tx1 + 1, tty1 + 1))
canvas.paste(title, (int(round(new_x((tx0 + tx1 + 1) / 2) - (tx1 - tx0 + 1) / 2)), tty0))
print(f"x tick labels re-placed at {[round(new_x(c), 1) for c, _, _ in TICK_LABELS]}")
print(f"axis title centre {(tx0+tx1+1)/2:.1f} -> {new_x((tx0+tx1+1)/2):.1f}")

canvas.save(TARGET)
print(f"\n[OK] {TARGET}   {canvas.size[0]} x {canvas.size[1]} px"
      f"   aspect {canvas.size[0]/canvas.size[1]:.3f}:1")
