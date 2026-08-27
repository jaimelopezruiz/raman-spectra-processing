"""Restore the PUBLISHED figure files in output/ from the manuscript's own
media.  Ticket C20 part A.

WHY. output/ is git-ignored, so a bad regeneration cannot be undone with git.
The authoritative copy of every published figure is therefore the raster that is
actually embedded in the Word documents. This script copies those bytes back
onto the published filenames, so output/ once again matches what the manuscript
shows. The C19 restyle survives only under *_restyled.png names.

VERIFIED AT WRITE TIME: main media/image6.png has md5 de59f0fbd214..., which is
byte-identical to the C18 build of the Figure 6 raster (the version carrying the
Koyanagi dotted line). Co-work had already swapped it in, so restoring Figure 6
from the doc restores C18's work exactly rather than the pre-C18 figure.

FILENAMES: the destinations below use the manuscript's TRUE figure numbers
(renamed in C22). Their pre-C22 names encoded the superseded typed numbers —
fig_9_fwhm_lo.png was Figure 6 and fig_11_annealing_4panel.png was Figure 9.

RESOLUTION CAVEAT. Word recompresses images on insert, so most of these media
are smaller than the rasters originally generated: e.g. Figure 3 is 1137x717 in
the doc where make_figures produced 1923x1213. Restoring from the doc therefore
returns the published *content* at the published *resolution* — which is what
the manuscript actually renders — but output/ no longer holds a higher-resolution
master for those figures. Figure 6 is unaffected (bit-identical).

NOT RESTORED (no source; neither document embeds them): fig_fto_fwhm.png,
fig_lo_vs_T.png, fig_annealing_overlay_au.png, fig_annealing_overlay_si300.png.
These are standalone reference panels, not figure slots; they are left in
whatever state the last generator run produced and are flagged in the report.

Run from the repo root:  .venv\\Scripts\\python.exe output\\restore_published_figures.py
"""
import hashlib
import io
import os
import sys
import zipfile

from PIL import Image

sys.stdout.reconfigure(errors="replace")

OUT = "output"
MAIN = "Paper WIP Draft - Lilly Edits Implemented.docx"
SUPP = "Raman Paper Supplementary Materials.docx"

# (destination in output/, source docx, media member, what it is)
RESTORE = [
    ("fig_6_fwhm_lo.png",                    MAIN, "word/media/image6.png",
     "Figure 6, FTO FWHM + folded-LO (C18 build, with the Koyanagi line)"),
    ("fig_survey_overlay.png",               MAIN, "word/media/image3.png",
     "Figure 3, polycrystalline survey overlay"),
    ("fig_9_annealing_4panel.png",           MAIN, "word/media/image9.png",
     "Figure 9, stepwise annealing, 4 panels"),
    ("fig_SM_annealing_overlay_au_full.png", SUPP, "word/media/image2.png",
     "Figure SM.2, Au full series"),
    ("fig_SM_annealing_overlay_ne_full.png", SUPP, "word/media/image3.png",
     "Figure SM.3, Ne full series"),
    ("fig_SM_annealing_overlay_si300_full.png", SUPP, "word/media/image4.png",
     "Figure SM.4, Si 2.5 dpa 300 °C full series"),
    ("fig_SM_annealing_overlay_si750_full.png", SUPP, "word/media/image5.png",
     "Figure SM.5, Si 2.5 dpa 750 °C full series"),
]
# created by C19, never published, nothing to restore it to
DELETE = ["fig_7_cc_morphology.png"]


def main():
    zips = {p: zipfile.ZipFile(p) for p in {src for _, src, _, _ in RESTORE}}
    ok = True
    for dest, src, member, what in RESTORE:
        raw = zips[src].read(member)
        path = os.path.join(OUT, dest)
        with open(path, "wb") as fh:
            fh.write(raw)
        want = hashlib.md5(raw).hexdigest()
        with open(path, "rb") as fh:
            got = hashlib.md5(fh.read()).hexdigest()
        size = Image.open(io.BytesIO(raw)).size
        flag = "OK " if got == want else "FAIL"
        ok &= got == want
        print(f"[{flag}] {dest:42s} <- {os.path.basename(src)[:18]:18s} {member[11:]:14s} "
              f"{size[0]:5d}x{size[1]:<5d} md5={got[:12]}  {what}")

    for name in DELETE:
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            os.remove(path)
            print(f"[DEL] {name} (C19-only, no published counterpart)")
        else:
            print(f"[--]  {name} already absent")

    print("\nRESTORE GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
