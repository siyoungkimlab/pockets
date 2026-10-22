# Finding ligand pockets in apo structures

A static site (GitHub Pages) for apo/holo pairs of the Schrödinger set (43) and the
adversarially challenging sites set (4): the apo
and holo structures, the top 10 sites Schrödinger MxMD + SiteMap finds (Schrödinger
set only), and the top 10 pockets CRUMB (`boonza cavity traj`, volume200, p_max)
finds in coarse-grained simulations of the apo protein with 18 amino-acid probes,
Martini 3 without and with a Go model, each in the MD frame it was found in.

Drawn with `boonza view` (boonza.viewer.export) by `build_pages.py`: `viewer/` is
the viewer's code, shared by every pair; each pair's folder holds its page and
`data/`.
