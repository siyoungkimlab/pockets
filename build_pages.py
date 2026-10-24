"""Pages of the pockets site (siyoungkimlab/pockets) for apo/holo pairs: the apo and
holo structures and the CRUMB pockets of chosen runs, each pocket with the MD frame it
was found in (boonza cavity traj, cavity_volume/), drawn with boonza.viewer.export.

    python build_pages.py SET_ROOT SITE PAIR [PAIR ...] [--mxmd MxMD_SiteMap]

--mxmd: Schrodinger MxMD + SiteMap results (mxmd_sitemap/PAIR.maegz, each site after
its MD frame, and mxmd_sitemap_result/PAIR.csv), the top sites by SiteScore drawn too.

Every run's coordinates are put in the frame of the first run's (its cavity_volume
apo.mae / holo.mae, already superposed on that run), fitted on the apo heavy atoms.
Each pocket is one model: the run's protein beads (gray) at the MD frame the pocket was
found in, and the pocket's medoid site (each rank its own colour).  The
shared viewer code goes to SITE/viewer/, each pair's page and data/ to SITE/PAIR/.
"""

import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

import boonza
from boonza.cavity.report import _drawable
from boonza.pockets.prepare import protein_ids
from boonza.system import System
from boonza.viewer import Model, Session, Viewer, bead_radii
from boonza.viewer.export import STATIC, export
from boonza.viewer.outputs import POCKET_COLOURS, points
from boonza.viewer.session import CG_POLYMER

#: (library, model, colour): the CRUMB runs drawn
RUNS = [("SingleAminoAcid18", "martini3", "#b07ae6"),
        ("SingleAminoAcid18_Go", "martini3", "#59a14f")]  # fmt: skip
TOP = 10  # pockets drawn per run (cavity traj writes the medoid frames of these)
SITE_RADIUS = 1.0  # A: a site point, half the grid spacing
POLAR = "(not hydrogen or withinbonds 1 of (element N O S))"
POCKET = 5.0  # A: residues this near the ligand are the holo site
CA_POCKET = 8.0  # A: CA to ligand heavy atom, the pocket of the pocket RMSD
COLUMNS = ["rank", "SiteScore", "Dscore", "volume", "probe", "rep", "frame", "p_max",
           "p_q90_open", "occupancy", "median_volume", "PPc"]
MXMD_COLOUR = "#59c9a5"
PROTEIN_COLOUR = "gray"  # every pocket's and site's protein; its points take its rank's colour


def cavity_dir(root: Path, lib: str, ff: str, pair: str) -> Path:
    pdb = pair.split("_")[0]
    for d in (root / "run" / lib / ff / pdb / "sim_000" / pair / "cavity_volume",
              root / "run" / lib / ff / pdb / pair / "cavity_volume"):  # fmt: skip
        if (d / "pockets.csv").is_file():
            return d
    raise FileNotFoundError(f"no cavity_volume for {lib}/{ff}/{pair}")


def table(s):
    """Per atom: chain name, resid, atom name, element number."""
    a, r, c = s.atoms.to_pandas(), s.residues.to_pandas(), s.chains.to_pandas()
    res = np.asarray(a["residue"])
    chain = np.asarray(c["name"])[np.asarray(r["chain"])[res]]
    return chain, np.asarray(r["resid"])[res], np.asarray(a["name"]), np.asarray(a["anum"])


def labels(s) -> list:
    """(chain, resid, atom name) per atom."""
    chain, resid, name, _ = table(s)
    return list(zip(chain, resid.tolist(), name))


def kabsch(mobile, ref):
    """R, t with mobile @ R + t closest to ref."""
    mc, rc = mobile.mean(0), ref.mean(0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (ref - rc))
    d = np.sign(np.linalg.det(u @ vt))
    R = u @ np.diag([1, 1, d]) @ vt
    return R, rc - mc @ R


def site_sel(site: dict) -> str:
    return " or ".join(f"(chain {c} and resid {' '.join(map(str, rs))})" for c, rs in site.items())


def residue_map(apo, holo) -> dict:
    """{holo residue index: apo residue index}, paired by aligning the sequences
    (boonza.align.superpose, match="sequence"): the two may be numbered differently."""
    from boonza.align import superpose

    fit = superpose(apo, holo, sel="protein and name CA", match="sequence", cutoff=None,
                    apply=False)  # fmt: skip
    ares = np.asarray(apo.atoms.to_pandas()["residue"])
    hres = np.asarray(holo.atoms.to_pandas()["residue"])
    return {int(hres[h]): int(ares[a]) for a, h in zip(fit.mobile_atoms, fit.reference_atoms)}


def residue_sel(system, residues) -> str:
    chain, resid, _, _ = table(system)
    res = np.asarray(system.atoms.to_pandas()["residue"])
    first = {}
    for i, r in enumerate(res):
        first.setdefault(int(r), i)
    site = {}
    for r in sorted(residues):
        i = first[r]
        site.setdefault(chain[i], []).append(int(resid[i]))
    return site_sel(site)


def pocket_rmsd(apo, holo, holo_res: list, rmap: dict):
    """Heavy-atom RMSD of the holo site's residues, apo vs holo (residues paired by
    sequence, atoms by name), fitted on their alpha carbons."""
    _, _, an, ae = table(apo)
    _, _, hn, he = table(holo)
    ares = np.asarray(apo.atoms.to_pandas()["residue"])
    hres = np.asarray(holo.atoms.to_pandas()["residue"])
    hk = {(int(hres[i]), hn[i]): i for i in range(len(hn)) if he[i] > 1}
    back = {a: h for h, a in rmap.items() if h in set(holo_res)}
    pairs = [(i, hk[(back[int(ares[i])], an[i])]) for i in range(len(an))
             if ae[i] > 1 and int(ares[i]) in back and (back[int(ares[i])], an[i]) in hk]
    if len(pairs) < 3:
        return None, 0
    a = np.array([p[0] for p in pairs])
    h = np.array([p[1] for p in pairs])
    is_ca = np.array([an[i] == "CA" for i in a])
    P, Q = apo.positions[a], holo.positions[h]
    R, t = kabsch(P[is_ca], Q[is_ca])
    return float(np.sqrt((((P @ R + t) - Q) ** 2).sum(1).mean())), len(pairs)


def pocket_ca_rmsd(apo, holo, lig: str, rmap: dict, cutoff: float = CA_POCKET):
    """(CA RMSD, pocket residues, paired residues): the holo residues whose CA is within
    ``cutoff`` of a ligand heavy atom, apo vs holo (residues paired by sequence), fitted on
    those same CAs."""
    lx = holo.positions[holo.select(f"({lig}) and not hydrogen").ids]
    hca = holo.select(f"protein and name CA and not ({lig})").ids
    d = np.sqrt(((holo.positions[hca][:, None] - lx[None]) ** 2).sum(-1)).min(1)
    hres = np.asarray(holo.atoms.to_pandas()["residue"])
    ares = np.asarray(apo.atoms.to_pandas()["residue"])
    pocket = [int(hres[i]) for i in hca[d <= cutoff]]
    apo_ca = {int(ares[i]): i for i in apo.select("protein and name CA").ids}
    holo_ca = {int(hres[i]): i for i in hca}
    pairs = [(apo_ca[rmap[r]], holo_ca[r]) for r in pocket if rmap.get(r) in apo_ca]
    if len(pairs) < 3:
        return None, len(pocket), len(pairs)
    P = apo.positions[[a for a, _ in pairs]]
    Q = holo.positions[[h for _, h in pairs]]
    R, t = kabsch(P, Q)
    return float(np.sqrt((((P @ R + t) - Q) ** 2).sum(1).mean())), len(pocket), len(pairs)


def burial(holo, lig: str) -> float:
    """1 - SASA(ligand in the complex) / SASA(ligand alone), on heavy atoms only."""
    from boonza.shrake_rupley import sasa

    heavy = holo.clone(holo.select("not hydrogen").ids)
    ids = heavy.select(f"({lig})").ids
    extra = {"I": 1.98, "Br": 1.85}  # A: Bondi radii mdtraj's table lacks
    in_complex = sasa(heavy, atoms=ids, radii=extra)[0][ids].sum()
    by_itself = sasa(heavy.clone(ids), radii=extra)[0].sum()
    return float(1 - in_complex / by_itself)


def mxmd_models(mxmd: Path, pair: str, ref_apo, top: int) -> list:
    """Schrodinger MxMD + SiteMap: the ``top`` sites by SiteScore, each one model -- its MD
    frame (all atoms, fitted onto ``ref_apo`` on the alpha carbons) and its site points."""
    import csv

    from boonza.align import superpose
    from boonza.io._maeparse import parse_mae, read_text
    from boonza.io.mae import _append_ct

    rows = sorted(csv.DictReader(open(mxmd / "mxmd_sitemap_result" / f"{pair}.csv")),
                  key=lambda r: int(r["rank"]))[:top]  # fmt: skip
    want = {r["site"] for r in rows} | {r["protein"] for r in rows}
    entries = {}
    maegz = mxmd / "mxmd_sitemap" / f"{pair}.maegz"
    if not maegz.is_file():  # one MxMD run for two ligands of a holo (5uxa_5igy_ery, _gdp)
        maegz = mxmd / "mxmd_sitemap" / f"{pair.rsplit('_', 1)[0]}.maegz"
    for ct in parse_mae(read_text(str(maegz))):
        title = ct.get("m_title")
        if title in want and title not in entries:
            entries[title] = System(title)
            _append_ct(entries[title], ct, False, True)
    models, fits = [], []
    for r in rows:
        k = int(r["rank"])
        system = entries[r["protein"]].clone()
        pts = entries[r["site"]].positions
        n = len(pts)
        system.append(System.from_arrays(np.asarray(pts, float), names=["X"] * n,
                                         anum=np.zeros(n, int), resnames=["PKT"] * n,
                                         resids=np.arange(1, n + 1)))  # fmt: skip
        fit = superpose(system, ref_apo, sel="protein and name CA", match="sequence")
        fits.append(fit.rmsd)
        lvc, pvn = float(r["ligand_volume_covered"]), float(r["pocket_volume_near_ligand"])
        probe, rep = r["site"].split("-rep")[0], int(r["site"].split("-rep")[1].split("-")[0])
        row = {"rank": k, "SiteScore": round(float(r["SiteScore"]), 3),
               "Dscore": round(float(r["Dscore"]), 3), "volume": round(float(r["volume"]), 1),
               "probe": probe, "rep": rep, "frame": int(r["Trajectory_frame"]),
               "PPc": "yes" if r["PPc"] == "True" else "no"}  # fmt: skip
        summary = overlap_summary(float(r["center_to_nearest_ligand_atom"]), lvc, pvn,
                                  "from the site's centre to the nearest ligand heavy atom "
                                  "(PPc: under 4 Å)")  # fmt: skip
        spot = POCKET_COLOURS[(k - 1) % len(POCKET_COLOURS)]

        def load_site(system=system, spot=spot, k=k):
            reps = [{"sel": "(protein or nucleic) and not resname PKT", "style": "cartoon",
                     "color": PROTEIN_COLOUR, "label": "protein"},
                    {"sel": "resname PKT", "style": "spheres", "color": spot,
                     "label": f"site {k}"}]  # fmt: skip
            return Session(system, title=f"site {k}", ligand="resname PKT", reps=reps,
                           sizes={"resname PKT": SITE_RADIUS})  # fmt: skip

        models.append(Model(f"site {k} · {probe} rep{rep}", load_site,
                            group="Schrödinger MxMD + SiteMap", shown=False, row=row,
                            label=str(k), color=PROTEIN_COLOUR, summary=summary))  # fmt: skip
    return models, fits


def overlap_summary(dist: float, lvc: float, pvn: float, how: str) -> list:
    return [
        {"label": "centroid → ligand", "value": f"{dist:.1f} Å", "title": how},
        {"label": "coverage", "value": f"{lvc:.0%}",
         "title": "share of the ligand's volume inside the site (LVC, |L∩P|/|L|)"},
        {"label": "precision", "value": f"{pvn:.0%}",
         "title": "share of the site's volume within 2 Å of the ligand (PVN)"},
        {"label": "fill", "value": f"{np.sqrt(lvc * pvn):.0%}", "title": "sqrt(LVC × PVN)"},
    ]  # fmt: skip


def reps_structure(site: str, ligand: str | None = None):
    reps = [{"sel": "(protein or nucleic) and not resname LIG", "style": "cartoon",
             "color": "white", "label": "polymer"}]  # fmt: skip
    if ligand:
        reps += [{"sel": f"({ligand}) and {POLAR}", "style": "sticks", "color": "element/#ffd9a0",
                  "label": "ligand atoms"},
                 {"sel": f"(same residue as ((protein or nucleic) and within {POCKET:g} of "
                         f"({ligand}))) and not ({ligand}) and {POLAR}", "style": "sticks",
                  "color": "element", "label": "pocket", "on": False}]  # fmt: skip
    reps.append({"sel": f"({site}) and {POLAR}", "style": "sticks", "color": "element",
                 "label": "holo site", "on": False})  # fmt: skip
    return reps


def page(root: Path, pair: str, out: Path, mxmd: Path | None = None) -> dict:
    import csv

    dirs = [cavity_dir(root, lib, ff, pair) for lib, ff, _ in RUNS]
    lig = json.loads((dirs[0] / "ligand.json").read_text())["selection"]
    apo, holo = boonza.load(dirs[0] / "apo.mae"), boonza.load(dirs[0] / "holo.mae")
    lig_xyz = holo.positions[holo.select(f"({lig}) and not hydrogen").ids]
    hres = np.asarray(holo.atoms.to_pandas()["residue"])
    near = holo.select(f"(same residue as (protein and within {POCKET:g} of ({lig}))) "
                       f"and not ({lig})").ids  # fmt: skip
    holo_res = sorted({int(hres[i]) for i in near})
    rmap = residue_map(apo, holo)
    sel_holo = residue_sel(holo, holo_res)
    sel_apo = residue_sel(apo, [rmap[r] for r in holo_res if r in rmap])
    rmsd, n_pocket, n_paired = pocket_ca_rmsd(apo, holo, lig, rmap)
    bur = burial(holo, lig)
    n_res = len(holo_res)

    models = [
        Model("apo", lambda: Session(apo, title="apo", reps=reps_structure(sel_apo)),
              group="structure", shown=False, color="#4e9be6"),
        Model("holo", lambda: Session(holo, title="holo", ligand=lig, carbon="#ffd9a0",
                                      reps=reps_structure(sel_holo, lig)),
              group="structure", shown=True, color="#f28e2b"),
    ]  # fmt: skip
    counts = []
    mxmd_fits = []
    if mxmd is not None:
        sites, mxmd_fits = mxmd_models(mxmd, pair, apo, TOP)
        models += sites
    ref_apo = apo.positions[apo.select("not hydrogen").ids]
    ref_ca = apo.positions[apo.select("name CA").ids]
    fits = []
    for (lib, ff, colour), d in zip(RUNS, dirs):
        run_apo = boonza.load(d / "apo.mae")
        R, t = kabsch(run_apo.positions[run_apo.select("not hydrogen").ids], ref_apo)
        run_holo = boonza.load(d / "holo.mae")  # this run's frame, for distances
        run_lig = run_holo.positions[run_holo.select(f"({lig}) and not hydrogen").ids]
        rows = {int(r["rank"]): r for r in csv.DictReader(open(d / "pockets.csv"))}
        frames = sorted(d.glob("frame_*_md*.mae"), key=lambda f: int(f.name.split("_")[1]))
        frames = frames[:TOP]
        tag = f"CRUMB · {lib} · {ff}"
        # the beads as cavity traj reads them: the run's solvated.dms, its protein beads.
        # A frame file is view.dms's atoms at those indices (the wrong atoms in a Go run,
        # whose view.dms is ordered differently), regrouped by residue when saved: its
        # positions go back to their beads by those labels
        w = d.parent.parent
        workdir = w / "md" if (w / "md" / "solvated.dms").is_file() else w
        system = boonza.load(workdir / "solvated.dms")
        ids = protein_ids(system)
        view = workdir / "view.dms"
        view = boonza.load(view) if view.is_file() else system
        if view.natoms != system.natoms:
            view = system
        slot = {lab: i for i, lab in enumerate(labels(view.clone(ids)))}
        if len(slot) != len(ids):
            raise ValueError(f"{workdir}: the protein beads' labels are not unique")
        for f in frames:
            k = int(f.name.split("_")[1])
            r = rows[k]
            frame = boonza.load(f)
            xyz = np.full((len(ids), 3), np.nan)
            for j, lab in enumerate(labels(frame)):
                xyz[slot[lab]] = frame.positions[j]
            if not np.isfinite(xyz).all():
                raise ValueError(f"{f}: not every protein bead of the run is in it")
            beads = _drawable(system, ids, xyz @ R + t)
            bb = beads.positions[beads.select("name BB GC").ids]
            if len(bb) == len(ref_ca):
                fits.append(float(np.sqrt(((bb - ref_ca) ** 2).sum(1).mean())))
            pts, _, _ = points(d / "sites.pqr", k)
            centroid = pts.mean(0)
            dist = float(np.sqrt(((run_lig - centroid) ** 2).sum(1)).min())
            lvc, pvn = float(r["LVC"] or 0), float(r["PVN"] or 0)
            row = {"rank": k, "p_max": round(float(r["p_max"]), 3),
                   "p_q90_open": round(float(r["p_q90_open"]), 3),
                   "occupancy": round(float(r["occupancy"]), 3),
                   "median_volume": float(r["median_volume"]), "frame": int(r["md_frame"]),
                   "PPc": "yes" if r["PPc"] == "True" else "no"}  # fmt: skip
            summary = overlap_summary(dist, lvc, pvn,
                                      "from the site's centroid to the nearest ligand heavy "
                                      "atom (PPc, at the pocket's centre: under 4 Å)")  # fmt: skip
            spot = POCKET_COLOURS[(k - 1) % len(POCKET_COLOURS)]

            def load_pocket(beads=beads, pts=pts @ R + t, colour=colour, spot=spot, k=k):
                s = beads.clone()
                radii = np.r_[bead_radii(beads), np.full(len(pts), SITE_RADIUS)]
                n = len(pts)
                s.append(System.from_arrays(np.asarray(pts, float), names=["X"] * n,
                                            anum=np.zeros(n, int), resnames=["PKT"] * n,
                                            resids=np.arange(1, n + 1)))  # fmt: skip
                reps = [{"sel": f"{CG_POLYMER} and not resname PKT", "style": "cartoon",
                         "color": PROTEIN_COLOUR, "label": "protein"},
                        {"sel": "resname PKT", "style": "spheres", "color": spot,
                         "label": f"pocket {k}"}]  # fmt: skip
                session = Session(s, title=f"pocket {k}", ligand="resname PKT", cg=True,
                                  reps=reps, radii=radii)  # fmt: skip
                # with radii given, boonza leaves the beads' secondary structure blank:
                # work it out from the protein's trace alone, the site's points none
                ss = Session(beads, cg=True).ss(0)
                session._ss[0] = ss + [""] * (s.nresidues - beads.nresidues)
                return session

            models.append(Model(f"pocket {k}", load_pocket, group=tag, shown=False, row=row,
                                label=str(k), color=PROTEIN_COLOUR, summary=summary))  # fmt: skip
        counts.append(len(frames))

    title = f"{pair}: apo/holo, Schrödinger and CRUMB" if mxmd else f"{pair}: apo/holo and CRUMB"
    v = Viewer(models, title=title, columns=[c for c in COLUMNS if any(c in m.row for m in models)])
    v.summary = [
        {"label": "pocket RMSD", "value": f"{rmsd:.2f} Å" if rmsd is not None else "—",
         "title": f"apo vs holo, CA RMSD of the {n_pocket} residues with a CA within "
                  f"{CA_POCKET:g} Å of a ligand heavy atom ({n_paired} in both), fitted on those CAs"},
        {"label": "burial", "value": f"{bur:.1%}",
         "title": "1 - SASA(ligand in the complex) / SASA(ligand alone), heavy atoms"},
    ]  # fmt: skip
    if out.exists():
        shutil.rmtree(out)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = export(v, out)
    # one copy of the viewer's code for every page, in SITE/viewer/
    for f in STATIC.iterdir():
        if f.is_file() and f.name != "index.html":
            (out / f.name).unlink(missing_ok=True)
    html = (out / "index.html").read_text()
    html = html.replace('href="style.css"', 'href="../viewer/style.css"')
    html = html.replace('src="main.js"', 'src="../viewer/main.js"')
    (out / "index.html").write_text(html)
    return {"pair": pair, "rmsd": rmsd, "burial": bur, "pockets": sum(counts),
            "mxmd_sites": len(mxmd_fits),
            "mxmd_fit_rmsd": [round(min(mxmd_fits), 2), round(max(mxmd_fits), 2)] if mxmd_fits else None,
            "BB_to_apo_CA_rmsd": [round(min(fits), 2), round(max(fits), 2)] if fits else None,
            **info}


def main():
    args = sys.argv[1:]
    mxmd = None
    if "--mxmd" in args:
        i = args.index("--mxmd")
        mxmd = Path(args[i + 1])
        del args[i : i + 2]
    root, site, pairs = Path(args[0]), Path(args[1]), args[2:]
    viewer = site / "viewer"
    viewer.mkdir(parents=True, exist_ok=True)
    for f in STATIC.iterdir():
        if f.is_file() and f.name != "index.html":
            shutil.copyfile(f, viewer / f.name)
    for pair in pairs:
        print(json.dumps(page(root, pair, site / pair, mxmd)), flush=True)


if __name__ == "__main__":
    main()
