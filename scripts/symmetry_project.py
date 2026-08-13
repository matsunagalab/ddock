"""Project homodimer poses onto exact C2 symmetry.

Report section 5.14.44 measured that same-UniProt homodimers are 28.4% of
`pinder_s` and sit 13.9 pp (acceptable) to 22.5 pp (high) below the rest, and
that every homodimer native whose chains can be put in correspondence is C2
(rotation angle median 180.0 deg). The search is a general six-degree-of-freedom
rigid docking that never uses that constraint.

This script does the cheap half of the fix: it takes poses that have already
been written out and moves the ligand rigidly so the complex becomes exactly
C2. No re-search, no training. If that alone recovers part of the deficit, a
constrained search (M-ZDOCK, Pierce/Tong/Weng 2005) is worth implementing.

The relation between the two subunits is frame independent. If both chains are
moved by a global rigid motion G, the transform T taking receptor onto ligand
becomes G T G^-1, and conjugation preserves both the rotation angle and the
screw translation. So the exported poses can be analysed in whatever frame the
docking left them in.

A rigid motion is an involution exactly when its rotation is by pi and it has
no translation along the axis. Projection therefore replaces the rotation by
the pi-rotation about the same axis and drops the axial part of the
translation. The correction is then applied to the ligand as a rigid body, so
the ligand keeps its own conformation instead of becoming a copy of the
receptor.

    uv run python scripts/symmetry_project.py audit \
        --ids-file <ids> --out data/scaling/c2_audit.csv

    uv run python scripts/symmetry_project.py project \
        --in-dir data/pinder_eval/trained_SYM_top50_refined \
        --out-dir data/pinder_eval/trained_SYM_top50_c2 \
        --ids-file <homodimer ids> --top-k 50
"""

from __future__ import annotations

import argparse
import csv
import shutil
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from biotite.structure.io.pdb import PDBFile  # noqa: E402

NATIVE_DIR = Path("external/pinder/pinder/2024-02/pdbs")


def is_homodimer(system_id: str) -> bool:
    """Both sides of a PINDER id carry the same UniProt accession."""
    try:
        a, b = system_id.split("--")
    except ValueError:
        return False
    return a.split("_")[-1] == b.split("_")[-1]


def _read(path: Path):
    return PDBFile.read(str(path)).get_structure(model=1)


def correspondence(arr, atom_name: str = "CA"):
    """Matched receptor/ligand coordinates, keyed on (res_id, atom_name).

    PINDER keeps the deposited numbering, so the two chains of a homodimer
    share residue ids. Matching on the id rather than on sequential order is
    what makes this robust to gaps and to unequal chain lengths -- ordering by
    position silently mispairs the chains whenever one has a missing loop.
    """
    sel = arr.atom_name == atom_name
    rec, lig = arr[sel & (arr.chain_id == "R")], arr[sel & (arr.chain_id == "L")]
    lig_by_res = {int(r): i for i, r in enumerate(lig.res_id)}
    ri, li = [], []
    for i, r in enumerate(rec.res_id):
        j = lig_by_res.get(int(r))
        if j is not None:
            ri.append(i)
            li.append(j)
    if not ri:
        return None, None
    return rec.coord[ri], lig.coord[li]


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid (rot, shift) minimising |rot @ p + shift - q|, no reflection."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    rot = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return rot, qc - rot @ pc


def rotation_angle(rot: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))))


def rotation_axis(rot: np.ndarray) -> np.ndarray:
    """Unit vector fixed by `rot`.

    The usual antisymmetric-part formula degenerates exactly at 180 degrees,
    which is the case this script cares about, so take the eigenvector for
    eigenvalue +1 instead.
    """
    w, v = np.linalg.eig(rot)
    k = int(np.argmin(np.abs(w - 1.0)))
    u = np.real(v[:, k])
    return u / np.linalg.norm(u)


def project_involution(rot: np.ndarray, shift: np.ndarray):
    """Nearest rigid motion with rot^2 = I and no screw component."""
    u = rotation_axis(rot)
    rot_pi = 2.0 * np.outer(u, u) - np.eye(3)
    screw = float(shift @ u)
    return rot_pi, shift - screw * u, u, screw


def analyse(rec: np.ndarray, lig: np.ndarray) -> dict:
    rot, shift = kabsch(rec, lig)
    rmsd = float(np.sqrt((((rec @ rot.T) + shift - lig) ** 2).sum(1).mean()))
    _, _, _, screw = project_involution(rot, shift)
    return {"angle": rotation_angle(rot), "screw": abs(screw), "rmsd": rmsd}


def symmetrise(arr):
    """Move chain L rigidly so the complex is exactly C2. Returns (arr, info)."""
    rec, lig = correspondence(arr)
    if rec is None or len(rec) < 8:
        return None, None
    rot, shift = kabsch(rec, lig)
    rot_pi, shift_pi, _, screw = project_involution(rot, shift)

    target = rec @ rot_pi.T + shift_pi
    # Best rigid move of the ligand *as it is* onto its symmetric position, so
    # the ligand conformation is preserved rather than replaced by the
    # receptor's.
    corr_rot, corr_shift = kabsch(lig, target)

    out = arr.copy()
    mask = out.chain_id == "L"
    out.coord[mask] = out.coord[mask] @ corr_rot.T + corr_shift
    info = {
        "angle_before": rotation_angle(rot),
        "screw_before": abs(screw),
        "corr_rot_deg": rotation_angle(corr_rot),
        "corr_shift_A": float(np.linalg.norm(corr_shift + corr_rot @ lig.mean(0) - lig.mean(0))),
    }
    return out, info


def cmd_audit(args) -> None:
    ids = [i.strip() for i in Path(args.ids_file).read_text().split() if i.strip()]
    rows = []
    for sid in ids:
        path = NATIVE_DIR / f"{sid}.pdb"
        if not path.exists():
            rows.append({"id": sid, "status": "missing"})
            continue
        rec, lig = correspondence(_read(path))
        if rec is None or len(rec) < 8:
            rows.append({"id": sid, "status": "no_correspondence"})
            continue
        row = {"id": sid, "status": "ok", "n_matched": len(rec)}
        row.update(analyse(rec, lig))
        rows.append(row)

    fields = ["id", "status", "n_matched", "angle", "screw", "rmsd"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    ok = [r for r in rows if r.get("status") == "ok"]
    ang = np.array([r["angle"] for r in ok])
    screw = np.array([r["screw"] for r in ok])
    c2 = (np.abs(ang - 180.0) <= args.angle_tol) & (screw <= args.screw_tol)
    print(f"analysed {len(ok)}/{len(ids)} (no correspondence: "
          f"{sum(1 for r in rows if r.get('status') == 'no_correspondence')})")
    if len(ok):
        print(f"  rotation angle : median {np.median(ang):.1f} deg, "
              f"min {ang.min():.1f}, max {ang.max():.1f}")
        print(f"  screw |t.u|    : median {np.median(screw):.2f} A, "
              f"p90 {np.percentile(screw, 90):.2f}")
        print(f"  C2 (|angle-180|<={args.angle_tol} and screw<={args.screw_tol} A): "
              f"{int(c2.sum())}/{len(ok)}")
    print(f"wrote {args.out}")


def cmd_project(args) -> None:
    ids = [i.strip() for i in Path(args.ids_file).read_text().split() if i.strip()]
    in_root, out_root = Path(args.in_dir), Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    done = skipped = copied = 0
    stats = []
    for sid in ids:
        src = in_root / sid / args.monomer / "models"
        dst = out_root / sid / args.monomer / "models"
        if not src.is_dir():
            skipped += 1
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for k in range(1, args.top_k + 1):
            fp = src / f"model_{k}.pdb"
            if not fp.exists():
                continue
            arr = _read(fp)
            out, info = symmetrise(arr)
            if out is None:
                shutil.copyfile(fp, dst / f"model_{k}.pdb")
                copied += 1
                continue
            pf = PDBFile()
            pf.set_structure(out)
            pf.write(str(dst / f"model_{k}.pdb"))
            done += 1
            if k == 1:
                stats.append(info)

    print(f"symmetrised {done} models, copied unchanged {copied}, "
          f"skipped {skipped} systems")
    if stats:
        a = np.array([s["angle_before"] for s in stats])
        r = np.array([s["corr_rot_deg"] for s in stats])
        t = np.array([s["corr_shift_A"] for s in stats])
        print(f"  rank-1 pose angle before : median {np.median(a):.1f} deg "
              f"(p10 {np.percentile(a, 10):.1f}, p90 {np.percentile(a, 90):.1f})")
        print(f"  correction applied       : rotation median {np.median(r):.1f} deg, "
              f"shift median {np.median(t):.2f} A")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit")
    a.add_argument("--ids-file", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--angle-tol", type=float, default=10.0)
    a.add_argument("--screw-tol", type=float, default=1.0)
    a.set_defaults(func=cmd_audit)

    p = sub.add_parser("project")
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ids-file", required=True)
    p.add_argument("--monomer", default="holo")
    p.add_argument("--top-k", type=int, default=50)
    p.set_defaults(func=cmd_project)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
