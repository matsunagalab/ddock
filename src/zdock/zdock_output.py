"""Python port of ZDOCK's `create_lig.cc` (Rong Chen 2002, updated BP 2010).

Reads a `.zd3.0.2.fg.fixed.out` file (54000 poses), parses the rigid-body
transformation parameters, and regenerates ligand coordinates for each pose
— all vectorized as a single tensor op so the full 54000-pose reconstruction
runs in <1 s on CPU for typical sizes.

BM4 inputs never use the switched-ligand branch (`rot_rec=1`), which we
confirmed by scanning all 176 `.out` headers. The implementation only
supports `rot_rec=0`; a clear error is raised for any file where the header
contains a switch flag. If we ever encounter a switched case, port
`createPDBrev` from `create_lig.cc`.

Verification:
  `tests/test_zdock_output.py` regenerates poses 1 / 50 / 100 from 1KXQ's
  `.out` file and asserts coord match against the C++-generated
  `complex.{1,50,100}.pdb` files to atol=1e-3 Å. That is the
  regression guard for this port.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ZDockOutput:
    """Parsed ZDOCK `.out` file.

    All rotation angles are in radians. Translation indices `trans_xyz` are
    raw integer grid offsets; the create_lig wrap-around (`t -= N` when
    `t >= N/2`) is applied in `generate_lig_coords`, not here — storing the
    raw values means the on-disk file is a faithful transcription.
    """
    N: int                         # grid size (cubic, N×N×N)
    spacing: float                 # grid spacing, Å
    lig_rand: np.ndarray           # (3,) initial ligand Euler angles (psi, theta, phi)
    rec_center: np.ndarray         # (3,) receptor translation "r1,r2,r3"
    lig_center: np.ndarray         # (3,) ligand translation "l1,l2,l3"
    rec_filename: str              # name recorded in line 3 (after lig_rand line)
    lig_filename: str              # name recorded in line 4
    # Per-pose transforms, shape (F, 3) for rotations / translations.
    pose_rot: np.ndarray           # (F, 3) pose Euler angles a1,a2,a3
    pose_trans: np.ndarray         # (F, 3) integer grid offsets t1,t2,t3
    pose_score: np.ndarray         # (F,) ZDOCK raw score


def parse_out_file(path: str | Path) -> ZDockOutput:
    """Parse a ZDOCK `.zd3.0.2.fg.fixed.out` file.

    Matches the layout expected by `create.pl`:

        Line 1:  N spacing [switch_num]
        Line 2:  [rec_rand1 rec_rand2 rec_rand3]  (only if switch_num present)
        Line 3:  lig_rand1 lig_rand2 lig_rand3
        Line 4:  rec_name r1 r2 r3
        Line 5:  lig_name l1 l2 l3
        Line 6+: angl_x angl_y angl_z trans_x trans_y trans_z score
    """
    path = Path(path)
    with open(path, "r") as fh:
        lines = [ln.rstrip("\n") for ln in fh.readlines()]

    header = lines[0].split()
    if len(header) == 2:
        N = int(header[0])
        spacing = float(header[1])
        switch_num = None
    elif len(header) == 3:
        N = int(header[0])
        spacing = float(header[1])
        switch_num = header[2]
        raise NotImplementedError(
            f"{path}: switched-ligand mode (rot_rec=1, switch_num={switch_num!r}) "
            "is not supported. No BM4 file uses it; if you hit this, port "
            "create_lig.cc's createPDBrev() branch."
        )
    else:
        raise ValueError(f"{path}: malformed header {header!r}")

    line_num = 1  # Per create.pl: skip rec_rand line only if switch_num present.
    # (The switch_num branch raises above, so line_num stays at 1 here.)
    lig_rand = np.asarray([float(v) for v in lines[line_num].split()])
    line_num += 1
    if lig_rand.shape != (3,):
        raise ValueError(f"{path}: lig_rand line malformed: {lines[1]!r}")

    rec_tokens = lines[line_num].split()
    line_num += 1
    rec_filename = rec_tokens[0]
    rec_center = np.asarray([float(v) for v in rec_tokens[1:4]])

    lig_tokens = lines[line_num].split()
    line_num += 1
    lig_filename = lig_tokens[0]
    lig_center = np.asarray([float(v) for v in lig_tokens[1:4]])

    n_pose = len(lines) - line_num
    pose_rot = np.empty((n_pose, 3), dtype=np.float64)
    pose_trans = np.empty((n_pose, 3), dtype=np.int64)
    pose_score = np.empty((n_pose,), dtype=np.float64)
    for i, ln in enumerate(lines[line_num:]):
        tok = ln.split()
        if len(tok) != 7:
            raise ValueError(
                f"{path}: pose line {line_num + i + 1} has {len(tok)} tokens, expected 7"
            )
        pose_rot[i] = [float(tok[0]), float(tok[1]), float(tok[2])]
        pose_trans[i] = [int(tok[3]), int(tok[4]), int(tok[5])]
        pose_score[i] = float(tok[6])

    return ZDockOutput(
        N=N,
        spacing=spacing,
        lig_rand=lig_rand,
        rec_center=rec_center,
        lig_center=lig_center,
        rec_filename=rec_filename,
        lig_filename=lig_filename,
        pose_rot=pose_rot,
        pose_trans=pose_trans,
        pose_score=pose_score,
    )


# ---------------------------------------------------------------------------
# Rotation matrix matching create_lig.cc::rotateAtom (rev=0).
# Parameters are (psi, theta, phi). Result is a 3x3 rotation matrix applied as
# R @ x for x shaped (3,).
# ---------------------------------------------------------------------------


def euler_rotation_matrix(
    euler: torch.Tensor,
) -> torch.Tensor:
    """Build rotation matrices following create_lig's Euler convention.

    Accepts `euler` of shape (..., 3) = (psi, theta, phi); returns matrices
    of shape (..., 3, 3). Matches the `rotateAtom` function in
    `create_lig.cc` (rev=0) exactly: the matrix entries below are a direct
    transcription.

    Applied as `R @ x` for x of shape (..., 3) — i.e. rotates a column vector.
    """
    psi = euler[..., 0]
    theta = euler[..., 1]
    phi = euler[..., 2]
    cp = torch.cos(psi); sp = torch.sin(psi)
    ct = torch.cos(theta); st = torch.sin(theta)
    cf = torch.cos(phi); sf = torch.sin(phi)

    r11 = cp * cf - sp * ct * sf
    r21 = sp * cf + cp * ct * sf
    r31 = st * sf
    r12 = -cp * sf - sp * ct * cf
    r22 = -sp * sf + cp * ct * cf
    r32 = st * cf
    r13 = sp * st
    r23 = -cp * st
    r33 = ct

    # Stack into (..., 3, 3).
    row1 = torch.stack([r11, r12, r13], dim=-1)
    row2 = torch.stack([r21, r22, r23], dim=-1)
    row3 = torch.stack([r31, r32, r33], dim=-1)
    return torch.stack([row1, row2, row3], dim=-2)


def generate_lig_coords(
    lig_xyz: torch.Tensor,
    zd: ZDockOutput,
    *,
    n_poses: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Regenerate ligand coordinates for every ZDOCK pose.

    Mirrors `createPDB` in `create_lig.cc` (rot_rec=0 branch), vectorized to
    produce all `F` poses in one tensor op.

    Pipeline per pose:
        xyz_atom = pdb_lig_xyz - lig_center
        xyz_rot1 = R(lig_rand) @ xyz_atom             # initial random rotation
        xyz_rot2 = R(pose_rot)  @ xyz_rot1            # pose rotation
        t_wrapped = pose_trans where t_wrapped[t >= N/2] = t - N
        xyz_out  = xyz_rot2 - t_wrapped * spacing + rec_center

    Args:
        lig_xyz: (N_lig, 3) raw ligand coordinates from the ligand PDB.
        zd: parsed ZDOCK output.
        n_poses: optional cap (top-N). Default: all poses in zd.pose_rot.
        dtype / device: torch destination. Use float64 for closest match to
            the C++ float32 reference (reconstruction rounds at write time).

    Returns:
        Tensor of shape (F, N_lig, 3).
    """
    if lig_xyz.ndim != 2 or lig_xyz.shape[1] != 3:
        raise ValueError(f"lig_xyz must be (N, 3), got {tuple(lig_xyz.shape)}")

    F_total = zd.pose_rot.shape[0]
    F = F_total if n_poses is None else min(n_poses, F_total)

    lig = lig_xyz.to(device=device, dtype=dtype)
    lig_center = torch.as_tensor(zd.lig_center, device=device, dtype=dtype)
    rec_center = torch.as_tensor(zd.rec_center, device=device, dtype=dtype)
    lig_rand = torch.as_tensor(zd.lig_rand, device=device, dtype=dtype)
    pose_rot = torch.as_tensor(zd.pose_rot[:F], device=device, dtype=dtype)
    pose_trans = torch.as_tensor(zd.pose_trans[:F], device=device, dtype=dtype)

    # Step 1: centering — subtract lig_center.
    lig_centered = lig - lig_center  # (N, 3)

    # Step 2: apply lig_rand rotation (same for all poses).
    R_init = euler_rotation_matrix(lig_rand)  # (3, 3)
    # R_init @ x for each atom x: result shape (N, 3).
    lig_rotated_init = lig_centered @ R_init.T  # (N, 3)

    # Step 3: per-pose rotation.
    R_pose = euler_rotation_matrix(pose_rot)  # (F, 3, 3)
    # Combined: for each pose, (R_pose @ R_init) @ centered_atom.
    # Equivalent and cheaper: rotate once then batched rotate.
    # lig_rotated[f, n, :] = R_pose[f] @ lig_rotated_init[n, :]
    # Use einsum for clarity: 'fij,nj->fni'.
    lig_rotated = torch.einsum("fij,nj->fni", R_pose, lig_rotated_init)

    # Step 4: translation wrap — elements >= N/2 become t - N.
    half = zd.N // 2
    pose_trans_wrapped = torch.where(
        pose_trans >= half, pose_trans - zd.N, pose_trans,
    )

    # Step 5: final shift. `- t*spacing + rec_center`, broadcast over atom dim.
    shift = -pose_trans_wrapped * zd.spacing + rec_center  # (F, 3)
    return lig_rotated + shift.unsqueeze(1)  # (F, N, 3)
