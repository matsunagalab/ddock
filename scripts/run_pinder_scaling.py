"""Interface-cluster **scaling law** for hard-negative mining on the PINDER
interface-deleaked split (EXPERIMENT_REPORT §5.6).

Question
--------
§5.5 found that hard-negative mining with 220 fit interface clusters produced
*zero* improvement on the held-out PINDER-S test set, and §5.5.1 listed
"not enough independent interfaces" as the leading (but unproven) explanation:
145 learnable parameters (alpha + 12x12 IFACE) against 220 clusters. This
script measures held-out performance and mining gain as a function of the
number of independent fit interface clusters N (220 / 500 / 1,000 / 2,000) at
fixed everything-else, so that a rising curve supports the data-limited
hypothesis and an early plateau supports the capacity/loss-limited one.

Streaming design (what changed vs ``run_pinder_hardneg.py``)
-----------------------------------------------------------
The old script held every ``PreparedProtein`` *and* every per-pose feature pool
on the GPU, which does not fit past a few hundred complexes. Here:

* prepared complexes live in a CPU **disk cache** (:mod:`zdock.prep_cache`),
  built once by ``scripts/prep_pinder_cache.py`` and shared by every seed;
* mining pages **one complex at a time** onto the GPU, runs the FFT search,
  labels + featurizes the poses, moves the ``(S_SC, T, S_ELEC, RMSD, DockQ)``
  features to CPU and frees the protein and the pose coordinates immediately;
* the feature pool therefore lives in **host** memory (~1-2 MB per complex);
* training moves only the current mini-batch to the GPU; validation and test
  evaluation stream one complex at a time as well;
* CUDA OOM is caught per stage, retried with halved chunk sizes, and finally
  recorded (id, atom counts, stage) and skipped rather than killing the run.

Scientific conditions are inherited unchanged from the stabilized §5.5 run:
parameters and Adam state carry over between rounds, scores are standardized
per complex *inside the loss only* (a positive affine map, so pose ranking and
therefore every reported metric is unchanged), ``alpha_lr=1e-5``,
``iface_lr=5e-4``, gradient clipping, ``0 <= alpha <= 0.1``, the positive set is
frozen at round 0, mining appends only ``DockQ < 0.23`` negatives, the
validation candidate pools are frozen at round 0 and never mined, early
stopping uses the fixed validation loss, and TEST is never used to pick a
checkpoint, a round, or a hyper-parameter.

Example
-------
    PINDER_BASE_DIR=$PWD/external/pinder CUDA_VISIBLE_DEVICES=0 \\
    uv run python -u scripts/run_pinder_scaling.py --n-fit 500 --seed 0 --rounds 1
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
# Must be set before the CUDA context is created. The FFT search allocates a
# few large, short-lived grids per complex while the run keeps a long-lived
# pool, which fragments the default caching allocator badly enough that a
# 1.4 GiB request failed on a 47 GiB card. Expandable segments let the
# allocator grow a single region instead of hoarding fixed-size blocks.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from zdock.atomtypes import charge_score as default_charge_score
from zdock.atomtypes import iface_ij
from zdock.dataset import (POSE_IDENTITY_MISSING, generate_decoys,
                           generate_pool_reachable,
                           has_pose_identity, label_decoys)
from zdock.evaluate import evaluate_ranking
from zdock.geom import grid_shape
from zdock.prep_cache import load_prepared
from zdock.score import (docking_score_elec, iface_score_matrix,
                         psc_score_from_terms, SC_REFERENCE_SPACING, SC_RHO)
from zdock.rotation_grid import hopf_quaternions, random_quaternions
from zdock.train import (loss_basin, loss_margin_hard_negatives,  # noqa: E501
                         loss_param_prior, loss_top_tail)

KS = (1, 5, 10, 50, 100)


# --------------------------------------------------------------------------
# per-complex feature pool (host-resident)
# --------------------------------------------------------------------------
class Feats:
    """Per-pose features + labels for one complex, normally on CPU.

    ``origin`` records the mining round each pose came from (0 = the round-0
    default-parameter candidate set), which is what lets us report pool
    composition (positives / random negatives / hard negatives) and assert that
    mining never grows the positive set.

    ``prov`` records *how* the pose was proposed: 0 = returned by the FFT
    search, 1 = enumerated on the search's own lattice near the native pose
    (see ``generate_pool_reachable``). Both are reachable; the flag exists so
    the report can say how many positives the search found unaided.

    ``sc`` is ``(F, 4)`` — the rho-independent PSC decomposition
    ``(c_pair, n_ss, n_sc, n_cc)`` — when the pool was built with
    ``--psc-decompose`` (the default), and ``(F,)`` for legacy caches.

    ``pose_key`` is ``(F, 4)``: the pose's exact identity as
    ``(rotation index, cell_x, cell_y, cell_z)``, which is what
    `generate_pool_reachable` already uses to drop duplicates between its two
    provenances. A later mining round needs it to tell a genuinely new pose
    from one already in the pool. Derived quantities cannot: two different
    poses can share an (RMSD, DockQ, ELEC) triple, and the same pose recomputed
    under different chunking can differ in the last bits and be counted as new.
    Four columns rather than one packed integer, because packing four values
    into an int64 overflowed and silently reduced 1944 rotations to two
    distinct keys. A row of -1 marks a pool built by a path that does not track
    identity (the legacy ``generate_decoys`` recipe); de-duplication refuses to
    run on those rather than guess.

    Rotation *indices* are only comparable against the same rotation grid, so
    they are only meaningful across rounds under ``--rot-set hopf``, which is
    round-independent; ``random`` reseeds per round. The CLI enforces that.
    """

    __slots__ = ("name", "sc", "T", "elec", "rmsd", "dockq", "origin", "prov",
                 "pose_key")

    def __init__(self, name, sc, T, elec, rmsd, dockq, origin, prov=None,
                 pose_key=None):
        self.name = name
        self.sc = sc
        self.T = T
        self.elec = elec
        self.rmsd = rmsd
        self.dockq = dockq
        self.origin = origin
        self.prov = (torch.zeros_like(origin) if prov is None else prov)
        self.pose_key = (torch.full((origin.shape[0], 4),
                                    POSE_IDENTITY_MISSING, dtype=torch.int64)
                         if pose_key is None else pose_key)

    @property
    def n(self) -> int:
        return int(self.sc.shape[0])

    def to(self, device, non_blocking: bool = False) -> "Feats":
        if self.sc.device == torch.device(device):
            return self
        return Feats(self.name,
                     self.sc.to(device, non_blocking=non_blocking),
                     self.T.to(device, non_blocking=non_blocking),
                     self.elec.to(device, non_blocking=non_blocking),
                     self.rmsd.to(device, non_blocking=non_blocking),
                     self.dockq.to(device, non_blocking=non_blocking),
                     self.origin.to(device, non_blocking=non_blocking),
                     self.prov.to(device, non_blocking=non_blocking),
                     self.pose_key.to(device, non_blocking=non_blocking))

    def cat(self, other: "Feats") -> None:
        self.sc = torch.cat([self.sc, other.sc])
        self.T = torch.cat([self.T, other.T])
        self.elec = torch.cat([self.elec, other.elec])
        self.rmsd = torch.cat([self.rmsd, other.rmsd])
        self.dockq = torch.cat([self.dockq, other.dockq])
        self.origin = torch.cat([self.origin, other.origin])
        self.prov = torch.cat([self.prov, other.prov])
        self.pose_key = torch.cat([self.pose_key, other.pose_key])

    def index(self, idx) -> "Feats":
        return Feats(self.name, self.sc[idx], self.T[idx], self.elec[idx],
                     self.rmsd[idx], self.dockq[idx], self.origin[idx],
                     self.prov[idx], self.pose_key[idx])

    def counts(self, dockq_thr: float) -> dict:
        pos = self.dockq >= dockq_thr
        r0 = self.origin == 0
        search = self.prov == 0
        return {"n": self.n,
                "n_pos": int(pos.sum()),
                "n_rand_neg": int((~pos & r0).sum()),
                "n_hard_neg": int((~pos & ~r0).sum()),
                "n_pos_from_search": int((pos & search).sum()),
                "n_pos_enumerated": int((pos & ~search).sum())}


class Params:
    """The trainable scoring parameters, bundled so adding one does not mean
    re-threading every signature in this file.

    The score is exactly linear in a 148-dimensional reparametrisation
    (verified numerically to 5e-12)::

        S = alpha*c_pair - (w_ss*n_ss + w_sc*n_sc + w_cc*n_cc)
              + sum_ij M_ij(e) T_ij + beta*S_ELEC

    Chen & Weng 2003 do not treat the three clash weights as free: they set
    ``Im in {0, rho, rho**2}``, so ``w_k = alpha * rho**k`` for k = 2, 3, 4.
    In log space that is the statement that the three log-weights are
    **collinear**::

        log w_k = log alpha + k * log rho          (2 degrees of freedom)

    ``--psc-mode``:

    * ``rho``  -- keep the collinearity. 146 trainable parameters
      (alpha, rho, 144 pair terms). Faithful to the paper.
    * ``free`` -- drop it: the three log-weights move independently. 148
      parameters. The published penalty *ratios* 1 : rho : rho**2 between
      surface-surface, surface-core and core-core become a hypothesis the fit
      can reject rather than an assumption.

    Both modes start from exactly the same point (``w_k = alpha0 * rho0**k``),
    so the comparison isolates the constraint. Weights are carried as logs:
    that keeps them positive (a clash must cost, never pay), makes the learning
    rate scale-free across weights spanning 12 to 150, and makes ``rho`` mode a
    literal linear subspace of ``free`` mode.

    ``beta`` and the 11 charges are deliberately NOT trained. After the ELEC
    mask/sign fixes they do receive gradient, but measured on the poses a ranker
    must actually discriminate among (1KXQ, Hopf nside=3, top-500),
    std(beta*S_ELEC) = 1.0 against std(S_IFACE) = 104 and std(S_PSC) = 102 --
    ELEC moves ~1% of the ranking, and beta = 3 is the published value.
    """

    #: exponents of rho for (surface-surface, surface-core, core-core)
    CLASH_POWERS = (2.0, 3.0, 4.0)

    #: every stored tensor. Which of them a given mode actually optimises is
    #: `tensors()`, not this.
    NAMES = ("alpha", "rho", "iface", "log_clash")

    def __init__(self, alpha, rho, iface, log_clash=None, mode="rho",
                 train_psc=True, iface_mode="full", iface0=None,
                 rowcol=None):
        self.alpha, self.rho, self.iface = alpha, rho, iface
        #: False freezes alpha and the clash weights at their initial (published)
        #: values and trains only the 144 pair terms. The cross-swap of a
        #: jointly-trained checkpoint showed the whole top-1 gain sitting in the
        #: IFACE block (+17 complexes of 236) while the PSC block alone cost 4,
        #: so the narrow claim "learning the pair potential helps" needs the PSC
        #: side held fixed rather than inferred from a post-hoc swap.
        self.train_psc = train_psc
        if log_clash is None:               # derive the paper's collinear point
            k = torch.tensor(self.CLASH_POWERS, device=rho.device,
                             dtype=rho.dtype)
            log_clash = (alpha.detach() * rho.detach().pow(k)).log()
        self.log_clash = log_clash          # (3,) log of the three weights
        self.mode = mode
        #: How the learned DIFFERENCE from the published table is parametrised.
        #:
        #: "full"   -- all 144 pair terms free.
        #: "add"    -- additive, `g + r_i + c_j`, 23 free directions.
        #: "sym"    -- symmetric additive, `g + a_i + a_j`, 12 free directions.
        #:
        #: A post-hoc decomposition of the full fit put 59% of its gain
        #: (+4.7 of +8.0 pp) in the additive subspace while the 144 pair
        #: residuals alone were not significant (p = 0.18), so "the model
        #: learned a chemical pair potential" needs the low-dimensional model
        #: fitted from scratch as a control, not read off a projection.
        #:
        #: That control has now been run (report section 5.14.12, 3 x 2 x 3
        #: cells on the frozen TEST pool). The p = 0.18 above was the wrong
        #: test -- it asked whether the residual ALONE beats the baseline, not
        #: whether it adds anything to the additive fit. Trained from scratch:
        #: success@1 73.6% (sym 12), 74.2% (add 23), 77.5% (full 144) against a
        #: 69.5% baseline. sym -> add is a flat null (2 wins, p >= 0.5); the
        #: increment that matters is add -> full, 9 wins 1 loss, exact McNemar
        #: p = 0.0215 in all three seeds. So the one-body part needs only 12
        #: directions, and the pair residual is doing real work after all.
        #:
        #: The coefficients live in an ORTHONORMAL zero-sum basis, not as raw
        #: (g, r, c). Two reasons. (a) `g + r_i + c_j` has a 2-dimensional
        #: gauge -- `r += a, c += b, g -= a+b` leaves the matrix unchanged --
        #: so raw coefficients leave flat directions for the optimiser.
        #: (b) With raw coefficients a single learning rate is not comparable
        #: across models: `g`'s gradient sums 144 cells, each `r_i` sums 12,
        #: and each entry of the full model sums 1. In this basis
        #: `||dE||_F^2 = ||theta||^2`, so identifiability, the prior and the
        #: learning-rate scale are all fixed at once, and the additive model is
        #: a metric subspace of the full one rather than merely a linear one.
        #:
        #: `sym` is the physically primary control: the published table is
        #: exactly symmetric, so asymmetric `r_i != c_j` could be exploiting
        #: PINDER's receptor/ligand role convention rather than chemistry.
        self.iface_mode = iface_mode
        self.iface0 = iface0                # frozen published table (rowcol)
        self.rowcol = rowcol                # (25,) = [g, r(12), c(12)]

    #: free directions per mode
    N_COEF = {"full": 144, "add": 23, "sym": 12}

    @staticmethod
    def _zero_sum_basis(n, device, dtype):
        """Orthonormal basis (n, n-1) of {v : sum v = 0} (Helmert)."""
        m = torch.eye(n, device=device, dtype=dtype) - 1.0 / n
        q, _ = torch.linalg.qr(m[:, : n - 1])
        return q

    def iface_vec(self) -> torch.Tensor:
        """The 144-vector actually contracted with T, in any mode."""
        if self.iface_mode == "full":
            return self.iface
        th = self.rowcol
        dev, dt = th.device, th.dtype
        V = self._zero_sum_basis(12, dev, dt)            # (12, 11), orthonormal
        J = torch.ones(12, 12, device=dev, dtype=dt)
        d = th[0] * (J / 12.0)                            # ||J/12||_F = 1
        if self.iface_mode == "sym":
            a = V @ th[1:12]                              # zero-sum, ||a|| = 1
            # ||a (x) 1 + 1 (x) a||_F^2 = 24 ||a||^2 for zero-sum a
            d = d + (a.unsqueeze(1) + a.unsqueeze(0)) / (24.0 ** 0.5)
        else:                                             # "add"
            r = V @ th[1:12]
            c = V @ th[12:23]
            d = d + r.unsqueeze(1) / (12.0 ** 0.5) + c.unsqueeze(0) / (12.0 ** 0.5)
        return self.iface0 + d.reshape(-1)

    def clash_weights(self) -> torch.Tensor:
        """``(w_ss, w_sc, w_cc)``, differentiable in whichever mode is active."""
        if self.mode == "rho":
            k = torch.tensor(self.CLASH_POWERS, device=self.rho.device,
                             dtype=self.rho.dtype)
            return self.alpha * self.rho.pow(k)
        return self.log_clash.exp()

    def _kw(self, fn):
        return dict(mode=self.mode, train_psc=self.train_psc,
                    iface_mode=self.iface_mode,
                    iface0=None if self.iface0 is None else fn(self.iface0),
                    rowcol=None if self.rowcol is None else fn(self.rowcol))

    def clone(self) -> "Params":
        f = lambda t: t.detach().clone()
        return Params(f(self.alpha), f(self.rho), f(self.iface),
                      f(self.log_clash), **self._kw(f))

    def cpu(self) -> "Params":
        """Detached CPU copy that KEEPS the mode.

        `cap_pool` used to be handed `Params(alpha, rho, iface)`, which silently
        rebuilt a `rho`-mode object and scored a `free`-mode run with the wrong
        clash weights.
        """
        f = lambda t: t.detach().cpu()
        return Params(f(self.alpha), f(self.rho), f(self.iface),
                      f(self.log_clash), **self._kw(f))

    def tensors(self):
        """Only what this mode actually optimises."""
        pair = [self.iface] if self.iface_mode == "full" else [self.rowcol]
        if not self.train_psc:
            return pair
        if self.mode == "rho":
            return [self.alpha, self.rho] + pair
        return [self.alpha, self.log_clash] + pair

    def requires_grad_(self, flag=True):
        for t in self.tensors():
            t.requires_grad_(flag)
        return self

    def state_dict(self):
        d = {n: getattr(self, n).detach().cpu() for n in self.NAMES}
        d["psc_mode"] = self.mode
        d["iface_mode"] = self.iface_mode
        d["clash_weights"] = self.clash_weights().detach().cpu()
        # always store the effective 144-vector so every evaluator is
        # parametrisation-agnostic
        d["iface"] = self.iface_vec().detach().cpu()
        if self.rowcol is not None:
            d["rowcol"] = self.rowcol.detach().cpu()
        return d

    @classmethod
    def initial(cls, args, device, dtype):
        a = torch.tensor(args.alpha0, device=device, dtype=dtype)
        r = torch.tensor(args.rho0, device=device, dtype=dtype)
        k = torch.tensor(cls.CLASH_POWERS, device=device, dtype=dtype)
        e0 = iface_ij(device=device, dtype=dtype, flat=True)
        im = getattr(args, "iface_mode", "full")
        return cls(a, r, e0, (a * r.pow(k)).log(),
                   getattr(args, "psc_mode", "rho"),
                   not getattr(args, "freeze_psc", False),
                   iface_mode=im, iface0=e0.clone(),
                   rowcol=torch.zeros(cls.N_COEF.get(im, 144),
                                      device=device, dtype=dtype))

    def summary(self, p0: "Params") -> dict:
        w = self.clash_weights()
        out = {"alpha": float(self.alpha), "rho": float(self.rho),
               "d_iface_norm": float((self.iface_vec() - p0.iface_vec()).norm()),
               "w_ss": float(w[0]), "w_sc": float(w[1]), "w_cc": float(w[2])}
        if self.mode == "free":
            # If the paper's collinearity still held, these three would agree.
            # Reporting them makes a violation visible rather than buried.
            k = torch.tensor(self.CLASH_POWERS, device=w.device, dtype=w.dtype)
            out["implied_rho"] = [float(x) for x in
                                  (w / self.alpha.clamp_min(1e-12)).pow(1.0 / k)]
        return out


def score_from_feats(f: Feats, p: Params, beta) -> torch.Tensor:
    imat = iface_score_matrix(p.iface_vec())
    if f.sc.ndim == 2:
        # (F, 4) = (c_pair, n_ss, n_sc, n_cc). The clash weights come from
        # `Params`, so `rho` and `free` mode share one scoring path.
        sc = p.alpha * f.sc[:, 0] - (f.sc[:, 1:4] * p.clash_weights()).sum(-1)
    else:
        # legacy (F,) cache with rho already baked in: no gradient to rho.
        sc = p.alpha * f.sc
    return sc + (imat * f.T).sum(dim=(-2, -1)) + beta * f.elec


def normalized_scores(f: Feats, p: Params, beta) -> torch.Tensor:
    """Ranking-preserving per-complex standardization, used *only* inside the
    loss (see §5.5: raw score std is 5e2-2e3 while the basin temperature is
    0.5). Centering and dividing by a detached positive scalar cannot change
    pose order, so the trained parameters rank exactly as the raw score does."""
    s = score_from_feats(f, p, beta)
    # `std()` of a 1-element tensor is NaN and `clamp_min` propagates it, so a
    # single one-pose complex would NaN the loss and Adam would permanently
    # poison alpha and iface. `unbiased=False` returns 0 there instead.
    scale = s.detach().std(unbiased=False).clamp_min(1.0)
    return (s - s.detach().mean()) / scale


# --------------------------------------------------------------------------
# streaming mining: one complex at a time on the GPU
# --------------------------------------------------------------------------
def _adaptive_pose_chunk(n_rec: int, n_lig: int, budget_elems: int) -> int:
    """DockQ builds a dense (chunk, N_rec, N_lig) tensor; keep it bounded."""
    per = max(1, n_rec * n_lig)
    return int(max(1, min(64, budget_elems // per)))


def _adaptive_frame_chunk(n_voxels: int, budget_elems: int, cap: int) -> int:
    """Featurisation builds ``L_count`` of shape ``(chunk * 12, nx, ny, nz)``.

    That is **twelve grids per pose**, so at the paper's 1.2 A spacing it, not
    the FFT search, is what exhausts the card: a 22.5M-voxel complex needs
    100 GiB at ``frame_chunk=100`` and still 25 GiB at 25. A fixed default
    profiled at the old 3.0 A spacing therefore killed the large complexes
    systematically -- measured on the first TEST pool build, 83 of 250 were lost
    to OOM and their voxel median was 4.1M against 1.7M for the survivors, i.e.
    exactly the survivorship bias the size cutoff exists to avoid.

    Scaling the chunk with the grid volume instead makes the peak roughly
    constant across the corpus.
    """
    per = max(1, 12 * n_voxels)
    return int(max(1, min(cap, budget_elems // per)))


def search_quaternions(args, round_idx: int, device, dtype):
    """The rotation set the FFT search runs over.

    Hopf (default): a deterministic, near-optimal covering of SO(3) with a
    measured covering radius of ~18.5 deg at 1944 points, against ~28.3 deg for
    the same number of uniform-random orientations. It carries no dependence on
    the native orientation, which is the point -- the previous recipe mixed a
    25 deg cone around q* into the *search* set and thereby leaked the answer
    into the candidate pool.
    """
    if args.rot_set == "hopf":
        q = hopf_quaternions(args.hopf_nside, device=device, dtype=dtype)
        return q / q.norm(dim=-1, keepdim=True)
    return random_quaternions(args.mine_random_rot,
                              seed=args.seed + 1000 * round_idx,
                              device=device, dtype=dtype)


@torch.no_grad()
def mine_complex(prot, p: Params, beta0, charge0, args, round_idx: int,
                 *, rot_chunk: int | None = None, frame_chunk: int | None = None,
                 pose_chunk: int | None = None):
    """FFT-search + label + featurize one complex; return host-side ``Feats``.

    Raises ``torch.cuda.OutOfMemoryError`` if even the smallest chunk sizes do
    not fit — the caller records it for the round's rescue pass and, if that
    also fails, skips the complex.
    """
    device = prot.rec_xyz.device
    dtype = prot.rec_xyz.dtype
    rot_chunk = args.rot_chunk if rot_chunk is None else rot_chunk
    nx, ny, nz = grid_shape(prot.rec_xyz, prot.lig_ref, spacing=args.spacing)
    n_voxels = nx * ny * nz
    if frame_chunk is None:
        frame_chunk = min(args.frame_chunk,
                          _adaptive_frame_chunk(n_voxels, args.feature_budget,
                                                args.frame_chunk))
    if pose_chunk is None:
        pose_chunk = _adaptive_pose_chunk(prot.n_rec, prot.n_lig, args.dockq_budget)
    quats = search_quaternions(args, round_idx, device, dtype)

    last_exc = None
    for attempt in range(args.oom_retries + 1):
        try:
            if args.pool == "reachable":
                poses, prov, pose_key = generate_pool_reachable(
                    prot, alpha=p.alpha, iface_ij_flat=p.iface_vec(), beta=beta0,
                    charge_score_lut=charge0, quats=quats,
                    ntop=args.mine_ntop, spacing=args.spacing,
                    rot_chunk_size=rot_chunk,
                    n_near_rot=args.near_rot, trans_cells=args.trans_cells,
                )
            else:                                    # legacy (section 5.6) recipe
                poses, _ = generate_decoys(
                    prot, alpha=p.alpha, iface_ij_flat=p.iface_vec(), beta=beta0,
                    charge_score_lut=charge0,
                    n_random_rot=args.mine_random_rot, n_cone=args.mine_cone,
                    ntop=args.mine_ntop, seed=args.seed + 1000 * round_idx,
                    rot_chunk_size=rot_chunk, spacing=args.spacing,
                )
                prov = torch.zeros(poses.shape[0], device=device,
                                   dtype=torch.int16)
                # this recipe does not track pose identity; the sentinel
                # makes de-duplication refuse rather than silently mis-count.
                # Same (F, 4) shape as a real identity, so nothing downstream
                # has to special-case the rank.
                pose_key = torch.full((poses.shape[0], 4),
                                      POSE_IDENTITY_MISSING,
                                      device=device, dtype=torch.int64)
            alpha_d = torch.zeros((), device=device, dtype=dtype)
            iface_d = iface_ij(device=device, dtype=dtype, flat=True)
            # `psc_decompose` caches (c_pair, n_ss, n_sc, n_cc) instead of the
            # collapsed S_PSC so that rho stays trainable from these features.
            sc, T, elec = docking_score_elec(
                prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
                prot.rec_atomtype_id, prot.rec_charge_id,
                poses, prot.lig_radius, prot.lig_sasa,
                prot.lig_atomtype_id, prot.lig_charge_id,
                alpha_d, iface_d, beta0, charge0,
                lig_xyz_for_grid=prot.lig_ref, spacing=args.spacing,
                frame_chunk_size=frame_chunk, return_components=True,
                psc_decompose=args.psc_decompose,
            )
            rmsd, dockq = label_decoys(prot, poses, pose_chunk=pose_chunk)
            out = Feats(prot.name, sc.cpu(), T.cpu(), elec.cpu(),
                        rmsd.cpu(), dockq.cpu(),
                        torch.full((sc.shape[0],), round_idx, dtype=torch.int16),
                        prov.cpu(), pose_key.cpu())
            del poses, prov, pose_key, sc, T, elec, rmsd, dockq
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return out
        except torch.cuda.OutOfMemoryError as exc:
            last_exc = exc
            if device.type == "cuda":
                torch.cuda.empty_cache()
            rot_chunk = max(1, rot_chunk // 2)
            frame_chunk = max(25, frame_chunk // 2)
            pose_chunk = max(1, pose_chunk // 2)
            args_retry = attempt + 1
            print(f"    [{prot.name}] OOM (attempt {args_retry}) -> retry with "
                  f"rot_chunk={rot_chunk} frame_chunk={frame_chunk} "
                  f"pose_chunk={pose_chunk}", flush=True)
    raise last_exc


# --------------------------------------------------------------------------
# pool bookkeeping
# --------------------------------------------------------------------------
#: Version of the pose-identity schema stored in a pool cache. Bump whenever
#: the meaning of `Feats.pose_key` changes: caches are keyed on it, so an old
#: pool can never be silently reused by code that reads its identities
#: differently. v2 = the (N, 4) `(rotation index, cell_x, cell_y, cell_z)`
#: tensor; v1 (unversioned) had no identity, and the packed-int64 form that
#: preceded it recorded only the rotation index's parity.
POSE_KEY_SCHEMA = 2


def _fresh_indices(cand: Feats, pool: Feats) -> tuple[torch.Tensor, int]:
    """``(indices of cand poses not already in pool, how many were dropped)``.

    Keyed on the pose's own identity (rotation index, translation cell),
    never on derived quantities: two different poses can share an
    (RMSD, DockQ, ELEC) triple, and the same pose recomputed under a
    different chunking can differ in the last bits and look new. Without
    this a pose the search returns in two rounds is stored twice and
    silently carries double weight in the loss -- an implicit reweighting
    that would be read as a mining effect.

    The candidate's own duplicates are dropped too: `generate_pool_reachable`
    de-duplicates its two provenances against each other but two rounds'
    proposals are only compared here.
    """
    if not (has_pose_identity(cand.pose_key) and has_pose_identity(pool.pose_key)):
        raise SystemExit(
            "de-duplication needs pose identities, but this pool was built "
            "by a path that does not record them (--pool decoys, or a "
            "cache written before pose_key existed). Re-mine round 0 with "
            "--pool reachable, or the round-1 pool will double-count every "
            "pose the search returns twice.")
    seen = set(map(tuple, pool.pose_key.tolist()))
    keep, dup_within = [], 0
    for i, row in enumerate(cand.pose_key.tolist()):
        k = tuple(row)
        if k in seen:
            dup_within += 1
            continue
        seen.add(k)
        keep.append(i)
    return torch.tensor(keep, dtype=torch.long), dup_within


def param_fingerprint(q: "Params") -> str:
    """Short hash of the parameters a mining round actually searches with.

    A round-0 pool is a function of the published parameters, so every seed
    and every model share one cache. A round-1 pool is a function of what
    round 0 *learned*, so it is specific to the seed, the dimension, the
    margin and the number of steps. Nothing in the round-0 key captures
    that. Hashing the searched parameters themselves makes it impossible to
    pair a cached round-1 pool with the checkpoint that did not produce it,
    however the run was invoked.
    """
    v = torch.cat([q.iface_vec().detach().cpu().reshape(-1).double(),
                   q.alpha.detach().cpu().reshape(-1).double(),
                   q.clash_weights().detach().cpu().reshape(-1).double()])
    return hashlib.sha1(v.numpy().tobytes()).hexdigest()[:12]


def cap_pool(f: Feats, cap: int, p: Params, beta, dockq_thr: float) -> Feats:
    """Keep every positive + the hardest (highest-scoring) current negatives."""
    if f.n <= cap:
        return f
    s = score_from_feats(f, p, beta)
    pos = (f.dockq >= dockq_thr).nonzero(as_tuple=True)[0]
    neg = (f.dockq < dockq_thr).nonzero(as_tuple=True)[0]
    n_keep = max(0, cap - pos.numel())
    if n_keep and neg.numel():
        top = torch.topk(s[neg], min(n_keep, neg.numel())).indices
        neg = neg[top]
    else:
        neg = neg[:0]
    return f.index(torch.cat([pos, neg]))


# --------------------------------------------------------------------------
# evaluation (streams complexes to the GPU one at a time)
# --------------------------------------------------------------------------
@torch.no_grad()
def aggregate(feats_list, p: Params, beta, device, rmsd_thr, dockq_thr):
    n = max(1, len(feats_list))
    succ_r = {k: 0 for k in KS}
    succ_d = {k: 0 for k in KS}
    top1 = 0.0
    for f in feats_list:
        g = f.to(device)
        s = score_from_feats(g, p, beta)
        rep = evaluate_ranking(s, g.rmsd, g.dockq, ks=KS,
                               rmsd_threshold=rmsd_thr, dockq_threshold=dockq_thr)
        for k in KS:
            succ_r[k] += int(rep.success_rmsd[k])
            succ_d[k] += int(rep.success_dockq[k])
        top1 += float(rep.best_dockq_at[1])
        del g, s
    return ({k: succ_r[k] / n for k in KS},
            {k: succ_d[k] / n for k in KS}, top1 / n)


def _fmt(tag, sr, sd, t1):
    out = [f"  [{tag}]  mean best-DockQ@top1 = {t1:.3f}",
           "     K   | success@K (RMSD<=thr) | success@K (DockQ>=thr)"]
    for k in KS:
        out.append(f"    {k:>4} |        {sr[k]*100:5.1f}%        |        {sd[k]*100:5.1f}%")
    return "\n".join(out)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def loss_view(f: Feats, loss_prov: str) -> Feats:
    """The subset of a complex's pool the LOSS is allowed to see.

    ``"search"`` keeps only poses the FFT search actually returned. The
    enumerated near-native poses stay in the pool for reachability accounting
    and for reporting, but they are not available at inference, and training on
    them is what destroyed the first N=220 run: they are 96% of the positives
    (median 180 of 199 per fit complex against 3 from the search), so the loss
    was dominated by "rank a hand-placed near-native pose above a search pose".
    The cheapest way to satisfy that is to switch the shape term off -- alpha
    went to 0.0014 -- after which the score is essentially a contact count. That
    separates enumerated from searched poses beautifully and is useless for
    choosing among poses the search actually offers: measured on 236 TEST
    complexes, success@1 fell 69.5% -> 34.7% (McNemar p = 4e-21, 86 complexes
    lost against 4 gained) while the same checkpoint *improved* on the pooled
    metric that includes the enumerated poses (65.9% -> 76.7%).
    """
    if loss_prov != "search":
        return f
    keep = (f.prov == 0).nonzero(as_tuple=True)[0]
    return f.index(keep)


def mean_objective(feats, p: Params, p0: Params, beta0, args,
                   charge_dummy, device):
    total = torch.zeros((), device=p.alpha.device, dtype=p.alpha.dtype)
    for f in feats:
        g = loss_view(f, args.loss_prov).to(device)
        s = normalized_scores(g, p, beta0)
        # Forward the pool's own threshold: the loss functions default to 0.23
        # independently of --dockq-threshold, so the two would silently diverge
        # the moment that flag is passed.
        total = total + loss_basin(s, g.dockq, temperature=args.basin_temp,
                                   positive_threshold=args.dockq_thr)
        # Which negative term (report section 5.14.26). `minanchor` is the
        # recipe every result up to 2026-07-28 was produced with; it anchors on
        # min(positive), which sits a median 7.10 SD below the pose that decides
        # Max(Top 1), and is active on a median 75% of the ~1494 negatives, so
        # it acts as a broad push-down rather than a hard-negative term.
        if args.loss_neg == "minanchor":
            total = total + args.lambda_margin * loss_margin_hard_negatives(
                s, g.dockq, margin=args.margin,
                positive_threshold=args.dockq_thr)
        elif args.loss_neg == "toptail":
            total = total + args.lambda_margin * loss_top_tail(
                s, g.dockq, margin=args.margin,
                positive_threshold=args.dockq_thr, k=args.toptail_k,
                tau_pos=args.tau_pos, tau_neg=args.tau_neg,
                tau_hinge=args.tau_hinge)
    total = total / max(1, len(feats))
    # rho is regularised towards its published initial value on the same
    # quadratic footing as alpha and the pair table.
    # Regularise the LEARNED DIFFERENCE, so the same lambda means the same
    # thing whether the difference lives in 144 dimensions or in 25.
    prior = loss_param_prior(p.alpha, p.iface_vec(), charge_dummy,
                             p0.alpha, p0.iface_vec(), charge_dummy)
    # Anchor whichever clash parametrisation is live. Both are in log space
    # (rho enters as log w_k = log alpha + k log rho), so the two priors are on
    # the same footing and the modes stay comparable.
    prior = prior + ((p.log_clash - p0.log_clash).pow(2).sum()
                     if p.mode == "free" else (p.rho - p0.rho).pow(2))
    return total + args.lambda_prior * prior


@torch.no_grad()
def _val_loss(val_feats, p: Params, p0: Params, beta0, args,
              charge_dummy, device):
    return float(mean_objective(val_feats, p, p0, beta0,
                                args, charge_dummy, device))


def train_params(fit_feats, val_feats, p: Params, p0: Params, beta0,
                 args, device, dtype, gen, n_steps, val_every,
                 optimizer_state=None):
    """Continue from the current parameters; select a checkpoint on the fixed
    validation loss only. Returns the trajectory for the report."""
    p.requires_grad_(True)
    charge_dummy = torch.zeros(0, device=device, dtype=dtype)
    pair = p.iface if p.iface_mode == "full" else p.rowcol
    groups = [{"params": [pair], "lr": args.iface_lr}]
    if p.train_psc:
        groups.append({"params": [p.alpha], "lr": args.alpha_lr})
        groups.append({"params": [p.rho], "lr": args.rho_lr} if p.mode == "rho"
                      else {"params": [p.log_clash], "lr": args.rho_lr})
    opt = torch.optim.Adam(groups)
    if optimizer_state is not None:
        opt.load_state_dict(optimizer_state)
    bs = min(args.batch_size, len(fit_feats))

    best_val = _val_loss(val_feats, p, p0, beta0, args, charge_dummy, device)
    best_params = p.clone()
    best_opt = copy.deepcopy(opt.state_dict())
    # The minibatch stream is part of the state a later round resumes, so it
    # has to be rewound to the accepted checkpoint along with the parameters
    # and Adam -- otherwise "continue from the best checkpoint" continues its
    # parameters but some later point of its data order.
    best_gen = gen.get_state().clone()
    traj = [{"step": 0, "fit_loss": float("nan"), "val_loss": best_val,
             "grad_norm": float("nan"), "accepted": 1, **p.summary(p0)}]
    stale = 0
    step = 0
    grad_norms = []
    for step in range(n_steps):
        opt.zero_grad()
        batch = torch.randperm(len(fit_feats), generator=gen)[:bs].tolist()
        total = mean_objective([fit_feats[i] for i in batch], p, p0, beta0,
                               args, charge_dummy, device)
        total.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(p.tensors(), args.grad_clip))
        grad_norms.append(gn)
        opt.step()
        with torch.no_grad():
            if p.train_psc:
                p.alpha.clamp_(min=0.0, max=args.alpha_max)
            # rho must stay strictly positive: at rho = 0 the open-space mask
            # `(im <= 0)` in psc_grids flips and S_PSC jumps discontinuously
            # (measured 259.0 vs the quartic's 152.0), so the cached quartic
            # stops being the true score there.
            if not p.train_psc:
                pass                       # alpha and the clash weights frozen
            elif p.mode == "rho":
                p.rho.clamp_(min=args.rho_min, max=args.rho_max)
            else:
                # keep every clash weight inside the range the constrained
                # model could have reached, so 'free' cannot win by escaping
                # to a regime 'rho' was forbidden from
                k = torch.tensor(Params.CLASH_POWERS, device=p.log_clash.device,
                                 dtype=p.log_clash.dtype)
                lo = (p.alpha.detach().clamp_min(1e-12).log()
                      + k * math.log(args.rho_min))
                hi = (p.alpha.detach().clamp_min(1e-12).log()
                      + k * math.log(args.rho_max))
                p.log_clash.clamp_(min=float(lo.min()), max=float(hi.max()))

        if step % val_every == 0 or step == n_steps - 1:
            val = _val_loss(val_feats, p, p0, beta0, args, charge_dummy, device)
            accepted = val < best_val - args.min_delta
            traj.append({"step": step + 1, "fit_loss": float(total.detach()),
                         "val_loss": val, "grad_norm": gn,
                         "accepted": int(accepted), **p.summary(p0)})
            if accepted:
                best_val = val
                best_params = p.clone()
                best_opt = copy.deepcopy(opt.state_dict())
                best_gen = gen.get_state().clone()
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break

    with torch.no_grad():
        for n in Params.NAMES:
            getattr(p, n).copy_(getattr(best_params, n))
        if best_params.rowcol is not None and p.rowcol is not None:
            p.rowcol.copy_(best_params.rowcol)
    opt.load_state_dict(best_opt)
    gen.set_state(best_gen)
    at_bound = {"alpha_at_bound": bool(float(p.alpha) in (0.0, args.alpha_max)),
                "rho_at_bound": bool(p.mode == "rho" and float(p.rho)
                                     in (args.rho_min, args.rho_max))}
    stats = {"best_val_loss": best_val, "steps_run": step + 1,
             "mean_grad_norm": sum(grad_norms) / max(1, len(grad_norms)),
             "max_grad_norm": max(grad_norms) if grad_norms else float("nan"),
             **at_bound}
    return p, stats, traj, opt.state_dict()


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def iface_coverage(feats_list) -> dict:
    """How many complexes actually observe each of the 144 IFACE components.

    ``T[f, i, j]`` is the pose-f contact statistic for atom-type pair (i, j);
    a component with zero mass in a complex contributes no gradient there, so
    the effective sample size per parameter is this count, not the number of
    complexes.
    """
    cov = torch.zeros(12, 12, dtype=torch.int64)
    mass = torch.zeros(12, 12, dtype=torch.float64)
    for f in feats_list:
        m = f.T.abs().sum(dim=0).double().cpu()
        cov += (m > 0).long()
        mass += m
    n = max(1, len(feats_list))
    flat = cov.flatten()
    return {
        "n_complexes": len(feats_list),
        "coverage_matrix": cov.tolist(),
        "coverage_frac_matrix": (cov.double() / n).tolist(),
        "total_mass_matrix": mass.tolist(),
        "n_components_zero": int((flat == 0).sum()),
        "n_components_lt10": int((flat < 10).sum()),
        "n_components_lt_1pct": int((flat.double() / n < 0.01).sum()),
        "min_coverage": int(flat.min()),
        "median_coverage": int(flat.median()),
        "mean_coverage_frac": float((cov.double() / n).mean()),
    }


# --------------------------------------------------------------------------
# dataset selection (deterministic, nested)
# --------------------------------------------------------------------------
def is_homodimer(pinder_id: str) -> bool:
    """Both partners are the same UniProt entry.

    PINDER ids look like ``8b3p__GA1_P69540--8b3p__PA1_P69540``; the trailing
    underscore-separated field of each half is the UniProt accession.

    Why this matters for training: the DockQ in this repository uses a fixed
    per-atom correspondence with no chain-permutation maximisation, so for a
    symmetric complex a pose that places the ligand on the *equivalent* partner
    site is scored against the wrong copy. Measured on 14 such complexes, the
    role-swapped pose -- which is the native complex atom-for-atom up to a rigid
    motion -- scores DockQ 1.000 for the 3 exact-C2 cases and **0.005-0.020 for
    the 11 non-involutory ones**. Roughly 20% of the corpus is same-UniProt and
    ~79% of those are non-C2, so ~15% of complexes admit perfectly correct poses
    that this metric labels as negatives. That is not noise, it is label noise
    pointing the wrong way: the loss is told to *down*-rank a correct pose.
    Excluding them is the cheap control until a symmetry-aware DockQ exists.
    """
    halves = pinder_id.split("--")
    if len(halves) != 2:
        return False
    return halves[0].rsplit("_", 1)[-1] == halves[1].rsplit("_", 1)[-1]


def select_split(args):
    """First ``n_total`` *usable* master-list clusters, split 80/20 by stride.

    Because both the usable list and the stride rule are prefix-stable, the fit
    and validation sets for N=500 are exact subsets of those for N=1,000 and
    N=2,000.
    """
    order = [ln.strip() for ln in Path(args.master_ids).read_text().splitlines()
             if ln.strip()]
    status = {}
    for line in Path(args.prep_manifest).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            status[r["id"]] = r

    # Deterministic grid-volume cutoff (see scripts/compute_grid_sizes.py).
    # A structural property, so identical across seeds and across N — unlike
    # excluding on observed OOM, which would depend on transient GPU pressure
    # and silently break the nested-subset design.
    max_vox = getattr(args, "max_grid_voxels", 0)
    # Complexes whose reference structure is not physically possible (receptor
    # and ligand heavy atoms closer than 2 A) or whose analytic native pose does
    # not reproduce native_lig. Produced by scripts/check_prep_cache.py; 83 of
    # 2900 in the current corpus. Training on them tells the model that an
    # interpenetrated pose is correct.
    bad_geometry: set[str] = set()
    if args.exclude_bad_geometry and Path(args.exclude_bad_geometry).exists():
        bad_geometry = {ln.strip() for ln
                        in Path(args.exclude_bad_geometry).read_text().splitlines()
                        if ln.strip()}

    voxels: dict[str, int] = {}
    if max_vox:
        vpath = Path(args.grid_voxels)
        if not vpath.exists():
            raise SystemExit(
                f"--max-grid-voxels is set but {vpath} is missing; run "
                f"scripts/compute_grid_sizes.py first")
        blob = json.loads(vpath.read_text())
        if isinstance(blob, dict) and "voxels" in blob:
            got = float(blob.get("spacing", float("nan")))
            if abs(got - float(args.spacing)) > 1e-9:
                raise SystemExit(
                    f"{vpath} was computed at spacing {got} A but this run uses "
                    f"{args.spacing} A. Voxel counts scale as spacing^-3, so the "
                    f"size cutoff would be off by ({got/args.spacing:.1f})^3 = "
                    f"{(got/args.spacing)**3:.0f}x and stop filtering. Run "
                    f"scripts/compute_grid_sizes.py --spacing {args.spacing}.")
            voxels = blob["voxels"]
        else:
            # Legacy table with no spacing recorded. Refuse rather than guess:
            # the default file is the 3.0 A one, and reading it at 1.2 A let
            # 61M-276M-voxel complexes through a 31.25M cap.
            raise SystemExit(
                f"{vpath} records no spacing, so it cannot be checked against "
                f"--spacing {args.spacing}. Re-run scripts/compute_grid_sizes.py "
                f"--spacing {args.spacing} --out {vpath}.")

    def eligible(pid: str) -> bool:
        if status.get(pid, {}).get("status") != "ok":
            return False
        if pid in bad_geometry:
            return False
        if args.exclude_homodimer and is_homodimer(pid):
            return False
        # Fail CLOSED on a missing voxel entry. `voxels.get(pid, 0)` treated an
        # absent id as 0 voxels, i.e. always eligible, so extending the prep
        # cache would let new complexes bypass the size cutoff entirely and
        # break the "identical filter for every N and seed" guarantee.
        if max_vox and pid not in voxels:
            raise SystemExit(
                f"{pid} is 'ok' in the prep manifest but absent from "
                f"{args.grid_voxels}; re-run scripts/compute_grid_sizes.py so "
                f"the size filter is applied to every candidate")
        if max_vox and voxels[pid] > max_vox:
            return False
        return True

    usable = [pid for pid in order if eligible(pid)]
    n_oversized = sum(1 for pid in order
                      if status.get(pid, {}).get("status") == "ok"
                      and max_vox and voxels.get(pid, 0) > max_vox)

    val_stride = int(round(1.0 / args.val_frac))
    n_total = args.n_total or int(round(args.n_fit / (1.0 - args.val_frac)))
    if len(usable) < n_total:
        raise SystemExit(
            f"need {n_total} usable clusters but the prep cache only has "
            f"{len(usable)} (attempted {len(status)} of {len(order)}); run "
            f"scripts/prep_pinder_cache.py with a larger --limit")
    sel = usable[:n_total]
    val_ids = [pid for i, pid in enumerate(sel) if i % val_stride == 0]
    fit_ids = [pid for i, pid in enumerate(sel) if i % val_stride != 0]
    info = {"n_master_scanned": len(order), "n_prepared_ok": len(status),
            "n_excluded_oversized": n_oversized, "n_usable": len(usable),
            "max_grid_voxels": max_vox}
    return sel, fit_ids, val_ids, status, info


def load_test_feats(path, dtype, *, require_psc_terms: bool):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    out = []
    for d in blob:
        n = d["sc"].shape[0]
        if require_psc_terms and d["sc"].ndim != 2:
            raise SystemExit(
                f"{path} holds a collapsed (F,) S_PSC, but rho is being "
                f"trained, which needs the (F, 4) decomposition. That cache "
                f"also predates the PSC/ELEC/binning fixes and the reachable "
                f"pool recipe, so it is not comparable with the fit pools "
                f"either. Rebuild it with scripts/build_test_pool.py.")
        out.append(Feats(d["name"], d["sc"].to(dtype), d["T"].to(dtype),
                         d["elec"].to(dtype), d["rmsd"].to(dtype),
                         d["dockq"].to(dtype),
                         torch.zeros(n, dtype=torch.int16),
                         d.get("prov", torch.zeros(n, dtype=torch.int16))))
    return out


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-fit", type=int, required=True, dest="n_fit",
                    help="number of FIT interface clusters (validation is extra)")
    ap.add_argument("--n-total", type=int, default=0, dest="n_total",
                    help="override total (fit+validation); default n_fit/(1-val_frac)")
    ap.add_argument("--val-frac", type=float, default=0.2, dest="val_frac")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=1,
                    help="number of hard-negative mining rounds after round 0")
    ap.add_argument("--mining-contract", choices=("hardneg", "none"),
                    default="hardneg", dest="mining_contract",
                    help="what a round > 0 does. 'hardneg': re-run the search "
                         "with the current parameters and add the new "
                         "NEGATIVES (the positive set stays frozen at round "
                         "0). 'none': the matched-budget control -- skip "
                         "mining entirely and spend the identical extra "
                         "optimizer budget on the unchanged round-0 pool. "
                         "Without the control, a round-0-to-round-1 difference "
                         "confounds mining with 1500 more steps of training.")
    ap.add_argument("--master-ids", default="data/scaling/master_ids.txt",
                    dest="master_ids")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache", dest="prep_cache")
    ap.add_argument("--prep-manifest", default="data/scaling/prep_manifest.jsonl",
                    dest="prep_manifest")
    ap.add_argument("--grid-voxels", default="data/scaling/grid_voxels.json",
                    dest="grid_voxels")
    # p95 of the corpus. Cost and peak VRAM scale with the FFT lattice volume;
    # the tail reaches 1.2e10 voxels, and a single 4.1e6-voxel complex stalled
    # four jobs for >20 min even after the OOM ladder dropped to rot_chunk=1.
    ap.add_argument("--exclude-bad-geometry",
                    default="data/scaling/excluded_bad_geometry.txt",
                    dest="exclude_bad_geometry",
                    help="ids whose reference structure is sterically "
                         "impossible; '' disables")
    ap.add_argument("--exclude-homodimer", dest="exclude_homodimer",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="drop same-UniProt complexes: the symmetry-blind DockQ "
                         "labels ~15%% of them with correct poses as negatives")
    ap.add_argument("--max-grid-voxels", type=int, default=2_000_000,
                    dest="max_grid_voxels",
                    help="drop complexes whose FFT lattice exceeds this many "
                         "voxels (0 disables the filter)")
    ap.add_argument("--test-cache", default="data/shards_pinder/test_feats.pt",
                    dest="test_cache")
    ap.add_argument("--test-ids", default="data/pinder_test_ids.txt", dest="test_ids")
    ap.add_argument("--out-dir", default="data/scaling/runs", dest="out_dir")
    # The round-0 pool is determined by the selection and the BASELINE
    # parameters, and `select_split` has no RNG -- so every --seed mines exactly
    # the same thing. Caching it turns "3 seeds" from 3x the GPU cost into 1x
    # plus three cheap training runs, and lets hyperparameters be re-tried
    # without re-mining. Mining rounds >= 1 are NOT cached: they depend on the
    # parameters reached so far, hence on the seed.
    ap.add_argument("--mine-shard", default="0/1", dest="mine_shard",
                    help="i/n -- mine every n-th remaining complex starting at "
                         "i, writing a shard-specific cache file, so several "
                         "GPUs can build one round-0 pool in parallel")
    ap.add_argument("--mine-only", dest="mine_only", action="store_true",
                    help="mine and cache, then exit before training")
    ap.add_argument("--resume-from", default="", dest="resume_from",
                    help="path to a round<r>_optstate.pt. Restores that "
                         "round's parameters AND Adam state and continues at "
                         "round r+1, instead of retraining the earlier rounds. "
                         "The mine and continue arms of a mining experiment "
                         "must branch from ONE round-0 state; re-running "
                         "round-0 training in each arm branches from two, and "
                         "GPU reductions do not guarantee they are identical.")
    ap.add_argument("--mine-from-ckpt", default="", dest="mine_from_ckpt",
                    help="path to a round-0 checkpoint. Skips round 0 entirely "
                         "and mines the LAST round with those parameters, so "
                         "several GPUs can shard a round-1 pool without each "
                         "re-running round-0 training. Use with --mine-only "
                         "and --mine-shard. The cache key carries a hash of "
                         "the loaded parameters, so a shard mined from the "
                         "wrong checkpoint cannot be picked up by mistake.")
    ap.add_argument("--pool-cache", default="data/scaling/pool_cache",
                    dest="pool_cache", help="'' disables")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # optimization (identical to the stabilized §5.5 configuration)
    ap.add_argument("--epoch-passes", type=int, default=100, dest="epoch_passes",
                    help="optimizer steps = max(min_steps, passes * ceil(N/bs))")
    ap.add_argument("--min-steps", type=int, default=1500, dest="min_steps")
    # The old 1e-5 was picked when alpha0 was 0.01, i.e. a relative step of
    # 1e-3. alpha0 is now a knob, so scale with it to keep that relative
    # step; a fixed 1e-5 against alpha0 = 1.0 is 100x too small and the
    # parameter simply does not move.
    ap.add_argument("--alpha-lr", type=float, default=0.0, dest="alpha_lr",
                    help="0 = 1e-3 * alpha0")
    ap.add_argument("--iface-lr", type=float, default=5e-4, dest="iface_lr")
    # alpha0 is a CLI knob and alpha_max defaults to 10*alpha0 so the box
    # constraint can never sit below the initial value. The old hardcoded
    # (0.01, max 0.1) pair capped alpha a factor of 10 BELOW the 1.0 that
    # Chen et al. 2003 Eq. (2) implies, i.e. the optimum was outside the
    # feasible set and "training did not help" was confounded with it.
    ap.add_argument("--alpha0", type=float, default=0.01, dest="alpha0")
    ap.add_argument("--alpha-max", type=float, default=0.0, dest="alpha_max",
                    help="0 = 10 * alpha0")
    ap.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    ap.add_argument("--lambda-margin", type=float, default=0.5, dest="lambda_margin")
    # The three-condition ablation of report section 5.14.26. `none` is the
    # mechanism control: it removes the negative term entirely, so a difference
    # between `none` and `minanchor` is what the min-anchor actually bought.
    ap.add_argument("--loss-neg", default="minanchor", dest="loss_neg",
                    choices=("minanchor", "none", "toptail"),
                    help="negative term: min(positive)-anchored hinge (the "
                         "recipe behind every result up to 2026-07-28), none, "
                         "or the soft top-tail penalty")
    ap.add_argument("--toptail-k", type=int, default=32, dest="toptail_k",
                    help="how deep into the negative tail --loss-neg toptail looks")
    ap.add_argument("--tau-pos", type=float, default=0.5, dest="tau_pos")
    ap.add_argument("--tau-neg", type=float, default=0.5, dest="tau_neg")
    ap.add_argument("--tau-hinge", type=float, default=1.0, dest="tau_hinge")
    ap.add_argument("--lambda-prior", type=float, default=0.1, dest="lambda_prior")
    ap.add_argument("--basin-temperature", type=float, default=0.5, dest="basin_temp")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--grad-clip", type=float, default=5.0, dest="grad_clip")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=1e-4, dest="min_delta")
    # search / labelling
    # Chunk defaults are set from a 12-complex profile (see report §5.6): the
    # FFT rotation batch dominates peak VRAM. rot_chunk 32 -> 8 cuts the worst
    # complex from 35.6 to 24.7 GiB at identical wall time (the search is
    # launch-bound, not bandwidth-bound, at this size); rot_chunk 4 halves
    # memory again but costs ~29% more time.
    # These were profiled at the OLD 3.0 A default. At the paper's 1.2 A a grid
    # has (3.0/1.2)^3 = 15.6x the cells, so peak VRAM scales with it: a
    # median-size complex (2.0M voxels) already peaks at 7.9 GiB with
    # rot_chunk=8, and nearly every TEST complex OOM'd at that setting on a
    # 47 GiB card. rot_chunk=2 is the 1.2 A default.
    ap.add_argument("--frame-chunk", type=int, default=100, dest="frame_chunk")
    ap.add_argument("--rot-chunk", type=int, default=2, dest="rot_chunk")
    # One spacing for BOTH the search that generates the pool and the scorer
    # that featurises it. These used to disagree (3.0 vs 1.2), so the top-N cut
    # was made under a different objective from the one being trained.
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    # --- candidate pool -------------------------------------------------
    ap.add_argument("--loss-prov", choices=("search", "all"), default="search",
                    dest="loss_prov",
                    help="which poses the LOSS may see. 'search' = only what "
                         "the FFT search returned (the deployment "
                         "distribution); 'all' also feeds it the enumerated "
                         "near-native poses, which the first N=220 run showed "
                         "collapses end-to-end success@1 from 69.5%% to 34.7%%")
    ap.add_argument("--pool", choices=("reachable", "legacy"), default="reachable",
                    help="reachable: Hopf search (no q* cone) + positives "
                         "enumerated on the search's own lattice. legacy: the "
                         "section 5.6 recipe (cone-seeded search + positives "
                         "injected at the exact, unreachable t*).")
    ap.add_argument("--rot-set", choices=("hopf", "uniform"), default="hopf",
                    dest="rot_set")
    ap.add_argument("--hopf-nside", type=int, default=3, dest="hopf_nside",
                    help="72*nside^3 rotations; 3 -> 1944, covering radius ~18.5 deg")
    ap.add_argument("--near-rot", type=int, default=8, dest="near_rot",
                    help="grid rotations nearest q* to enumerate positives from")
    # +/-1, not +/-2. The radius sets how many near-native poses are
    # enumerated, and 8 rotations x (2R+1)^3 translations floods the positive
    # class fast: measured over 24 TEST complexes, R=2 yields a median 982
    # positives of which only 14 clear DockQ 0.6, while R=1 yields 216 with 8 --
    # a 4.6x smaller pool, the same ceiling (best DockQ 0.671 vs 0.686) and a
    # BETTER median positive (0.383 vs 0.352). The marginal ones are poses
    # pushed up to 2.4 A into the receptor; labelling them "correct" teaches the
    # model that interpenetration is fine, and it made the pooled AUC
    # meaningless (0.085 at DockQ>=0.23 against 0.990 at >=0.8).
    ap.add_argument("--trans-cells", type=int, default=1, dest="trans_cells",
                    help="+/- this many lattice cells around the snapped t*")
    ap.add_argument("--psc-decompose", dest="psc_decompose",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="cache (c_pair, n_ss, n_sc, n_cc) so rho is trainable")
    # --- trainable parameters -------------------------------------------
    ap.add_argument("--iface-mode", choices=("full", "add", "sym"),
                    default="full", dest="iface_mode",
                    help="full: all 144 pair terms. add: additive update "
                         "g + r_i + c_j (23 free directions). sym: symmetric "
                         "additive g + a_i + a_j (12) -- the physically primary "
                         "control, since the published table is symmetric. "
                         "Coefficients are in an orthonormal zero-sum basis so "
                         "the prior and the learning rate mean the same thing "
                         "in every mode.")
    ap.add_argument("--freeze-psc", dest="freeze_psc", action="store_true",
                    help="hold alpha and the clash weights at the published "
                         "values and train only the 144 pair terms")
    ap.add_argument("--psc-mode", choices=("rho", "free"), default="rho",
                    dest="psc_mode",
                    help="rho: keep Chen & Weng's w_k = alpha*rho^k, i.e. the "
                         "three log clash weights stay collinear (146 params). "
                         "free: let them move independently (148), so the "
                         "published 1:rho:rho^2 penalty ratio becomes testable")
    ap.add_argument("--rho0", type=float, default=SC_RHO)
    ap.add_argument("--rho-lr", type=float, default=1e-3, dest="rho_lr")
    ap.add_argument("--rho-min", type=float, default=0.5, dest="rho_min",
                    help="strictly > 0: S_PSC is discontinuous at rho = 0")
    ap.add_argument("--rho-max", type=float, default=9.0, dest="rho_max")
    # Elements in the (frame_chunk*12, nx, ny, nz) featurisation tensor.
    # 1e9 float32 = 4 GiB, which leaves room for the receptor grids and the
    # FFT buffers on a 47 GiB card.
    ap.add_argument("--feature-budget", type=int, default=1_000_000_000,
                    dest="feature_budget",
                    help="max elements in the (chunk*12, nx, ny, nz) IFACE tensor")
    ap.add_argument("--dockq-budget", type=int, default=50_000_000,
                    dest="dockq_budget",
                    help="max elements in the dense DockQ (chunk,N_rec,N_lig) tensor")
    ap.add_argument("--oom-retries", type=int, default=3, dest="oom_retries")
    ap.add_argument("--rmsd-threshold", type=float, default=5.0, dest="rmsd_thr")
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="dockq_thr")
    ap.add_argument("--pool-cap", type=int, default=4000, dest="pool_cap")
    ap.add_argument("--mine-random-rot", type=int, default=1500, dest="mine_random_rot")
    ap.add_argument("--mine-cone", type=int, default=400, dest="mine_cone")
    ap.add_argument("--mine-ntop", type=int, default=1500, dest="mine_ntop")
    args = ap.parse_args()
    if args.mine_from_ckpt and args.rounds < 1:
        raise SystemExit(
            "--mine-from-ckpt --rounds 0 would mine with TRAINED parameters "
            "and write the result into the round-0 cache, whose key carries no "
            "parameter fingerprint and is shared by every seed and model. Pass "
            "--rounds 1 or higher.")
    if args.rounds > 0 and args.mining_contract == "hardneg" \
            and args.rot_set != "hopf":
        raise SystemExit(
            "a mining round de-duplicates poses by (rotation index, "
            f"translation cell), and --rot-set {args.rot_set} reseeds its "
            "rotations per round, so the same index is a different rotation in "
            "round 1 than in round 0 and the keys are not comparable. Use "
            "--rot-set hopf, which is round-independent.")
    if args.alpha_lr <= 0:
        args.alpha_lr = 1e-3 * args.alpha0
    if args.alpha_max <= 0:
        args.alpha_max = 10.0 * args.alpha0
    assert args.alpha_max >= args.alpha0, (
        f"--alpha-max {args.alpha_max} is below --alpha0 {args.alpha0}: "
        "the initial value would be outside the feasible set")

    t_start = time.time()
    device = torch.device(args.device)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    gen = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)

    # Namespace the run directory by what distinguishes concurrent processes.
    # Three sharded miners and the two arms of a mining experiment all have the
    # same N and seed; without this they race on one split.json / skipped.jsonl
    # and each arm overwrites the other's round-1 artefacts.
    run_tag = f"N{args.n_fit}_seed{args.seed}"
    if args.mine_only and args.mine_shard != "0/1":
        run_tag += f"_mine{args.mine_shard.replace('/', 'of')}"
    if args.rounds > 0 and not args.mine_only:
        run_tag += f"_{args.mining_contract}"
    run_dir = Path(args.out_dir) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    skip_log = open(run_dir / "skipped.jsonl", "w", buffering=1)

    beta0 = torch.tensor(3.0, device=device, dtype=dtype)
    charge0 = default_charge_score(device=device, dtype=dtype)

    # ---- dataset selection + leakage assertions --------------------------
    sel, fit_ids, val_ids, prep_status, sel_info = select_split(args)
    test_ids = set(ln.strip() for ln in Path(args.test_ids).read_text().splitlines()
                   if ln.strip())
    assert len(set(sel)) == len(sel), "duplicate id in selection"
    assert not (set(fit_ids) & set(val_ids)), "fit/validation overlap"
    assert set(fit_ids) | set(val_ids) == set(sel), "split does not cover selection"
    assert not (set(sel) & test_ids), "TEST id leaked into fit/validation"
    assert len(fit_ids) == args.n_fit, (
        f"expected {args.n_fit} fit clusters, got {len(fit_ids)}")
    print(f"selection: total={len(sel)} fit={len(fit_ids)} val={len(val_ids)} "
          f"(usable={sel_info['n_usable']} of {sel_info['n_prepared_ok']} prepared; "
          f"{sel_info['n_excluded_oversized']} excluded as oversized "
          f"> {sel_info['max_grid_voxels']} voxels)", flush=True)

    (run_dir / "split.json").write_text(json.dumps(
        {"n_fit": len(fit_ids), "n_val": len(val_ids), "n_total": len(sel),
         "selection": sel_info,
         "fit_ids": fit_ids, "val_ids": val_ids, "config": vars(args)}, indent=1))

    # ---- fixed TEST pool (built once with default parameters, §5.4) ------
    print("loading FIXED deleaked TEST pool ...", flush=True)
    test_feats = load_test_feats(args.test_cache, dtype,
                                 require_psc_terms=args.psc_decompose)
    assert set(f.name for f in test_feats) <= test_ids, "TEST cache holds non-test ids"
    print(f"  test complexes: {len(test_feats)}", flush=True)

    val_set = set(val_ids)
    fit_set = set(fit_ids)
    pools: dict[str, Feats] = {}
    p0 = Params.initial(args, device, dtype)
    p = p0.clone()
    optimizer_state = None
    history = []
    n_skipped_total = 0

    def pool_cache_key(rnd: int) -> str | None:
        if not args.pool_cache:
            return None
        Path(args.pool_cache).mkdir(parents=True, exist_ok=True)
        # The round-0 pool depends on the selection and the BASELINE parameters,
        # neither of which depends on --seed (select_split has no RNG and the
        # split files are byte-identical across seeds). So every seed would mine
        # exactly the same pool. Key the cache on what actually determines it.
        key = (f"n{args.n_fit}_r{rnd}_sp{args.spacing}_{args.pool}"
               f"_{args.rot_set}{args.hopf_nside}_ntop{args.mine_ntop}"
               f"_nr{args.near_rot}_tc{args.trans_cells}"
               f"_a{args.alpha0}_rho{args.rho0}"
               f"_hd{int(args.exclude_homodimer)}_mv{args.max_grid_voxels}"
               f"_bg{int(bool(args.exclude_bad_geometry))}"
               # Pose-identity schema. v1 pools carry no identity at all, and
               # the packing before that recorded only the rotation index's
               # parity. Reusing either under the same key would look like a
               # cache hit and fail hours later, inside a round-1 absorb.
               f"_pk{POSE_KEY_SCHEMA}")
        if rnd > 0:
            key += f"_p{param_fingerprint(p)}"
        return key

    def resume_identity() -> dict:
        """What must match for a resumed round to be the same experiment.

        The Adam state alone does not say which model, loss or split produced
        it, so resuming across a changed `--iface-mode`, `--lambda-margin` or
        selection would silently continue a different run.
        """
        keys = ("iface_mode", "lambda_margin", "margin", "lambda_prior",
                "loss_neg", "toptail_k", "tau_pos", "tau_neg", "tau_hinge",
                "iface_lr", "alpha_lr", "rho_lr", "freeze_psc", "psc_mode",
                "loss_prov", "basin_temp", "batch_size", "dockq_thr",
                "pool_cap", "n_fit", "seed", "spacing", "rot_set",
                "hopf_nside", "mine_ntop", "near_rot", "trans_cells",
                # continuation rule: two arms that stop by different rules are
                # not matched, and nothing else here would notice
                "min_steps", "epoch_passes", "patience", "min_delta",
                "grad_clip", "alpha_max", "rho_min", "rho_max",
                # what the pool and the score even are
                "alpha0", "rho0", "psc_decompose", "pool")
        return {"config": {k: getattr(args, k) for k in keys},
                "split": hashlib.sha1(
                    ("\n".join(fit_ids) + "|" + "\n".join(val_ids))
                    .encode()).hexdigest()[:16]}

    def restore_params(state: dict) -> None:
        """Put a saved `Params.state_dict()` back into `p`, in place.

        One implementation for both --resume-from and --mine-from-ckpt: if the
        miner and the trainer restored a checkpoint even slightly differently
        they would search with different parameters, and the fingerprint that
        is supposed to prevent exactly that would be computed from the wrong
        vector on one side.
        """
        with torch.no_grad():
            p.alpha.copy_(state["alpha"].to(device=device, dtype=dtype))
            if "rho" in state:
                p.rho.copy_(state["rho"].to(device=device, dtype=dtype))
            if "clash_weights" in state:
                p.log_clash.copy_(state["clash_weights"]
                                  .to(device=device, dtype=dtype).log())
            if p.iface_mode == "full":
                p.iface.copy_(state["iface"].to(device=device, dtype=dtype))
            else:
                # `iface_vec()` rebuilds the 144-vector from iface0 + rowcol, so
                # restoring rowcol is what restores the table -- iface0 is p0's
                # published table and is not written by training.
                p.rowcol.copy_(state["rowcol"].to(device=device, dtype=dtype))
        # Elementwise, not just the norm: a permuted or transposed table has the
        # same norm and would search a different scoring function.
        want_vec = state["iface"].to(device=device, dtype=dtype)
        worst = float((p.iface_vec() - want_vec).abs().max())
        if worst > 1e-9:
            raise SystemExit(
                f"restored IFACE table differs from the checkpoint by up to "
                f"{worst:.3e} elementwise; the parametrisation does not "
                f"round-trip and the mined pool would not be the one this "
                f"checkpoint implies")
        got = float((p.iface_vec() - p0.iface_vec()).norm())
        print(f"restored from checkpoint: ||d_iface||={got:.4f} "
              f"alpha={float(p.alpha):.4f} "
              f"fingerprint={param_fingerprint(p)}", flush=True)

    resume_round = -1
    if args.mine_from_ckpt:
        if not args.mine_only:
            raise SystemExit("--mine-from-ckpt is for sharded miners: it skips "
                             "round-0 training, so the parameters it would go "
                             "on to train are not the ones it loaded. Pass "
                             "--mine-only.")
        if args.resume_from:
            raise SystemExit("pass --resume-from or --mine-from-ckpt, not both")
        restore_params(torch.load(args.mine_from_ckpt, map_location="cpu",
                                  weights_only=True))
    elif args.resume_from:
        blob = torch.load(args.resume_from, map_location="cpu",
                          weights_only=False)
        want, have = blob.get("resume_identity"), resume_identity()
        if want is None:
            raise SystemExit(
                f"{args.resume_from} predates the resume-identity record. "
                f"Re-run round 0 with the current code.")
        if want != have:
            diff = [k for k in have["config"]
                    if want["config"].get(k) != have["config"][k]]
            raise SystemExit(
                f"--resume-from was produced by a different run: "
                + (f"config differs in {diff}. " if diff else "")
                + ("the fit/validation split differs. "
                   if want["split"] != have["split"] else "")
                + "Resuming would continue someone else's experiment.")
        restore_params(blob["param_state"])
        optimizer_state = blob["optimizer_state"]
        resume_round = int(blob["round"])
        # Restore the minibatch stream too. Without it a resumed round replays
        # round 0's sequence, so "continue for 1500 more steps" would not be the
        # continuation it claims to be -- and the two arms would only agree
        # because they replay the same wrong sequence.
        if "generator_state" in blob:
            gen.set_state(blob["generator_state"])
        print(f"--resume-from {args.resume_from}: continuing after round "
              f"{resume_round} with its Adam state and minibatch stream",
              flush=True)

    def pool_cache_path(rnd: int) -> Path | None:
        """Where THIS process writes. Sharded miners must not share a file."""
        key = pool_cache_key(rnd)
        if key is None:
            return None
        suffix = "" if args.mine_shard == "0/1" else \
            "." + args.mine_shard.replace("/", "of")
        return Path(args.pool_cache) / f"{key}{suffix}.pt"

    def pool_cache_read(rnd: int) -> list[Path]:
        """Every file that can contribute, whole-set and sharded alike."""
        key = pool_cache_key(rnd)
        if key is None:
            return []
        d = Path(args.pool_cache)
        return sorted(d.glob(f"{key}.pt")) + sorted(d.glob(f"{key}.*of*.pt"))

    #: per-round mining bookkeeping, written into round<r>_metrics.json
    mine_stats: dict[int, dict] = {}

    def absorb(pid: str, nf: Feats, rnd: int) -> None:
        """Install a freshly mined candidate set into the complex's pool."""
        if rnd == 0:
            pools[pid] = nf
            return
        # hard-*negative* mining: never re-add the near-native cone positives,
        # so the positive set stays frozen at round 0.
        neg = (nf.dockq < args.dockq_thr).nonzero(as_tuple=True)[0]
        st = mine_stats.setdefault(
            rnd, {"n_complexes": 0, "n_proposed_neg": 0, "n_duplicate_neg": 0,
                  "n_added_neg": 0, "n_new_survived_cap": 0,
                  "n_old_neg_evicted": 0, "pool_before": 0, "pool_after": 0})
        if neg.numel() == 0 or pid not in pools:
            return
        cand = nf.index(neg)
        keep, n_dup = _fresh_indices(cand, pools[pid])
        st["n_complexes"] += 1
        st["n_proposed_neg"] += int(neg.numel())
        st["n_duplicate_neg"] += n_dup
        st["pool_before"] += pools[pid].n
        if keep.numel() == 0:
            st["pool_after"] += pools[pid].n
            return
        before_pos = int((pools[pid].dockq >= args.dockq_thr).sum())
        old_neg_keys = set(map(tuple, pools[pid].pose_key[
            pools[pid].dockq < args.dockq_thr].tolist()))
        pools[pid].cat(cand.index(keep))
        st["n_added_neg"] += int(keep.numel())
        pools[pid] = cap_pool(pools[pid], args.pool_cap, p.cpu(),
                              beta0.detach().cpu(), args.dockq_thr)
        # The cap does not just refuse new poses, it can evict old ones -- so
        # `pool_after - pool_before` is not the number of new poses that
        # survived. Count each side directly.
        after_keys = set(map(tuple, pools[pid].pose_key.tolist()))
        st["n_new_survived_cap"] += sum(
            1 for k in map(tuple, cand.pose_key[keep].tolist())
            if k in after_keys)
        st["n_old_neg_evicted"] += sum(
            1 for k in old_neg_keys if k not in after_keys)
        st["pool_after"] += pools[pid].n
        after_pos = int((pools[pid].dockq >= args.dockq_thr).sum())
        assert after_pos == before_pos, (
            f"{pid}: positives changed {before_pos} -> {after_pos} "
            f"during mining round {rnd}")

    def try_mine(pid: str, rnd: int, **chunks):
        """Return ``(Feats, meta)`` on success or ``(None, meta)`` on failure."""
        prot_cpu = load_prepared(args.prep_cache, pid)
        if prot_cpu is None:
            # `load_prepared` returns None for a MISSING **or CORRUPT** entry,
            # and the rescue pass only retries `stage == "mine"`. Silently
            # skipping here would drop the complex from the fit set while
            # `len(fit_ids) == n_fit` still held (that assertion checks the id
            # list, not the loaded pool), i.e. the run would quietly train at
            # N-1. The prep cache is meant to be complete, so this is fatal.
            raise SystemExit(
                f"prep cache entry for {pid} is missing or unreadable under "
                f"{args.prep_cache}. Re-prepare it (scripts/prep_pinder_cache.py "
                f"--force) rather than running at a silently reduced N.")
        meta = {"stage": "mine", "n_rec": prot_cpu.n_rec, "n_lig": prot_cpu.n_lig}
        prot = None
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            prot = prot_cpu.to(device, dtype=dtype)
            nf = mine_complex(prot, p, beta0, charge0, args, rnd,
                              **chunks)
            return nf, meta
        except torch.cuda.OutOfMemoryError as exc:
            meta["reason"] = f"OOM: {str(exc)[:160]}"
            return None, meta
        except Exception as exc:  # noqa: BLE001
            meta["reason"] = f"{type(exc).__name__}: {exc}"[:200]
            return None, meta
        finally:
            del prot, prot_cpu
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # A sharded miner already holds the parameters round 0 would have produced,
    # so it goes straight to the round it was asked to mine; --resume-from
    # picks up after the round its checkpoint recorded.
    rounds_to_run = ([args.rounds] if args.mine_from_ckpt
                     else list(range(resume_round + 1, args.rounds + 1)))
    if not rounds_to_run:
        raise SystemExit(
            f"nothing to do: --resume-from is at round {resume_round} and "
            f"--rounds is {args.rounds}. Raise --rounds.")

    def preflight_identities(pool_map, ids, where: str) -> None:
        """Fail before the expensive part if a later round could not dedup.

        Placed at every entry point that precedes hours of mining -- Phase A's
        round 0, and a sharded miner's start -- because the alternative is
        discovering it in Phase C's first `absorb`, after the mining is spent.
        """
        if args.pool != "reachable":
            return
        bad = [q for q in ids
               if q in pool_map and not has_pose_identity(pool_map[q].pose_key)]
        if bad:
            raise SystemExit(
                f"{len(bad)} of {len(ids)} round-0 pools carry no pose "
                f"identity (e.g. {bad[0]}), so a later round could not tell a "
                f"new pose from one it already holds. Re-mine round 0 with the "
                f"current code (cache schema v{POSE_KEY_SCHEMA}); {where}.")
        missing = [q for q in ids if q not in pool_map]
        if missing:
            raise SystemExit(
                f"{len(missing)} of {len(ids)} round-0 pools are absent "
                f"(e.g. {missing[0]}); {where}.")

    if args.mine_from_ckpt and args.pool == "reachable":
        # This worker mines round > 0 without ever loading the round-0 pool, so
        # nothing else here would notice that Phase C will not be able to use
        # what it produces. Read the cache once and check, before hours of FFT.
        r0: dict[str, Feats] = {}
        for src in pool_cache_read(0):
            blob = torch.load(src, map_location="cpu", weights_only=True)
            for d in blob["pools"]:
                r0[d["name"]] = Feats(
                    d["name"], d["sc"], d["T"], d["elec"], d["rmsd"],
                    d["dockq"], d["origin"], d["prov"], d.get("pose_key"))
        preflight_identities(
            r0, fit_ids,
            "a sharded miner refuses to spend hours on a pool round 1 could "
            "not absorb")
        print(f"miner preflight: {len(fit_ids)} round-0 pools carry pose "
              f"identities", flush=True)
        del r0

    if resume_round >= 0:
        # Skipping round 0 also skips the read that fills `pools`, and a later
        # round trains on `pools` -- so without this the resumed run would fit
        # on an empty set while every assertion about ids still passed.
        for src in pool_cache_read(0):
            blob = torch.load(src, map_location="cpu", weights_only=True)
            for d in blob["pools"]:
                pools[d["name"]] = Feats(d["name"], d["sc"], d["T"], d["elec"],
                                         d["rmsd"], d["dockq"], d["origin"],
                                         d["prov"], d.get("pose_key"))
        missing = [q for q in sel if q not in pools]
        if missing:
            raise SystemExit(
                f"--resume-from needs the round-0 pool, but {len(missing)} of "
                f"{len(sel)} complexes are absent from {args.pool_cache}. "
                f"Re-run round 0 with the same --pool-cache first.")
        preflight_identities(pools, sel, "checked when resuming")
        print(f"resumed run: loaded {len(pools)} round-0 pools from cache",
              flush=True)
    for rnd in rounds_to_run:
        t_round = time.time()
        targets = sel if rnd == 0 else fit_ids
        if rnd > 0 and args.mining_contract == "none":
            # Matched-BUDGET control: same checkpoint, same Adam state, same
            # minibatch stream, same step budget and stopping rule, same pool.
            # Not "matched compute" -- `--patience` can stop the two arms at
            # different `steps_run`. Whatever this round changes is what further
            # optimisation alone buys, which is what the mining arm must beat.
            targets = []
            print(f"  round {rnd}: --mining-contract none, pool unchanged "
                  f"({len(fit_ids)} fit complexes), training only", flush=True)
        t_mine = time.time()

        cpath = pool_cache_path(rnd)
        n_from_cache = 0
        # A round > 0 cache holds the RAW mined candidate sets, not pools: the
        # de-duplication, the positive freeze and the cap all belong to
        # `absorb`, and running them here as well would apply them twice.
        mined_raw: dict[str, Feats] = {}
        sources = pool_cache_read(rnd) if targets else []
        if sources:
            for src in sources:
                blob = torch.load(src, map_location="cpu", weights_only=True)
                for d in blob["pools"]:
                    f = Feats(d["name"], d["sc"], d["T"], d["elec"],
                              d["rmsd"], d["dockq"], d["origin"], d["prov"],
                              d.get("pose_key"))
                    tgt = pools if rnd == 0 else mined_raw
                    if f.name in tgt and not torch.equal(
                            tgt[f.name].pose_key, f.pose_key):
                        raise SystemExit(
                            f"{f.name} appears in two cache files for round "
                            f"{rnd} with different poses. Silently keeping the "
                            f"last one would make the pool depend on glob "
                            f"order. Delete the stale shard.")
                    tgt[f.name] = f
            for pid, f in mined_raw.items():
                if pid in fit_set:
                    absorb(pid, f, rnd)
        # Only what the cache lacks is mined. A changed selection (a new
        # exclusion list, a different N) overlaps heavily with the old one, and
        # re-mining the shared part costs hours for nothing. Entries the
        # current selection no longer wants stay in the cache but are simply
        # not read into fit/val below.
        #
        # Shard FIRST, then drop what is already cached. The other order makes
        # a shard's membership depend on what happened to be cached when it
        # started, so a worker launched late, or re-run after a failure, owns a
        # different set and the union stops covering the fit list exactly once.
        # At round 0 the pool is seed-independent so sharding is pure
        # wall-clock; at round > 0 it is a function of the trained parameters,
        # which is why the cache key carries their fingerprint -- a shard mined
        # from a different checkpoint lands in a different file and can never
        # be silently mixed in.
        si, sn = (int(x) for x in args.mine_shard.split("/"))
        if sn > 1 and targets:
            targets = targets[si::sn]
            print(f"  round {rnd}: mining shard {si}/{sn} -> {len(targets)} "
                  f"complexes (before removing cached)", flush=True)
        if sources:
            have = pools if rnd == 0 else mined_raw
            n_from_cache = sum(1 for q in targets if q in have)
            targets = [q for q in targets if q not in have]
            print(f"  round {rnd}: {n_from_cache} candidate set(s) reused from "
                  f"{len(sources)} cache file(s), {len(targets)} to mine",
                  flush=True)
        failed: list[tuple[str, dict]] = []
        for ci, pid in enumerate(targets):
            nf, meta = try_mine(pid, rnd)
            if nf is None:
                failed.append((pid, meta))
            elif rnd == 0:
                absorb(pid, nf, rnd)
            else:
                mined_raw[pid] = nf
                absorb(pid, nf, rnd)
            if (ci + 1) % 50 == 0:
                el = time.time() - t_mine
                print(f"  round {rnd}: mined {ci+1}/{len(targets)} "
                      f"({el/(ci+1):.2f}s/complex, {el/60:.1f} min, "
                      f"{len(failed)} pending rescue)", flush=True)

        # Rescue pass. Whether a complex OOMs depends on how fragmented the
        # allocator happens to be when its turn comes, which would make set
        # membership depend on transient GPU pressure — i.e. differ between
        # seeds and between N for reasons unrelated to the science. Re-running
        # the failures at the minimum chunk sizes, with the pool already built
        # and the cache emptied, makes membership deterministic in practice.
        n_skipped = 0
        if failed:
            print(f"  round {rnd}: rescue pass over {len(failed)} complex(es) "
                  f"at rot_chunk=1", flush=True)
            for pid, meta in failed:
                if meta["stage"] != "mine":
                    skip_log.write(json.dumps({"round": rnd, "id": pid, **meta})
                                   + "\n")
                    n_skipped += 1
                    continue
                nf, meta2 = try_mine(pid, rnd, rot_chunk=1, frame_chunk=25,
                                     pose_chunk=1)
                if nf is None:
                    skip_log.write(json.dumps(
                        {"round": rnd, "id": pid, "rescued": False,
                         "first_reason": meta.get("reason", ""), **meta2}) + "\n")
                    n_skipped += 1
                    print(f"    [{pid}] SKIP after rescue "
                          f"(n_rec={meta2.get('n_rec')} n_lig={meta2.get('n_lig')})",
                          flush=True)
                else:
                    if rnd > 0:
                        mined_raw[pid] = nf
                    absorb(pid, nf, rnd)
                    skip_log.write(json.dumps(
                        {"round": rnd, "id": pid, "rescued": True,
                         "first_reason": meta.get("reason", ""),
                         "stage": "mine", "n_rec": meta["n_rec"],
                         "n_lig": meta["n_lig"]}) + "\n")
                    print(f"    [{pid}] rescued at rot_chunk=1", flush=True)
        n_skipped_total += n_skipped
        mine_seconds = time.time() - t_mine
        if cpath is not None and (targets or n_from_cache == 0):
            # round 0 caches the pools themselves; a later round caches the raw
            # candidate sets it mined, keyed by the parameters that found them.
            store = pools.values() if rnd == 0 else \
                [mined_raw[p] for p in targets if p in mined_raw]
            if rnd == 0 or store:
                torch.save({"n_skipped": n_skipped,
                            "pools": [{"name": f.name, "sc": f.sc, "T": f.T,
                                       "elec": f.elec, "rmsd": f.rmsd,
                                       "dockq": f.dockq, "origin": f.origin,
                                       "prov": f.prov, "pose_key": f.pose_key}
                                      for f in store]}, cpath)
                print(f"  round {rnd}: cached {len(list(store))} candidate "
                      f"set(s) -> {cpath}", flush=True)

        if args.mine_only:
            print(f"  --mine-only: wrote {cpath}; stopping before training",
                  flush=True)
            return

        if rnd == 0:
            # Phase A checks what Phase B and C will depend on, while it is
            # still a second of start-up rather than hours of spent mining.
            preflight_identities(pools, sel, "checked after mining round 0")
        fit_pools = [pools[p] for p in fit_ids if p in pools]
        val_pools = [pools[p] for p in val_ids if p in pools]
        assert not (set(f.name for f in fit_pools) & set(f.name for f in val_pools))
        assert not (set(f.name for f in fit_pools) & test_ids)
        assert not (set(f.name for f in val_pools) & test_ids)

        if rnd == 0:
            # Baseline (default ZDOCK parameters) on the same fixed pools.
            b_sr, b_sd, b_t1 = aggregate(test_feats, p0, beta0,
                                         device, args.rmsd_thr, args.dockq_thr)
            print("=" * 62)
            print("BASELINE (default ZDOCK params) on fixed deleaked TEST")
            print("=" * 62)
            print(_fmt("baseline/TEST", b_sr, b_sd, b_t1), flush=True)
            cov = {"fit": iface_coverage(fit_pools),
                   "val": iface_coverage(val_pools),
                   "test": iface_coverage(test_feats)}
            (run_dir / "coverage.json").write_text(json.dumps(cov, indent=1))
            print(f"  IFACE coverage: fit zero-components="
                  f"{cov['fit']['n_components_zero']}/144  "
                  f"median complexes/component={cov['fit']['median_coverage']}"
                  f" of {cov['fit']['n_complexes']}", flush=True)
            (run_dir / "baseline_test.json").write_text(json.dumps(
                {"success_rmsd": {str(k): b_sr[k] for k in KS},
                 "success_dockq": {str(k): b_sd[k] for k in KS},
                 "mean_best_dockq_at1": b_t1}, indent=1))

        n_steps = max(args.min_steps,
                      args.epoch_passes * -(-len(fit_pools) // args.batch_size))
        val_every = max(50, n_steps // 100)
        t_train = time.time()
        p, stats, traj, optimizer_state = train_params(
            fit_pools, val_pools, p, p0, beta0, args,
            device, dtype, gen, n_steps, val_every, optimizer_state)
        train_seconds = time.time() - t_train

        with open(run_dir / f"round{rnd}_trajectory.csv", "w") as fh:
            fh.write("step,fit_loss,val_loss,grad_norm,accepted,"
                     "alpha,rho,d_iface_norm\n")
            for r in traj:
                fh.write(f"{r['step']},{r['fit_loss']:.6f},{r['val_loss']:.6f},"
                         f"{r['grad_norm']:.6f},{r['accepted']},{r['alpha']:.6f},"
                         f"{r['rho']:.6f},{r['d_iface_norm']:.6f}\n")

        sr, sd, t1 = aggregate(test_feats, p, beta0, device,
                               args.rmsd_thr, args.dockq_thr)
        srt, sdt, t1t = aggregate(fit_pools, p, beta0, device,
                                  args.rmsd_thr, args.dockq_thr)
        n_usable_loss = sum(
            1 for f in fit_pools
            if int((loss_view(f, args.loss_prov).dockq >= args.dockq_thr).sum()) > 0)
        print(f"  complexes contributing a gradient under --loss-prov "
              f"{args.loss_prov}: {n_usable_loss}/{len(fit_pools)}", flush=True)
        comp = [f.counts(args.dockq_thr) for f in fit_pools]
        pool_keys = ("n", "n_pos", "n_rand_neg", "n_hard_neg",
                     "n_pos_from_search", "n_pos_enumerated")
        pool_stats = {k: sum(c[k] for c in comp) / max(1, len(comp))
                      for k in pool_keys}
        # A complex with no positive at all has no reachable near-native pose
        # under this rotation grid and translation lattice: no amount of
        # parameter fitting can make the search return one. Count it.
        pool_stats["n_complexes_without_positive"] = sum(
            1 for c in comp if c["n_pos"] == 0)
        rejected = [r for r in traj if r["step"] > 0 and not r["accepted"]]
        accepted = [r for r in traj if r["step"] > 0 and r["accepted"]]
        peak_mem = (torch.cuda.max_memory_allocated() / 2**30
                    if device.type == "cuda" else 0.0)

        rec = {
            "round": rnd, "seed": args.seed, "n_fit": len(fit_pools),
            "n_val": len(val_pools), "n_test": len(test_feats),
            "n_skipped_this_round": n_skipped, "n_skipped_total": n_skipped_total,
            **p.summary(p0),
            "alpha_at_bound": stats["alpha_at_bound"],
            "rho_at_bound": stats["rho_at_bound"],
            "val_loss": stats["best_val_loss"], "steps_run": stats["steps_run"],
            "steps_budget": n_steps, "val_every": val_every,
            "mean_grad_norm": stats["mean_grad_norm"],
            "max_grad_norm": stats["max_grad_norm"],
            "n_accepted_checkpoints": len(accepted),
            "n_rejected_checkpoints": len(rejected),
            "rejected_mean_fit_loss": (sum(r["fit_loss"] for r in rejected)
                                       / len(rejected)) if rejected else None,
            "rejected_mean_val_loss": (sum(r["val_loss"] for r in rejected)
                                       / len(rejected)) if rejected else None,
            "pool_mean": pool_stats,
            "mining_contract": args.mining_contract,
            # How much of what the search proposed this round was actually new,
            # and how much of that survived the cap. A round whose proposals are
            # mostly duplicates has not mined anything, however long it ran.
            "mining": mine_stats.get(rnd),
            "test_success_rmsd": {str(k): sr[k] for k in KS},
            "test_success_dockq": {str(k): sd[k] for k in KS},
            "test_mean_best_dockq_at1": t1,
            "fit_success_rmsd": {str(k): srt[k] for k in KS},
            "fit_success_dockq": {str(k): sdt[k] for k in KS},
            "fit_mean_best_dockq_at1": t1t,
            "mine_seconds": mine_seconds, "train_seconds": train_seconds,
            "round_seconds": time.time() - t_round,
            "peak_gpu_gib": peak_mem,
        }
        (run_dir / f"round{rnd}_metrics.json").write_text(json.dumps(rec, indent=1))
        torch.save({"round": rnd, "seed": args.seed, "n_fit": len(fit_pools),
                    **p.state_dict(),
                    "val_loss": stats["best_val_loss"], "config": vars(args)},
                   run_dir / f"round{rnd}_ckpt.pt")
        # Separate file: the optimizer state is large, and every consumer of the
        # checkpoint so far (compare_conditions, the analysis scripts) loads it
        # with weights_only=True and wants only the parameters. Saving it lets a
        # later round resume this exact state instead of retraining round 0 --
        # which matters because the mine and continue arms must branch from ONE
        # round-0 state, not from two independent re-runs of it.
        torch.save({"round": rnd, "seed": args.seed,
                    "optimizer_state": optimizer_state,
                    "param_state": p.state_dict(),
                    # Enough identity that resuming with a different model,
                    # loss or split fails loudly instead of producing a number.
                    "resume_identity": resume_identity(),
                    "generator_state": gen.get_state()},
                   run_dir / f"round{rnd}_optstate.pt")

        tag = ("round 0 (no mining)" if rnd == 0
               else f"round {rnd} (mined)" if args.mining_contract == "hardneg"
               else f"round {rnd} (matched-budget control, no mining)")
        print("=" * 62)
        print(f"{tag} | N_fit={len(fit_pools)} | mean pool={pool_stats['n']:.0f} "
              f"(pos {pool_stats['n_pos']:.0f} "
              f"[search {pool_stats['n_pos_from_search']:.0f} / "
              f"enumerated {pool_stats['n_pos_enumerated']:.0f}] / neg "
              f"{pool_stats['n_rand_neg']:.0f})")
        if pool_stats["n_complexes_without_positive"]:
            print(f"  {pool_stats['n_complexes_without_positive']} of "
                  f"{len(fit_pools)} fit complexes have NO reachable positive "
                  f"- that is a hard ceiling on what training can do")
        print(f"  alpha={float(p.alpha):.4f} rho={float(p.rho):.4f} ||dIface||="
              f"{float((p.iface_vec()-p0.iface_vec()).norm()):.3f} "
              f"val_loss={stats['best_val_loss']:.4f} "
              f"steps={stats['steps_run']}/{n_steps} skipped={n_skipped} "
              f"peakGPU={peak_mem:.1f} GiB")
        if stats["alpha_at_bound"] or stats["rho_at_bound"]:
            print(f"  WARNING: a parameter sits on its box constraint "
                  f"(alpha {stats['alpha_at_bound']}, rho {stats['rho_at_bound']}) "
                  f"- the reported optimum is a clamp artifact", flush=True)
        print("=" * 62)
        print(_fmt("TEST(deleaked)", sr, sd, t1))
        print(_fmt("FIT", srt, sdt, t1t))
        print(flush=True)
        history.append(rec)

    summary_csv = run_dir / "summary.csv"
    with open(summary_csv, "w") as fh:
        fh.write("round,seed,n_fit,n_val,n_test,"
                 + ",".join(f"test_dockq@{k}" for k in KS) + ","
                 + ",".join(f"test_rmsd@{k}" for k in KS)
                 + ",test_mean_best_dockq1,val_loss,alpha,d_iface,"
                   "mean_pool,n_pos,n_rand_neg,n_hard_neg,n_skipped,"
                   "mine_s,train_s,peak_gpu_gib\n")
        for r in history:
            fh.write(",".join([
                str(r["round"]), str(r["seed"]), str(r["n_fit"]), str(r["n_val"]),
                str(r["n_test"]),
                *[f"{r['test_success_dockq'][str(k)]:.6f}" for k in KS],
                *[f"{r['test_success_rmsd'][str(k)]:.6f}" for k in KS],
                f"{r['test_mean_best_dockq_at1']:.6f}", f"{r['val_loss']:.6f}",
                f"{r['alpha']:.6f}", f"{r['d_iface_norm']:.6f}",
                f"{r['pool_mean']['n']:.1f}", f"{r['pool_mean']['n_pos']:.1f}",
                f"{r['pool_mean']['n_rand_neg']:.1f}",
                f"{r['pool_mean']['n_hard_neg']:.1f}",
                str(r["n_skipped_this_round"]), f"{r['mine_seconds']:.1f}",
                f"{r['train_seconds']:.1f}", f"{r['peak_gpu_gib']:.2f}"]) + "\n")

    skip_log.close()
    print("\n=== DockQ success@K on held-out DELEAKED TEST (PINDER-S) ===")
    print(f"{'round':<20} {'top1':>7} {'top5':>7} {'top10':>7} {'top50':>7} {'top100':>7}")
    for r in history:
        d = r["test_success_dockq"]
        print(f"{'round '+str(r['round']):<20} "
              + " ".join(f"{d[str(k)]*100:6.1f}%" for k in KS))
    if len(history) > 1:
        g = {k: (history[-1]["test_success_dockq"][str(k)]
                 - history[0]["test_success_dockq"][str(k)]) * 100 for k in KS}
        print("mining gain (pp)     " + " ".join(f"{g[k]:6.1f} " for k in KS))
    print(f"\ntotal wall time: {(time.time()-t_start)/60:.1f} min  "
          f"-> {run_dir}")


if __name__ == "__main__":
    main()
