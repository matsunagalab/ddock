"""Fixed-parameter smooth intermolecular LJ and screened charge potential.

Coordinates/sigma are Angstrom, energies kJ/mol, charges elementary charge.
This is an intermolecular search potential, not an implicit-solvent binding
free energy.  Only the divergent core is changed: the LJ well and zero remain
exact outside 0.9 sigma.  A radial C2 continuation is used at both cores.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import math

import torch
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True, eq=False)
class TypedParameters:
    sigma: torch.Tensor
    epsilon: torch.Tensor
    charges: torch.Tensor
    diagnostics: dict = field(default_factory=dict)

    def validate(self, n_atoms: int | None = None):
        if self.sigma.ndim != 1 or not len(self.sigma):
            raise ValueError('typed parameters must be nonempty vectors')
        for value in (self.sigma, self.epsilon, self.charges):
            if value.shape != self.sigma.shape or value.dtype != torch.float64 or value.device.type != 'cpu':
                raise TypeError('typed potential requires matching CPU float64 parameter vectors')
            if not bool(torch.isfinite(value).all()):
                raise ValueError('typed parameters must be finite')
        if n_atoms is not None and len(self.sigma) != n_atoms:
            raise ValueError('typed parameter count differs from coordinates')
        if bool((self.epsilon < 0).any()) or bool((self.sigma < 0).any()):
            raise ValueError('sigma and epsilon must be nonnegative')
        if bool(((self.epsilon > 0) & (self.sigma <= 0)).any()):
            raise ValueError('positive epsilon requires positive sigma')


@dataclass(frozen=True)
class TypedPotentialConfig:
    lj_core_ratio: float = .9
    dielectric: float = 78.5
    kappa: float = .1
    coulomb_core: float = .5
    switch_start: float = 10.
    cutoff: float = 12.
    atom_chunk_size: int = 32
    checkpoint_direct: bool = True
    # Mass assignment is a coarse proposal, not an exact atomistic score.
    # The optional cardinal correction can create deep overshoot wells on
    # underresolved grids and is deliberately NOT the default.
    grid_interpolation: str = 'mass'

    def validate(self):
        for name in ('lj_core_ratio', 'dielectric', 'kappa', 'coulomb_core', 'switch_start', 'cutoff'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite real scalar')
        if not 0 < self.lj_core_ratio < 1:
            raise ValueError('LJ core ratio must lie in (0,1)')
        if self.dielectric <= 0 or self.kappa < 0 or self.coulomb_core <= 0:
            raise ValueError('invalid dielectric, screening, or Coulomb core')
        if not self.coulomb_core < self.switch_start < self.cutoff:
            raise ValueError('require Coulomb core < switch_start < cutoff')
        if type(self.atom_chunk_size) is not int or self.atom_chunk_size < 1:
            raise ValueError('atom_chunk_size must be a positive integer')
        if type(self.checkpoint_direct) is not bool:
            raise ValueError('checkpoint_direct must be boolean')
        if self.grid_interpolation not in ('mass', 'cardinal'):
            raise ValueError('grid_interpolation must be mass or cardinal')

    def to_dict(self):
        return dict(asdict(self), length_unit='angstrom', energy_unit='kJ/mol',
                    coulomb_constant_kj_angstrom_per_mol_e2=1389.35456,
                    scope='intermolecular C2 LJ plus screened dielectric charge; no solvent burial response')


def _validate_xyz(xyz, params):
    params.validate(len(xyz))
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.dtype != torch.float64 or xyz.device.type != 'cpu':
        raise TypeError('coordinates must have shape (N,3), CPU float64')
    if not bool(torch.isfinite(xyz).all()):
        raise ValueError('coordinates must be finite')


def extract_typed_parameters(protein) -> TypedParameters:
    """Rebuild the SAME prepared topology/ff14SB assignment; retain LJ data.

    Rebuilding avoids changing legacy ChargedProtein. Atom order and charges
    are checked. No positions are minimized or hydrogens regenerated here.
    """
    from .atomistic_score import _topology
    from .chemistry import _load_openmm, FORCEFIELD_FILE
    openmm, app, unit = _load_openmm()
    topology = _topology([protein], app)
    ff = app.ForceField(FORCEFIELD_FILE)
    system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=None,
                             rigidWater=False, removeCMMotion=False)
    forces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    if len(forces) != 1 or forces[0].getNumParticles() != protein.monomer.n_atoms:
        raise ValueError('expected one ff14SB particle assignment in prepared atom order')
    rows = [[q.value_in_unit(unit.elementary_charge), sigma.value_in_unit(unit.angstrom),
             epsilon.value_in_unit(unit.kilojoule_per_mole)]
            for q, sigma, epsilon in (forces[0].getParticleParameters(i)
                                      for i in range(forces[0].getNumParticles()))]
    values = torch.tensor(rows, dtype=torch.float64)
    if not torch.allclose(values[:, 0], protein.monomer.charges.detach().cpu().double(), atol=1e-12, rtol=0):
        raise ValueError('reconstructed ff14SB charges differ from prepared protein')
    digest = hashlib.sha256(values.numpy().tobytes()).hexdigest()
    result = TypedParameters(values[:, 1].clone(), values[:, 2].clone(), values[:, 0].clone(),
        dict(forcefield=FORCEFIELD_FILE, openmm_version=str(openmm.__version__),
             parameter_sha256=digest, n_atoms=len(rows), n_exceptions=forces[0].getNumExceptions(),
             parameter_columns=['charge_e', 'sigma_angstrom', 'epsilon_kj_per_mol'],
             role='intermolecular pair parameters; intramolecular exclusions are NOT implied'))
    result.validate()
    return result


def c2_switch(distance2: torch.Tensor, config: TypedPotentialConfig) -> torch.Tensor:
    r = distance2.clamp_min(1e-24).sqrt()
    u = ((r - config.switch_start) / (config.cutoff-config.switch_start)).clamp(0., 1.)
    return (1-u).pow(3) * (1+3*u+6*u.square())


def lj_potential(distance2: torch.Tensor, sigma: torch.Tensor, epsilon: torch.Tensor,
                 config: TypedPotentialConfig | None = None) -> torch.Tensor:
    """Mixed-pair C2 LJ; arguments broadcast, including zero-epsilon sites."""
    config = config or TypedPotentialConfig()
    uc = config.lj_core_ratio ** 2
    safe_sigma = torch.where(epsilon > 0, sigma, torch.ones_like(sigma))
    u = distance2 / safe_sigma.square()
    exterior = u.clamp_min(uc)
    inv3 = exterior.pow(-3)
    h = 4*(inv3.square()-inv3)
    hc = 4*(uc**-6-uc**-3)
    hp = 4*(-6*uc**-7+3*uc**-4)
    hpp = 4*(42*uc**-8-12*uc**-5)
    core = hc + hp*(u-uc) + .5*hpp*(u-uc).square()
    return epsilon * torch.where(u < uc, core, h) * c2_switch(distance2, config)


def charge_kernel(distance2: torch.Tensor, config: TypedPotentialConfig | None = None) -> torch.Tensor:
    """kJ/mol per e² with a finite C2 continuation in squared distance."""
    config = config or TypedPotentialConfig()
    rc, k = config.coulomb_core, config.kappa
    sc = rc*rc
    r = distance2.clamp_min(sc).sqrt()
    exterior = torch.exp(-k*r)/r
    e = math.exp(-k*rc)
    v = e/rc
    vp = -e*(k*rc+1)/(2*rc**3)
    vpp = e*(k*k*rc*rc+3*k*rc+3)/(4*rc**5)
    core = v + vp*(distance2-sc) + .5*vpp*(distance2-sc).square()
    return (1389.35456/config.dielectric) * torch.where(distance2 < sc, core, exterior) * c2_switch(distance2, config)


def direct_pair_energy(rx: torch.Tensor, lx: torch.Tensor, rparams: TypedParameters,
                       lparams: TypedParameters, config: TypedPotentialConfig | None = None) -> torch.Tensor:
    """Differentiable direct intermolecular energy; no grid and no FFT.

    All cross-partner pairs are included. Intramolecular ff14SB exceptions
    must be handled by the monomer prior, not by this function.
    Reverse-mode AD recomputes one chunk at a time, avoiding retained all-pair
    intermediates. Forward AD bypasses checkpoint (PyTorch checkpoint does
    not support JVP transforms). This switch does not alter the potential.
    """
    config = config or TypedPotentialConfig()
    config.validate()
    _validate_xyz(rx, rparams)
    _validate_xyz(lx, lparams)
    has_forward_tangent = any(torch.autograd.forward_ad.unpack_dual(x).tangent is not None
                              for x in (rx, lx))
    use_checkpoint = (config.checkpoint_direct and torch.is_grad_enabled()
                      and (rx.requires_grad or lx.requires_grad) and not has_forward_tangent)
    result = rx.sum()*0 + lx.sum()*0
    for start in range(0, len(rx), config.atom_chunk_size):
        end = start + config.atom_chunk_size
        args = (rx[start:end], lx, rparams.sigma[start:end], rparams.epsilon[start:end],
                rparams.charges[start:end], lparams.sigma, lparams.epsilon, lparams.charges, config)
        if use_checkpoint:
            value = checkpoint(_direct_pair_chunk, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            value = _direct_pair_chunk(*args)
        result = result + value
    return result


def _direct_pair_chunk(rx, lx, rsigma, repsilon, rcharge, lsigma, lepsilon, lcharge, config):
    d2 = (rx[:, None, :] - lx[None, :, :]).square().sum(-1)
    sigma = .5*(rsigma[:, None] + lsigma[None, :])
    epsilon = (repsilon[:, None] * lepsilon[None, :]).sqrt()
    charge = rcharge[:, None] * lcharge[None, :]
    return (lj_potential(d2, sigma, epsilon, config) + charge*charge_kernel(d2, config)).sum()
