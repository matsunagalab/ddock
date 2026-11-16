"""A stationary apo reference for an ff14SB/OBC2 torsional deformation prior.

All energies are kJ/mol and coordinates are Angstrom.  The explicit Cartesian
anchor is part of the model, not a numerical Hessian shift.  The CA spring
prior is not added.  One apo-only reference is shared across all docking poses.

OpenMM supplies exact Cartesian first derivatives.  The PyTorch bridge supports
ordinary first-order reverse AD; it deliberately does not pretend to supply
second derivatives.  Hessian-vector products difference the complete torsional
gradient, including the nonlinear coordinate map.  No native coordinates enter
this module.  Reference contexts are CPU float64 and not thread-safe.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from .atomistic_score import PROTOCOL, _particle_parameters, _split_force_groups, _topology
from .chemistry import ChargedProtein, FORCEFIELD_FILE, _forcefield_sources, _load_openmm
from .hydrogen_kinematics import HydrogenatedKinematics


POLISH_PROTOCOL = dict(method='matrix_free_Newton_CG', maximum_iterations=5,
    maximum_cg_iterations=100, cg_relative_tolerance=.01, hvp_step_rad=1e-4,
    energy_roundoff_kJ_per_mol=1e-7, maximum_backtracks=12,
    required_torque_ratio=.8, physical_hessian_materialized=False)


class PhysicalPreparationError(RuntimeError):
    """A failed apo preparation, retaining its complete numerical diagnostics."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _positive(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _tensor(value, shape, name):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float64:
        raise TypeError(f"{name} must be a CPU float64 tensor")
    if value.shape != shape or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must have finite shape {shape}")


def _input_fingerprint(protein):
    def digest(value):
        return hashlib.sha256(value).hexdigest()
    return dict(xyz_sha256=digest(protein.monomer.xyz.detach().contiguous().numpy().tobytes()),
        charges_sha256=digest(protein.monomer.charges.detach().contiguous().numpy().tobytes()),
        atom_keys_sha256=digest(json.dumps([asdict(key) for key in protein.monomer.atom_keys], sort_keys=True).encode()),
        bonds_sha256=digest(json.dumps(protein.bonds).encode()),
        heavy_indices_sha256=digest(protein.heavy_indices.numpy().tobytes()))


class _ForceContext:
    def __init__(self, protein):
        openmm, app, unit = _load_openmm()
        data = Path(app.__file__).parent / "data"
        ff_path, implicit_path = data / FORCEFIELD_FILE, data / PROTOCOL["implicit_solvent"]
        hashes = _forcefield_sources(ff_path)
        if (protein.provenance.get("forcefield_file") != FORCEFIELD_FILE
                or sorted(protein.provenance.get("forcefield_xml_sha256", {}).values()) != sorted(hashes.values())
                or protein.provenance.get("openmm_version") != openmm.__version__):
            raise ValueError("Physical prior chemistry differs from prepared ff14SB/OpenMM provenance")
        forcefield = app.ForceField(str(ff_path), str(implicit_path))
        system = forcefield.createSystem(_topology((protein,), app), nonbondedMethod=app.NoCutoff,
            constraints=None, rigidWater=False, removeCMMotion=False, soluteDielectric=1.,
            solventDielectric=78.5, implicitSolventKappa=0. / unit.nanometer, sasaMethod="ACE")
        parameters, _ = _particle_parameters(system, openmm, unit)
        if not np.allclose(parameters[:, 0], protein.monomer.charges.numpy(), rtol=0., atol=1e-12):
            raise ValueError("Physical prior reconstructed different atom charges")
        self.provenance = dict(forcefield=FORCEFIELD_FILE, implicit_solvent=PROTOCOL["implicit_solvent"],
            forcefield_xml_sha256={**hashes, **_forcefield_sources(implicit_path)},
            system_xml_sha256=hashlib.sha256(openmm.XmlSerializer.serialize(system).encode()).hexdigest(),
            openmm_version=openmm.__version__, platform="Reference", coordinate_unit="angstrom",
            energy_unit="kJ/mol", force_conversion="gradient_A = -0.1 * force_kJ_per_mol_nm",
            solute_dielectric=1., solvent_dielectric=78.5, kappa_per_nm=0., sasa="ACE",
            nonbonded_method="NoCutoff", original_charges_validated=True)
        _split_force_groups(system, openmm, unit)
        self.unit, self.n_atoms = unit, len(parameters)
        self.integrator = openmm.VerletIntegrator(.001 * unit.picoseconds)
        self.context = openmm.Context(system, self.integrator, openmm.Platform.getPlatformByName("Reference"))
        self.evaluations = 0

    def evaluate(self, xyz, *, components=False):
        if self.context is None:
            raise RuntimeError("Physical prior OpenMM context has been closed")
        _tensor(xyz, (self.n_atoms, 3), "coordinates")
        self.context.setPositions(xyz.detach().numpy() * self.unit.angstrom)
        state = self.context.getState(getEnergy=True, getForces=True)
        energy = float(state.getPotentialEnergy().value_in_unit(self.unit.kilojoule_per_mole))
        gradient = -.1 * np.asarray(state.getForces(asNumpy=True).value_in_unit(
            self.unit.kilojoule_per_mole / self.unit.nanometer), dtype=np.float64)
        self.evaluations += 1
        if not math.isfinite(energy) or not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite ff14SB/OBC2 energy or force")
        terms = None
        if components:
            terms = {key: float(self.context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
                               .value_in_unit(self.unit.kilojoule_per_mole))
                     for i, key in enumerate(("bonded", "vdw", "coulomb", "gbsa"))}
        return energy, torch.from_numpy(gradient.copy()), terms

    def close(self):
        self.context = None
        self.integrator = None


class _OpenMMEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, force_context):
        energy, gradient, _ = force_context.evaluate(xyz)
        ctx.save_for_backward(gradient)
        return xyz.new_tensor(energy)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        gradient, = ctx.saved_tensors
        return upstream * gradient, None


@dataclass(frozen=True)
class PreparedTorsionMap:
    base: object
    theta0: torch.Tensor
    reference: torch.Tensor

    @property
    def n_torsions(self):
        return self.base.n_torsions

    @property
    def ca_indices(self):
        return self.base.ca_indices

    def coordinates(self, delta_angles):
        return self.base.coordinates(self.theta0 + delta_angles)


class PhysicalPrior:
    """ff14SB/OBC2 plus 0.5*k*sum_all_atoms |x-x_input|².

    ``theta`` always means total angles relative to the original supplied
    kinematics. ``theta0`` is set only by successful ``prepare``.  The anchor
    defaults to 10 kJ/mol/A² per atom: a declared input-structure prior, fixed
    without native or docking outcomes.  It is not a physical solvent term.
    """

    def __init__(self, protein: ChargedProtein, *, kinematics=None, anchor_k=10.):
        self.anchor_k = _positive(anchor_k, "anchor_k")
        _tensor(protein.monomer.xyz, protein.monomer.xyz.shape, "protein coordinates")
        self.protein = protein
        self.kinematics = HydrogenatedKinematics.from_prepared(protein) if kinematics is None else kinematics
        if not torch.equal(self.kinematics.reference, protein.monomer.xyz):
            raise ValueError("Physical prior kinematics must start from prepared all-atom input")
        self.anchor_reference = protein.monomer.xyz.detach().clone()
        self._context = _ForceContext(protein)
        self.theta0 = None
        self.reference = None
        self.baseline_energy = None
        self.prepared_kinematics = None
        self.diagnostics = {"status": "unprepared", "anchor_k_kJ_per_mol_A2": self.anchor_k,
            "anchor_atoms": "all prepared atoms including hydrogen", "physics": self._context.provenance,
            "prior_definition": "ff14SB/OBC2/ACE plus input Cartesian anchor; no CA spring", "native_access": False,
            "input_fingerprint": _input_fingerprint(protein)}

    def _validate_theta(self, theta):
        _tensor(theta, (self.kinematics.n_torsions,), "theta")

    def energy_xyz(self, xyz):
        _tensor(xyz, self.anchor_reference.shape, "coordinates")
        return _OpenMMEnergy.apply(xyz, self._context) + .5 * self.anchor_k * (xyz - self.anchor_reference).square().sum()

    def energy(self, theta):
        self._validate_theta(theta)
        return self.energy_xyz(self.kinematics.coordinates(theta))

    def energy_and_gradient(self, theta):
        self._validate_theta(theta)
        with torch.enable_grad():
            variable = theta.detach().clone().requires_grad_(True)
            energy = self.energy(variable)
            if variable.numel():
                gradient, = torch.autograd.grad(energy, variable)
            else:
                gradient = variable.detach().clone()
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Nonfinite full torsional gradient")
        return float(energy.detach()), gradient.detach()

    def hessian_vector(self, theta, vector, *, step=1e-4):
        self._validate_theta(theta)
        _tensor(vector, theta.shape, "HVP vector")
        step = _positive(step, "HVP step")
        norm = float(vector.norm())
        if norm == 0:
            return torch.zeros_like(vector)
        offset = vector * (step / norm)
        plus = self.energy_and_gradient(theta + offset)[1]
        minus = self.energy_and_gradient(theta - offset)[1]
        return (plus - minus) * (norm / (2 * step))

    def components(self, theta):
        self._validate_theta(theta)
        xyz = self.kinematics.coordinates(theta).detach()
        energy, _, terms = self._context.evaluate(xyz, components=True)
        terms["anchor"] = float(.5 * self.anchor_k * (xyz-self.anchor_reference).square().sum())
        terms["total"] = energy + terms["anchor"]
        return terms

    def _variable_transform(self, method):
        """Fixed input anchor metric; this variable change leaves U unchanged.

        The temporary torsion-coordinate Jacobian is formed once. The dense
        torsion metric is not a physical Hessian and is discarded after the
        change of variables. Its identity ridge belongs only to the optimizer.
        """
        kin, xyz = self.kinematics, self.anchor_reference
        tangents = []
        if hasattr(kin, "joints"):
            for joint in kin.joints:
                axis = xyz[joint.axis_end] - xyz[joint.axis_start]
                axis = axis / axis.norm()
                relative = xyz[list(joint.moving_indices)] - xyz[joint.axis_end]
                tangent = torch.zeros_like(xyz)
                tangent[list(joint.moving_indices)] = torch.linalg.cross(axis.expand_as(relative), relative)
                tangents.append(tangent.reshape(-1))
        else:
            zero = xyz.new_zeros(kin.n_torsions)
            for i in range(kin.n_torsions):
                direction = torch.zeros_like(zero)
                direction[i] = 1.
                tangent = torch.func.jvp(kin.coordinates, (zero,), (direction,))[1]
                tangents.append(tangent.reshape(-1))
        if not tangents:
            return xyz.new_empty((0, 0)), 1., dict(method=method, dimension=0)
        jacobian = torch.stack(tangents)
        if method == 'diagonal':
            values = self.anchor_k * jacobian.square().sum(-1) + 1.
            transform = torch.diag(values.rsqrt())
            inverse_row_bound = float(values.sqrt().max())
        else:
            metric = self.anchor_k * (jacobian @ jacobian.T) + torch.eye(kin.n_torsions, dtype=xyz.dtype)
            values, vectors = torch.linalg.eigh(metric)
            if not bool((values > 0).all()):
                raise FloatingPointError('Input anchor metric is not positive definite')
            transform = (vectors * values.rsqrt()) @ vectors.T
            inverse_transform = (vectors * values.sqrt()) @ vectors.T
            inverse_row_bound = float(inverse_transform.abs().sum(-1).max())
        diagnostics = dict(method=method, dimension=kin.n_torsions, numerical_metric_ridge=1.,
            formula='G = anchor_k J_input.T J_input + I; theta = G^(-1/2) z' if method=='anchor_metric' else 'diagonal of G only',
            eigenvalues_or_diagonal=values.tolist(),
            transform_sha256=hashlib.sha256(transform.numpy().tobytes()).hexdigest(),
            temporary_jacobian_bytes=jacobian.numel()*jacobian.element_size(),
            materialized_physical_hessian=False,
            interpretation='optimizer variable change only; energy, anchor and stationarity gate unchanged')
        return transform, inverse_row_bound, diagnostics

    def prepare(self, *, max_iterations=1000, max_evaluations=5000,
                stationarity_tolerance=1e-3, max_heavy_displacement=1., progress=None,
                preconditioner='anchor_metric', polish=True):
        """Apo-only L-BFGS reference, with explicit failure instead of clipping.

        Optimization is unconstrained in torsions and local from zero. The
        final heavy displacement is an acceptance gate, not an unreported
        projection that would invalidate stationarity. All evaluations are
        recorded. A failed call cannot leave a usable prepared reference.
        """
        from scipy.optimize import minimize
        if self.theta0 is not None:
            raise ValueError("Physical prior is already prepared; create a fresh prior for a new attempt")
        if progress is not None and not callable(progress):
            raise ValueError("progress must be a callable or None")
        if preconditioner not in {'diagonal', 'anchor_metric'}:
            raise ValueError('preconditioner must be diagonal or anchor_metric')
        if type(polish) is not bool:
            raise ValueError('polish must be a boolean')
        for value, label in ((max_iterations, "max_iterations"), (max_evaluations, "max_evaluations")):
            if type(value) is not int or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        tolerance = _positive(stationarity_tolerance, "stationarity_tolerance")
        bound = _positive(max_heavy_displacement, "max_heavy_displacement")
        started = time.monotonic()
        history = []
        self.diagnostics.update(status="preparing", maximum_iterations=max_iterations,
            maximum_evaluations=max_evaluations, stationarity_tolerance_kJ_per_mol_rad=tolerance,
            maximum_heavy_displacement_gate_angstrom=bound, evaluations=history,
            optimizer="scipy L-BFGS-B with no bounds, fixed input-Jacobian variable transform")
        zero = self.anchor_reference.new_zeros(self.kinematics.n_torsions)
        initial_evaluations = self._context.evaluations
        try:
            transform, inverse_row_bound, transform_diagnostics = self._variable_transform(preconditioner)
            self.diagnostics['preconditioner'] = transform_diagnostics

            def budget_check():
                if self._context.evaluations-initial_evaluations >= max_evaluations:
                    raise RuntimeError('Physical apo preparation reached the energy-evaluation budget')

            def evaluate(theta, phase):
                budget_check()
                energy, gradient = self.energy_and_gradient(theta)
                history.append(dict(phase=phase, energy=energy,
                    maximum_abs_torque=float(gradient.abs().max()) if len(gradient) else 0.,
                    maximum_abs_angle=float(theta.abs().max()) if len(theta) else 0.))
                if progress is not None and (len(history) == 1 or len(history) % 25 == 0):
                    progress(self.diagnostics)
                return energy, gradient

            budget_check()
            self.diagnostics["initial_components"] = self.components(zero)

            def objective(z):
                theta = transform @ torch.from_numpy(np.asarray(z, dtype=np.float64).copy())
                energy, gradient = evaluate(theta, 'lbfgs')
                return energy, (transform.T @ gradient).numpy()

            if len(zero):
                result = minimize(objective, np.zeros(len(zero)), method="L-BFGS-B", jac=True,
                    options={"maxiter": max_iterations, "maxfun": max_evaluations,
                             "gtol": tolerance / inverse_row_bound * .1, "ftol": 1e-15,
                             "maxls": 40, "maxcor": 30})
                theta = transform @ torch.from_numpy(np.asarray(result.x, dtype=np.float64).copy())
                optimizer = dict(success=bool(result.success), status=int(result.status), message=str(result.message),
                                 iterations=int(result.nit), function_evaluations=int(result.nfev))
            else:
                theta = zero
                optimizer = dict(success=True, status=0, message="no torsional degrees of freedom", iterations=0, function_evaluations=0)
            self.diagnostics['lbfgs_result'] = optimizer.copy()
            final_energy, gradient = evaluate(theta, 'post_lbfgs')
            self.diagnostics.update(post_lbfgs_energy=final_energy,
                post_lbfgs_maximum_abs_torque=float(gradient.abs().max()) if len(gradient) else 0.,
                post_lbfgs_angles=theta.tolist())
            if polish and len(theta):
                theta, final_energy, gradient = self._polish_torque(
                    theta, transform, evaluate, tolerance)
                optimizer = dict(success=float(gradient.abs().max()) <= tolerance,
                    method='L-BFGS followed by matrix-free Newton-CG torque polishing',
                    lbfgs=optimizer, polish=self.diagnostics['polishing'])
            xyz = self.kinematics.coordinates(theta).detach()
            delta = xyz - self.anchor_reference
            maximum = float(delta[self.protein.heavy_indices].norm(dim=-1).max())
            torque = float(gradient.abs().max()) if len(gradient) else 0.
            budget_check()
            self.diagnostics.update(optimizer=optimizer, final_energy=final_energy,
                final_components=self.components(theta), maximum_abs_torque=torque,
                final_maximum_heavy_displacement_angstrom=maximum,
                final_maximum_atom_displacement_angstrom=float(delta.norm(dim=-1).max()),
                final_ca_rmsd_angstrom=float(delta[self.kinematics.ca_indices].square().sum(-1).mean().sqrt()),
                final_angles=theta.tolist())
            if not optimizer["success"]:
                raise RuntimeError(f"Apo reference optimizer failed: {optimizer['message']}")
            if torque > tolerance:
                raise RuntimeError(f"Apo reference is not stationary: max torque {torque:.6g} > {tolerance:.6g}")
            if maximum > bound:
                raise RuntimeError(f"Apo preparation moved a heavy atom {maximum:.6g} A > {bound:.6g} A")
            self.theta0, self.reference, self.baseline_energy = theta.clone(), xyz.clone(), final_energy
            self.prepared_kinematics = PreparedTorsionMap(self.kinematics, self.theta0, self.reference)
            self.diagnostics["status"] = "prepared"
        except Exception as error:
            self.diagnostics.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise PhysicalPreparationError(str(error), self.diagnostics) from error
        finally:
            self.diagnostics["wall_seconds"] = time.monotonic() - started
            self.diagnostics["openmm_evaluations"] = self._context.evaluations
        return self

    def _polish_torque(self, theta, transform, evaluate, tolerance):
        """Solve the true torque root when energy line searches hit roundoff.

        Newton systems use only full-torque finite-difference HVPs. The fixed
        input metric is a variable transform, never a physical Hessian shift.
        Every force evaluation, including failed trials, shares the preparation
        budget. Nonpositive curvature causes an explicit failed preparation.
        """
        record = copy.deepcopy(POLISH_PROTOCOL)
        record['iterations'] = []
        self.diagnostics['polishing'] = record
        energy, gradient = evaluate(theta, 'polish_start')
        for iteration in range(record['maximum_iterations']):
            torque = float(gradient.abs().max())
            if torque <= tolerance:
                record.update(success=True, final_maximum_abs_torque=torque)
                return theta, energy, gradient
            step_record = dict(iteration=iteration, initial_energy=energy,
                initial_maximum_abs_torque=torque, cg_curvatures=[], trials=[])
            record['iterations'].append(step_record)
            rhs = -transform.T @ gradient
            solution = torch.zeros_like(rhs)
            residual = rhs.clone()
            search = residual.clone()
            squared = float(residual @ residual)
            initial_norm = math.sqrt(squared)
            for cg_iteration in range(record['maximum_cg_iterations']):
                direction = transform @ search
                norm = float(direction.norm())
                if norm == 0:
                    raise RuntimeError('Newton-CG encountered a zero search direction before convergence')
                offset = direction * (record['hvp_step_rad']/norm)
                plus = evaluate(theta+offset, 'polish_hvp_plus')[1]
                minus = evaluate(theta-offset, 'polish_hvp_minus')[1]
                product = transform.T @ ((plus-minus) * (norm/(2*record['hvp_step_rad'])))
                curvature = float(search @ product)
                step_record['cg_curvatures'].append(curvature)
                if not math.isfinite(curvature) or curvature <= 0:
                    raise RuntimeError(f'Newton-CG detected nonpositive curvature: {curvature}')
                alpha = squared/curvature
                solution += alpha*search
                residual -= alpha*product
                next_squared = float(residual @ residual)
                relative = math.sqrt(next_squared)/initial_norm
                step_record.update(cg_iterations=cg_iteration+1, cg_relative_residual=relative)
                if relative <= record['cg_relative_tolerance']:
                    break
                search = residual+(next_squared/squared)*search
                squared = next_squared
            else:
                raise RuntimeError('Newton-CG reached its inner iteration limit')
            increment = transform @ solution
            for backtrack in range(record['maximum_backtracks']):
                fraction = .5**backtrack
                proposed = theta+fraction*increment
                proposed_energy, proposed_gradient = evaluate(proposed, 'polish_trial')
                proposed_torque = float(proposed_gradient.abs().max())
                accepted = (proposed_energy-energy <= record['energy_roundoff_kJ_per_mol']
                    and (proposed_torque <= tolerance or proposed_torque < record['required_torque_ratio']*torque))
                step_record['trials'].append(dict(fraction=fraction, energy=proposed_energy,
                    energy_change=proposed_energy-energy, maximum_abs_torque=proposed_torque, accepted=accepted))
                if accepted:
                    theta, energy, gradient = proposed, proposed_energy, proposed_gradient
                    break
            else:
                raise RuntimeError('Newton-CG could not reduce the true torque within the energy roundoff allowance')
        final_torque = float(gradient.abs().max())
        record.update(success=final_torque <= tolerance, final_maximum_abs_torque=final_torque)
        if not record['success']:
            raise RuntimeError('Newton-CG reached its outer iteration limit')
        return theta, energy, gradient

    def close(self):
        self._context.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def prepare_physical_prior(protein, *, kinematics=None, anchor_k=10., **preparation_options):
    prior = PhysicalPrior(protein, kinematics=kinematics, anchor_k=anchor_k)
    try:
        return prior.prepare(**preparation_options)
    except Exception:
        prior.close()
        raise


def load_prepared_physical_prior(protein, report_path, *, anchor_k=10.,
                                stationarity_tolerance=1e-3, max_heavy_displacement=1.):
    """Restore a successful verification artifact and independently recheck it.

    The artifact is valid only for identical code, input atoms/coordinates,
    chemistry and declared physical gates. Its coordinates, energy and full
    torque are recomputed before a usable reference is exposed. Saved modes
    are not trusted or restored: callers rebuild their desired directions.
    """
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    if report.get('status') != 'passed' or report.get('preparation', {}).get('status') != 'prepared':
        raise ValueError('Only a passed physical-prior verification artifact can be restored')
    registered = report['registration']
    configuration = registered['configuration']
    for key, value in [('anchor_k', anchor_k), ('stationarity_tolerance', stationarity_tolerance),
                       ('max_heavy_displacement', max_heavy_displacement)]:
        _positive(value, key)
        if configuration.get(key) != value:
            raise ValueError(f'Cached physical prior has a different {key}')
    sources = registered['source_sha256']
    repository = Path(__file__).resolve().parents[3]
    required = {'src/zdock/inducedfit/' + name for name in
                ['physical_prior.py', 'atomistic_score.py', 'hydrogen_kinematics.py',
                 'kinematics.py', 'chemistry.py', 'pdb_input.py', 'types.py']}
    if not required.issubset(sources):
        raise ValueError('Cached physical prior lacks required source hashes')
    for relative, expected in sources.items():
        path = (repository/relative).resolve()
        if not path.is_relative_to(repository) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Cached physical prior source mismatch: {relative}')
    archive_path = report_path.parent/'arrays.npz'
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != report.get('arrays_sha256'):
        raise ValueError('Cached physical prior array hash mismatch')
    with np.load(archive_path, allow_pickle=False) as archive:
        theta = torch.from_numpy(archive['theta0'].copy())
        saved_reference = torch.from_numpy(archive['reference'].copy())
    saved = report['preparation']
    if saved.get('input_fingerprint') != _input_fingerprint(protein):
        raise ValueError('Cached physical prior input fingerprint mismatch')
    prior = PhysicalPrior(protein, anchor_k=anchor_k)
    try:
        if saved.get('physics') != prior.diagnostics['physics']:
            raise ValueError('Cached physical prior forcefield/system provenance mismatch')
        prior._validate_theta(theta)
        energy, gradient = prior.energy_and_gradient(theta)
        reference = prior.kinematics.coordinates(theta).detach()
        _tensor(saved_reference, reference.shape, 'cached reference')
        if not torch.allclose(reference, saved_reference, atol=1e-10, rtol=0.):
            raise ValueError('Cached physical prior reference differs from reconstructed torsions')
        torque = float(gradient.abs().max()) if gradient.numel() else 0.
        maximum = float((reference-prior.anchor_reference)[protein.heavy_indices].norm(dim=-1).max())
        if not math.isclose(energy, saved['final_energy'], abs_tol=1e-8, rel_tol=1e-12):
            raise ValueError('Cached physical prior energy differs from recomputed energy')
        if torque > stationarity_tolerance or maximum > max_heavy_displacement:
            raise ValueError('Cached physical prior fails recomputed stationarity or displacement gate')
        prior.theta0, prior.reference, prior.baseline_energy = theta, reference, energy
        prior.prepared_kinematics = PreparedTorsionMap(prior.kinematics, theta, reference)
        prior.diagnostics = copy.deepcopy(saved)
        prior.diagnostics['restore_validation'] = dict(report_path=str(report_path),
            report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
            source_hashes_checked=True, array_hash_checked=True, input_fingerprint_checked=True,
            forcefield_checked=True, maximum_abs_torque=torque,
            maximum_heavy_displacement_angstrom=maximum, energy=energy,
            evaluations=prior._context.evaluations)
        return prior
    except Exception:
        prior.close()
        raise


@dataclass(frozen=True)
class PhysicalModeModel:
    prior: PhysicalPrior
    angle_basis: torch.Tensor
    basis: torch.Tensor
    stiffness: torch.Tensor
    diagnostics: dict

    @property
    def n_modes(self):
        return self.angle_basis.shape[1]

    @property
    def reference(self):
        return self.prior.reference

    @property
    def ca_indices(self):
        return self.prior.kinematics.ca_indices

    @property
    def kinematics(self):
        return self.prior.prepared_kinematics

    def coordinates(self, amplitudes):
        if amplitudes.shape != (self.n_modes,) or amplitudes.dtype != self.angle_basis.dtype or amplitudes.device != self.angle_basis.device:
            raise ValueError("Physical mode amplitudes must match mode count, dtype and device")
        return self.kinematics.coordinates(self.angle_basis @ amplitudes)

    def deformation_energy(self, amplitudes):
        return self.prior.energy_xyz(self.coordinates(amplitudes)) - self.prior.baseline_energy


def build_physical_modes(prior, angle_basis, *, fd_step=1e-4, fd_relative_tolerance=5e-3,
                         metric_rtol=1e-10):
    """Normalize supplied torsion directions at the prepared apo reference.

    Compare stiffness from step h and h/2. Reject unresolved asymmetry, finite
    difference disagreement, singular CA metric or nonpositive K. There is no
    eigenvalue clipping, rank truncation, CA-spring addition or gradient shift.
    """
    if prior.theta0 is None or prior.diagnostics.get("status") != "prepared":
        raise ValueError("Physical modes require a successfully prepared stationary prior")
    if not isinstance(angle_basis, torch.Tensor) or angle_basis.ndim != 2:
        raise ValueError("angle_basis must be a matrix (N_torsions, M)")
    _tensor(angle_basis, (prior.kinematics.n_torsions, angle_basis.shape[1]), "angle_basis")
    if angle_basis.shape[1] < 1:
        raise ValueError("At least one physical mode is required")
    fd_step = _positive(fd_step, "fd_step")
    fd_relative_tolerance = _positive(fd_relative_tolerance, "fd_relative_tolerance")
    metric_rtol = _positive(metric_rtol, "metric_rtol")
    m = angle_basis.shape[1]
    directions = angle_basis.detach().clone()

    def coordinate_basis(values):
        return torch.stack([torch.func.jvp(prior.kinematics.coordinates, (prior.theta0,), (column,))[1].detach()
                            for column in values.T])

    raw_basis = coordinate_basis(directions)
    ca = raw_basis[:, prior.kinematics.ca_indices].reshape(m, -1)
    metric = ca @ ca.T / len(prior.kinematics.ca_indices)
    values, vectors = torch.linalg.eigh(.5*(metric+metric.T))
    if not bool((values > metric_rtol * values.abs().max()).all()):
        raise ValueError("Physical mode CA metric is singular or rank deficient")
    inverse_root = (vectors * values.rsqrt()) @ vectors.T
    directions = directions @ inverse_root
    basis = coordinate_basis(directions)

    def stiffness_at(step):
        return torch.stack([directions.T @ prior.hessian_vector(prior.theta0, column, step=step)
                            for column in directions.T], dim=1)

    coarse, fine = stiffness_at(fd_step), stiffness_at(.5*fd_step)
    denominator = max(float(fine.norm()), 1e-12)
    disagreement = float((fine-coarse).norm()) / denominator
    asymmetry = float((fine-fine.T).norm()) / denominator
    if not bool(torch.isfinite(fine).all()) or disagreement > fd_relative_tolerance or asymmetry > fd_relative_tolerance:
        raise ValueError(f"Physical stiffness finite differences unresolved: relative change={disagreement}, asymmetry={asymmetry}")
    stiffness = .5*(fine+fine.T)
    eigenvalues = torch.linalg.eigvalsh(stiffness)
    if not bool((eigenvalues > 0).all()):
        raise ValueError(f"Physical stiffness is not positive definite: minimum={float(eigenvalues.min())}")
    normalized = basis[:, prior.kinematics.ca_indices].reshape(m, -1)
    metric_error = float((normalized @ normalized.T / len(prior.kinematics.ca_indices)-torch.eye(m, dtype=torch.float64)).abs().max())
    if metric_error > 1e-7:
        raise ValueError("Physical mode CA normalization failed")
    _, torque = prior.energy_and_gradient(prior.theta0)
    diagnostics = dict(method="stationary_ff14sb_obc2_cartesian_anchor", energy_unit="kJ/mol",
        actual_modes=m, CA_metric_max_abs_error=metric_error, stiffness_eigenvalues=eigenvalues.tolist(),
        stiffness_fd_step_rad=fd_step, stiffness_fd_relative_change=disagreement,
        stiffness_fd_relative_asymmetry=asymmetry, stiffness_fd_relative_tolerance=fd_relative_tolerance,
        baseline_projected_gradient=(directions.T @ torque).tolist(),
        anchor_k_kJ_per_mol_A2=prior.anchor_k, baseline_energy=prior.baseline_energy,
        reference_preparation=prior.diagnostics,
        energy_policy="same full nonlinear ff14SB/OBC2/ACE plus Cartesian anchor minus common apo baseline; no clipping")
    return PhysicalModeModel(prior, directions, basis, stiffness, diagnostics)
