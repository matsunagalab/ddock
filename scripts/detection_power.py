"""Exact paired-endpoint power audit for the fixed-pool docking experiments.

The exact McNemar test discards concordant pairs.  Write

    q     = P(new succeeds, base fails) + P(new fails, base succeeds)
    delta = P(new succeeds) - P(base succeeds)

Then a pair is discordant with probability q and, conditional on discordance,
the new method wins with probability (q + delta) / (2 q).  For N independent
interface clusters this script averages the exact two-sided binomial rejection
rule over M ~ Binomial(N, q).  It does not use a normal approximation.

The historical win/loss counts below are transcribed from
EXPERIMENT_REPORT.md sections 5.14.45, 5.14.47, and 5.14.50--54.  They are
descriptive observations from different cohorts and sometimes different
endpoints; they are not exchangeable replicates from which q can be estimated
with high precision.  Product and log-contact arms did not save per-complex
hit vectors, so their 76 -> 76 totals cannot identify q and are explicitly
listed as unavailable rather than silently treated as 0/0.

Usage:

    uv run python scripts/detection_power.py
    uv run python scripts/detection_power.py --json /tmp/detection_power.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binom


@dataclass(frozen=True)
class Comparison:
    name: str
    cohort: str
    endpoint: str
    n: int
    wins: int | None
    losses: int | None
    source: str
    selected: bool = True
    observed_exact_p_override: float | None = None

    @property
    def discordant(self) -> int | None:
        if self.wins is None or self.losses is None:
            return None
        return self.wins + self.losses

    @property
    def q(self) -> float | None:
        m = self.discordant
        return None if m is None else m / self.n


# One row per reported comparison.  Sensitivity-path rows are retained because
# they bound how much the endpoint can move when the added coefficients are not
# shrunk, but they are not independent experiments and are not used as if they
# were iid samples of a q distribution.
COMPARISONS = (
    Comparison("C2 projection", "homodimer 71", "acceptable", 71, 0, 9,
               "5.14.45"),
    Comparison("C2 filter 10deg/5A", "homodimer 71", "acceptable", 71, 4, 8,
               "5.14.45"),
    Comparison("trans_per_rotation 1 -> 10", "pilot 30", "acceptable", 30,
               0, 0, "5.14.47"),
    Comparison("refine K=50 -> 1000", "pilot 45", "acceptable", 45, 2, 2,
               "5.14.50"),
    Comparison("correlation products", "fixed validation", "acceptable", 152,
               None, None, "5.14.51", observed_exact_p_override=1.0),
    Comparison("additive log-contact", "fixed validation", "acceptable", 152,
               None, None, "5.14.51", observed_exact_p_override=1.0),
    Comparison("3 shells, CV-selected", "fixed validation", "acceptable", 152,
               0, 0, "5.14.52"),
    Comparison("3 shells, lambda=1e-4", "fixed validation", "acceptable", 152,
               4, 4, "5.14.52", selected=False),
    Comparison("outer bands, CV-selected", "fixed validation", "acceptable",
               152, 0, 0, "5.14.53"),
    Comparison("outer bands, lambda=1e-4", "fixed validation", "acceptable",
               152, 2, 2, "5.14.53", selected=False),
    Comparison("element bands, sym", "OOM-survivor validation", "acceptable",
               140, 2, 3, "5.14.54"),
)


def min_attainable_p(discordant: int) -> float:
    """Smallest two-sided exact-binomial p value for M discordant pairs."""
    if discordant <= 0:
        return 1.0
    return min(1.0, 2.0 ** (1 - discordant))


def exact_mcnemar_p(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p value (exact binomial sign test)."""
    m = wins + losses
    if m == 0:
        return 1.0
    return min(1.0, float(2.0 * binom.cdf(min(wins, losses), m, 0.5)))


def lower_critical_value(m: int, alpha: float = 0.05) -> int:
    """Largest k for which the exact two-sided null rejects at W <= k."""
    if m <= 0:
        return -1
    # ppf returns the first k whose CDF reaches alpha/2.  Exact rejection needs
    # the CDF to be no larger than alpha/2, hence the correction loops.
    k = int(binom.ppf(alpha / 2.0, m, 0.5))
    while k >= 0 and 2.0 * binom.cdf(k, m, 0.5) > alpha + 1e-15:
        k -= 1
    while k + 1 <= m // 2 and 2.0 * binom.cdf(k + 1, m, 0.5) <= alpha + 1e-15:
        k += 1
    return k


def conditional_rejection_probability(
    discordant: int, win_probability: float, alpha: float = 0.05
) -> float:
    """Exact rejection probability conditional on M discordant pairs."""
    k = lower_critical_value(discordant, alpha)
    if k < 0:
        return 0.0
    lower = binom.cdf(k, discordant, win_probability)
    upper = binom.sf(discordant - k - 1, discordant, win_probability)
    return float(lower + upper)


def exact_unconditional_power(
    n: int, discordance_rate: float, effect: float, alpha: float = 0.05
) -> float:
    """Power of the two-sided exact McNemar test, averaging over M.

    ``effect`` is the absolute marginal success-rate improvement, not an odds
    ratio.  A proposed pair (q, effect) is impossible when |effect| > q.
    """
    q = float(discordance_rate)
    delta = float(effect)
    if not (0.0 <= q <= 1.0):
        raise ValueError("discordance_rate must lie in [0, 1]")
    if abs(delta) > q + 1e-15:
        raise ValueError(
            f"effect {delta:.6g} is impossible with discordance rate {q:.6g}"
        )
    if q == 0.0:
        return 0.0

    pi = (q + delta) / (2.0 * q)
    m = np.arange(n + 1)
    weights = binom.pmf(m, n, q)
    reject = np.fromiter(
        (conditional_rejection_probability(int(mm), pi, alpha) for mm in m),
        dtype=np.float64,
        count=n + 1,
    )
    return float(weights @ reject)


def gain_gate_probability(
    n: int,
    discordance_rate: float,
    true_effect: float,
    gate_effect: float,
) -> float:
    """Probability that the observed paired gain reaches a point-estimate gate."""
    q = float(discordance_rate)
    if abs(true_effect) > q + 1e-15:
        raise ValueError("true effect cannot exceed discordance rate")
    if q == 0.0:
        return float(gate_effect <= 0.0)
    pi = (q + true_effect) / (2.0 * q)
    total = 0.0
    for m in range(n + 1):
        # (wins - losses)/N = (2*wins - m)/N >= gate_effect.
        need = math.ceil((m + n * gate_effect) / 2.0 - 1e-12)
        cond = 1.0 if need <= 0 else float(binom.sf(need - 1, m, pi))
        total += float(binom.pmf(m, n, q)) * cond
    return total


def required_n(
    discordance_rate: float,
    effect: float,
    target_power: float = 0.80,
    alpha: float = 0.05,
    max_n: int = 100_000,
) -> int:
    """Smallest planning N reaching target power (exact unconditional rule)."""
    q = float(discordance_rate)
    if abs(effect) > q + 1e-15:
        raise ValueError("effect cannot exceed discordance rate")
    hi = 8
    while exact_unconditional_power(hi, q, effect, alpha) < target_power:
        hi *= 2
        if hi > max_n:
            raise RuntimeError(f"required N exceeds {max_n}")

    # Do not assume monotonicity of a non-randomized exact test: discreteness can
    # produce small sawteeth.  Cache the conditional rejection probabilities and
    # scan every N up to a known crossing, so the returned N is genuinely first.
    pi = (q + effect) / (2.0 * q)
    reject = np.fromiter(
        (conditional_rejection_probability(m, pi, alpha) for m in range(hi + 1)),
        dtype=np.float64,
        count=hi + 1,
    )
    for n in range(1, hi + 1):
        m = np.arange(n + 1)
        power = float(binom.pmf(m, n, q) @ reject[: n + 1])
        if power >= target_power:
            return n
    raise AssertionError("known power crossing was not found")


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval without continuity correction."""
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - alpha / 2.0))
    p = successes / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return centre - half, centre + half


def cluster_counts(
    split_path: Path, index_path: Path
) -> dict[str, dict[str, int | float]]:
    """Verify that the source split has one complex per interface cluster."""
    import pandas as pd

    split = json.loads(split_path.read_text())
    ix = pd.read_parquet(
        index_path, columns=["id", "cluster_id", "pinder_s", "pinder_xl"]
    ).set_index("id")
    out: dict[str, dict[str, int | float]] = {}
    for label, key, analysed in (("fit", "fit_ids", 637), ("validation", "val_ids", 152)):
        ids = split[key]
        counts = ix.loc[ids, "cluster_id"].astype(str).value_counts().to_numpy()
        neff = len(ids) ** 2 / float(np.sum(counts * counts))
        out[label] = {
            "source_complexes": len(ids),
            "source_clusters": int(len(counts)),
            "analysed_complexes": analysed,
            # The analysed set is a subset of a one-per-cluster source set.
            "analysed_clusters": analysed,
            "kish_neff_perfect_within_cluster_correlation": float(analysed),
            "source_kish_neff": neff,
        }
    current = set(ix.loc[split["fit_ids"] + split["val_ids"], "cluster_id"].astype(str))
    xl = ix[ix["pinder_xl"] & ~ix["pinder_s"]]
    xl_clusters = set(xl["cluster_id"].astype(str))
    out["pinder_xl_not_s"] = {
        "source_complexes": int(len(xl)),
        "source_clusters": int(len(xl_clusters)),
        "analysed_complexes": int(len(xl)),
        "analysed_clusters": int(len(xl_clusters)),
        "kish_neff_perfect_within_cluster_correlation": float(len(xl_clusters)),
        "source_kish_neff": float(len(xl_clusters)),
        "cluster_overlap_with_current_split": int(len(current & xl_clusters)),
    }
    return out


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def build_results(args: argparse.Namespace) -> dict:
    scenarios = {
        # Directly observed on fixed validation; this is the only selected
        # nonzero q among the stored fine-feature comparisons.
        "element_sym_5_of_140": 5 / 140,
        # Largest observed endpoint movement on the shell sensitivity path.
        "small_intervention_upper_8_of_152": 8 / 152,
        # Refinement pilot: different cohort, useful as a higher-q sensitivity.
        "refine_4_of_45": 4 / 45,
        # C2 default filter: different cohort and mechanism, highest observed q.
        "c2_filter_12_of_71": 12 / 71,
    }
    effects = [x / 100.0 for x in args.effects_pp]
    expected_transport_eligible = int(round(args.transport_n * args.eligibility_rate))
    powers: dict[str, dict[str, dict[str, float | str]]] = {}
    for name, q in scenarios.items():
        powers[name] = {}
        for effect in effects:
            key = f"{effect * 100:g}pp"
            if effect > q + 1e-15:
                powers[name][key] = {"status": "infeasible: effect > q"}
            else:
                powers[name][key] = {
                    f"N={args.validation_n}": exact_unconditional_power(
                        args.validation_n, q, effect, args.alpha
                    ),
                    f"N={args.fit_n}": exact_unconditional_power(
                        args.fit_n, q, effect, args.alpha
                    ),
                    f"N={args.transport_n}": exact_unconditional_power(
                        args.transport_n, q, effect, args.alpha
                    ),
                    f"N={expected_transport_eligible}": exact_unconditional_power(
                        expected_transport_eligible, q, effect, args.alpha
                    ),
                }

    needed = {}
    for name, q in scenarios.items():
        n = required_n(q, 0.02, args.target_power, args.alpha)
        needed[name] = {
            "q": q,
            "required_clusters": n,
            "add_to_validation_152": max(0, n - args.validation_n),
            "can_reach_by_moving_at_most_637_fit": n <= args.validation_n + args.fit_n,
            "fit_remaining_if_moved": max(0, args.fit_n - max(0, n - args.validation_n)),
            "raw_complexes_at_current_eligibility_rate": math.ceil(
                n / args.eligibility_rate
            ),
            "pinder_xl_1705_raw_sufficient_at_current_eligibility_rate": (
                args.transport_n * args.eligibility_rate >= n
            ),
        }

    lo, hi = wilson_interval(76, 152, args.alpha)
    comparisons = []
    for comp in COMPARISONS:
        row = asdict(comp)
        row.update(
            discordant=comp.discordant,
            discordance_rate=comp.q,
            min_attainable_p=(
                None if comp.discordant is None else min_attainable_p(comp.discordant)
            ),
            observed_exact_p=(
                comp.observed_exact_p_override
                if comp.observed_exact_p_override is not None
                else (
                    None
                    if comp.wins is None or comp.losses is None
                    else exact_mcnemar_p(comp.wins, comp.losses)
                )
            ),
        )
        comparisons.append(row)

    available_m = sorted(
        comp.discordant for comp in COMPARISONS if comp.discordant is not None
    )
    selected_m = sorted(
        comp.discordant
        for comp in COMPARISONS
        if comp.selected and comp.discordant is not None
    )

    clusters = cluster_counts(args.split, args.index)
    gate = {}
    for name, q in scenarios.items():
        gate[name] = {
            "true_2pp_crosses_OOF_2pp_gate": gain_gate_probability(
                args.fit_n, q, 0.02, 0.02
            ),
            "true_0pp_crosses_OOF_2pp_gate": gain_gate_probability(
                args.fit_n, q, 0.0, 0.02
            ),
        }
    return {
        "alpha_two_sided": args.alpha,
        "target_power": args.target_power,
        "base_validation": {
            "successes": 76,
            "n": 152,
            "wilson_95": [lo, hi],
        },
        "comparisons": comparisons,
        "discordance_count_distribution": {
            "all_available_including_sensitivity_paths": available_m,
            "selected_or_default_comparisons": selected_m,
        },
        "q_scenarios": scenarios,
        "power": powers,
        "required_n_for_2pp": needed,
        "OOF_2pp_point_gate": gate,
        "clusters": clusters,
        "current_fixed_pool_eligibility_rate": args.eligibility_rate,
        "expected_eligible_from_pinder_xl_not_s": expected_transport_eligible,
        "pinder_xl_not_s": {
            "complexes": clusters["pinder_xl_not_s"]["source_complexes"],
            "interface_clusters": clusters["pinder_xl_not_s"]["source_clusters"],
            "cluster_overlap_with_current_fit_or_validation": clusters[
                "pinder_xl_not_s"
            ]["cluster_overlap_with_current_split"],
        },
    }


def print_report(results: dict, validation_n: int, fit_n: int, transport_n: int) -> None:
    base = results["base_validation"]
    print("BASE")
    print(
        f"  {base['successes']}/{base['n']} = 50.00%; Wilson 95% "
        f"[{fmt_pct(base['wilson_95'][0])}, {fmt_pct(base['wilson_95'][1])}]"
    )
    print("\nOBSERVED DISCORDANCE")
    for r in results["comparisons"]:
        if r["discordant"] is None:
            p_text = (
                "unavailable"
                if r["observed_exact_p"] is None
                else f"{r['observed_exact_p']:.6g}"
            )
            print(
                f"  {r['name']:<34} N={r['n']:3d}  "
                f"wins/losses and M unavailable; p={p_text}"
            )
        else:
            print(
                f"  {r['name']:<34} N={r['n']:3d}  "
                f"W/L={r['wins']}/{r['losses']}  M={r['discordant']:2d}  "
                f"q={fmt_pct(r['discordance_rate'])}  "
                f"p={r['observed_exact_p']:.6g}  min-p={r['min_attainable_p']:.6g}"
            )
    dist = results["discordance_count_distribution"]
    print(
        "  M distribution (all available, incl. sensitivity): "
        f"{dist['all_available_including_sensitivity_paths']}"
    )
    print(
        "  M distribution (selected/default only): "
        f"{dist['selected_or_default_comparisons']}"
    )

    print("\nEXACT UNCONDITIONAL POWER")
    expected = results["expected_eligible_from_pinder_xl_not_s"]
    for scenario, rows in results["power"].items():
        print(f"  {scenario}  q={fmt_pct(results['q_scenarios'][scenario])}")
        for effect, values in rows.items():
            if "status" in values:
                print(f"    true {effect:>4}: {values['status']}")
            else:
                print(
                    f"    true {effect:>4}: N={validation_n} "
                    f"{fmt_pct(values[f'N={validation_n}'])}, N={fit_n} "
                    f"{fmt_pct(values[f'N={fit_n}'])}, N={transport_n} "
                    f"{fmt_pct(values[f'N={transport_n}'])}, N={expected} "
                    f"{fmt_pct(values[f'N={expected}'])}"
                )

    print("\nOOF +2pp POINT-ESTIMATE GATE (N=637)")
    for scenario, row in results["OOF_2pp_point_gate"].items():
        print(
            f"  {scenario:<39} true +2pp pass="
            f"{fmt_pct(row['true_2pp_crosses_OOF_2pp_gate'])}  "
            f"true 0pp false-pass={fmt_pct(row['true_0pp_crosses_OOF_2pp_gate'])}"
        )

    print("\nN FOR 80% POWER AT +2pp")
    for scenario, row in results["required_n_for_2pp"].items():
        print(
            f"  {scenario:<39} q={fmt_pct(row['q'])}  "
            f"N={row['required_clusters']}  add={row['add_to_validation_152']}  "
            f"raw@63.1%={row['raw_complexes_at_current_eligibility_rate']}  "
            f"fit-transfer={'yes' if row['can_reach_by_moving_at_most_637_fit'] else 'no'}"
        )

    print("\nCLUSTERS")
    for label, row in results["clusters"].items():
        print(
            f"  {label}: source {row['source_complexes']} complexes / "
            f"{row['source_clusters']} clusters; analysed "
            f"{row['analysed_complexes']} / {row['analysed_clusters']}"
        )
    print(
        f"  expected eligible XL at current {fmt_pct(results['current_fixed_pool_eligibility_rate'])} "
        f"rate: {results['expected_eligible_from_pinder_xl_not_s']}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validation-n", type=int, default=152)
    ap.add_argument("--fit-n", type=int, default=637)
    ap.add_argument("--transport-n", type=int, default=1705)
    ap.add_argument("--eligibility-rate", type=float, default=789 / 1250)
    ap.add_argument("--effects-pp", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target-power", type=float, default=0.80)
    ap.add_argument(
        "--split",
        type=Path,
        default=Path("data/scaling/runs_nfixed/N1000_seed0/split.json"),
    )
    ap.add_argument(
        "--index",
        type=Path,
        default=Path("external/pinder/pinder/2024-02/index.parquet"),
    )
    ap.add_argument("--json", type=Path, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # Regression checks for the exact two-sided convention used in the report.
    assert math.isclose(min_attainable_p(4), 0.125)
    assert math.isclose(min_attainable_p(6), 0.03125)
    assert math.isclose(exact_mcnemar_p(4, 0), 0.125)
    assert math.isclose(exact_mcnemar_p(4, 4), 1.0)
    assert lower_critical_value(5, args.alpha) == -1
    assert lower_critical_value(6, args.alpha) == 0

    results = build_results(args)
    print_report(results, args.validation_n, args.fit_n, args.transport_n)
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
