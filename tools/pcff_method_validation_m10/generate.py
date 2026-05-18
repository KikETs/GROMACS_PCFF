#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import median


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent

DEFAULT_RESULTS_ROOT = WORKSPACE_ROOT / "DL" / "gromacs" / "eval_top10_bottom10_stratified100" / "results"
DEFAULT_REFERENCE_AGGREGATE = WORKSPACE_ROOT / "DL" / "gromacs" / "simulation-trajectory-aggregate.csv"
DEFAULT_LOCAL_GROMACS_ROOT = WORKSPACE_ROOT / "DL" / "gromacs"
DEFAULT_LAMMPS_ROOT = WORKSPACE_ROOT / "DL" / "LAMMPS_NEW"
DEFAULT_OUT_ROOT = REPO_ROOT / "tests" / "reference_results" / "m10"
DEFAULT_GMX = Path("gmx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate M10 PCFF method-validation summaries.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--reference-aggregate", type=Path, default=DEFAULT_REFERENCE_AGGREGATE)
    parser.add_argument("--local-gromacs-root", type=Path, default=DEFAULT_LOCAL_GROMACS_ROOT)
    parser.add_argument("--lammps-root", type=Path, default=DEFAULT_LAMMPS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return float(text)


def safe_log10_abs_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a <= 0.0 or b <= 0.0:
        return None
    return abs(math.log10(a / b))


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def rmse(values: list[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(v * v for v in values) / len(values))


def rank_with_average_ties(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(rank_with_average_ties(xs), rank_with_average_ties(ys))


def top_k_overlap(ref_pairs: list[tuple[str, float]], pred_pairs: list[tuple[str, float]], k: int) -> dict[str, object]:
    ref_sorted = sorted(ref_pairs, key=lambda item: item[1], reverse=True)
    pred_sorted = sorted(pred_pairs, key=lambda item: item[1], reverse=True)
    ref_top = {traj for traj, _ in ref_sorted[:k]}
    pred_top = {traj for traj, _ in pred_sorted[:k]}
    overlap = ref_top & pred_top
    return {
        "k": k,
        "overlap_count": len(overlap),
        "overlap_fraction": len(overlap) / float(k) if k else None,
        "reference_only": sorted(ref_top - pred_top),
        "predicted_only": sorted(pred_top - ref_top),
        "shared": sorted(overlap),
    }


def choose_sigma_prediction(row: dict[str, str]) -> float | None:
    source = row.get("sigma_pred_source", "")
    sigma_cne = parse_float(row.get("sigma_cNE_htpmd_S_cm_pred"))
    sigma_ne = parse_float(row.get("sigma_NE_htpmd_S_cm_pred"))
    if source == "cNE" and sigma_cne is not None:
        return sigma_cne
    if source == "NE" and sigma_ne is not None:
        return sigma_ne
    if sigma_cne is not None:
        return sigma_cne
    return sigma_ne


def sigma_prediction_source(row: dict[str, str]) -> str:
    source = row.get("sigma_pred_source", "")
    sigma_cne = parse_float(row.get("sigma_cNE_htpmd_S_cm_pred"))
    sigma_ne = parse_float(row.get("sigma_NE_htpmd_S_cm_pred"))
    if source == "cNE" and sigma_cne is not None:
        return "cNE"
    if source == "NE" and sigma_ne is not None:
        return "NE"
    if sigma_cne is not None:
        return "cNE_fallback"
    if sigma_ne is not None:
        return "NE_fallback"
    return "missing"


def summarize_scalar_errors(per_system: list[dict[str, object]], error_key: str) -> dict[str, float | int | None]:
    errors = [entry[error_key] for entry in per_system if entry.get(error_key) is not None]
    values = [float(v) for v in errors]
    return {
        "n_compared": len(values),
        "mean_abs_error": average(values),
        "median_abs_error": median(values) if values else None,
        "max_abs_error": max(values) if values else None,
    }


def summarize_log_errors(per_system: list[dict[str, object]], error_key: str) -> dict[str, float | int | None]:
    errors = [entry[error_key] for entry in per_system if entry.get(error_key) is not None]
    values = [float(v) for v in errors]
    return {
        "n_compared": len(values),
        "mean_abs_log10_error": average(values),
        "median_abs_log10_error": median(values) if values else None,
        "max_abs_log10_error": max(values) if values else None,
    }


def count_statuses(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["status"])] += 1
    return dict(sorted(counts.items()))


def parse_molecule_counts(topol_path: Path) -> dict[str, int]:
    section = None
    counts: dict[str, int] = {}
    for raw in topol_path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            section = line.strip("[] ").lower()
            continue
        if section == "molecules":
            parts = line.split()
            counts[parts[0]] = int(parts[1])
    return counts


def parse_all_atomtype_masses(path: Path) -> dict[str, float]:
    masses: dict[str, float] = {}
    if not path.exists():
        return masses
    section = None
    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            section = line.strip("[] ").lower()
            continue
        if section == "atomtypes":
            parts = line.split()
            if len(parts) >= 3:
                try:
                    masses[parts[0]] = float(parts[2])
                except ValueError:
                    pass
    return masses


def parse_itp_molecule_mass(itp_path: Path, all_atomtype_masses: dict[str, float]) -> float | None:
    if not itp_path.exists():
        return None
    section = None
    total = 0.0
    saw_atoms = False
    for raw in itp_path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            section = line.strip("[] ").lower()
            continue
        if section == "atoms":
            saw_atoms = True
            parts = line.split()
            if len(parts) < 7:
                continue
            mass = None
            if len(parts) >= 8:
                try:
                    mass = float(parts[7])
                except ValueError:
                    mass = None
            if mass is None:
                mass = all_atomtype_masses.get(parts[1])
            if mass is None and parts[1] == "Li":
                mass = 6.941
            if mass is None:
                return None
            total += mass
    return total if saw_atoms else None


def parse_packmol_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    current_structure = None
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "structure" and len(parts) >= 2:
            current_structure = Path(parts[1]).name.lower()
            continue
        if parts[0] == "number" and current_structure is not None and len(parts) >= 2:
            label = current_structure
            if "chain" in label or "polymer" in label:
                counts["polymer"] = int(parts[1])
            elif label.startswith("li"):
                counts["LI"] = int(parts[1])
            elif "tfsi" in label:
                counts["tfsi"] = int(parts[1])
    return counts


def parse_pdb_atom_signature(path: Path) -> list[tuple[str, str, str]]:
    signature = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip()
            element = (line[76:78].strip() or atom_name[:1]).strip()
            signature.append((atom_name, residue_name, element))
    return signature


def load_reference_aggregate(path: Path) -> dict[str, dict[str, str]]:
    return {row["Trajectory ID"]: row for row in load_csv_rows(path)}


def build_pcff_paired_provenance_gate(local_gromacs_root: Path, lammps_root: Path) -> list[dict[str, object]]:
    runs_root = local_gromacs_root / "eval_top10_bottom10_stratified100" / "runs"
    phase_script = local_gromacs_root / "eval_top10_bottom10_stratified100" / "phase_scripts" / "gromacs_new_phase_atomtyping.py"
    phase_text = phase_script.read_text() if phase_script.exists() else ""
    global_pipeline_uses_acpype = "run_acpype(" in phase_text
    global_pipeline_mentions_gaff2 = "gaff2" in phase_text.lower()
    global_pipeline_li_fallback = "forced amber99sb-ildn Li fallback" in phase_text

    rows: list[dict[str, object]] = []
    for traj_id in ["14748", "27670"]:
        run_root = runs_root / f"Traj_{traj_id}"
        topol_path = run_root / "topology" / "topol.top"
        polymer_itp = run_root / "topology" / "polymer_GMX.itp"
        polymer_acpype_log = run_root / "topology" / "polymer.acpype" / "acpype.log"
        atomtyping_logs = sorted(run_root.glob("atomtyping_attempt*.log"))
        lammps_prod_in = lammps_root / f"Traj_{traj_id}" / "MD" / "production.in"

        polymer_itp_header = polymer_itp.read_text().splitlines()[0] if polymer_itp.exists() else ""
        polymer_acpype_text = polymer_acpype_log.read_text() if polymer_acpype_log.exists() else ""
        atomtyping_text = "\n".join(path.read_text() for path in atomtyping_logs if path.exists())
        lammps_text = lammps_prod_in.read_text() if lammps_prod_in.exists() else ""

        gromacs_preparation = "unknown"
        exclusion_reason = "insufficient provenance evidence"
        if topol_path.exists() and (
            "created by acpype" in polymer_itp_header.lower()
            or "gaff2" in polymer_acpype_text.lower()
            or "acpype" in polymer_acpype_text.lower()
        ):
            gromacs_preparation = "acpype_gaff2_topology"
            exclusion_reason = "GROMACS paired topology was generated with ACPYPE/GAFF2, not PCFF"
        elif (not topol_path.exists()) and ("gaff2" in atomtyping_text.lower() or "acpype" in atomtyping_text.lower()):
            gromacs_preparation = "acpype_gaff2_atomtyping_failed"
            exclusion_reason = "GROMACS paired topology is missing and the preserved typing attempt is ACPYPE/GAFF2"

        lammps_reference = "unknown"
        if (
            "pair_style      lj/class2/coul/long" in lammps_text
            or "pair_style\t\tlj/class2/coul/long" in lammps_text
        ) and "bond_style" in lammps_text and "class2" in lammps_text:
            lammps_reference = "pcff_class2"

        keep_for_pcff_strict_parity = gromacs_preparation == "pcff" and lammps_reference == "pcff_class2"
        status = "eligible" if keep_for_pcff_strict_parity else gromacs_preparation
        rows.append(
            {
                "trajectory_id": traj_id,
                "status": status,
                "global_gromacs_pipeline_uses_acpype": global_pipeline_uses_acpype,
                "global_gromacs_pipeline_mentions_gaff2": global_pipeline_mentions_gaff2,
                "global_gromacs_pipeline_li_fallback": global_pipeline_li_fallback,
                "gromacs_topol_exists": topol_path.exists(),
                "gromacs_preparation": gromacs_preparation,
                "lammps_reference": lammps_reference,
                "keep_for_pcff_strict_parity": keep_for_pcff_strict_parity,
                "exclusion_reason": None if keep_for_pcff_strict_parity else exclusion_reason,
            }
        )
    return rows


def build_strict_parity(
    results_root: Path,
    local_gromacs_root: Path,
    lammps_root: Path,
    per_traj_rows: list[dict[str, str]],
    provenance_gate_rows: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rdf_rows = load_csv_rows(results_root / "li_n_rdf_comparison.csv")
    nn_rows = load_csv_rows(results_root / "li_n_nn_summary.csv")
    balance_rows = load_csv_rows(results_root / "li_tfsiO_vs_polyO_contact_balance.csv")
    pop_rows = load_csv_rows(results_root / "popmat_bin_compare_top_available.csv")
    hybrid_rows = load_csv_rows(results_root / "hybrid_conductivity_decomposition.csv")

    rdf_by_traj_source: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rdf_rows:
        rdf_by_traj_source[row["traj"]][row["source"]].append(row)

    nn_by_traj_source: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in nn_rows:
        nn_by_traj_source[row["traj"]][row["source"]] = row

    balance_by_traj_engine: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in balance_rows:
        balance_by_traj_engine[row["traj"]][row["engine"]] = row

    pop_by_traj: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pop_rows:
        pop_by_traj[row["Trajectory ID"]].append(row)

    hybrid_by_traj: dict[str, dict[str, str]] = {row["Trajectory ID"]: row for row in hybrid_rows}
    per_traj_by_id = {row["Trajectory ID"]: row for row in per_traj_rows if row["status"] == "ok"}

    candidate_paired_trajs = sorted(
        traj.replace("Traj_", "")
        for traj, sources in rdf_by_traj_source.items()
        if {"gromacs_40_70ns_sampled", "lammps_production_sampled"} <= set(sources)
    )
    keep_by_id = {row["trajectory_id"]: bool(row["keep_for_pcff_strict_parity"]) for row in provenance_gate_rows}
    paired_trajs = [traj_id for traj_id in candidate_paired_trajs if keep_by_id.get(traj_id, False)]
    rejected_candidates = [
        {
            "trajectory_id": row["trajectory_id"],
            "gromacs_preparation": row["gromacs_preparation"],
            "lammps_reference": row["lammps_reference"],
            "reason": row["exclusion_reason"],
        }
        for row in provenance_gate_rows
        if row["trajectory_id"] in candidate_paired_trajs and not row["keep_for_pcff_strict_parity"]
    ]

    per_system_rows: list[dict[str, object]] = []
    for traj_id in paired_trajs:
        traj_label = f"Traj_{traj_id}"
        g_curve = sorted(rdf_by_traj_source[traj_label]["gromacs_40_70ns_sampled"], key=lambda row: float(row["r_nm"]))
        l_curve = sorted(rdf_by_traj_source[traj_label]["lammps_production_sampled"], key=lambda row: float(row["r_nm"]))
        if len(g_curve) != len(l_curve):
            raise ValueError(f"RDF grids differ for {traj_label}")

        g_diffs = []
        cn_diffs = []
        for g_row, l_row in zip(g_curve, l_curve):
            if g_row["r_nm"] != l_row["r_nm"]:
                raise ValueError(f"RDF radius mismatch for {traj_label}: {g_row['r_nm']} vs {l_row['r_nm']}")
            g_diffs.append(float(g_row["g_r"]) - float(l_row["g_r"]))
            cn_diffs.append(float(g_row["coord_num"]) - float(l_row["coord_num"]))

        g_peak = max(g_curve, key=lambda row: float(row["g_r"]))
        l_peak = max(l_curve, key=lambda row: float(row["g_r"]))

        g_nn = nn_by_traj_source[traj_label]["gromacs_40_70ns_sampled"]
        l_nn = nn_by_traj_source[traj_label]["lammps_production_sampled"]
        g_balance = balance_by_traj_engine[traj_label]["gromacs_40_70ns_sampled"]
        l_balance = balance_by_traj_engine[traj_label]["lammps_production_sampled"]
        pop_entries = pop_by_traj[traj_id]
        hybrid = hybrid_by_traj[traj_id]
        transport = per_traj_by_id[traj_id]

        rg_path = lammps_root / f"Traj_{traj_id}" / "MD" / "radgyr.txt"
        rg_available = rg_path.exists()
        lammps_rg_mean_nm = read_lammps_radgyr_mean_nm(rg_path) if rg_available else None

        topol_path = local_gromacs_root / "eval_top10_bottom10_stratified100" / "runs" / f"Traj_{traj_id}" / "topology" / "topol.top"
        topol_counts = None
        subset_count_note = None
        if topol_path.exists():
            section = None
            counts: dict[str, int] = {}
            for raw in topol_path.read_text().splitlines():
                line = raw.split(";", 1)[0].strip()
                if not line:
                    continue
                if line.startswith("["):
                    section = line.strip("[] ").lower()
                    continue
                if section == "molecules":
                    parts = line.split()
                    counts[parts[0]] = int(parts[1])
            topol_counts = counts
            if int(float(g_nn["n_li"])) * 2 == counts.get("LI", -1):
                subset_count_note = "sampled structural CSV uses half-count subset relative to full topol.top"

        chosen_sigma = choose_sigma_prediction(transport)
        per_system_rows.append(
            {
                "trajectory_id": traj_id,
                "rdf_curve_rmse": rmse(g_diffs),
                "rdf_curve_max_abs_delta": max(abs(v) for v in g_diffs),
                "coord_curve_rmse": rmse(cn_diffs),
                "rdf_peak_r_nm_gromacs": float(g_peak["r_nm"]),
                "rdf_peak_r_nm_lammps": float(l_peak["r_nm"]),
                "rdf_peak_r_nm_abs_delta": abs(float(g_peak["r_nm"]) - float(l_peak["r_nm"])),
                "rdf_peak_g_gromacs": float(g_peak["g_r"]),
                "rdf_peak_g_lammps": float(l_peak["g_r"]),
                "rdf_peak_g_abs_delta": abs(float(g_peak["g_r"]) - float(l_peak["g_r"])),
                "coord_mean_n_per_li_gromacs": float(g_nn["mean_n_per_li"]),
                "coord_mean_n_per_li_lammps": float(l_nn["mean_n_per_li"]),
                "coord_mean_n_per_li_abs_delta": abs(float(g_nn["mean_n_per_li"]) - float(l_nn["mean_n_per_li"])),
                "coord_frac_li_ge2_gromacs": float(g_nn["frac_li_ge2N"]),
                "coord_frac_li_ge2_lammps": float(l_nn["frac_li_ge2N"]),
                "coord_frac_li_ge2_abs_delta": abs(float(g_nn["frac_li_ge2N"]) - float(l_nn["frac_li_ge2N"])),
                "coord_mean_nn_nm_gromacs": float(g_nn["mean_nn_nm"]),
                "coord_mean_nn_nm_lammps": float(l_nn["mean_nn_nm"]),
                "coord_mean_nn_nm_abs_delta": abs(float(g_nn["mean_nn_nm"]) - float(l_nn["mean_nn_nm"])),
                "li_tfsiO_mean_contacts_gromacs": float(g_balance["li_tfsiO_mean_contacts"]),
                "li_tfsiO_mean_contacts_lammps": float(l_balance["li_tfsiO_mean_contacts"]),
                "li_tfsiO_mean_contacts_abs_delta": abs(float(g_balance["li_tfsiO_mean_contacts"]) - float(l_balance["li_tfsiO_mean_contacts"])),
                "li_polyO_mean_contacts_gromacs": float(g_balance["li_polyO_mean_contacts"]),
                "li_polyO_mean_contacts_lammps": float(l_balance["li_polyO_mean_contacts"]),
                "li_polyO_mean_contacts_abs_delta": abs(float(g_balance["li_polyO_mean_contacts"]) - float(l_balance["li_polyO_mean_contacts"])),
                "popmat_total_abs_diff_10x10": sum(float(row["abs_diff"]) for row in pop_entries),
                "popmat_max_abs_diff_10x10": max(float(row["abs_diff"]) for row in pop_entries),
                "popmat_top_bin_count": len(pop_entries),
                "sigma_cne_gromacs": parse_float(hybrid["g_sigma_cNE"]),
                "sigma_cne_lammps_hybrid": parse_float(hybrid["l_sigma_cNE"]),
                "sigma_ne_gromacs": parse_float(hybrid["g_sigma_NE"]),
                "sigma_ne_lammps_hybrid": parse_float(hybrid["l_sigma_NE"]),
                "diffusivity_li_gromacs": parse_float(hybrid["g_D_Li"]),
                "diffusivity_li_lammps": parse_float(hybrid["l_D_Li"]),
                "diffusivity_anion_gromacs": parse_float(hybrid["g_D_an"]),
                "diffusivity_anion_lammps": parse_float(hybrid["l_D_an"]),
                "reference_conductivity": parse_float(transport["CONDUCTIVITY"]),
                "gromacs_selected_conductivity": chosen_sigma,
                "conductivity_abs_log10_error": safe_log10_abs_ratio(chosen_sigma, parse_float(transport["CONDUCTIVITY"])),
                "reference_tplus": parse_float(transport["Transference Number"]),
                "gromacs_tplus_ne": parse_float(transport["tplus_NE_pred"]),
                "tplus_ne_abs_error": abs(parse_float(transport["tplus_NE_pred"]) - parse_float(transport["Transference Number"])),
                "gromacs_tplus_cne": parse_float(transport["c_tn_htpmd_pred"]),
                "reference_d_li": parse_float(transport["Li Diffusivity"]),
                "gromacs_d_li": parse_float(transport["D_Li_cm2s_pred"]),
                "d_li_abs_log10_error": safe_log10_abs_ratio(parse_float(transport["D_Li_cm2s_pred"]), parse_float(transport["Li Diffusivity"])),
                "reference_d_anion": parse_float(transport["TFSI Diffusivity"]),
                "gromacs_d_anion": parse_float(transport["D_an_cm2s_pred"]),
                "d_anion_abs_log10_error": safe_log10_abs_ratio(parse_float(transport["D_an_cm2s_pred"]), parse_float(transport["TFSI Diffusivity"])),
                "density_status": "unavailable",
                "density_reason": "paired-system density provenance is unresolved: sampled structural CSV counts do not match topol.top molecule counts",
                "chain_size_status": "unavailable",
                "lammps_reference_rg_mean_nm": lammps_rg_mean_nm,
                "chain_size_reason": (
                    f"LAMMPS radgyr available at {rg_path}, but no matched GROMACS production Rg artifact was found"
                    if rg_available
                    else "no matched GROMACS/LAMMPS chain-size artifact pair was found"
                ),
                "topology_molecule_counts": topol_counts,
                "structural_subset_note": subset_count_note,
            }
        )

    strict_summary = {
        "status": "limited" if paired_trajs else "blocked_by_pcff_provenance",
        "candidate_paired_system_ids": candidate_paired_trajs,
        "paired_system_ids": paired_trajs,
        "n_paired_systems": len(paired_trajs),
        "rejected_candidate_systems": rejected_candidates,
        "strict_metric_scope": [
            "rdf_li_n_curve",
            "coordination_li_n",
            "li_tfsiO_vs_li_polyO_contacts",
            "population_matrix_bins",
            "conductivity",
            "li_diffusivity",
            "anion_diffusivity",
            "transference_number",
        ],
        "missing_strict_metrics": [
            {
                "metric": "pcff_provenance_gate",
                "status": "blocked" if not paired_trajs else "qualified",
                "reason": (
                    "no paired system passed the GROMACS PCFF provenance gate"
                    if not paired_trajs
                    else "paired systems passed the provenance gate"
                ),
            },
            {
                "metric": "density",
                "status": "unavailable",
                "reason": "paired-system density provenance is unresolved because sampled structural CSV counts do not match the full GROMACS topol.top molecule counts",
            },
            {
                "metric": "chain_size_observables",
                "status": "unavailable",
                "reason": "LAMMPS radgyr.txt exists for paired systems but no matched GROMACS production Rg artifact was found",
            },
            {
                "metric": "polymer_diffusivity",
                "status": "unavailable",
                "reason": "no matched paired-system GROMACS-vs-LAMMPS polymer diffusivity artifact was found",
            },
        ],
        "aggregates": {
            "rdf_curve": {
                "n_compared": len(per_system_rows),
                "mean_rmse": average([float(row["rdf_curve_rmse"]) for row in per_system_rows]),
                "max_rmse": max(float(row["rdf_curve_rmse"]) for row in per_system_rows) if per_system_rows else None,
                "mean_peak_r_abs_delta_nm": average([float(row["rdf_peak_r_nm_abs_delta"]) for row in per_system_rows]),
                "mean_peak_g_abs_delta": average([float(row["rdf_peak_g_abs_delta"]) for row in per_system_rows]),
            },
            "coordination": {
                "mean_abs_delta_mean_n_per_li": average([float(row["coord_mean_n_per_li_abs_delta"]) for row in per_system_rows]),
                "mean_abs_delta_frac_li_ge2": average([float(row["coord_frac_li_ge2_abs_delta"]) for row in per_system_rows]),
                "mean_abs_delta_mean_nn_nm": average([float(row["coord_mean_nn_nm_abs_delta"]) for row in per_system_rows]),
            },
            "li_oxygen_contacts": {
                "mean_abs_delta_li_tfsiO_contacts": average([float(row["li_tfsiO_mean_contacts_abs_delta"]) for row in per_system_rows]),
                "mean_abs_delta_li_polyO_contacts": average([float(row["li_polyO_mean_contacts_abs_delta"]) for row in per_system_rows]),
            },
            "population_matrix": {
                "mean_total_abs_diff_10x10": average([float(row["popmat_total_abs_diff_10x10"]) for row in per_system_rows]),
                "max_bin_abs_diff_10x10": max(float(row["popmat_max_abs_diff_10x10"]) for row in per_system_rows) if per_system_rows else None,
            },
            "transport": {
                "conductivity": summarize_log_errors(per_system_rows, "conductivity_abs_log10_error"),
                "li_diffusivity": summarize_log_errors(per_system_rows, "d_li_abs_log10_error"),
                "anion_diffusivity": summarize_log_errors(per_system_rows, "d_anion_abs_log10_error"),
                "tplus_ne": summarize_scalar_errors(per_system_rows, "tplus_ne_abs_error"),
            },
        },
    }
    return strict_summary, per_system_rows


def build_screening_usefulness(results_root: Path, reference_aggregate: Path, local_gromacs_root: Path) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    per_traj_rows = load_csv_rows(results_root / "per_traj_eval.csv")
    reference_rows = load_reference_aggregate(reference_aggregate)

    ok_rows = [row for row in per_traj_rows if row["status"] == "ok"]
    completion = {
        "n_total": len(per_traj_rows),
        "n_completed": len(ok_rows),
        "n_failed": sum(1 for row in per_traj_rows if row["status"] != "ok"),
        "per_group_completed": {
            group: sum(1 for row in ok_rows if row["sample_group"] == group)
            for group in sorted({row["sample_group"] for row in per_traj_rows})
        },
    }

    metric_specs = [
        ("conductivity", "CONDUCTIVITY", choose_sigma_prediction, "log10"),
        ("li_diffusivity", "Li Diffusivity", lambda row: parse_float(row["D_Li_cm2s_pred"]), "log10"),
        ("anion_diffusivity", "TFSI Diffusivity", lambda row: parse_float(row["D_an_cm2s_pred"]), "log10"),
        ("transference_number", "Transference Number", lambda row: parse_float(row["tplus_NE_pred"]), "absolute"),
    ]

    metric_table: list[dict[str, object]] = []
    screening_metrics: dict[str, dict[str, object]] = {}
    for metric_name, reference_key, predicted_fn, error_mode in metric_specs:
        ref_pairs: list[tuple[str, float]] = []
        pred_pairs: list[tuple[str, float]] = []
        per_system = []
        source_counts: dict[str, int] = defaultdict(int)
        for row in ok_rows:
            ref_value = parse_float(row[reference_key])
            pred_value = predicted_fn(row)
            if ref_value is None or pred_value is None:
                continue
            error = safe_log10_abs_ratio(pred_value, ref_value) if error_mode == "log10" else abs(pred_value - ref_value)
            if metric_name == "conductivity":
                source_counts[sigma_prediction_source(row)] += 1
            entry = {
                "metric": metric_name,
                "trajectory_id": row["Trajectory ID"],
                "sample_group": row["sample_group"],
                "reference_value": ref_value,
                "predicted_value": pred_value,
                "abs_error": error if error_mode == "absolute" else None,
                "abs_log10_error": error if error_mode == "log10" else None,
            }
            metric_table.append(entry)
            per_system.append(entry)
            ref_pairs.append((row["Trajectory ID"], ref_value))
            pred_pairs.append((row["Trajectory ID"], pred_value))

        ref_values = [pair[1] for pair in ref_pairs]
        pred_values = [pair[1] for pair in pred_pairs]
        screening_metrics[metric_name] = {
            "n_compared": len(per_system),
            "spearman_rank_correlation": spearman(ref_values, pred_values),
            "top10_overlap": top_k_overlap(ref_pairs, pred_pairs, min(10, len(per_system))) if per_system else None,
            "bottom10_overlap": top_k_overlap(
                [(traj, -value) for traj, value in ref_pairs],
                [(traj, -value) for traj, value in pred_pairs],
                min(10, len(per_system)),
            )
            if per_system
            else None,
        }
        if metric_name == "conductivity":
            screening_metrics[metric_name]["prediction_source_counts"] = dict(sorted(source_counts.items()))
        if error_mode == "log10":
            screening_metrics[metric_name].update(summarize_log_errors(per_system, "abs_log10_error"))
        else:
            screening_metrics[metric_name].update(summarize_scalar_errors(per_system, "abs_error"))

    density_rows: list[dict[str, object]] = []
    for final_summary in sorted(local_gromacs_root.glob("Traj_*/analysis/final_summary.csv")):
        rows = load_csv_rows(final_summary)
        if not rows:
            continue
        row = rows[0]
        traj_id = row["Trajectory ID"]
        reference = reference_rows.get(traj_id)
        if reference is None:
            continue
        ref_density = parse_float(reference["Density"])
        pred_density = parse_float(row["Density"])
        if ref_density is None or pred_density is None:
            continue
        density_rows.append(
            {
                "metric": "density",
                "trajectory_id": traj_id,
                "reference_value": ref_density,
                "predicted_value": pred_density,
                "abs_error": abs(pred_density - ref_density),
            }
        )

    density_ref = [entry["reference_value"] for entry in density_rows]
    density_pred = [entry["predicted_value"] for entry in density_rows]
    screening_metrics["density"] = {
        "n_compared": len(density_rows),
        "availability_scope": "local final_summary subset only",
        "spearman_rank_correlation": spearman(density_ref, density_pred),
        "top3_overlap": top_k_overlap(
            [(entry["trajectory_id"], entry["reference_value"]) for entry in density_rows],
            [(entry["trajectory_id"], entry["predicted_value"]) for entry in density_rows],
            min(3, len(density_rows)),
        )
        if density_rows
        else None,
        **summarize_scalar_errors(density_rows, "abs_error"),
    }

    screening_summary = {
        "status": "not_pcff_qualified",
        "completion": completion,
        "provenance_status": {
            "gromacs_pipeline_preparation": "ACPYPE/GAFF2 with Li amber99sb-ildn fallback",
            "pcff_qualified_for_current_m10_claim": False,
            "reason": "the screening pipeline prepares GROMACS systems with ACPYPE/GAFF2 rather than PCFF class2 parameters",
        },
        "strict_vs_screening_separation": {
            "strict_parity_basis": "paired-system direct comparisons on representative systems",
            "screening_usefulness_basis": "cohort-level ranking and error analysis on completed GROMACS screening runs",
        },
        "metrics": screening_metrics,
        "limitations": [
            "screening outputs are not PCFF-qualified because the local GROMACS preparation pipeline uses ACPYPE/GAFF2",
            "conductivity usefulness mixes cNE and NE-derived GROMACS conductivity outputs depending on artifact availability",
            "density usefulness is based on the local final_summary subset only",
            "screening transference uses tplus_NE because cNE transference outputs are frequently unphysical",
        ],
    }
    return screening_summary, metric_table, density_rows


def build_density_provenance(results_root: Path, local_gromacs_root: Path) -> list[dict[str, object]]:
    stage_rows = load_csv_rows(results_root / "li_tfsiO_stage_trace.csv")
    stage_by_key = {(row["traj"], row["engine"], row["stage"]): row for row in stage_rows}
    nn_rows = load_csv_rows(results_root / "li_n_nn_summary.csv")
    structural_counts: dict[str, dict[str, str]] = {}
    for row in nn_rows:
        if row["source"] == "gromacs_40_70ns_sampled":
            structural_counts[row["traj"]] = row

    rows: list[dict[str, object]] = []
    for traj_label in ["Traj_14748", "Traj_27670"]:
        traj_id = traj_label.replace("Traj_", "")
        run_topology = local_gromacs_root / "eval_top10_bottom10_stratified100" / "runs" / traj_label / "topology"
        run_root = local_gromacs_root / "eval_top10_bottom10_stratified100" / "runs" / traj_label
        topol_path = run_topology / "topol.top"
        packmol_counts = parse_packmol_counts(run_root / "packmol" / "packmol.inp")
        alltypes = parse_all_atomtype_masses(run_topology / "all_atomtypes.itp")
        molecule_counts = parse_molecule_counts(topol_path) if topol_path.exists() else None
        polymer_mass = parse_itp_molecule_mass(run_topology / "polymer_clean.itp", alltypes)
        tfsi_mass = parse_itp_molecule_mass(run_topology / "tfsi_clean.itp", alltypes)
        li_mass = parse_itp_molecule_mass(run_topology / "li_clean.itp", alltypes)

        g_stage = stage_by_key.get((traj_label, "gromacs", "production"))
        l_stage = stage_by_key.get((traj_label, "lammps_new", "production_sampled_avg"))
        g_struct = structural_counts.get(traj_label)

        topol_density = None
        if molecule_counts and polymer_mass and tfsi_mass and li_mass and g_stage and g_stage["box_volume_nm3"]:
            total_mass_amu = (
                molecule_counts.get("polymer", 0) * polymer_mass
                + molecule_counts.get("tfsi", 0) * tfsi_mass
                + molecule_counts.get("LI", 0) * li_mass
            )
            topol_density = total_mass_amu * 1.66053906660e-24 / (float(g_stage["box_volume_nm3"]) * 1e-21)

        status = "unavailable"
        reason = "missing full paired provenance"
        if not topol_path.exists():
            if packmol_counts:
                reason = "GROMACS paired topology is missing, but packmol input still records the intended molecule counts"
            else:
                reason = "GROMACS paired topology is missing"
        elif g_struct is None:
            reason = "GROMACS structural sampled-count CSV is missing"
        elif molecule_counts is not None and g_struct is not None and molecule_counts.get("LI") is not None:
            sampled_li = int(float(g_struct["n_li"]))
            if sampled_li != molecule_counts.get("LI"):
                status = "inconsistent"
                reason = "GROMACS sampled structural counts do not match the paired topol.top molecule counts"
            else:
                status = "available"
                reason = "counts are internally consistent"

        rows.append(
            {
                "trajectory_id": traj_id,
                "status": status,
                "reason": reason,
                "gromacs_topol_exists": topol_path.exists(),
                "gromacs_topol_counts": molecule_counts,
                "packmol_counts": packmol_counts or None,
                "gromacs_structural_sampled_n_li": int(float(g_struct["n_li"])) if g_struct else None,
                "gromacs_structural_sampled_n_n": int(float(g_struct["n_n"])) if g_struct else None,
                "gromacs_production_box_volume_nm3": float(g_stage["box_volume_nm3"]) if g_stage and g_stage["box_volume_nm3"] else None,
                "lammps_production_box_volume_nm3": None if l_stage is None or not l_stage["box_volume_nm3"] else float(l_stage["box_volume_nm3"]),
                "gromacs_topol_density_g_cm3_if_used": topol_density,
            }
        )
    return rows


def run_grompp_dry_run(gmx_path: Path, donor_topology_dir: Path, conf_gro: Path) -> dict[str, object]:
    if not gmx_path.exists():
        return {"status": "unavailable", "reason": f"gmx not found at {gmx_path}"}
    if not donor_topology_dir.exists() or not conf_gro.exists():
        return {"status": "unavailable", "reason": "missing donor topology or target GRO"}

    with tempfile.TemporaryDirectory(prefix="pcff_m10_recover_") as tmp:
        tmpdir = Path(tmp)
        topology_dir = tmpdir / "topology"
        shutil.copytree(donor_topology_dir, topology_dir)
        target_gro = tmpdir / "conf_initial.gro"
        shutil.copy2(conf_gro, target_gro)
        em_mdp = tmpdir / "em.mdp"
        em_mdp.write_text(
            "\n".join(
                [
                    "integrator = steep",
                    "emtol = 100",
                    "emstep = 0.01",
                    "nsteps = 10",
                    "cutoff-scheme = Verlet",
                    "coulombtype = PME",
                    "rcoulomb = 1.2",
                    "vdwtype = Cut-off",
                    "rvdw = 1.2",
                    "pbc = xyz",
                    "",
                ]
            )
        )
        output_tpr = tmpdir / "test.tpr"
        result = subprocess.run(
            [
                str(gmx_path),
                "grompp",
                "-f",
                str(em_mdp),
                "-c",
                str(target_gro),
                "-p",
                str(topology_dir / "topol.top"),
                "-o",
                str(output_tpr),
                "-maxwarn",
                "2",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        merged_output = (result.stdout or "") + "\n" + (result.stderr or "")
        warning_count = merged_output.count("WARNING")
        atom_mismatch_line = next(
            (line.strip() for line in merged_output.splitlines() if "non-matching atom names" in line),
            None,
        )
        return {
            "status": "ok" if result.returncode == 0 else "failed",
            "return_code": result.returncode,
            "warning_count": warning_count,
            "atom_mismatch_note": atom_mismatch_line,
        }


def build_paired_topology_recovery(results_root: Path, local_gromacs_root: Path) -> list[dict[str, object]]:
    manifest_rows = load_csv_rows(results_root / "sample_manifest.csv")
    manifest_by_id = {row["Trajectory ID"]: row for row in manifest_rows}
    runs_root = local_gromacs_root / "eval_top10_bottom10_stratified100" / "runs"

    def donor_candidates(target_id: str) -> list[str]:
        target = manifest_by_id[target_id]
        candidates = []
        for row in manifest_rows:
            donor_id = row["Trajectory ID"]
            if donor_id == target_id:
                continue
            same_spec = (
                row["SMILES"] == target["SMILES"]
                and row["Degree of Polymerization"] == target["Degree of Polymerization"]
                and row["Density"] == target["Density"]
                and row["Molality"] == target["Molality"]
            )
            if not same_spec:
                continue
            donor_topol = runs_root / f"Traj_{donor_id}" / "topology" / "topol.top"
            if donor_topol.exists():
                candidates.append(donor_id)
        return sorted(candidates)

    rows: list[dict[str, object]] = []
    for traj_id in ["14748", "27670"]:
        run_root = runs_root / f"Traj_{traj_id}"
        topology_dir = run_root / "topology"
        topol_path = topology_dir / "topol.top"
        target_sig_path = run_root / "structures" / f"Traj_{traj_id}_chain_fix_acpypefix.pdb"
        donor_ids = donor_candidates(traj_id)
        donor_id = donor_ids[0] if donor_ids else None
        donor_sig_match = None
        dry_run = None
        if donor_id is not None:
            donor_root = runs_root / f"Traj_{donor_id}"
            donor_sig_path = donor_root / "structures" / f"Traj_{donor_id}_chain_fix_acpypefix.pdb"
            if donor_sig_path.exists() and target_sig_path.exists():
                donor_sig_match = parse_pdb_atom_signature(donor_sig_path) == parse_pdb_atom_signature(target_sig_path)
            dry_run = run_grompp_dry_run(
                DEFAULT_GMX,
                donor_root / "topology",
                run_root / "md" / "conf_initial.gro",
            )

        rows.append(
            {
                "trajectory_id": traj_id,
                "topol_exists": topol_path.exists(),
                "sample_manifest_smiles": manifest_by_id[traj_id]["SMILES"],
                "sample_manifest_n_repeat": manifest_by_id[traj_id]["Degree of Polymerization"],
                "sample_manifest_density": manifest_by_id[traj_id]["Density"],
                "sample_manifest_molality": manifest_by_id[traj_id]["Molality"],
                "donor_trajectory_id": donor_id,
                "donor_atom_signature_match": donor_sig_match,
                "donor_topology_dry_run_status": None if dry_run is None else dry_run["status"],
                "donor_topology_dry_run_rc": None if dry_run is None else dry_run.get("return_code"),
                "donor_topology_dry_run_warning_count": None if dry_run is None else dry_run.get("warning_count"),
                "donor_topology_dry_run_atom_mismatch_note": None if dry_run is None else dry_run.get("atom_mismatch_note"),
            }
        )
    return rows


def build_chain_size_artifact_status(local_gromacs_root: Path, lammps_root: Path) -> list[dict[str, object]]:
    run_results_path = local_gromacs_root / "eval_top10_bottom10_stratified100" / "results" / "run_results.csv"
    run_results_by_id = {row["Trajectory ID"]: row for row in load_csv_rows(run_results_path)} if run_results_path.exists() else {}
    rows: list[dict[str, object]] = []
    for traj_id in ["14748", "27670"]:
        run_root = local_gromacs_root / "eval_top10_bottom10_stratified100" / "runs" / f"Traj_{traj_id}"
        production_dir = run_root / "md" / "production"
        lammps_rg = lammps_root / f"Traj_{traj_id}" / "MD" / "radgyr.txt"
        run_results_row = run_results_by_id.get(traj_id)
        analysis_csv = Path(run_results_row["analysis_csv"]) if run_results_row and run_results_row.get("analysis_csv") else None
        gromacs_candidates = sorted(
            str(path)
            for path in local_gromacs_root.glob(f"**/Traj_{traj_id}/**/*")
            if path.is_file() and ("rg" in path.name.lower() or "gyr" in path.name.lower())
        )
        gromacs_prod_dirs = [str(production_dir)] if production_dir.is_dir() else []
        production_tpr = production_dir / "production.tpr"
        production_xtc = production_dir / "production.xtc"
        production_gro = production_dir / "production.gro"
        status = "unavailable"
        reason = "no matched GROMACS production chain-size artifact found"
        if not lammps_rg.exists():
            reason = "LAMMPS reference radgyr.txt is missing"
        elif production_tpr.exists() and production_xtc.exists() and gromacs_candidates:
            status = "available"
            reason = "matched artifacts found"
        elif production_dir.is_dir():
            reason = "GROMACS production directory exists but required trajectory artifacts for Rg are missing"
        rows.append(
            {
                "trajectory_id": traj_id,
                "status": status,
                "reason": reason,
                "lammps_rg_path": str(lammps_rg) if lammps_rg.exists() else None,
                "lammps_rg_mean_nm": read_lammps_radgyr_mean_nm(lammps_rg) if lammps_rg.exists() else None,
                "gromacs_rg_candidates": gromacs_candidates,
                "gromacs_production_dirs": gromacs_prod_dirs,
                "gromacs_production_tpr_exists": production_tpr.exists(),
                "gromacs_production_xtc_exists": production_xtc.exists(),
                "gromacs_production_gro_exists": production_gro.exists(),
                "run_results_analysis_csv_path": str(analysis_csv) if analysis_csv else None,
                "run_results_analysis_csv_exists": analysis_csv.exists() if analysis_csv else False,
                "gromacs_rg_generation_ready": production_tpr.exists() and production_xtc.exists(),
            }
        )
    return rows


def build_paired_artifact_registry_audit(results_root: Path, local_gromacs_root: Path) -> list[dict[str, object]]:
    run_results_path = local_gromacs_root / "eval_top10_bottom10_stratified100" / "results" / "run_results.csv"
    run_results_by_id = {row["Trajectory ID"]: row for row in load_csv_rows(run_results_path)} if run_results_path.exists() else {}
    stage_rows = load_csv_rows(results_root / "li_tfsiO_stage_trace.csv")
    stage_keys = {(row["traj"], row["engine"], row["stage"]) for row in stage_rows}
    rdf_rows = load_csv_rows(results_root / "li_n_rdf_comparison.csv")
    rdf_keys = {(row["traj"], row["source"]) for row in rdf_rows}
    nn_rows = load_csv_rows(results_root / "li_n_nn_summary.csv")
    nn_keys = {(row["traj"], row["source"]) for row in nn_rows}

    rows: list[dict[str, object]] = []
    for traj_id in ["14748", "27670"]:
        traj_label = f"Traj_{traj_id}"
        run_root = local_gromacs_root / "eval_top10_bottom10_stratified100" / "runs" / traj_label
        production_dir = run_root / "md" / "production"
        production_tpr = production_dir / "production.tpr"
        production_xtc = production_dir / "production.xtc"
        production_gro = production_dir / "production.gro"
        topol_path = run_root / "topology" / "topol.top"

        run_results_row = run_results_by_id.get(traj_id, {})
        analysis_csv = Path(run_results_row["analysis_csv"]) if run_results_row.get("analysis_csv") else None
        analysis_csv_exists = analysis_csv.exists() if analysis_csv else False
        production_stage_trace_present = (traj_label, "gromacs", "production") in stage_keys
        sampled_rdf_present = (traj_label, "gromacs_40_70ns_sampled") in rdf_keys
        sampled_nn_present = (traj_label, "gromacs_40_70ns_sampled") in nn_keys
        raw_artifacts_present = production_tpr.exists() or production_xtc.exists() or production_gro.exists()
        derived_artifacts_present = production_stage_trace_present or sampled_rdf_present or sampled_nn_present

        status = "unavailable"
        reason = "no run-results registry row was found"
        if run_results_row:
            if run_results_row.get("analysis_status") == "ok" and not analysis_csv_exists and not raw_artifacts_present:
                if derived_artifacts_present:
                    status = "derived_metrics_without_raw_artifacts"
                    reason = (
                        "run_results.csv reports completed analysis and derived production/sample summaries exist, "
                        "but the referenced analysis CSV and raw production artifacts are missing"
                    )
                else:
                    status = "false_positive_registry"
                    reason = (
                        "run_results.csv reports completed analysis, but neither the referenced analysis CSV nor "
                        "raw/derived artifacts were found"
                    )
            elif run_results_row.get("analysis_status") == "ok" and analysis_csv_exists:
                status = "consistent"
                reason = "run_results.csv points to an existing analysis CSV"
            else:
                status = "incomplete"
                reason = "run_results.csv does not claim a completed analysis with locally present artifacts"

        rows.append(
            {
                "trajectory_id": traj_id,
                "status": status,
                "reason": reason,
                "run_results_status": run_results_row.get("status"),
                "run_results_analysis_status": run_results_row.get("analysis_status"),
                "run_results_analysis_last_stage": run_results_row.get("analysis_last_stage"),
                "run_results_analysis_csv_path": str(analysis_csv) if analysis_csv else None,
                "run_results_analysis_csv_exists": analysis_csv_exists,
                "topol_exists": topol_path.exists(),
                "gromacs_production_dir_exists": production_dir.is_dir(),
                "gromacs_production_tpr_exists": production_tpr.exists(),
                "gromacs_production_xtc_exists": production_xtc.exists(),
                "gromacs_production_gro_exists": production_gro.exists(),
                "gromacs_production_stage_trace_present": production_stage_trace_present,
                "gromacs_sampled_rdf_present": sampled_rdf_present,
                "gromacs_sampled_nn_present": sampled_nn_present,
            }
        )
    return rows


def build_transport_decomposition(results_root: Path) -> list[dict[str, object]]:
    hybrid_rows = load_csv_rows(results_root / "hybrid_conductivity_decomposition.csv")
    topbin_rows = load_csv_rows(results_root / "hybrid_top_bin_contributions.csv")
    charge_rows = load_csv_rows(results_root / "ion_charge_compare.csv")
    ion_lj_rows = load_csv_rows(results_root / "ion_lj_compare.csv")
    pair_rows = load_csv_rows(results_root / "mixed_pair_compare.csv")

    topbin_by_traj_source: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in topbin_rows:
        topbin_by_traj_source[(row["Trajectory ID"], row["source"])].append(row)

    charge_by_traj: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in charge_rows:
        charge_by_traj[row["traj"]].append(row)

    ion_lj_by_traj: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ion_lj_rows:
        ion_lj_by_traj[row["traj"]].append(row)

    pair_by_traj: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pair_rows:
        pair_by_traj[row["traj"]].append(row)

    rows: list[dict[str, object]] = []
    for row in hybrid_rows:
        traj_id = row["Trajectory ID"]
        if traj_id not in {"14748", "27670"}:
            continue
        traj_label = f"Traj_{traj_id}"
        ref_sigma = parse_float(row["ref_sigma"])
        g_cne = parse_float(row["g_sigma_cNE"])
        g_ne = parse_float(row["g_sigma_NE"])
        l_cne = parse_float(row["l_sigma_cNE"])
        l_ne = parse_float(row["l_sigma_NE"])
        g_li = parse_float(row["g_D_Li"])
        l_li = parse_float(row["l_D_Li"])
        g_an = parse_float(row["g_D_an"])
        l_an = parse_float(row["l_D_an"])
        g_v = parse_float(row["g_V_cm3"])
        l_v = parse_float(row["l_V_cm3"])

        g_top = max(topbin_by_traj_source[(traj_id, "G")], key=lambda entry: float(entry["sigma_fraction"]))
        l_top = max(topbin_by_traj_source[(traj_id, "L")], key=lambda entry: float(entry["sigma_fraction"]))

        li_o = next((entry for entry in pair_by_traj[traj_label] if entry["pair"] == "Li-O"), None)
        li_n = next((entry for entry in pair_by_traj[traj_label] if entry["pair"] == "Li-N"), None)
        site_charges = {entry["site"]: entry for entry in charge_by_traj[traj_label]}
        site_lj = {entry["site"]: entry for entry in ion_lj_by_traj[traj_label]}

        rows.append(
            {
                "trajectory_id": traj_id,
                "reference_sigma": ref_sigma,
                "gromacs_sigma_cNE": g_cne,
                "gromacs_sigma_NE": g_ne,
                "lammps_sigma_cNE_hybrid": l_cne,
                "lammps_sigma_NE_hybrid": l_ne,
                "sigma_cNE_abs_log10_error": safe_log10_abs_ratio(g_cne, ref_sigma),
                "sigma_NE_abs_log10_error": safe_log10_abs_ratio(g_ne, ref_sigma),
                "li_diffusivity_ratio_g_over_l": (g_li / l_li) if g_li and l_li else None,
                "anion_diffusivity_ratio_g_over_l": (g_an / l_an) if g_an and l_an else None,
                "volume_ratio_g_over_l": (g_v / l_v) if g_v and l_v else None,
                "top_gromacs_population_bin": f"({g_top['bin_i']},{g_top['bin_j']})",
                "top_gromacs_population_sigma_fraction": float(g_top["sigma_fraction"]),
                "top_lammps_population_bin": f"({l_top['bin_i']},{l_top['bin_j']})",
                "top_lammps_population_sigma_fraction": float(l_top["sigma_fraction"]),
                "li_o_qprod_gromacs": parse_float(li_o["g_qprod"]) if li_o else None,
                "li_o_qprod_lammps": parse_float(li_o["l_qprod"]) if li_o else None,
                "li_o_sigma_gromacs_nm": parse_float(li_o["g_sigma_nm_LB"]) if li_o else None,
                "li_o_sigma_lammps_nm": parse_float(li_o["l_sigma_nm_6th"]) if li_o else None,
                "li_n_qprod_gromacs": parse_float(li_n["g_qprod"]) if li_n else None,
                "li_n_qprod_lammps": parse_float(li_n["l_qprod"]) if li_n else None,
                "charge_n_gromacs": parse_float(site_charges["N"]["g_charge"]) if "N" in site_charges else None,
                "charge_n_lammps": parse_float(site_charges["N"]["l_charge"]) if "N" in site_charges else None,
                "charge_o_gromacs": parse_float(site_charges["O"]["g_charge"]) if "O" in site_charges else None,
                "charge_o_lammps": parse_float(site_charges["O"]["l_charge"]) if "O" in site_charges else None,
                "charge_s_gromacs": parse_float(site_charges["S"]["g_charge"]) if "S" in site_charges else None,
                "charge_s_lammps": parse_float(site_charges["S"]["l_charge"]) if "S" in site_charges else None,
                "lj_li_sigma_gromacs_nm": parse_float(site_lj["Li"]["g_sigma_nm"]) if "Li" in site_lj else None,
                "lj_li_sigma_lammps_nm": parse_float(site_lj["Li"]["l_sigma_nm"]) if "Li" in site_lj else None,
                "lj_li_epsilon_gromacs_kj": parse_float(site_lj["Li"]["g_epsilon_kj"]) if "Li" in site_lj else None,
                "lj_li_epsilon_lammps_kj": parse_float(site_lj["Li"]["l_epsilon_kj"]) if "Li" in site_lj else None,
                "lj_n_sigma_gromacs_nm": parse_float(site_lj["N"]["g_sigma_nm"]) if "N" in site_lj else None,
                "lj_n_sigma_lammps_nm": parse_float(site_lj["N"]["l_sigma_nm"]) if "N" in site_lj else None,
                "lj_o_sigma_gromacs_nm": parse_float(site_lj["O"]["g_sigma_nm"]) if "O" in site_lj else None,
                "lj_o_sigma_lammps_nm": parse_float(site_lj["O"]["l_sigma_nm"]) if "O" in site_lj else None,
            }
        )
    return rows


def summarize_transport_decomposition(rows: list[dict[str, object]]) -> dict[str, object]:
    def mean_abs_delta(key_a: str, key_b: str) -> float | None:
        values = []
        for row in rows:
            a = row.get(key_a)
            b = row.get(key_b)
            if a is not None and b is not None:
                values.append(abs(float(a) - float(b)))
        return average(values)

    def mean_relative_delta(key_a: str, key_b: str) -> float | None:
        values = []
        for row in rows:
            a = row.get(key_a)
            b = row.get(key_b)
            if a is not None and b is not None and float(b) != 0.0:
                values.append(abs(float(a) - float(b)) / abs(float(b)))
        return average(values)

    top_bin_match_count = sum(
        1
        for row in rows
        if row.get("top_gromacs_population_bin") is not None
        and row.get("top_gromacs_population_bin") == row.get("top_lammps_population_bin")
    )

    electrostatics_score = average(
        [
            value
            for value in [
                mean_relative_delta("charge_n_gromacs", "charge_n_lammps"),
                mean_relative_delta("charge_o_gromacs", "charge_o_lammps"),
                mean_relative_delta("charge_s_gromacs", "charge_s_lammps"),
                mean_relative_delta("li_o_qprod_gromacs", "li_o_qprod_lammps"),
                mean_relative_delta("li_n_qprod_gromacs", "li_n_qprod_lammps"),
            ]
            if value is not None
        ]
    )
    lj_score = average(
        [
            value
            for value in [
                mean_relative_delta("lj_li_sigma_gromacs_nm", "lj_li_sigma_lammps_nm"),
                mean_relative_delta("lj_li_epsilon_gromacs_kj", "lj_li_epsilon_lammps_kj"),
                mean_relative_delta("lj_n_sigma_gromacs_nm", "lj_n_sigma_lammps_nm"),
                mean_relative_delta("lj_o_sigma_gromacs_nm", "lj_o_sigma_lammps_nm"),
                mean_relative_delta("li_o_sigma_gromacs_nm", "li_o_sigma_lammps_nm"),
            ]
            if value is not None
        ]
    )
    volume_score = average(
        [abs(float(row["volume_ratio_g_over_l"]) - 1.0) for row in rows if row.get("volume_ratio_g_over_l") is not None]
    )
    heuristic_driver_scores = {
        "electrostatics_relative_delta_score": electrostatics_score,
        "lj_relative_delta_score": lj_score,
        "volume_ratio_delta_score": volume_score,
    }
    heuristic_driver_order = [
        item[0]
        for item in sorted(
            [(key, value) for key, value in heuristic_driver_scores.items() if value is not None],
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    return {
        "n_paired_systems": len(rows),
        "conductivity_error_modes": {
            "sigma_cNE_mean_abs_log10_error": average(
                [float(row["sigma_cNE_abs_log10_error"]) for row in rows if row.get("sigma_cNE_abs_log10_error") is not None]
            ),
            "sigma_NE_mean_abs_log10_error": average(
                [float(row["sigma_NE_abs_log10_error"]) for row in rows if row.get("sigma_NE_abs_log10_error") is not None]
            ),
        },
        "population_bin_agreement": {
            "top_bin_match_count": top_bin_match_count,
            "top_bin_match_fraction": (top_bin_match_count / len(rows)) if rows else None,
        },
        "pair_parameter_deltas": {
            "mean_abs_delta_li_o_qprod": mean_abs_delta("li_o_qprod_gromacs", "li_o_qprod_lammps"),
            "mean_abs_delta_li_n_qprod": mean_abs_delta("li_n_qprod_gromacs", "li_n_qprod_lammps"),
            "mean_abs_delta_li_o_sigma_nm": mean_abs_delta("li_o_sigma_gromacs_nm", "li_o_sigma_lammps_nm"),
        },
        "site_charge_deltas": {
            "mean_abs_delta_n_charge": mean_abs_delta("charge_n_gromacs", "charge_n_lammps"),
            "mean_abs_delta_o_charge": mean_abs_delta("charge_o_gromacs", "charge_o_lammps"),
            "mean_abs_delta_s_charge": mean_abs_delta("charge_s_gromacs", "charge_s_lammps"),
        },
        "ion_lj_deltas": {
            "mean_abs_delta_li_sigma_nm": mean_abs_delta("lj_li_sigma_gromacs_nm", "lj_li_sigma_lammps_nm"),
            "mean_abs_delta_li_epsilon_kj": mean_abs_delta("lj_li_epsilon_gromacs_kj", "lj_li_epsilon_lammps_kj"),
            "mean_abs_delta_n_sigma_nm": mean_abs_delta("lj_n_sigma_gromacs_nm", "lj_n_sigma_lammps_nm"),
            "mean_abs_delta_o_sigma_nm": mean_abs_delta("lj_o_sigma_gromacs_nm", "lj_o_sigma_lammps_nm"),
        },
        "heuristic_driver_scores": heuristic_driver_scores,
        "heuristic_driver_order": heuristic_driver_order,
        "state_variable_ratios": {
            "mean_li_diffusivity_ratio_g_over_l": average(
                [float(row["li_diffusivity_ratio_g_over_l"]) for row in rows if row.get("li_diffusivity_ratio_g_over_l") is not None]
            ),
            "mean_anion_diffusivity_ratio_g_over_l": average(
                [float(row["anion_diffusivity_ratio_g_over_l"]) for row in rows if row.get("anion_diffusivity_ratio_g_over_l") is not None]
            ),
            "mean_volume_ratio_g_over_l": average(
                [float(row["volume_ratio_g_over_l"]) for row in rows if row.get("volume_ratio_g_over_l") is not None]
            ),
        },
    }


def build_method_readiness(
    strict_summary: dict[str, object],
    screening_summary: dict[str, object],
    artifact_registry_rows: list[dict[str, object]],
    provenance_gate_rows: list[dict[str, object]],
) -> dict[str, object]:
    conductivity_rho = screening_summary["metrics"]["conductivity"]["spearman_rank_correlation"]
    li_rho = screening_summary["metrics"]["li_diffusivity"]["spearman_rank_correlation"]
    an_rho = screening_summary["metrics"]["anion_diffusivity"]["spearman_rank_correlation"]
    density_rho = screening_summary["metrics"]["density"]["spearman_rank_correlation"]
    n_pcff_qualified = sum(1 for row in provenance_gate_rows if row["keep_for_pcff_strict_parity"])

    readiness = {
        "overall_status": "pcff_provenance_blocked",
        "strict_parity_readiness": {
            "status": "blocked_by_provenance",
            "basis": [
                f"PCFF-qualified paired systems available: {n_pcff_qualified}",
                "density strict parity is unavailable because production-density provenance is unresolved",
                "chain-size parity is unavailable because matched GROMACS Rg artifacts were not found",
                "paired raw-production provenance is incomplete even where derived GROMACS sampled metrics exist",
            ],
        },
        "screening_usefulness_readiness": {
            "status": "not_pcff_qualified",
            "basis": [
                "screening cohort comes from a GROMACS ACPYPE/GAFF2 preparation pipeline rather than PCFF class2",
                f"conductivity screening rho={conductivity_rho}",
                f"Li diffusivity screening rho={li_rho}",
                f"anion diffusivity screening rho={an_rho}",
                f"density subset rho={density_rho}",
            ],
        },
        "recommended_use": [
            "use the current M10 artifacts only for provenance debugging and workflow scaffolding",
            "do not interpret the current screening cohort as a PCFF-vs-LAMMPS validation set",
            "rebuild M10 on GROMACS systems prepared with actual PCFF parameters before making any transport claim",
        ],
        "blocking_gaps": [
            "no paired system currently passes the GROMACS PCFF provenance gate",
            "screening cohort is prepared with ACPYPE/GAFF2 rather than PCFF",
            "paired density provenance is unresolved",
            "paired chain-size observables are unavailable on the GROMACS side",
            "paired raw production artifacts are missing while run_results.csv still reports completed analysis",
            "polymer diffusivity parity is unavailable",
            "strict parity coverage is limited to two representative systems",
        ],
        "artifact_registry_status_counts": count_statuses(artifact_registry_rows),
        "pcff_provenance_gate_status_counts": count_statuses(provenance_gate_rows),
    }
    return readiness


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_lammps_radgyr_mean_nm(path: Path) -> float | None:
    values: list[float] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    row_index = int(parts[0])
                    value_angstrom = float(parts[1])
                except ValueError:
                    continue
                if row_index >= 1:
                    values.append(value_angstrom / 10.0)
    return average(values)


def main() -> None:
    args = parse_args()
    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    per_traj_rows = load_csv_rows(args.results_root / "per_traj_eval.csv")
    provenance_gate_rows = build_pcff_paired_provenance_gate(args.local_gromacs_root, args.lammps_root)
    strict_summary, strict_rows = build_strict_parity(
        args.results_root, args.local_gromacs_root, args.lammps_root, per_traj_rows, provenance_gate_rows
    )
    screening_summary, screening_rows, density_rows = build_screening_usefulness(
        args.results_root, args.reference_aggregate, args.local_gromacs_root
    )
    density_provenance_rows = build_density_provenance(args.results_root, args.local_gromacs_root)
    topology_recovery_rows = build_paired_topology_recovery(args.results_root, args.local_gromacs_root)
    chain_size_rows = build_chain_size_artifact_status(args.local_gromacs_root, args.lammps_root)
    artifact_registry_rows = build_paired_artifact_registry_audit(args.results_root, args.local_gromacs_root)
    transport_decomposition_rows = build_transport_decomposition(args.results_root)
    readiness = build_method_readiness(strict_summary, screening_summary, artifact_registry_rows, provenance_gate_rows)

    comparison_summary = {
        "milestone": "M10",
        "results_root": str(args.results_root),
        "reference_aggregate": str(args.reference_aggregate),
        "local_gromacs_root": str(args.local_gromacs_root),
        "lammps_root": str(args.lammps_root),
        "strict_parity": strict_summary,
        "screening_usefulness": screening_summary,
        "provenance_diagnostics": {
            "pcff_paired_provenance_gate": {
                "n_systems": len(provenance_gate_rows),
                "status_counts": count_statuses(provenance_gate_rows),
                "eligible_for_pcff_strict_parity": [
                    row["trajectory_id"] for row in provenance_gate_rows if row["keep_for_pcff_strict_parity"]
                ],
            },
            "paired_density_provenance": {
                "n_systems": len(density_provenance_rows),
                "status_counts": count_statuses(density_provenance_rows),
            },
            "paired_topology_recovery": {
                "n_systems": len(topology_recovery_rows),
                "n_missing_topologies": sum(1 for row in topology_recovery_rows if not row["topol_exists"]),
                "recoverable_with_donor_topology": [
                    row["trajectory_id"]
                    for row in topology_recovery_rows
                    if (not row["topol_exists"]) and row.get("donor_topology_dry_run_status") == "ok"
                ],
            },
            "paired_chain_size_artifacts": {
                "n_systems": len(chain_size_rows),
                "status_counts": count_statuses(chain_size_rows),
            },
            "paired_artifact_registry_audit": {
                "n_systems": len(artifact_registry_rows),
                "status_counts": count_statuses(artifact_registry_rows),
            },
        },
        "transport_mismatch_diagnostics": summarize_transport_decomposition(transport_decomposition_rows),
        "method_readiness": readiness,
    }

    write_json(out_root / "comparison_summary.json", comparison_summary)
    write_json(out_root / "strict_parity_summary.json", strict_summary)
    write_json(out_root / "screening_usefulness_summary.json", screening_summary)
    write_json(out_root / "method_readiness_summary.json", readiness)
    write_csv(out_root / "strict_parity_metrics.csv", strict_rows)
    write_csv(out_root / "screening_metric_rows.csv", screening_rows)
    write_csv(out_root / "density_local_subset.csv", density_rows)
    write_csv(out_root / "pcff_paired_provenance_gate.csv", provenance_gate_rows)
    write_csv(out_root / "paired_density_provenance.csv", density_provenance_rows)
    write_csv(out_root / "paired_topology_recovery.csv", topology_recovery_rows)
    write_csv(out_root / "chain_size_artifact_status.csv", chain_size_rows)
    write_csv(out_root / "paired_artifact_registry_audit.csv", artifact_registry_rows)
    write_csv(out_root / "transport_decomposition.csv", transport_decomposition_rows)


if __name__ == "__main__":
    main()
