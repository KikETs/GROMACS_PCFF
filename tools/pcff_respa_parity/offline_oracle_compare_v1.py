from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ENTRYPOINT = REPO_ROOT / "tools" / "pcff_respa_parity" / "run.py"
COMPARE_ENTRYPOINT = REPO_ROOT / "tools" / "pcff_respa_parity" / "compare.py"
DEFAULT_FIXTURE = "dense_oligomer"
DEFAULT_DT_LABEL = "dt_0p0005"
DEFAULT_OUT_ROOT = REPO_ROOT / "tests" / "reference_results" / "step30_offline_oracle_compare_rule_fix"
LEGACY_STEP3_DIR = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "step3_md_refresh_restore_review_summary"
    / DEFAULT_FIXTURE
    / DEFAULT_DT_LABEL
)
LEGACY_STEP4_DIR = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "step4_nextstep_workload_review_summary"
    / DEFAULT_FIXTURE
    / DEFAULT_DT_LABEL
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate versioned plain-facing compare tables with explicit semantic-class "
            "selection for the exact r-RESPA dense_oligomer fixture."
        )
    )
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--dt-label", default=DEFAULT_DT_LABEL)
    parser.add_argument(
        "--boundary-table",
        help="Exact refresh/restore boundary table. Default: step3 review summary artifact.",
    )
    parser.add_argument(
        "--workload-table",
        help="Next-step workload refresh table. Default: step4 review summary artifact.",
    )
    parser.add_argument(
        "--final-kick-trace",
        help="Final-kick direct-consumption trace. Default: step24 raw trace artifact.",
    )
    parser.add_argument(
        "--plain-stage-trace",
        help="Plain Verlet stage trace. Default: step4 multistep plain trace artifact.",
    )
    parser.add_argument(
        "--out",
        help="Output directory. Default: tests/reference_results/step30_offline_oracle_compare_rule_fix/<fixture>/<dt-label>",
    )
    parser.add_argument(
        "--allow-unvalidated-fixture",
        action="store_true",
        help="Allow running on fixtures other than dense_oligomer. The selection rule is only validated for dense_oligomer.",
    )
    return parser.parse_args()


def default_boundary_table(fixture: str, dt_label: str) -> Path:
    return (
        REPO_ROOT
        / "tests"
        / "reference_results"
        / "step3_md_refresh_restore_review_summary"
        / fixture
        / dt_label
        / "refresh_restore_boundary_table.tsv"
    )


def default_workload_table(fixture: str, dt_label: str) -> Path:
    return (
        REPO_ROOT
        / "tests"
        / "reference_results"
        / "step4_nextstep_workload_review_summary"
        / fixture
        / dt_label
        / "nextstep_workload_refresh_table.tsv"
    )


def default_final_kick_trace(fixture: str, dt_label: str) -> Path:
    return (
        REPO_ROOT
        / "tests"
        / "reference_results"
        / "step24_final_kick_schedule_contract_trace"
        / fixture
        / dt_label
        / "exact_three_level"
        / "final_kick_consumer_contract_trace.txt"
    )


def default_plain_stage_trace(fixture: str, dt_label: str) -> Path:
    return (
        REPO_ROOT
        / "tests"
        / "reference_results"
        / "step4_nextstep_workload_review_multistep_v2"
        / fixture
        / dt_label
        / "plain_verlet"
        / "multistep_xvf_stage_trace.txt"
    )


def format_vec(vec: tuple[float, float, float]) -> str:
    return ",".join(f"{component:+.15f}" for component in vec)


def parse_vec(value: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(f"Expected 3-vector, got {value!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def absmax_delta(lhs: tuple[float, float, float], rhs: tuple[float, float, float]) -> float:
    return max(abs(lhs[i] - rhs[i]) for i in range(3))


def load_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_boundary_rows(path: Path) -> dict[tuple[int, int, int, str], dict[str, str]]:
    rows: dict[tuple[int, int, int, str], dict[str, str]] = {}
    for row in load_tsv_rows(path):
        key = (int(row["step"]), int(row["next_step"]), int(row["atom"]), row["stage"])
        rows.setdefault(key, row)
    return rows


def parse_key_value_line(line: str) -> dict[str, str]:
    record: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        record[key] = value
    return record


def load_final_kick_trace(path: Path) -> dict[tuple[int, int], dict[str, object]]:
    rows: dict[tuple[int, int], dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = parse_key_value_line(stripped)
            if record.get("phase") != "Final":
                continue
            key = (int(record["step"]), int(record["atom"]))
            rows[key] = {
                "line_number": line_number,
                "step": int(record["step"]),
                "atom": int(record["atom"]),
                "finalKickLevels": record["finalKickLevels"],
                "highestFinalLevel": int(record["highestFinalLevel"]),
                "producer_safe_total": parse_vec(record["producer_safe_total"]),
                "consumed_total": parse_vec(record["final_kick_consumed_total"]),
                "consumer_level2": parse_vec(record["consumer_level2"]),
                "code_location": record.get("code_location", ""),
            }
    return rows


def load_plain_post_force_rows(path: Path) -> dict[tuple[int, int], dict[str, object]]:
    rows: dict[tuple[int, int], dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = parse_key_value_line(stripped)
            if record.get("side") != "PLAIN":
                continue
            stage = record.get("stage", "")
            if not stage.endswith("_POST_FORCE_XVF"):
                continue
            key = (int(record["step"]), int(record["atom"]))
            rows[key] = {
                "line_number": line_number,
                "stage": stage,
                "fx": float(record["fx"]),
                "fy": float(record["fy"]),
                "fz": float(record["fz"]),
                "xyz": (float(record["fx"]), float(record["fy"]), float(record["fz"])),
                "code_location": record.get("code_location", ""),
            }
    return rows


def build_refresh_restore_compare(
    boundary_rows: dict[tuple[int, int, int, str], dict[str, str]],
    final_kick_rows: dict[tuple[int, int], dict[str, object]],
    plain_rows: dict[tuple[int, int], dict[str, object]],
) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    candidate_keys = sorted({(step, next_step, atom) for step, next_step, atom, _ in boundary_rows.keys()})
    required_stages = (
        "after_refresh_before_restore",
        "after_restore_before_final_kick",
        "after_final_kick_handoff",
    )
    for step, next_step, atom in candidate_keys:
        if any((step, next_step, atom, stage) not in boundary_rows for stage in required_stages):
            continue
        if (step, atom) not in final_kick_rows or (next_step, atom) not in plain_rows:
            continue

        after_refresh = boundary_rows[(step, next_step, atom, "after_refresh_before_restore")]
        after_restore = boundary_rows[(step, next_step, atom, "after_restore_before_final_kick")]
        after_handoff = boundary_rows[(step, next_step, atom, "after_final_kick_handoff")]
        final_kick = final_kick_rows[(step, atom)]
        plain = plain_rows[(next_step, atom)]

        after_refresh_vec = (
            float(after_refresh["F_shared_plus_slow_levels_x"]),
            float(after_refresh["F_shared_plus_slow_levels_y"]),
            float(after_refresh["F_shared_plus_slow_levels_z"]),
        )
        after_restore_vec = (
            float(after_restore["F_shared_plus_slow_levels_x"]),
            float(after_restore["F_shared_plus_slow_levels_y"]),
            float(after_restore["F_shared_plus_slow_levels_z"]),
        )
        after_handoff_vec = (
            float(after_handoff["F_shared_plus_slow_levels_x"]),
            float(after_handoff["F_shared_plus_slow_levels_y"]),
            float(after_handoff["F_shared_plus_slow_levels_z"]),
        )

        producer_safe = final_kick["producer_safe_total"]
        consumed = final_kick["consumed_total"]
        plain_xyz = plain["xyz"]
        producer_safe_vs_plain = absmax_delta(producer_safe, plain_xyz)
        restored_vs_plain = absmax_delta(after_restore_vec, plain_xyz)

        output_rows.append(
            {
                "sampled_step": step,
                "next_plain_force_step": next_step,
                "atom": atom,
                "exact_selected_label": "producer_safe_total",
                "exact_selected_semantic_class": "plain_facing_exact_comparator",
                "exact_selected_xyz": format_vec(producer_safe),
                "after_refresh_xyz": format_vec(after_refresh_vec),
                "after_refresh_semantic_class": "refresh_output_f_shared_plus_slow_levels_audit_only",
                "after_restore_xyz": format_vec(after_restore_vec),
                "after_restore_semantic_class": "restored_live_consumed_state_audit_only",
                "after_handoff_xyz": format_vec(after_handoff_vec),
                "after_handoff_semantic_class": "post_final_kick_handoff_consumed_state_audit_only",
                "consumed_total_xyz": format_vec(consumed),
                "plain_stage": plain["stage"],
                "plain_semantic_class": "plain_post_force_visible_state",
                "plain_post_force_xyz": format_vec(plain_xyz),
                "absmax_selected_vs_plain": f"{producer_safe_vs_plain:.15f}",
                "absmax_after_restore_vs_plain": f"{restored_vs_plain:.15f}",
                "absmax_after_refresh_vs_plain": f"{absmax_delta(after_refresh_vec, plain_xyz):.15f}",
                "producer_safe_available": "true",
                "producer_safe_selected": "true",
                "selection_rule_summary": (
                    "plain STEP{next}_POST_FORCE_XVF may only join to exact semantic class "
                    "plain_facing_exact_comparator=producer_safe_total for dense_oligomer"
                ),
                "classification": (
                    "plain_facing_comparator_selected"
                    if producer_safe_vs_plain <= restored_vs_plain
                    else "unexpected_selection_regression"
                ),
            }
        )
    return output_rows


def build_nextstep_workload_compare(
    workload_rows: list[dict[str, str]],
    boundary_rows: dict[tuple[int, int, int, str], dict[str, str]],
    final_kick_rows: dict[tuple[int, int], dict[str, object]],
    plain_rows: dict[tuple[int, int], dict[str, object]],
) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for row in workload_rows:
        step = int(row["step"])
        next_step = int(row["next_step"])
        atom = int(row["atom"])
        if (step, atom) not in final_kick_rows or (next_step, atom) not in plain_rows:
            continue
        after_refresh = boundary_rows.get((step, next_step, atom, "after_refresh_before_restore"))
        if after_refresh is None:
            continue

        legacy_refresh = (
            float(after_refresh["F_shared_plus_slow_levels_x"]),
            float(after_refresh["F_shared_plus_slow_levels_y"]),
            float(after_refresh["F_shared_plus_slow_levels_z"]),
        )
        producer_safe = final_kick_rows[(step, atom)]["producer_safe_total"]
        plain_xyz = plain_rows[(next_step, atom)]["xyz"]

        output_rows.append(
            {
                "step": step,
                "next_step": next_step,
                "atom": atom,
                "selected_force_levels": row["selected_force_levels"],
                "trace_refreshed_force_levels": row["trace_refreshed_force_levels"],
                "legacy_refresh_xyz": format_vec(legacy_refresh),
                "legacy_refresh_semantic_class": "refresh_output_f_shared_plus_slow_levels_audit_only",
                "exact_selected_label": "producer_safe_total",
                "exact_selected_semantic_class": "plain_facing_exact_comparator",
                "exact_selected_xyz": format_vec(producer_safe),
                "plain_stage": plain_rows[(next_step, atom)]["stage"],
                "plain_semantic_class": "plain_post_force_visible_state",
                "plain_post_force_xyz": format_vec(plain_xyz),
                "absmax_legacy_refresh_vs_plain": f"{absmax_delta(legacy_refresh, plain_xyz):.15f}",
                "absmax_selected_vs_plain": f"{absmax_delta(producer_safe, plain_xyz):.15f}",
                "producer_safe_selected": "true",
                "selection_rule_summary": (
                    "selected_force_levels is preserved for workload audit, "
                    "but plain-facing join must use producer_safe_total instead of legacy refresh row"
                ),
            }
        )
    return output_rows


def build_decisive_mapping_trace(
    refresh_restore_rows: list[dict[str, object]],
    final_kick_rows: dict[tuple[int, int], dict[str, object]],
    plain_rows: dict[tuple[int, int], dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_key = {(int(row["sampled_step"]), int(row["atom"])): row for row in refresh_restore_rows}
    for atom in (0, 5):
        row = by_key[(1, atom)]
        final_kick = final_kick_rows[(1, atom)]
        plain = plain_rows[(2, atom)]
        rows.append(
            {
                "step": 1,
                "atom": atom,
                "exact_selected_label": row["exact_selected_label"],
                "exact_selected_semantic_class": row["exact_selected_semantic_class"],
                "exact_selected_xyz": row["exact_selected_xyz"],
                "audit_after_restore_xyz": row["after_restore_xyz"],
                "audit_after_restore_semantic_class": row["after_restore_semantic_class"],
                "consumed_total_xyz": format_vec(final_kick["consumed_total"]),
                "plain_row_label": plain["stage"],
                "plain_semantic_class": "plain_post_force_visible_state",
                "plain_post_force_xyz": format_vec(plain["xyz"]),
                "mapping_rule_summary": row["selection_rule_summary"],
                "ignored_semantic_class": row["after_restore_semantic_class"],
                "producer_safe_available": "true",
                "producer_safe_selected": "true",
                "first_wrong_action_removed": "true",
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_frozen_checklist(path: Path) -> None:
    rows = [
        {"scope": "pair_0_1_masking", "status": "closed", "basis": "step29 frozen_runtime_family_checklist.tsv"},
        {"scope": "excluded_energy_admission", "status": "closed", "basis": "step29 frozen_runtime_family_checklist.tsv"},
        {"scope": "startup_initial_kick", "status": "closed", "basis": "step29 frozen_runtime_family_checklist.tsv"},
        {"scope": "step0_reconstructed_physical_total_near_match", "status": "closed", "basis": "step29 frozen_runtime_family_checklist.tsv"},
        {"scope": "dominant_pair_force_closure", "status": "closed", "basis": "step29 frozen_runtime_family_checklist.tsv"},
        {"scope": "fixed_sink_behavior", "status": "closed", "basis": "step29 frozen_runtime_family_checklist.tsv"},
        {"scope": "canonical_gate_scope", "status": "closed", "basis": "step29 frozen_runtime_family_checklist.tsv"},
        {"scope": "active_duplicate1_gate_scope", "status": "closed", "basis": "step29 frozen_runtime_family_checklist.tsv"},
        {"scope": "active_duplicate2_gate_scope", "status": "closed", "basis": "step29 frozen_runtime_family_checklist.tsv"},
        {"scope": "restored_equals_consumed_sampled_rows", "status": "closed", "basis": "step27 comparator_semantic_trace_table.tsv"},
        {"scope": "producer_safe_better_than_restored_on_decisive_rows", "status": "closed", "basis": "step27 comparator_semantic_trace_table.tsv"},
    ]
    write_tsv(path, rows)


def build_workflow_table(script_path: Path, out_dir: Path) -> list[dict[str, object]]:
    step3_note = LEGACY_STEP3_DIR / "LEGACY_COMPARE_DEPRECATED.md"
    step4_note = LEGACY_STEP4_DIR / "LEGACY_COMPARE_DEPRECATED.md"
    legacy_m6_dir = out_dir.parent / "legacy_m6_parity"
    return [
        {
            "path": str(RUN_ENTRYPOINT),
            "role": "workflow_entrypoint",
            "old_status": "m6_parity_only_entrypoint",
            "new_status": "entrypoint_wired_to_authoritative_offline_compare_for_dense_oligomer",
            "current_fixture_authoritative": "true",
            "entrypoint_wired": "true",
            "downstream_consumer_updated": "true",
            "failfast_or_warning_added": "true",
            "notes": "run.py defaults to --offline-oracle-mode auto and may run --offline-oracle-mode only for the current fixture.",
        },
        {
            "path": str(script_path),
            "role": "generator",
            "old_status": "new_versioned_candidate",
            "new_status": "authoritative_for_dense_oligomer_plain_facing_compare",
            "current_fixture_authoritative": "true",
            "entrypoint_wired": "true",
            "downstream_consumer_updated": "true",
            "failfast_or_warning_added": "true",
            "notes": "Explicit semantic-class rule: plain joins only against producer_safe_total for dense_oligomer.",
        },
        {
            "path": str(out_dir / "refresh_restore_oracle_compare.tsv"),
            "role": "authoritative_output",
            "old_status": "absent",
            "new_status": "authoritative_plain_facing_compare_output",
            "current_fixture_authoritative": "true",
            "entrypoint_wired": "true",
            "downstream_consumer_updated": "true",
            "failfast_or_warning_added": "true",
            "notes": "Replaces legacy semantic-mismatched refresh_restore_oracle_compare.tsv for dense_oligomer truth use.",
        },
        {
            "path": str(COMPARE_ENTRYPOINT),
            "role": "legacy_parallel_entrypoint",
            "old_status": "ambiguous_vs_plain_facing_compare_role",
            "new_status": "m6_parity_only_non_authoritative_for_dense_oligomer_plain_compare",
            "current_fixture_authoritative": "false",
            "entrypoint_wired": "false",
            "downstream_consumer_updated": "n/a",
            "failfast_or_warning_added": "true",
            "notes": "CLI/help text now states compare.py is not the authoritative dense_oligomer plain-facing comparator path and direct use requires an explicit override.",
        },
        {
            "path": str(legacy_m6_dir),
            "role": "legacy_output_root",
            "old_status": "legacy_outputs_lived_in_default_workflow_root",
            "new_status": "legacy_outputs_moved_under_dedicated_legacy_subdir",
            "current_fixture_authoritative": "false",
            "entrypoint_wired": "true",
            "downstream_consumer_updated": "true",
            "failfast_or_warning_added": "true",
            "notes": "Default workflow root keeps plain_facing_truth_source.json as the single root-level truth pointer while M6 parity JSONs live under legacy_m6_parity/.",
        },
        {
            "path": str(LEGACY_STEP3_DIR / "refresh_restore_oracle_compare.tsv"),
            "role": "legacy_generator_output",
            "old_status": "implicitly_treated_as_truth_source",
            "new_status": "legacy_non_authoritative_diagnostic_only",
            "current_fixture_authoritative": "false",
            "entrypoint_wired": "false",
            "downstream_consumer_updated": "false",
            "failfast_or_warning_added": "true" if step3_note.exists() else "false",
            "notes": "Legacy exact-row semantic mapping compares after_refresh/after_restore/after_handoff directly to plain post-force.",
        },
        {
            "path": str(LEGACY_STEP4_DIR / "nextstep_workload_oracle_compare.tsv"),
            "role": "legacy_downstream_consumer_output",
            "old_status": "downstream_repeat_of_legacy_truth",
            "new_status": "legacy_non_authoritative_diagnostic_only",
            "current_fixture_authoritative": "false",
            "entrypoint_wired": "false",
            "downstream_consumer_updated": "true",
            "failfast_or_warning_added": "true" if step4_note.exists() else "false",
            "notes": "Downstream repeat; preserved for audit but not truth selection.",
        },
    ]


def main() -> None:
    args = parse_args()
    if args.fixture != DEFAULT_FIXTURE and not args.allow_unvalidated_fixture:
        raise SystemExit(
            "producer_safe_total as the plain-facing comparator is only validated for dense_oligomer. "
            "Pass --allow-unvalidated-fixture to override."
        )

    boundary_table = Path(args.boundary_table).resolve() if args.boundary_table else default_boundary_table(args.fixture, args.dt_label)
    workload_table = Path(args.workload_table).resolve() if args.workload_table else default_workload_table(args.fixture, args.dt_label)
    final_kick_trace = Path(args.final_kick_trace).resolve() if args.final_kick_trace else default_final_kick_trace(args.fixture, args.dt_label)
    plain_stage_trace = Path(args.plain_stage_trace).resolve() if args.plain_stage_trace else default_plain_stage_trace(args.fixture, args.dt_label)
    out_dir = Path(args.out).resolve() if args.out else (DEFAULT_OUT_ROOT / args.fixture / args.dt_label)

    boundary_rows = load_boundary_rows(boundary_table)
    workload_rows = load_tsv_rows(workload_table)
    final_kick_rows = load_final_kick_trace(final_kick_trace)
    plain_rows = load_plain_post_force_rows(plain_stage_trace)

    refresh_restore_rows = build_refresh_restore_compare(boundary_rows, final_kick_rows, plain_rows)
    nextstep_rows = build_nextstep_workload_compare(workload_rows, boundary_rows, final_kick_rows, plain_rows)
    decisive_rows = build_decisive_mapping_trace(refresh_restore_rows, final_kick_rows, plain_rows)
    workflow_rows = build_workflow_table(Path(__file__).resolve(), out_dir)

    write_frozen_checklist(out_dir / "frozen_runtime_family_checklist.tsv")
    write_tsv(out_dir / "refresh_restore_oracle_compare.tsv", refresh_restore_rows)
    write_tsv(out_dir / "nextstep_workload_oracle_compare.tsv", nextstep_rows)
    write_tsv(out_dir / "decisive_row_mapping_trace.tsv", decisive_rows)
    write_tsv(out_dir / "authoritative_legacy_workflow_table.tsv", workflow_rows)

    decisive_atom0 = next(row for row in refresh_restore_rows if row["sampled_step"] == 1 and row["atom"] == 0)
    decisive_atom5 = next(row for row in refresh_restore_rows if row["sampled_step"] == 1 and row["atom"] == 5)
    summary = {
        "verdict": "NEW_VERSIONED_OFFLINE_COMPARE_SCRIPT_ADDED",
        "fixture": args.fixture,
        "dt_label": args.dt_label,
        "current_fixture_only_rule": args.fixture == DEFAULT_FIXTURE,
        "existing_generator_found": False,
        "primary_input_boundary_table": str(boundary_table),
        "primary_input_final_kick_trace": str(final_kick_trace),
        "primary_input_plain_stage_trace": str(plain_stage_trace),
        "first_wrong_action": {
            "consumer": "refresh_restore_oracle_compare",
            "action": "legacy offline join treated after_refresh/after_restore/after_handoff exact rows as directly comparable to plain STEP{next}_POST_FORCE_XVF",
            "replacement_rule": "join plain post-force only against exact producer_safe_total semantic class",
        },
        "producer_safe_available_but_previously_ignored": True,
        "decisive_row": {
            "step": 1,
            "next_step": 2,
            "atom0": {
                "selected_label": decisive_atom0["exact_selected_label"],
                "absmax_selected_vs_plain": float(decisive_atom0["absmax_selected_vs_plain"]),
                "absmax_after_restore_vs_plain": float(decisive_atom0["absmax_after_restore_vs_plain"]),
            },
            "atom5": {
                "selected_label": decisive_atom5["exact_selected_label"],
                "absmax_selected_vs_plain": float(decisive_atom5["absmax_selected_vs_plain"]),
                "absmax_after_restore_vs_plain": float(decisive_atom5["absmax_after_restore_vs_plain"]),
            },
        },
        "runtime_scope_touched": False,
        "notes": [
            "This script encodes the semantic-class rule explicitly rather than inferring it from legacy stage labels.",
            "restored/consumed rows remain in the output as audit-only classes and are excluded from the plain-facing join.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "offline_oracle_compare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
