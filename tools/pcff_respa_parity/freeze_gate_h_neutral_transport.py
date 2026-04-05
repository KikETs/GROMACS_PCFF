from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from freeze_gate_a_oracle import REPO_ROOT, write_text
from validate_gate_b_nb_gpu import load_json


DEFAULT_PILOT_RESULT = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_h_neutral_transport_pilot"
    / "summaries"
    / "pilot_result.json"
)
DEFAULT_ENTRY_RESULT = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_h_neutral_entry_validation_longer"
    / "summaries"
    / "entry_result.json"
)
DEFAULT_SCAFFOLD_MANIFEST = (
    REPO_ROOT
    / "tests"
    / "reference_results"
    / "gate_h_fixture_scaffold"
    / "gate_h_dense_oligomer_2x2x2"
    / "fixture_manifest.json"
)
DEFAULT_OUT = REPO_ROOT / "tests" / "reference_results" / "gate_h_neutral_transport_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the scope-limited official Gate H-neutral manifest from the large-neutral transport pilot."
    )
    parser.add_argument("--pilot-result", default=str(DEFAULT_PILOT_RESULT), help="Large-neutral pilot result JSON.")
    parser.add_argument("--entry-result", default=str(DEFAULT_ENTRY_RESULT), help="Large-neutral entry result JSON.")
    parser.add_argument("--scaffold-manifest", default=str(DEFAULT_SCAFFOLD_MANIFEST), help="Large-neutral scaffold manifest.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory.")
    return parser.parse_args()


def build_markdown(manifest: dict[str, object]) -> str:
    comparison = manifest["comparison"]
    lines = [
        "# Gate H-Neutral Transport Validation",
        "",
        f"- Verdict: `{manifest['status']}`",
        f"- Scope: {manifest['scope']}",
        f"- Scope-limited recommendation: `{manifest['scope_limited_recommendation']}`",
        f"- Overall transport production recommendation: `{manifest['overall_transport_production_recommendation']}`",
        f"- Replica count per layout: `{manifest['run_settings']['replicas']}`",
        f"- Equilibration / production: `{manifest['run_settings']['equil_ps']} ps / {manifest['run_settings']['prod_ps']} ps`",
        "",
        "## Observable",
        f"- `{manifest['observable']}`: `{comparison['classification']}` / passes=`{comparison['passes']}`",
        f"- CPU mean: `{comparison['cpu_mean']:.10g}` cm^2/s",
        f"- GPU mean: `{comparison['gpu_mean']:.10g}` cm^2/s",
        f"- Mean diff: `{comparison['mean_diff']:.10g}` cm^2/s",
        f"- Combined uncertainty: `{comparison['combined_uncertainty']:.10g}` cm^2/s",
        "",
        "## Boundaries",
        f"- Full official Gate H remains out of scope: `{manifest['full_gate_h_status']}`",
        "- Charged conductivity/cNE are not covered by this manifest.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    pilot_result = load_json(Path(args.pilot_result))
    entry_result = load_json(Path(args.entry_result))
    scaffold_manifest = load_json(Path(args.scaffold_manifest))

    if pilot_result.get("status") != "PASS":
        raise ValueError("Large-neutral transport pilot is not PASS; cannot freeze Gate H-neutral.")
    if entry_result.get("status") != "PASS":
        raise ValueError("Large-neutral entry result is not PASS; cannot freeze Gate H-neutral.")

    out_root = Path(args.out).resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    summaries_dir = out_root / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    comparison = pilot_result["comparison"]
    manifest = {
        "schema_version": 1,
        "scope": "Official Gate H-neutral only: large-neutral scaffold self-diffusion on the standalone exact r-RESPA full-GPU path.",
        "status": "PASS",
        "observable": "oligomer_diffusivity_cm2_s",
        "comparison": comparison,
        "run_settings": pilot_result["run_settings"],
        "analysis_settings": pilot_result["analysis_settings"],
        "scope_limited_recommendation": "GO",
        "overall_transport_production_recommendation": "NO-GO",
        "full_gate_h_status": "FAIL",
        "source_artifacts": {
            "pilot_result": str(Path(args.pilot_result).resolve()),
            "entry_result": str(Path(args.entry_result).resolve()),
            "scaffold_manifest": str(Path(args.scaffold_manifest).resolve()),
            "pilot_transport_table": str(
                Path(args.pilot_result).resolve().parent / "transport_comparison.tsv"
            ),
        },
        "scaffold": {
            "derived_system": scaffold_manifest["derived_system"],
            "natoms": scaffold_manifest["natoms"],
            "box_nm": scaffold_manifest["box_nm"],
        },
        "limitations": [
            "This manifest does not close charged transport, conductivity, or cNE.",
            "This manifest does not override the official full Gate H FAIL/NO-GO result.",
            "This manifest is limited to the large-neutral scaffold and molecule-wise MSD/diffusivity only.",
        ],
    }

    (out_root / "run_commands.sh").write_text(
        "#!/usr/bin/env bash\n"
        f"# Source pilot artifact: {Path(args.pilot_result).resolve()}\n"
        f"# Source entry artifact: {Path(args.entry_result).resolve()}\n",
        encoding="utf-8",
    )
    transport_src = Path(args.pilot_result).resolve().parent / "transport_comparison.tsv"
    transport_dst = summaries_dir / "transport_comparison.tsv"
    shutil.copy2(transport_src, transport_dst)
    (summaries_dir / "gate_h_neutral_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_text(summaries_dir / "gate_h_neutral_manifest.md", build_markdown(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
