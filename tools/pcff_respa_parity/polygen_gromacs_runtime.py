from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

ATM_TO_BAR = 1.01325
POLYGEN_EQ_CHUNK_STEPS = 100_000


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _render_template(text: str, replacements: Mapping[str, str]) -> str:
    rendered = text
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def _stage_manifest_path(base_dir: Path) -> Path:
    return base_dir / "gromacs_pcff" / "stage_manifest.json"


def _compute_stage_nsteps(stage: dict[str, Any], *, eqfactor: float, production_total_ns: float) -> int:
    dt_ps = float(stage["dt_ps"])
    if dt_ps <= 0:
        return max(1, int(stage.get("nsteps", 50000)))
    if stage["phase"] == "production":
        duration_ps = float(production_total_ns) * 1000.0
    else:
        duration_ps = float(stage["duration_ps"]) * float(eqfactor)
    return max(1, int(round(duration_ps / dt_ps)))


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _replace_mdp_key(text: str, key: str, value: str) -> str:
    import re

    pattern = rf"^(\s*{re.escape(key)}\s*=).*$"
    out, n = re.subn(pattern, rf"\1 {value}", text, count=1, flags=re.M)
    if n:
        return out
    return text.rstrip() + f"\n{key} = {value}\n"


def _polygen_stage_nsteps(duration_ps: float, base_dt_ps: float, *, eqfactor: float) -> int:
    nsteps = max(4, int(round(float(duration_ps) * float(eqfactor) / float(base_dt_ps))))
    return nsteps + ((-nsteps) % 4)


def _polygen_common_nonbonded(*, ewald_rtol: float = 4.47100039e-03, nstlist: int = 80) -> list[str]:
    pme_order = int(os.environ.get("GROMACS_BATCH_PME_ORDER", "5"))
    # LAMMPS uses a 0.95 nm cutoff with `neighbor 3.0 bin`.  Keep the same
    # 0.30 nm neighbor skin.  At the default nstlist=80 and r-RESPA factor 4,
    # GROMACS rebuilds every 20 LAMMPS outer steps; the completed 1474101
    # reference rebuilt every 22.5 outer steps on average with `check yes`.
    rlist_nm = float(os.environ.get("GROMACS_BATCH_RLIST_NM", "1.250"))
    # LAMMPS compute temp removes the three translational center-of-mass DOF.
    # A very large nstcomm gives GROMACS the same 3N-3 thermostat DOF without
    # periodically perturbing the trajectory after the initial COM cleanup.
    nstcomm = int(os.environ.get("GROMACS_BATCH_NSTCOMM", "1000000000"))
    return [
        "constraints             = none",
        "cutoff-scheme           = Verlet",
        f"nstlist                 = {int(nstlist)}",
        f"rlist                   = {rlist_nm:.3f}",
        "rvdw                    = 0.950",
        "rcoulomb                = 0.950",
        "vdwtype                 = Cut-off",
        "vdw-modifier            = none",
        "DispCorr                = AllEnerPres",
        "coulombtype             = PME",
        "coulomb-modifier        = none",
        f"ewald-rtol              = {float(ewald_rtol):.8e}",
        f"pme-order               = {pme_order}",
        "fourierspacing          = 0.120",
        "epsilon-r               = 1",
        "pbc                     = xyz",
        "comm-mode               = Linear",
        "comm-grps               = System",
        f"nstcomm                 = {nstcomm}",
        "verlet-buffer-tolerance = -1",
    ]


def _polygen_exact_respa_lines() -> list[str]:
    pair14_level = int(os.environ.get("GROMACS_BATCH_EXACT_RESPA_PAIR14_LEVEL", "2"))
    if pair14_level not in (1, 2):
        raise ValueError(
            "GROMACS_BATCH_EXACT_RESPA_PAIR14_LEVEL must be 1 or 2; "
            f"got {pair14_level}"
        )
    # The reference PolyGen/LAMMPS lane uses `run_style respa 2 4`: bonded
    # terms are evaluated every 0.5 fs and pair, listed 1-4, and kspace terms
    # are all evaluated on the outer 2 fs step. Use the native two-level
    # representation. A nominal three-level outer-only representation is
    # dynamically equivalent, but was measured to be more than twice slower.
    return [
        "exact-respa             = yes",
        "exact-respa-levels      = 2",
        "exact-respa-level2-factor = 4",
        "exact-respa-bond-level  = 1",
        "exact-respa-angle-level = 1",
        "exact-respa-dihedral-level = 1",
        "exact-respa-improper-level = 1",
        f"exact-respa-pair14-level = {pair14_level}",
        "exact-respa-pair-level  = 2",
        "exact-respa-kspace-level = 2",
    ]


def _polygen_md_mdp(
    *,
    name: str,
    ensemble: str,
    duration_ps: float,
    base_dt_ps: float,
    eqfactor: float,
    temp_k: float,
    pressure_atm: float | None,
    tau_t_ps: float,
    tau_p_ps: float | None = None,
    gen_vel: bool = False,
    gen_seed: int | None = None,
    production: bool = False,
    nsteps_override: int | None = None,
    temp_start_k: float | None = None,
    temp_end_k: float | None = None,
    pressure_start_atm: float | None = None,
    pressure_end_atm: float | None = None,
    nstlist: int = 80,
    init_step: int = 0,
    schedule_start_ps: float = 0.0,
    thermostat: str = "nose-hoover",
) -> tuple[str, int, float]:
    nsteps = int(nsteps_override) if nsteps_override is not None else _polygen_stage_nsteps(duration_ps, base_dt_ps, eqfactor=eqfactor)
    final_state_interval = int(init_step) + int(nsteps)
    if final_state_interval <= 0:
        raise ValueError(
            f"PolyGen MD stage {name} has invalid final step {final_state_interval}"
        )
    sample = 40000
    energy_sample = sample
    if name == "eq12_npt_1200ps":
        # LAMMPS fix ave/time samples lx on every 2 fs outer step. Store the
        # same cadence in EDR instead of a full-system XTC stream.
        box_sample_ps = float(
            os.environ.get(
                "GROMACS_BATCH_EQ12_BOX_SAMPLE_PS",
                f"{4.0 * float(base_dt_ps):.12g}",
            )
        )
        energy_sample = max(4, int(round(box_sample_ps / float(base_dt_ps))))
        energy_sample += (-energy_sample) % 4
    thermostat = str(thermostat).strip().lower().replace("_", "-")
    if thermostat not in {"nose-hoover", "none"}:
        raise ValueError(
            f"Unsupported PolyGen thermostat {thermostat!r} for stage {name}"
        )
    temp_start = float(temp_k if temp_start_k is None else temp_start_k)
    temp_end = float(temp_k if temp_end_k is None else temp_end_k)
    ref_temp = temp_end
    ref_pressure = pressure_atm
    if pressure_end_atm is not None:
        ref_pressure = float(pressure_end_atm)
    lines = [
        f"title                   = PolyGen EM handoff {name} exact r-RESPA",
        "integrator              = md-vv",
        f"dt                      = {base_dt_ps:.9f}",
        f"nsteps                  = {nsteps}",
        f"init-step               = {int(init_step)}",
        *_polygen_common_nonbonded(nstlist=int(nstlist)),
        *_polygen_exact_respa_lines(),
        f"nstcalcenergy           = {energy_sample}",
        f"nstenergy               = {energy_sample}",
        f"nstlog                  = {sample}",
        f"nstvout                 = {0 if production else final_state_interval}",
        "nstfout                 = 0",
    ]
    if production:
        # The legacy production_stage00_nvt.in dumps every 2000 outer
        # 2 fs steps, so preserve its 4 ps trajectory cadence.
        traj_interval = max(4, int(round(4.0 / base_dt_ps)))
        traj_interval += (-traj_interval) % 4
        lines += [
            f"nstxout                 = {traj_interval}",
            f"nstxout-compressed      = {traj_interval}",
        ]
    else:
        lines += [
            f"nstxout                 = {final_state_interval}",
            "nstxout-compressed      = 0",
        ]
    lines.append("compressed-x-precision  = 1000")
    if thermostat == "nose-hoover":
        lines += [
            "tcoupl                  = nose-hoover",
            "tc-grps                 = System",
            f"tau-t                   = {float(tau_t_ps):.6f}",
            f"ref-t                   = {ref_temp:.6f}",
            "nsttcouple              = 4",
            "nh-chain-length         = 3",
            "print-nose-hoover-chain-variables = no",
        ]
    else:
        lines.append("tcoupl                  = no")
    lines.append(f"gen-vel                 = {'yes' if gen_vel else 'no'}")
    if gen_vel:
        seed = (
            int(gen_seed)
            if gen_seed is not None
            else int(os.environ.get("GROMACS_BATCH_GEN_SEED", "520419"))
        )
        lines += [
            f"gen-temp                = {temp_start:.6f}",
            f"gen-seed                = {seed}",
        ]
    lines.append("continuation            = no" if gen_vel else "continuation            = yes")
    if not math.isclose(temp_start, temp_end):
        active_ps = max(0.0, float(nsteps) * float(base_dt_ps))
        schedule_end_ps = float(schedule_start_ps) + active_ps
        lines += [
            "annealing               = single",
            "annealing-npoints       = 2",
            f"annealing-time          = {float(schedule_start_ps):.9f} {schedule_end_ps:.9f}",
            f"annealing-temp          = {temp_start:.6f} {temp_end:.6f}",
        ]
    if ensemble == "npt":
        if ref_pressure is None or tau_p_ps is None:
            raise ValueError(f"NPT stage {name} requires pressure_atm and tau_p_ps")
        if (
            pressure_start_atm is not None
            and pressure_end_atm is not None
            and not math.isclose(float(pressure_start_atm), float(pressure_end_atm))
        ):
            lines.append(
                f"; NOTE: pressure ramp {float(pressure_start_atm):g}->{float(pressure_end_atm):g} atm is applied during mdrun via GMX_PCFF_REFP_RAMP_*; static ref-p is ramp end."
            )
        lines += [
            "pcoupl                  = MTTK",
            "pcoupltype              = isotropic",
            f"tau-p                   = {float(tau_p_ps):.6f}",
            f"ref-p                   = {float(ref_pressure) * ATM_TO_BAR:.6f}",
            "compressibility         = 4.5e-05",
            "nstpcouple              = 4",
            "refcoord-scaling        = no",
        ]
    else:
        lines.append("pcoupl                  = no")
    return "\n".join(lines) + "\n", nsteps, base_dt_ps


def _polygen_em_mdp(name: str, *, nsteps: int = 1000) -> tuple[str, int, float]:
    lines = [
        f"title                   = PolyGen EM handoff {name}",
        "integrator              = cg",
        "emtol                   = 4.184e-05",
        "emstep                  = 0.01",
        f"nsteps                  = {int(nsteps)}",
        "nstcgsteep              = 0",
        *_polygen_common_nonbonded(),
        "tcoupl                  = no",
        "pcoupl                  = no",
        "nstenergy               = 100",
        "nstlog                  = 100",
    ]
    return "\n".join(lines) + "\n", int(nsteps), 0.0


def _polygen_interp(start: float, end: float, fraction: float) -> float:
    return float(start) + (float(end) - float(start)) * float(fraction)


def _polygen_lammps_chunk_boundaries_ps(
    duration_ps: float,
    outer_timestep_fs: float,
    *,
    eqfactor: float = 1.0,
) -> list[float]:
    """Return the physical chunk boundaries used by the LAMMPS runner.

    LAMMPS first applies ``eqfactor`` while converting the nominal duration to
    an integer outer-step count, then splits that count into 100k-step chunks.
    Applying eqfactor independently after nominal-duration chunking changes
    both chunk lengths and pressure/temperature ramp endpoints.
    """

    raw_steps = (
        float(eqfactor) * float(duration_ps) * 1000.0 / float(outer_timestep_fs)
    )
    # Decimal protocol values such as 0.83*10 ps/0.5 fs are exact integers in
    # the LAMMPS input, but can land a few ulps below that integer in Python.
    total_steps = int(math.floor(raw_steps + 1.0e-9))
    if total_steps <= 0:
        raise ValueError(
            f"PolyGen stage has no LAMMPS outer steps: duration_ps={duration_ps}, "
            f"outer_timestep_fs={outer_timestep_fs}, eqfactor={eqfactor}"
        )
    boundaries = [0.0]
    current = 0
    while current < total_steps:
        current = min(total_steps, current + POLYGEN_EQ_CHUNK_STEPS)
        boundaries.append(float(current) * float(outer_timestep_fs) / 1000.0)
    return boundaries


def _expand_polygen_equil_stage_specs(
    stage_specs: list[dict[str, Any]],
    *,
    eqfactor: float = 1.0,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for base_index, spec in enumerate(stage_specs, start=1):
        if spec.get("phase") != "equilibration" or spec.get("kind") != "md":
            out = dict(spec)
            out["base_index"] = base_index
            out["segment_index"] = 1
            out["segment_count"] = 1
            out["segment_start_ps"] = 0.0
            expanded.append(out)
            continue
        outer_dt_fs = float(
            spec.get("outer_timestep_fs", float(spec["base_dt_ps"]) * 4000.0)
        )
        stage_eqfactor = 1.0 if bool(spec.get("eqfactor_applied", False)) else float(eqfactor)
        lammps_boundaries = _polygen_lammps_chunk_boundaries_ps(
            float(spec["duration_ps"]),
            outer_dt_fs,
            eqfactor=stage_eqfactor,
        )
        effective_duration_ps = lammps_boundaries[-1]
        if str(spec.get("name")) == "eq12_npt_1200ps" or bool(
            spec.get("disable_chunking", False)
        ):
            boundaries = [0.0, effective_duration_ps]
        else:
            boundaries = lammps_boundaries
        nseg = len(boundaries) - 1
        if nseg <= 1:
            out = dict(spec)
            out["nominal_duration_ps"] = float(spec["duration_ps"])
            out["duration_ps"] = effective_duration_ps
            out["eqfactor_applied"] = True
            out["base_index"] = base_index
            out["segment_index"] = 1
            out["segment_count"] = 1
            out["segment_start_ps"] = 0.0
            expanded.append(out)
            continue
        temp_start = float(spec.get("temp_start_k", spec.get("temp_k", 353.0)))
        temp_end = float(spec.get("temp_end_k", spec.get("temp_k", temp_start)))
        press_start = spec.get("pressure_start_atm", spec.get("pressure_atm"))
        press_end = spec.get("pressure_end_atm", spec.get("pressure_atm", press_start))
        for seg_index in range(1, nseg + 1):
            seg_start = float(boundaries[seg_index - 1])
            seg_end = float(boundaries[seg_index])
            frac0 = seg_start / effective_duration_ps
            frac1 = seg_end / effective_duration_ps
            out = dict(spec)
            out["name"] = f"{spec['name']}_chunk{seg_index:04d}"
            out["nominal_duration_ps"] = float(spec["duration_ps"])
            out["duration_ps"] = seg_end - seg_start
            out["eqfactor_applied"] = True
            out["temp_start_k"] = _polygen_interp(temp_start, temp_end, frac0)
            out["temp_end_k"] = _polygen_interp(temp_start, temp_end, frac1)
            out["temp_k"] = out["temp_end_k"]
            if press_start is not None and press_end is not None:
                out["pressure_start_atm"] = _polygen_interp(float(press_start), float(press_end), frac0)
                out["pressure_end_atm"] = _polygen_interp(float(press_start), float(press_end), frac1)
                out["pressure_atm"] = out["pressure_end_atm"]
            out["gen_vel"] = bool(spec.get("gen_vel", False)) and seg_index == 1
            out["base_index"] = base_index
            out["segment_index"] = seg_index
            out["segment_count"] = nseg
            out["segment_start_ps"] = seg_start
            expanded.append(out)
    return expanded


def _polygen_legacy_eq12_average_window(
    base_stage_specs: list[dict[str, Any]],
    *,
    eqfactor: float,
) -> dict[str, Any]:
    """Reconstruct the actual legacy LAMMPS ``fix ave/time`` sample window.

    Eq04 resets the global LAMMPS timestep to zero.  Eq12 does not reset it,
    and ``fix ave/time`` aligns output to global multiples of ``nave``.  Thus
    the retained Eq12 value is generally not an average over the final
    ``500*eqfactor`` ps of Eq12.  This reconstructs the uninterrupted generated
    schedule; a rescued reference whose restart carries a different global
    step must provide explicit Eq12-relative bounds.
    """

    eqfactor = float(eqfactor)
    if not math.isfinite(eqfactor) or eqfactor <= 0.0:
        raise ValueError(f"eqfactor must be positive; got {eqfactor}")

    reset_name = "eq04_npt_compress_500ps"
    eq12_name = "eq12_npt_1200ps"
    try:
        reset_index = next(
            index
            for index, spec in enumerate(base_stage_specs)
            if str(spec.get("name")) == reset_name
        )
        eq12_index = next(
            index
            for index, spec in enumerate(base_stage_specs)
            if str(spec.get("name")) == eq12_name
        )
    except StopIteration as exc:
        raise ValueError("PolyGen schedule is missing Eq04 or Eq12") from exc
    if eq12_index <= reset_index:
        raise ValueError("PolyGen Eq12 must follow the Eq04 timestep reset")

    def outer_steps(spec: Mapping[str, Any]) -> int:
        timestep_fs = float(spec["outer_timestep_fs"])
        stage_eqfactor = 1.0 if bool(spec.get("eqfactor_applied", False)) else eqfactor
        raw_steps = (
            stage_eqfactor
            * float(spec["duration_ps"])
            * 1000.0
            / timestep_fs
        )
        steps = int(math.floor(raw_steps + 1.0e-9))
        if steps <= 0:
            raise ValueError(
                f"PolyGen stage {spec.get('name')} has no LAMMPS outer steps"
            )
        return steps

    stage_steps: dict[str, int] = {}
    stage_start_step = 0
    for spec in base_stage_specs[reset_index:eq12_index]:
        if spec.get("phase") != "equilibration" or spec.get("kind") != "md":
            continue
        steps = outer_steps(spec)
        stage_steps[str(spec["name"])] = steps
        stage_start_step += steps

    eq12_spec = base_stage_specs[eq12_index]
    timestep_fs = float(eq12_spec["outer_timestep_fs"])
    eq12_steps = outer_steps(eq12_spec)
    stage_end_step = stage_start_step + eq12_steps
    nave = int(math.floor(eqfactor * 500.0 * 1000.0 / timestep_fs))
    nrepeat = nave - 1
    nskip = int(math.floor(eqfactor * 200.0 * 1000.0 / timestep_fs))
    if nave <= 2:
        raise ValueError(
            f"Eq12 legacy average needs nave > 2; got nave={nave}"
        )

    # With Nevery=1, Nrepeat=nave-1, Nfreq=nave, LAMMPS reports on global
    # multiples of nave.  The latest complete report is the value subsequently
    # read through f_4 and used by change_box.
    output_step = (stage_end_step // nave) * nave
    first_sample_step = output_step - (nrepeat - 1)
    if output_step <= stage_start_step or output_step > stage_end_step:
        raise ValueError(
            "Eq12 legacy fix ave/time has no output inside the stage: "
            f"stage={stage_start_step}..{stage_end_step}, output={output_step}"
        )
    if first_sample_step <= stage_start_step or first_sample_step < nskip:
        raise ValueError(
            "Eq12 legacy fix ave/time sample window predates fix availability: "
            f"stage_start={stage_start_step}, first_sample={first_sample_step}, "
            f"nskip={nskip}"
        )

    relative_start_ps = (
        float(first_sample_step - stage_start_step) * timestep_fs / 1000.0
    )
    relative_end_ps = (
        float(output_step - stage_start_step) * timestep_fs / 1000.0
    )
    return {
        "mode": "legacy_actual",
        "timestep_reset_stage": reset_name,
        "stage_start_global_outer_step": stage_start_step,
        "stage_end_global_outer_step": stage_end_step,
        "eq12_outer_steps": eq12_steps,
        "outer_timestep_fs": timestep_fs,
        "nave": nave,
        "nrepeat": nrepeat,
        "nfreq": nave,
        "nskip": nskip,
        "first_sample_global_outer_step": first_sample_step,
        "output_global_outer_step": output_step,
        "relative_start_ps": relative_start_ps,
        "relative_end_ps": relative_end_ps,
        "sample_count": nrepeat,
        "sample_spacing_outer_steps": 1,
        "preceding_stage_outer_steps": stage_steps,
    }


def _setup_polygen_em_handoff_environment(
    proj: Path,
    *,
    prepare_report: Mapping[str, Any],
    temperature_k: float,
    eqfactor: float,
    production_total_ns: float,
    nproc: int,
    gromacs_stage_layouts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    md_dir = proj / "MD_GMX"
    prod_base_dt_ps = 0.0005
    prod_nsteps = max(4, int(round(float(production_total_ns) * 1000.0 / prod_base_dt_ps)))
    prod_nsteps += (-prod_nsteps) % 4
    eq09_mode = (
        os.environ.get("GROMACS_BATCH_EQ09_RESCUE_MODE", "standard")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if eq09_mode not in {"standard", "actual_1474101"}:
        raise ValueError(
            "GROMACS_BATCH_EQ09_RESCUE_MODE must be standard or "
            f"actual_1474101; got {eq09_mode!r}"
        )
    if eq09_mode == "actual_1474101" and not math.isclose(
        float(eqfactor), 0.83, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            "The actual 1474101 Eq09 rescue is defined only for eqfactor=0.83; "
            f"got {eqfactor}"
        )

    eq01_specs: list[dict[str, Any]] = [
        {
            "name": "eq01_soft_langevin_10ps",
            "phase": "equilibration",
            "kind": "md",
            "ensemble": "nve_langevin",
            "thermostat": "none",
            "duration_ps": 10.0,
            "base_dt_ps": 0.000125,
            "outer_timestep_fs": 0.5,
            "tau_t_ps": 0.05,
            "gen_vel": True,
            "gen_seed": 520419,
            "kspace_compute": "no",
            "temp_start_k": 353.0,
            "temp_end_k": 353.0,
            "soft_start": {
                "displacement_limit_nm": 0.010,
                "langevin_temp_k": 353.0,
                "langevin_damp_ps": 0.05,
                "langevin_seed": 97531,
                "zero_net_force": True,
            },
        },
        {
            "name": "eq01_nvt_40ps",
            "phase": "equilibration",
            "kind": "md",
            "ensemble": "nvt",
            "duration_ps": 40.0,
            "base_dt_ps": 0.000125,
            "outer_timestep_fs": 0.5,
            "tau_t_ps": 0.05,
            "kspace_compute": "no",
            "temp_start_k": 353.0,
            "temp_end_k": 353.0,
        },
    ]
    if eq09_mode == "actual_1474101":
        eq09_specs: list[dict[str, Any]] = [
            {
                "name": "eq09_rescue_recondition_50ps",
                "phase": "equilibration",
                "kind": "md",
                "ensemble": "nve_langevin",
                "thermostat": "none",
                "duration_ps": 50.0,
                "eqfactor_applied": True,
                "disable_chunking": True,
                "base_dt_ps": 0.0000625,
                "outer_timestep_fs": 0.25,
                "tau_t_ps": 0.05,
                "temp_start_k": 353.0,
                "temp_end_k": 353.0,
                "soft_start": {
                    "displacement_limit_nm": 0.0005,
                    "langevin_temp_k": 353.0,
                    "langevin_damp_ps": 0.05,
                    "langevin_seed": 1474101,
                    "zero_net_force": True,
                },
            },
        ]
        for segment_index, (pressure_start, pressure_end) in enumerate(
            ((1.0, 500.0), (500.0, 1500.0), (1500.0, 2500.0), (2500.0, 3213.04819)),
            start=1,
        ):
            eq09_specs.append(
                {
                    "name": f"eq09_rescue_npt_segment{segment_index:02d}_50ps",
                    "phase": "equilibration",
                    "kind": "md",
                    "ensemble": "npt",
                    "duration_ps": 50.0,
                    "eqfactor_applied": True,
                    "disable_chunking": True,
                    "base_dt_ps": 0.0000625,
                    "outer_timestep_fs": 0.25,
                    "tau_t_ps": 0.1,
                    "tau_p_ps": 5.0,
                    "pressure_start_atm": pressure_start,
                    "pressure_end_atm": pressure_end,
                    "pressure_atm": pressure_end,
                    "temp_start_k": 353.0,
                    "temp_end_k": 353.0,
                    "lammps_fix_nh_drag": 4.0,
                }
            )
        eq09_specs.append(
            {
                "name": "eq09_npt_compress_chunk0002_49ps",
                "phase": "equilibration",
                "kind": "md",
                "ensemble": "npt",
                "duration_ps": 49.0,
                "eqfactor_applied": True,
                "disable_chunking": True,
                "base_dt_ps": 0.0005,
                "outer_timestep_fs": 2.0,
                "tau_t_ps": 0.2,
                "tau_p_ps": 2.0,
                "pressure_start_atm": 3213.04819,
                "pressure_end_atm": 4000.0,
                "pressure_atm": 4000.0,
                "temp_start_k": 353.0,
                "temp_end_k": 353.0,
            }
        )
    else:
        eq09_specs = [
            {"name": "eq09_npt_compress_300ps", "phase": "equilibration", "kind": "md", "ensemble": "npt", "duration_ps": 300.0, "base_dt_ps": 0.0005, "outer_timestep_fs": 2.0, "tau_t_ps": 0.2, "tau_p_ps": 2.0, "pressure_start_atm": 1.0, "pressure_end_atm": 4000.0, "pressure_atm": 4000.0, "temp_start_k": 353.0, "temp_end_k": 353.0},
        ]
    base_stage_specs: list[dict[str, Any]] = [
        *eq01_specs,
        {"name": "eq02_npt_compress_100ps", "phase": "equilibration", "kind": "md", "ensemble": "npt", "duration_ps": 100.0, "base_dt_ps": 0.000125, "outer_timestep_fs": 0.5, "tau_t_ps": 0.05, "tau_p_ps": 0.5, "pressure_start_atm": 1.0, "pressure_end_atm": 1000.0, "pressure_atm": 1000.0, "kspace_compute": "no", "temp_start_k": 353.0, "temp_end_k": 353.0},
        {"name": "eq03_pre_2fs_minimize", "phase": "equilibration", "kind": "em"},
        {"name": "eq04_npt_compress_500ps", "phase": "equilibration", "kind": "md", "ensemble": "npt", "duration_ps": 500.0, "base_dt_ps": 0.0005, "outer_timestep_fs": 2.0, "tau_t_ps": 0.2, "tau_p_ps": 2.0, "pressure_start_atm": 1.0, "pressure_end_atm": 4000.0, "pressure_atm": 4000.0, "gen_vel": True, "gen_seed": 63862, "temp_start_k": 353.0, "temp_end_k": 353.0},
        {"name": "eq05_npt_hold_hi_400ps", "phase": "equilibration", "kind": "md", "ensemble": "npt", "duration_ps": 400.0, "base_dt_ps": 0.0005, "outer_timestep_fs": 2.0, "tau_t_ps": 0.2, "tau_p_ps": 2.0, "pressure_start_atm": 4000.0, "pressure_end_atm": 4000.0, "pressure_atm": 4000.0, "temp_start_k": 353.0, "temp_end_k": 353.0},
        {"name": "eq06_npt_decompress_600ps", "phase": "equilibration", "kind": "md", "ensemble": "npt", "duration_ps": 600.0, "base_dt_ps": 0.0005, "outer_timestep_fs": 2.0, "tau_t_ps": 0.2, "tau_p_ps": 2.0, "pressure_start_atm": 4000.0, "pressure_end_atm": 1.0, "pressure_atm": 1.0, "temp_start_k": 353.0, "temp_end_k": 353.0},
        {"name": "eq07_npt_heat_400ps", "phase": "equilibration", "kind": "md", "ensemble": "npt", "duration_ps": 400.0, "base_dt_ps": 0.0005, "outer_timestep_fs": 2.0, "tau_t_ps": 0.2, "tau_p_ps": 2.0, "pressure_start_atm": 1.0, "pressure_end_atm": 1.0, "pressure_atm": 1.0, "temp_start_k": 353.0, "temp_end_k": 453.0},
        {"name": "eq08_npt_cool_400ps", "phase": "equilibration", "kind": "md", "ensemble": "npt", "duration_ps": 400.0, "base_dt_ps": 0.0005, "outer_timestep_fs": 2.0, "tau_t_ps": 0.2, "tau_p_ps": 2.0, "pressure_start_atm": 1.0, "pressure_end_atm": 1.0, "pressure_atm": 1.0, "temp_start_k": 453.0, "temp_end_k": 353.0},
        *eq09_specs,
        {"name": "eq10_npt_decompress_300ps", "phase": "equilibration", "kind": "md", "ensemble": "npt", "duration_ps": 300.0, "base_dt_ps": 0.0005, "outer_timestep_fs": 2.0, "tau_t_ps": 0.2, "tau_p_ps": 2.0, "pressure_start_atm": 4000.0, "pressure_end_atm": 1.0, "pressure_atm": 1.0, "temp_start_k": 353.0, "temp_end_k": 353.0},
        {"name": "eq11_nvt_800ps", "phase": "equilibration", "kind": "md", "ensemble": "nvt", "duration_ps": 800.0, "base_dt_ps": 0.0005, "outer_timestep_fs": 2.0, "tau_t_ps": 0.2, "temp_start_k": 353.0, "temp_end_k": 353.0},
        {"name": "eq12_npt_1200ps", "phase": "equilibration", "kind": "md", "ensemble": "npt", "duration_ps": 1200.0, "base_dt_ps": 0.0005, "outer_timestep_fs": 2.0, "tau_t_ps": 0.2, "tau_p_ps": 2.0, "pressure_start_atm": 1.0, "pressure_end_atm": 1.0, "pressure_atm": 1.0, "temp_start_k": 353.0, "temp_end_k": 353.0},
        {"name": "eq13_nvt_fixed_volume_1000ps", "phase": "equilibration", "kind": "md", "ensemble": "nvt", "duration_ps": 1000.0, "base_dt_ps": 0.0005, "outer_timestep_fs": 2.0, "tau_t_ps": 0.2, "temp_start_k": 353.0, "temp_end_k": 353.0},
        {"name": "prod_nvt", "phase": "production", "kind": "md", "ensemble": "nvt", "duration_ps": float(production_total_ns) * 1000.0, "base_dt_ps": prod_base_dt_ps, "tau_t_ps": 0.2, "nsteps_override": prod_nsteps, "production": True},
    ]
    stage_specs = _expand_polygen_equil_stage_specs(
        base_stage_specs,
        eqfactor=float(eqfactor),
    )
    eq12_window_mode = (
        os.environ.get("GROMACS_BATCH_EQ12_AVERAGE_WINDOW_MODE", "legacy_actual")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if eq12_window_mode not in {"legacy_actual", "final"}:
        raise ValueError(
            "GROMACS_BATCH_EQ12_AVERAGE_WINDOW_MODE must be legacy_actual or final; "
            f"got {eq12_window_mode!r}"
        )
    eq12_legacy_window = _polygen_legacy_eq12_average_window(
        base_stage_specs,
        eqfactor=float(eqfactor),
    )
    eq12_stage_spec = next(
        spec for spec in stage_specs if str(spec.get("name")) == "eq12_npt_1200ps"
    )
    eq12_duration_ps = float(eq12_stage_spec["duration_ps"])
    override_start_raw = os.environ.get("GROMACS_BATCH_EQ12_WINDOW_START_PS")
    override_end_raw = os.environ.get("GROMACS_BATCH_EQ12_WINDOW_END_PS")
    if (override_start_raw is None) != (override_end_raw is None):
        raise ValueError(
            "GROMACS_BATCH_EQ12_WINDOW_START_PS and "
            "GROMACS_BATCH_EQ12_WINDOW_END_PS must be set together"
        )
    if override_start_raw is not None and override_end_raw is not None:
        if eq12_window_mode != "legacy_actual":
            raise ValueError(
                "Explicit Eq12 window bounds require "
                "GROMACS_BATCH_EQ12_AVERAGE_WINDOW_MODE=legacy_actual"
            )
        try:
            eq12_window_start_ps = float(override_start_raw)
            eq12_window_end_ps = float(override_end_raw)
        except ValueError as exc:
            raise ValueError("Eq12 window overrides must be finite numbers in ps") from exc
        eq12_window_source = "environment_override"
    elif eq12_window_mode == "legacy_actual":
        eq12_window_start_ps = float(eq12_legacy_window["relative_start_ps"])
        eq12_window_end_ps = float(eq12_legacy_window["relative_end_ps"])
        eq12_window_source = "generated_schedule"
    else:
        eq12_window_end_ps = eq12_duration_ps
        eq12_window_start_ps = eq12_window_end_ps - 500.0 * float(eqfactor)
        eq12_window_source = "final_window_opt_in"
    if (
        not math.isfinite(eq12_window_start_ps)
        or not math.isfinite(eq12_window_end_ps)
        or eq12_window_start_ps < 0.0
        or eq12_window_end_ps <= eq12_window_start_ps
        or eq12_window_end_ps > eq12_duration_ps + 1.0e-9
    ):
        raise ValueError(
            "Eq12 average window must lie inside the effective stage: "
            f"window={eq12_window_start_ps}..{eq12_window_end_ps} ps, "
            f"stage=0..{eq12_duration_ps} ps"
        )
    for spec in stage_specs:
        if str(spec.get("name")) == "eq12_npt_1200ps":
            spec["eq12_average_window_mode"] = eq12_window_mode
            spec["eq12_average_window_source"] = eq12_window_source
            spec["eq12_average_window_start_ps"] = eq12_window_start_ps
            spec["eq12_average_window_end_ps"] = eq12_window_end_ps
            spec["eq12_legacy_average_window"] = eq12_legacy_window

    rendered_stages: list[dict[str, Any]] = []
    for spec in stage_specs:
        segment_start_ps = float(spec.get("segment_start_ps", 0.0) or 0.0)
        base_dt_ps = float(spec.get("base_dt_ps", 0.0) or 0.0)
        init_step = (
            int(round(segment_start_ps / base_dt_ps))
            if spec["kind"] == "md" and base_dt_ps > 0.0
            else 0
        )
        if spec["kind"] == "em":
            text, nsteps, dt_ps = _polygen_em_mdp(str(spec["name"]))
        else:
            text, nsteps, dt_ps = _polygen_md_mdp(
                name=str(spec["name"]),
                ensemble=str(spec["ensemble"]),
                duration_ps=float(spec["duration_ps"]),
                base_dt_ps=float(spec["base_dt_ps"]),
                eqfactor=(
                    1.0
                    if spec["phase"] == "production" or bool(spec.get("eqfactor_applied", False))
                    else float(eqfactor)
                ),
                temp_k=float(spec.get("temp_k", temperature_k)),
                pressure_atm=None if spec.get("pressure_atm") is None else float(spec["pressure_atm"]),
                tau_t_ps=float(spec["tau_t_ps"]),
                tau_p_ps=None if spec.get("tau_p_ps") is None else float(spec["tau_p_ps"]),
                gen_vel=bool(spec.get("gen_vel", False)),
                gen_seed=None if spec.get("gen_seed") is None else int(spec["gen_seed"]),
                production=bool(spec.get("production", False)),
                nsteps_override=spec.get("nsteps_override"),
                temp_start_k=None if spec.get("temp_start_k") is None else float(spec["temp_start_k"]),
                temp_end_k=None if spec.get("temp_end_k") is None else float(spec["temp_end_k"]),
                pressure_start_atm=None if spec.get("pressure_start_atm") is None else float(spec["pressure_start_atm"]),
                pressure_end_atm=None if spec.get("pressure_end_atm") is None else float(spec["pressure_end_atm"]),
                nstlist=int(spec.get("nstlist", 80)),
                init_step=init_step,
                schedule_start_ps=segment_start_ps,
                thermostat=str(spec.get("thermostat", "nose-hoover")),
            )
        out_name = f"{spec['name']}.mdp"
        rendered_path = md_dir / out_name
        rendered_path.write_text(text, encoding="utf-8")
        rendered_stages.append(
            {
                "name": spec["name"],
                "phase": spec["phase"],
                "kind": spec["kind"],
                "mdp_path": str(rendered_path),
                "nsteps": int(nsteps),
                "init_step": int(init_step),
                "dt_ps": float(dt_ps),
                "outer_timestep_fs": float(spec.get("outer_timestep_fs", dt_ps * 4000.0)),
                "ensemble": spec.get("ensemble"),
                "thermostat": spec.get("thermostat", "nose-hoover"),
                "tau_t_ps": spec.get("tau_t_ps"),
                "tau_p_ps": spec.get("tau_p_ps"),
                "temp_start_k": spec.get("temp_start_k", spec.get("temp_k", temperature_k)),
                "temp_end_k": spec.get("temp_end_k", spec.get("temp_k", temperature_k)),
                "pressure_start_atm": spec.get("pressure_start_atm"),
                "pressure_end_atm": spec.get("pressure_end_atm", spec.get("pressure_atm")),
                "kspace_compute": spec.get("kspace_compute"),
                "gen_seed": spec.get("gen_seed") if bool(spec.get("gen_vel", False)) else None,
                "base_index": spec.get("base_index"),
                "segment_index": spec.get("segment_index", 1),
                "segment_count": spec.get("segment_count", 1),
                "segment_start_ps": segment_start_ps,
                "soft_start": spec.get("soft_start"),
                "lammps_fix_nh_drag": spec.get("lammps_fix_nh_drag", 0.0),
                "eq12_average_window_mode": spec.get("eq12_average_window_mode"),
                "eq12_average_window_source": spec.get("eq12_average_window_source"),
                "eq12_average_window_start_ps": spec.get("eq12_average_window_start_ps"),
                "eq12_average_window_end_ps": spec.get("eq12_average_window_end_ps"),
                "eq12_legacy_average_window": spec.get("eq12_legacy_average_window"),
            }
        )

    final_equil_stage = next(
        (stage for stage in reversed(rendered_stages) if stage["phase"] == "equilibration"),
        None,
    )
    production_start_structure = (
        f"{final_equil_stage['name']}.gro" if final_equil_stage is not None else "05_nvt_relaxed.gro"
    )
    production_start_state_trr = (
        f"{final_equil_stage['name']}_final_state.trr"
        if final_equil_stage is not None and final_equil_stage.get("kind") == "md"
        else None
    )
    if eq12_window_mode == "legacy_actual":
        eq12_parity_note = (
            "The eq12 cell handoff uses the legacy LAMMPS fix ave/time-aligned "
            f"window ({eq12_window_start_ps:.6f}..{eq12_window_end_ps:.6f} ps "
            f"relative to Eq12; source={eq12_window_source}). "
            "GROMACS stores Box-X/Y/Z in EDR on every matching outer step and "
            "uses the same discrete arithmetic sample window as LAMMPS."
        )
    else:
        eq12_parity_note = (
            "GROMACS_BATCH_EQ12_AVERAGE_WINDOW_MODE=final opts into averaging the final "
            "500*eqfactor ps; this does not reproduce the actual legacy LAMMPS fix "
            "ave/time alignment."
        )

    context = {
        "schema_name": "gromacs_pcff_batch_runtime_context",
        "schema_version": 7,
        "schedule": "polygen_em_handoff",
        "project_dir": str(proj),
        "md_dir": str(md_dir),
        "gmx_binary": str(prepare_report["binary"]["batch_binary"]),
        "gmxlib": prepare_report["binary"].get("gmxlib"),
        "temperature_k": float(temperature_k),
        "eqfactor": float(eqfactor),
        "eq09_rescue_mode": eq09_mode,
        "production_total_ns": float(production_total_ns),
        "nproc": int(nproc),
        "gromacs_stage_layouts": dict(gromacs_stage_layouts or {}),
        "prepare_report_path": str(proj / "build" / "gromacs_pcff" / "gromacs_pcff_prepare_report.json"),
        "stages": rendered_stages,
        "production_start_structure": production_start_structure,
        "production_start_state_trr": production_start_state_trr,
        "overall_status": "ready_for_md",
        "failure_reason": None,
        "parity_limitations": [
            "Eq01 uses the LAMMPS-style 20% displacement-limited Langevin prelude followed by a fresh 80% Nose-Hoover NVT stage; stochastic coordinates are not expected to be byte-identical across engines.",
            (
                "Eq09 reproduces the actual completed 1474101 0.25 fs rescue schedule."
                if eq09_mode == "actual_1474101"
                else "Eq09 uses the generated standard schedule rather than a trajectory-specific rescue."
            ),
            "This worker schedule does not reproduce LAMMPS velocity-create dumps byte-for-byte.",
            "Temperature ramps use simulated annealing and pressure ramps use GMX_PCFF_REFP_RAMP_*; exact stochastic trajectory identity with LAMMPS is not expected.",
            eq12_parity_note,
        ],
    }
    if not (md_dir / "conf.gro").exists() or not (md_dir / "topol.top").exists():
        context["overall_status"] = "blocked"
        context["failure_reason"] = "missing assembled conf.gro/topol.top in MD_GMX"
    _write_json(md_dir / "gromacs_runtime_context.json", context)
    return context


def setup_gromacs_environment(
    proj: Path | str,
    *,
    base_dir: Path | str,
    prepare_report: Mapping[str, Any],
    temperature_k: float,
    eqfactor: float,
    production_total_ns: float,
    nproc: int,
    gromacs_stage_layouts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    proj = Path(proj).resolve()
    base_dir = Path(base_dir).resolve()
    md_dir = proj / "MD_GMX"
    md_dir.mkdir(parents=True, exist_ok=True)
    schedule = os.environ.get("GROMACS_BATCH_SCHEDULE", "stage_manifest").strip() or "stage_manifest"
    if schedule == "polygen_em_handoff":
        return _setup_polygen_em_handoff_environment(
            proj,
            prepare_report=prepare_report,
            temperature_k=temperature_k,
            eqfactor=eqfactor,
            production_total_ns=production_total_ns,
            nproc=nproc,
            gromacs_stage_layouts=gromacs_stage_layouts,
        )
    if schedule != "stage_manifest":
        raise ValueError(f"Unsupported GROMACS_BATCH_SCHEDULE={schedule!r}")
    stage_root = base_dir / "gromacs_pcff"
    manifest = _read_json(_stage_manifest_path(base_dir))

    rendered_stages: list[dict[str, Any]] = []
    for stage in manifest["stages"]:
        template_path = stage_root / stage["template"]
        out_name = stage["output_mdp"]
        rendered_path = md_dir / out_name
        template_text = template_path.read_text(encoding="utf-8")
        nsteps = _compute_stage_nsteps(stage, eqfactor=eqfactor, production_total_ns=production_total_ns)
        rendered_text = _render_template(
            template_text,
            {
                "__REF_T__": f"{float(temperature_k):.3f}",
                "__GEN_TEMP__": f"{float(temperature_k):.3f}",
                "__NSTEPS__": str(nsteps),
            },
        )
        rendered_path.write_text(rendered_text, encoding="utf-8")
        rendered_stages.append(
            {
                "name": stage["name"],
                "phase": stage["phase"],
                "mdp_path": str(rendered_path),
                "nsteps": nsteps,
                "dt_ps": float(stage["dt_ps"]),
            }
        )

    context = {
        "schema_name": "gromacs_pcff_batch_runtime_context",
        "schema_version": 1,
        "project_dir": str(proj),
        "md_dir": str(md_dir),
        "gmx_binary": str(prepare_report["binary"]["batch_binary"]),
        "gmxlib": prepare_report["binary"].get("gmxlib"),
        "temperature_k": float(temperature_k),
        "eqfactor": float(eqfactor),
        "production_total_ns": float(production_total_ns),
        "nproc": int(nproc),
        "gromacs_stage_layouts": dict(gromacs_stage_layouts or {}),
        "prepare_report_path": str(proj / "build" / "gromacs_pcff" / "gromacs_pcff_prepare_report.json"),
        "stages": rendered_stages,
        "overall_status": "ready_for_md",
        "failure_reason": None,
    }

    prepare_status = str(prepare_report["workflow"]["overall_status"])
    if prepare_status not in {"ready_for_mixed_system_assembly", "ready_for_md"}:
        context["overall_status"] = "blocked"
        context["failure_reason"] = str(prepare_report["workflow"]["failure_reason"])
    elif not (md_dir / "conf.gro").exists() or not (md_dir / "topol.top").exists():
        context["overall_status"] = "blocked"
        context["failure_reason"] = "missing assembled conf.gro/topol.top in MD_GMX"

    _write_json(md_dir / "gromacs_runtime_context.json", context)
    return context


def _run_cmd(cmd: list[str], *, cwd: Path, env: Mapping[str, str] | None = None) -> None:
    print(f"\n$ (cwd={cwd}) {' '.join(map(shlex.quote, cmd))}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=dict(env) if env is not None else None, check=True)


def _gromacs_checkpoint_state(
    *,
    gmx: str,
    checkpoint: Path,
    cwd: Path,
    env: Mapping[str, str],
) -> dict[str, int]:
    """Read the authoritative integration step and part from a checkpoint."""

    proc = subprocess.run(
        [gmx, "dump", "-cp", str(checkpoint)],
        cwd=str(cwd),
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Cannot inspect GROMACS checkpoint {checkpoint}; gmx dump returned "
            f"{proc.returncode}: {proc.stdout[-2000:]}"
        )
    step_match = re.search(r"^step\s*=\s*(-?\d+)\s*$", proc.stdout, flags=re.MULTILINE)
    part_match = re.search(
        r"^simulation part #\s*=\s*(\d+)\s*$",
        proc.stdout,
        flags=re.MULTILINE,
    )
    if step_match is None:
        raise RuntimeError(f"Cannot find integration step in GROMACS checkpoint {checkpoint}")
    if part_match is None:
        raise RuntimeError(f"Cannot find simulation part in GROMACS checkpoint {checkpoint}")
    return {
        "step": int(step_match.group(1)),
        "simulation_part": int(part_match.group(1)),
    }


def _gromacs_checkpoint_step(
    *,
    gmx: str,
    checkpoint: Path,
    cwd: Path,
    env: Mapping[str, str],
) -> int:
    return _gromacs_checkpoint_state(
        gmx=gmx,
        checkpoint=checkpoint,
        cwd=cwd,
        env=env,
    )["step"]


def _stage_expected_checkpoint_step(stage: Mapping[str, Any]) -> int:
    return int(stage.get("init_step", 0) or 0) + int(stage.get("nsteps", 0) or 0)


def _final_state_trr_path(md_dir: Path, stage: Mapping[str, Any]) -> Path:
    return md_dir / f"{stage['name']}_final_state.trr"


def _materialize_final_state_trr(
    *,
    gmx: str,
    md_dir: Path,
    stage: Mapping[str, Any],
    checkpoint: Path,
    env: Mapping[str, str],
) -> Path:
    """Export only x/v/box from a verified final checkpoint to one-frame TRR.

    A checkpoint also contains Nose-Hoover/MTTK extended state, which must not
    leak across different LAMMPS base stages because those stages recreate
    their fixes.  A TRR frame carries the full-precision physical state but no
    thermostat/barostat history, making it the appropriate ``grompp -t``
    handoff alongside the ordinary ``-c`` structure.
    """

    checkpoint_state = _gromacs_checkpoint_state(
        gmx=gmx,
        checkpoint=checkpoint,
        cwd=md_dir,
        env=env,
    )
    expected_step = _stage_expected_checkpoint_step(stage)
    if checkpoint_state["step"] < expected_step:
        raise RuntimeError(
            f"Cannot export final state for {stage.get('name')}: checkpoint step "
            f"{checkpoint_state['step']} is before {expected_step}"
        )
    destination = _final_state_trr_path(md_dir, stage)
    temporary = destination.with_name(f".{destination.name}.tmp.trr")
    temporary.unlink(missing_ok=True)
    _run_cmd(
        [
            gmx,
            "convert-trj",
            "-f",
            str(checkpoint),
            "-o",
            str(temporary),
            "-vel",
            "always",
            "-force",
            "never",
        ],
        cwd=md_dir,
        env=env,
    )
    if not temporary.exists() or temporary.stat().st_size == 0:
        raise RuntimeError(
            f"GROMACS did not create the final-state TRR for {stage.get('name')}"
        )
    temporary.replace(destination)
    _write_json(
        destination.with_suffix(".json"),
        {
            "schema_name": "gromacs_full_precision_stage_handoff",
            "schema_version": 1,
            "stage": str(stage.get("name")),
            "source_checkpoint": str(checkpoint),
            "checkpoint_step": checkpoint_state["step"],
            "simulation_part": checkpoint_state["simulation_part"],
            "output_trr": str(destination),
            "payload": "coordinates_velocities_box_only",
        },
    )
    return destination


def _canonicalize_cross_chunk_gro(
    *,
    gmx: str,
    md_dir: Path,
    stage: Mapping[str, Any],
    checkpoint: Path,
    env: Mapping[str, str],
) -> tuple[Path, str]:
    """Copy a ``-noappend`` part GRO to the stable stage-name GRO path."""

    deffnm = str(stage["name"])
    canonical = md_dir / f"{deffnm}.gro"
    if int(stage.get("segment_index", 1) or 1) <= 1:
        return canonical, deffnm
    if not checkpoint.exists():
        return canonical, deffnm
    checkpoint_state = _gromacs_checkpoint_state(
        gmx=gmx,
        checkpoint=checkpoint,
        cwd=md_dir,
        env=env,
    )
    output_stem = f"{deffnm}.part{checkpoint_state['simulation_part']:04d}"
    part_gro = md_dir / f"{output_stem}.gro"
    if not part_gro.exists():
        raise RuntimeError(
            f"Checkpoint {checkpoint} reports simulation part "
            f"{checkpoint_state['simulation_part']}, but {part_gro} is missing"
        )
    shutil.copy2(part_gro, canonical)
    return canonical, output_stem


def _require_completed_md_stage(
    *,
    gmx: str,
    md_dir: Path,
    stage: Mapping[str, Any],
    gro_path: Path,
    checkpoint: Path,
    env: Mapping[str, str],
) -> int:
    """Require both a final structure and a checkpoint at the requested step."""

    if not gro_path.exists():
        checkpoint_note = (
            f" Checkpoint preserved at {checkpoint}; rerun with resume-existing."
            if checkpoint.exists()
            else " No checkpoint was produced."
        )
        raise RuntimeError(
            f"GROMACS mdrun returned without the required final structure {gro_path}."
            f"{checkpoint_note}"
        )
    if not checkpoint.exists():
        raise RuntimeError(
            f"Cannot verify completed MD stage {stage.get('name')}: missing checkpoint {checkpoint}"
        )
    checkpoint_step = _gromacs_checkpoint_step(
        gmx=gmx,
        checkpoint=checkpoint,
        cwd=md_dir,
        env=env,
    )
    expected_step = _stage_expected_checkpoint_step(stage)
    if checkpoint_step < expected_step:
        raise RuntimeError(
            f"GROMACS stage {stage.get('name')} stopped at step {checkpoint_step} before "
            f"requested step {expected_step}; checkpoint is resumable with resume-existing"
        )
    return checkpoint_step


def _read_gromacs_box_series(path: Path) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8", errors="ignore").splitlines(),
        start=1,
    ):
        stripped = raw.strip()
        if not stripped or stripped.startswith(("#", "@")):
            continue
        values = stripped.split()
        if len(values) < 4:
            continue
        try:
            row = tuple(float(value) for value in values[:4])
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid Box-X/Box-Y/Box-Z row at {path}:{line_number}: {raw!r}"
            ) from exc
        if not all(math.isfinite(value) for value in row):
            raise RuntimeError(
                f"Non-finite Box-X/Box-Y/Box-Z row at {path}:{line_number}: {raw!r}"
            )
        if any(value <= 0.0 for value in row[1:]):
            raise RuntimeError(
                f"Non-positive box length at {path}:{line_number}: {raw!r}"
            )
        if rows and row[0] < rows[-1][0] - 1.0e-9:
            raise RuntimeError(
                f"Non-monotonic box time series at {path}:{line_number}: "
                f"{row[0]} ps follows {rows[-1][0]} ps"
            )
        if rows and math.isclose(row[0], rows[-1][0], rel_tol=0.0, abs_tol=1.0e-9):
            # An appended trajectory can contain the checkpoint frame twice.
            # Keep the later record so the integration interval is not zero.
            rows[-1] = row
        else:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No Box-X/Box-Y/Box-Z rows found in {path}")
    return rows


def _late_window_box_average(
    rows: list[tuple[float, float, float, float]],
    *,
    requested_window_ps: float,
) -> dict[str, Any]:
    """Return a time-weighted late-window box average.

    LAMMPS samples ``lx`` every integration step. GROMACS box streams are
    lower frequency and can end with a short, irregular interval, so an
    unweighted mean of saved frames biases that final interval. Linear
    interpolation at the window boundary followed by trapezoidal integration
    is the closest reproducible approximation available from the saved stream.
    """

    requested_window_ps = float(requested_window_ps)
    if not math.isfinite(requested_window_ps) or requested_window_ps <= 0.0:
        raise RuntimeError(
            f"eq12 average-cell window must be positive; got {requested_window_ps} ps"
        )

    if len(rows) < 2:
        raise RuntimeError("eq12 average-cell handoff requires at least two box samples")
    first_time_ps = float(rows[0][0])
    end_time_ps = float(rows[-1][0])
    available_window_ps = end_time_ps - first_time_ps
    if available_window_ps <= 0.0:
        raise RuntimeError(
            f"eq12 box series has no positive time span: {first_time_ps}..{end_time_ps} ps"
        )
    window_ps = min(available_window_ps, requested_window_ps)
    result = _box_window_average(
        rows,
        start_time_ps=end_time_ps - window_ps,
        end_time_ps=end_time_ps,
    )
    result["requested_window_ps"] = requested_window_ps
    return result


def _box_window_average(
    rows: list[tuple[float, float, float, float]],
    *,
    start_time_ps: float,
    end_time_ps: float,
) -> dict[str, Any]:
    """Time-average a box series between two possibly unsampled times."""

    if len(rows) < 2:
        raise RuntimeError("eq12 average-cell handoff requires at least two box samples")
    start_time_ps = float(start_time_ps)
    end_time_ps = float(end_time_ps)
    if not math.isfinite(start_time_ps) or not math.isfinite(end_time_ps):
        raise RuntimeError(
            f"eq12 average-cell bounds must be finite: {start_time_ps}..{end_time_ps} ps"
        )
    if end_time_ps <= start_time_ps:
        raise RuntimeError(
            f"eq12 average-cell window must have positive span: "
            f"{start_time_ps}..{end_time_ps} ps"
        )
    source_start_ps = float(rows[0][0])
    source_end_ps = float(rows[-1][0])
    if start_time_ps < source_start_ps - 1.0e-9 or end_time_ps > source_end_ps + 1.0e-9:
        raise RuntimeError(
            f"eq12 requested box window {start_time_ps}..{end_time_ps} ps lies outside "
            f"the saved series {source_start_ps}..{source_end_ps} ps"
        )

    def interpolate(time_ps: float) -> tuple[float, float, float, float]:
        for index, row in enumerate(rows):
            if math.isclose(row[0], time_ps, rel_tol=0.0, abs_tol=1.0e-9):
                return (time_ps, row[1], row[2], row[3])
            if row[0] > time_ps:
                if index == 0:
                    break
                left = rows[index - 1]
                span = row[0] - left[0]
                if span <= 0.0:
                    break
                fraction = (time_ps - left[0]) / span
                return (
                    time_ps,
                    *(
                        left[axis] + (row[axis] - left[axis]) * fraction
                        for axis in range(1, 4)
                    ),
                )
        raise RuntimeError(f"Cannot interpolate eq12 box at {time_ps} ps")

    points = [interpolate(start_time_ps)]
    points.extend(row for row in rows if start_time_ps < row[0] < end_time_ps)
    points.append(interpolate(end_time_ps))

    integrals = [0.0, 0.0, 0.0]
    for left, right in zip(points, points[1:]):
        dt_ps = right[0] - left[0]
        if dt_ps <= 0.0:
            raise RuntimeError("eq12 average-cell integration points are not strictly increasing")
        for axis in range(3):
            integrals[axis] += 0.5 * (left[axis + 1] + right[axis + 1]) * dt_ps
    window_ps = end_time_ps - start_time_ps
    means = [integral / window_ps for integral in integrals]
    return {
        "window_start_ps": start_time_ps,
        "window_end_ps": end_time_ps,
        "window_ps": window_ps,
        "source_sample_count": sum(
            start_time_ps - 1.0e-9 <= row[0] <= end_time_ps + 1.0e-9 for row in rows
        ),
        "integration_point_count": len(points),
        "averaging_method": "linear_interpolation_trapezoidal_time_weighted",
        "mean_box_x_nm": means[0],
        "mean_box_y_nm": means[1],
        "mean_box_z_nm": means[2],
    }


def _box_discrete_sample_average(
    rows: list[tuple[float, float, float, float]],
    *,
    start_time_ps: float,
    end_time_ps: float,
    sample_spacing_ps: float,
    expected_count: int,
) -> dict[str, Any]:
    """Match LAMMPS fix ave/time's inclusive arithmetic sample mean."""

    spacing = float(sample_spacing_ps)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise RuntimeError(f"Invalid discrete Eq12 sample spacing {spacing} ps")
    tolerance = max(1.0e-9, spacing * 1.0e-6)
    selected = [
        row
        for row in rows
        if start_time_ps - tolerance <= row[0] <= end_time_ps + tolerance
    ]
    if not selected:
        raise RuntimeError(
            f"No Eq12 samples in discrete window {start_time_ps}..{end_time_ps} ps"
        )
    if not math.isclose(selected[0][0], start_time_ps, rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(
            f"Eq12 first discrete sample is {selected[0][0]} ps, expected {start_time_ps} ps"
        )
    if not math.isclose(selected[-1][0], end_time_ps, rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(
            f"Eq12 last discrete sample is {selected[-1][0]} ps, expected {end_time_ps} ps"
        )
    if len(selected) != int(expected_count):
        raise RuntimeError(
            f"Eq12 discrete sample count is {len(selected)}, expected {expected_count}"
        )
    for left, right in zip(selected, selected[1:]):
        if not math.isclose(
            right[0] - left[0], spacing, rel_tol=0.0, abs_tol=tolerance
        ):
            raise RuntimeError(
                "Eq12 discrete box stream has a cadence gap: "
                f"{left[0]} -> {right[0]} ps, expected {spacing} ps"
            )
    means = [
        sum(row[axis] for row in selected) / len(selected)
        for axis in range(1, 4)
    ]
    return {
        "window_start_ps": float(start_time_ps),
        "window_end_ps": float(end_time_ps),
        "window_ps": float(end_time_ps) - float(start_time_ps),
        "source_sample_count": len(selected),
        "integration_point_count": len(selected),
        "averaging_method": "inclusive_discrete_arithmetic_mean",
        "sample_spacing_ps": spacing,
        "mean_box_x_nm": means[0],
        "mean_box_y_nm": means[1],
        "mean_box_z_nm": means[2],
    }


def _write_isotropically_remapped_gro(source: Path, destination: Path, target_box_nm: float) -> dict[str, Any]:
    lines = source.read_text(encoding="utf-8", errors="strict").splitlines()
    if len(lines) < 4:
        raise RuntimeError(f"Malformed GRO file: {source}")
    natoms = int(lines[1].strip())
    if len(lines) < natoms + 3:
        raise RuntimeError(f"Truncated GRO file: {source}")
    box_values = [float(value) for value in lines[2 + natoms].split()]
    source_box = box_values[:3]
    if len(source_box) != 3 or any(value <= 0.0 for value in source_box):
        raise RuntimeError(f"Invalid orthorhombic box in {source}: {source_box}")
    if len(box_values) not in (3, 9):
        raise RuntimeError(f"Unsupported GRO box record in {source}: {box_values}")
    if len(box_values) == 9 and any(abs(value) > 1.0e-12 for value in box_values[3:]):
        raise RuntimeError(
            f"eq12 isotropic remap does not support a triclinic source box in {source}: {box_values}"
        )
    if not math.isfinite(float(target_box_nm)) or float(target_box_nm) <= 0.0:
        raise RuntimeError(f"Invalid eq12 target box length: {target_box_nm}")
    scale = [float(target_box_nm) / value for value in source_box]

    out = [f"{lines[0]} | eq12 time-averaged isotropic cell", lines[1]]
    for atom_index, raw in enumerate(lines[2 : 2 + natoms], start=1):
        if len(raw) < 44:
            raise RuntimeError(f"Malformed GRO atom line {atom_index} in {source}: {raw!r}")
        try:
            coordinates = [float(raw[start : start + 8]) for start in (20, 28, 36)]
        except ValueError as exc:
            raise RuntimeError(f"Cannot parse GRO coordinates on atom line {atom_index} in {source}") from exc
        remapped = [value * factor for value, factor in zip(coordinates, scale)]
        velocity_fields = [raw[start : start + 8] for start in (44, 52, 60)] if len(raw) >= 68 else []
        velocities: list[float] = []
        if velocity_fields and all(field.strip() for field in velocity_fields):
            velocities = [float(field) for field in velocity_fields]
        rendered = raw[:20] + "".join(f"{value:18.12f}" for value in remapped)
        if velocities:
            rendered += "".join(f"{value:18.12f}" for value in velocities)
        out.append(rendered)
    out.append("".join(f"{float(target_box_nm):18.12f}" for _ in range(3)))
    destination.write_text("\n".join(out) + "\n", encoding="utf-8")
    return {
        "source_box_nm": source_box,
        "target_box_nm": float(target_box_nm),
        "coordinate_scale": scale,
    }


def _write_isotropically_remapped_trr(
    source: Path,
    destination: Path,
    target_box_nm: float,
) -> dict[str, Any]:
    """Scale x and box in a single-frame TRR while preserving full-precision v.

    The final-state TRR is deliberately one frame exported from a checkpoint.
    Updating the XDR payload in place avoids a text structure round-trip and
    leaves velocities, time, step, and lambda byte-for-byte unchanged.
    """

    if not math.isfinite(float(target_box_nm)) or float(target_box_nm) <= 0.0:
        raise RuntimeError(f"Invalid eq12 target box length: {target_box_nm}")
    payload = bytearray(source.read_bytes())
    offset = 0

    def read_int() -> int:
        nonlocal offset
        if offset + 4 > len(payload):
            raise RuntimeError(f"Truncated TRR header in {source}")
        value = struct.unpack_from(">i", payload, offset)[0]
        offset += 4
        return value

    if read_int() != 1993:
        raise RuntimeError(f"Invalid GROMACS TRR magic number in {source}")
    version_capacity = read_int()
    version_length = read_int()
    if version_capacity <= 0 or version_length <= 0 or version_length >= version_capacity:
        raise RuntimeError(f"Invalid TRR version string header in {source}")
    padded_version_length = version_length + ((-version_length) % 4)
    if offset + padded_version_length > len(payload):
        raise RuntimeError(f"Truncated TRR version string in {source}")
    version = bytes(payload[offset : offset + version_length])
    offset += padded_version_length
    if version != b"GMX_trn_file":
        raise RuntimeError(f"Unsupported TRR version {version!r} in {source}")

    (
        ir_size,
        energy_size,
        box_size,
        virial_size,
        pressure_size,
        topology_size,
        symmetry_size,
        coordinate_size,
        velocity_size,
        force_size,
    ) = (read_int() for _ in range(10))
    natoms = read_int()
    step = read_int()
    _nre = read_int()
    if natoms <= 0:
        raise RuntimeError(f"Invalid atom count {natoms} in {source}")
    if any(value != 0 for value in (ir_size, energy_size, topology_size, symmetry_size)):
        raise RuntimeError(f"Unsupported legacy payload blocks in final-state TRR {source}")

    precision_candidates: list[int] = []
    if box_size:
        if box_size % 9:
            raise RuntimeError(f"Invalid box payload size in {source}")
        precision_candidates.append(box_size // 9)
    for block_size in (coordinate_size, velocity_size, force_size):
        if block_size:
            denominator = natoms * 3
            if block_size % denominator:
                raise RuntimeError(f"Invalid atom-vector payload size in {source}")
            precision_candidates.append(block_size // denominator)
    if not precision_candidates or any(
        value != precision_candidates[0] for value in precision_candidates
    ):
        raise RuntimeError(f"Inconsistent floating-point precision in {source}")
    precision = precision_candidates[0]
    if precision not in (4, 8):
        raise RuntimeError(f"Unsupported TRR floating-point width {precision} in {source}")
    if box_size != 9 * precision or coordinate_size != natoms * 3 * precision:
        raise RuntimeError(f"Final-state TRR lacks box or coordinates: {source}")
    if velocity_size != natoms * 3 * precision:
        raise RuntimeError(f"Final-state TRR lacks full velocities: {source}")

    # Header time and lambda use the same real width as frame vectors.
    if offset + 2 * precision > len(payload):
        raise RuntimeError(f"Truncated TRR time/lambda header in {source}")
    offset += 2 * precision
    box_offset = offset
    coordinate_offset = box_offset + box_size + virial_size + pressure_size
    expected_size = (
        coordinate_offset + coordinate_size + velocity_size + force_size
    )
    if expected_size != len(payload):
        raise RuntimeError(
            f"Expected one complete TRR frame in {source}; parsed {expected_size} of "
            f"{len(payload)} bytes"
        )

    real_format = ">d" if precision == 8 else ">f"

    def read_real(position: int) -> float:
        return float(struct.unpack_from(real_format, payload, position)[0])

    def write_real(position: int, value: float) -> None:
        struct.pack_into(real_format, payload, position, float(value))

    box = [read_real(box_offset + index * precision) for index in range(9)]
    source_box = [box[0], box[4], box[8]]
    off_diagonal = [box[index] for index in (1, 2, 3, 5, 6, 7)]
    if any(value <= 0.0 for value in source_box) or any(
        abs(value) > 1.0e-12 for value in off_diagonal
    ):
        raise RuntimeError(
            f"eq12 isotropic TRR remap requires an orthorhombic source box: {box}"
        )
    scale = [float(target_box_nm) / value for value in source_box]
    for index in range(9):
        write_real(
            box_offset + index * precision,
            float(target_box_nm) if index in (0, 4, 8) else 0.0,
        )
    for atom_index in range(natoms):
        atom_offset = coordinate_offset + atom_index * 3 * precision
        for axis in range(3):
            position = atom_offset + axis * precision
            write_real(position, read_real(position) * scale[axis])

    destination.write_bytes(payload)
    return {
        "source_trr": str(source),
        "output_trr": str(destination),
        "source_box_nm": source_box,
        "target_box_nm": float(target_box_nm),
        "coordinate_scale": scale,
        "natoms": natoms,
        "step": step,
        "precision_bytes": precision,
        "velocities_preserved": True,
    }


def _apply_eq12_average_cell(
    *,
    gmx: str,
    md_dir: Path,
    stage: Mapping[str, Any],
    source_gro: Path,
    source_state_trr: Path | None = None,
    env: Mapping[str, str],
    eqfactor: float,
) -> Path:
    deffnm = str(stage["name"])
    averaged_gro = md_dir / f"{deffnm}_average_cell.gro"
    averaged_state_trr = md_dir / f"{deffnm}_average_cell_final_state.trr"
    edr_path = md_dir / f"{deffnm}.edr"
    xtc_path = md_dir / f"{deffnm}.xtc"
    tpr_path = md_dir / f"{deffnm}.tpr"
    xvg_path = md_dir / f"{deffnm}_box_series.xvg"
    log_path = md_dir / f"{deffnm}_box_series.log"
    if not ((xtc_path.exists() and tpr_path.exists()) or edr_path.exists()):
        raise RuntimeError(f"eq12 average-cell handoff requires {xtc_path} or {edr_path}")

    expected_end_time_ps = float(stage.get("nsteps", 0) or 0) * float(
        stage.get("dt_ps", 0.0) or 0.0
    )
    extraction_logs: list[str] = []
    source_series: Path | None = None
    rows: list[tuple[float, float, float, float]] | None = None
    attempts: list[tuple[Path, list[str], str]] = []
    if xtc_path.exists() and tpr_path.exists():
        attempts.append(
            (
                xtc_path,
                [
                    gmx,
                    "traj",
                    "-f",
                    str(xtc_path),
                    "-s",
                    str(tpr_path),
                    "-ob",
                    str(xvg_path),
                    "-xvg",
                    "none",
                ],
                "System\n",
            )
        )
    if edr_path.exists():
        # The parity schedule writes Eq12 box energies every outer step.  This
        # preserves the exact 2 fs discrete sample stream used by LAMMPS
        # fix ave/time without emitting a very large coordinate trajectory.
        attempts.append(
            (
                edr_path,
                [gmx, "energy", "-f", str(edr_path), "-o", str(xvg_path), "-xvg", "none"],
                "Box-X\nBox-Y\nBox-Z\n0\n",
            )
        )

    for candidate, command, selection in attempts:
        proc = subprocess.run(
            command,
            cwd=str(md_dir),
            env=dict(env),
            input=selection,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        extraction_logs.append(f"$ {' '.join(map(shlex.quote, command))}\n{proc.stdout}")
        if proc.returncode != 0:
            continue
        try:
            candidate_rows = _read_gromacs_box_series(xvg_path)
        except RuntimeError as exc:
            extraction_logs.append(f"Rejected {candidate}: {exc}\n")
            continue
        end_tolerance_ps = max(
            1.0e-6,
            0.5 * float(stage.get("dt_ps", 0.0) or 0.0),
        )
        if (
            expected_end_time_ps > 0.0
            and not math.isclose(
                candidate_rows[-1][0],
                expected_end_time_ps,
                rel_tol=0.0,
                abs_tol=end_tolerance_ps,
            )
        ):
            extraction_logs.append(
                f"Rejected {candidate}: last box time {candidate_rows[-1][0]} ps "
                f"does not match expected stage end {expected_end_time_ps} ps\n"
            )
            continue
        source_series = candidate
        rows = candidate_rows
        break

    log_path.write_text("\n".join(extraction_logs), encoding="utf-8")
    if source_series is None or rows is None:
        raise RuntimeError(f"Cannot extract a complete eq12 box series; see {log_path}")

    window_mode = str(stage.get("eq12_average_window_mode") or "").strip()
    window_source = str(stage.get("eq12_average_window_source") or "").strip()
    legacy_reference = stage.get("eq12_legacy_average_window")
    if window_mode not in {"legacy_actual", "final"} or not window_source:
        raise RuntimeError(
            "eq12 runtime context has no supported average-window mode; "
            "regenerate it before running equilibration"
        )
    if window_mode == "legacy_actual" and not isinstance(legacy_reference, Mapping):
        raise RuntimeError(
            "eq12 legacy_actual averaging requires regenerated runtime context "
            "with eq12_legacy_average_window metadata"
        )
    try:
        window_start_ps = float(stage["eq12_average_window_start_ps"])
        window_end_ps = float(stage["eq12_average_window_end_ps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Malformed Eq12 selected-window metadata in runtime context"
        ) from exc
    if window_mode == "legacy_actual":
        assert isinstance(legacy_reference, Mapping)
        sample_spacing_ps = (
            float(stage.get("outer_timestep_fs", 0.0) or 0.0)
            * float(legacy_reference.get("sample_spacing_outer_steps", 1) or 1)
            / 1000.0
        )
        average = _box_discrete_sample_average(
            rows,
            start_time_ps=window_start_ps,
            end_time_ps=window_end_ps,
            sample_spacing_ps=sample_spacing_ps,
            expected_count=int(legacy_reference["sample_count"]),
        )
    else:
        average = _box_window_average(
            rows,
            start_time_ps=window_start_ps,
            end_time_ps=window_end_ps,
        )
    # LAMMPS averages v_cella (= lx), then applies that scalar to x/y/z.
    # Isotropic GROMACS coupling should keep all three equal, but use Box-X
    # explicitly to preserve the schedule definition.
    target_box_nm = float(average["mean_box_x_nm"])
    remap = _write_isotropically_remapped_gro(source_gro, averaged_gro, target_box_nm)
    state_remap: dict[str, Any] | None = None
    if source_state_trr is not None:
        if not source_state_trr.exists():
            raise RuntimeError(
                f"eq12 full-precision remap source is missing: {source_state_trr}"
            )
        state_remap = _write_isotropically_remapped_trr(
            source_state_trr,
            averaged_state_trr,
            target_box_nm,
        )
    report = {
        "schema_name": "gromacs_eq12_average_cell_handoff",
        "schema_version": 5,
        "source_box_series": str(source_series),
        "source_gro": str(source_gro),
        "source_state_trr": (
            str(source_state_trr) if source_state_trr is not None else None
        ),
        "output_gro": str(averaged_gro),
        "output_state_trr": (
            str(averaged_state_trr) if state_remap is not None else None
        ),
        "eqfactor": float(eqfactor),
        "window_mode": window_mode,
        "window_source": window_source,
        "selected_window_start_ps": window_start_ps,
        "selected_window_end_ps": window_end_ps,
        "nominal_lammps_average_duration_ps": 500.0 * float(eqfactor),
        "legacy_reference": (
            dict(legacy_reference)
            if isinstance(legacy_reference, Mapping)
            else None
        ),
        **average,
        **remap,
        "full_precision_state_remap": state_remap,
    }
    _write_json(md_dir / f"{deffnm}_average_cell.json", report)
    return averaged_gro


def _eq12_average_cell_is_current(
    *,
    md_dir: Path,
    stage: Mapping[str, Any],
    source_gro: Path,
    source_state_trr: Path | None = None,
    eqfactor: float,
) -> bool:
    """Return whether a cached Eq12 remap matches its inputs and window."""

    deffnm = str(stage["name"])
    averaged_gro = md_dir / f"{deffnm}_average_cell.gro"
    averaged_state_trr = md_dir / f"{deffnm}_average_cell_final_state.trr"
    report_path = md_dir / f"{deffnm}_average_cell.json"
    if not averaged_gro.exists() or not report_path.exists():
        return False
    try:
        report = _read_json(report_path)
        if int(report.get("schema_version", 0)) < 5:
            return False
        if Path(str(report.get("source_gro", ""))).resolve() != source_gro.resolve():
            return False
        if source_state_trr is not None:
            if not averaged_state_trr.exists():
                return False
            if (
                Path(str(report.get("source_state_trr", ""))).resolve()
                != source_state_trr.resolve()
            ):
                return False
            if (
                Path(str(report.get("output_state_trr", ""))).resolve()
                != averaged_state_trr.resolve()
            ):
                return False
        elif report.get("source_state_trr") is not None:
            return False
        if not math.isclose(
            float(report.get("eqfactor")),
            float(eqfactor),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            return False
        window_mode = str(stage.get("eq12_average_window_mode") or "").strip()
        if str(report.get("window_mode") or "").strip() != window_mode:
            return False
        window_source = str(stage.get("eq12_average_window_source") or "").strip()
        if str(report.get("window_source") or "").strip() != window_source:
            return False
        for report_key, stage_key in (
            ("selected_window_start_ps", "eq12_average_window_start_ps"),
            ("selected_window_end_ps", "eq12_average_window_end_ps"),
            ("window_start_ps", "eq12_average_window_start_ps"),
            ("window_end_ps", "eq12_average_window_end_ps"),
        ):
            if not math.isclose(
                float(report.get(report_key)),
                float(stage.get(stage_key)),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                return False
        if window_mode == "legacy_actual":
            legacy_reference = stage.get("eq12_legacy_average_window")
            report_reference = report.get("legacy_reference")
            if not isinstance(legacy_reference, Mapping) or not isinstance(
                report_reference, Mapping
            ):
                return False
            for key in (
                "stage_start_global_outer_step",
                "stage_end_global_outer_step",
                "first_sample_global_outer_step",
                "output_global_outer_step",
                "nave",
                "nrepeat",
                "nfreq",
                "nskip",
                "sample_count",
            ):
                if int(report_reference.get(key)) != int(legacy_reference.get(key)):
                    return False
            for key in ("outer_timestep_fs", "relative_start_ps", "relative_end_ps"):
                if not math.isclose(
                    float(report_reference.get(key)),
                    float(legacy_reference.get(key)),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ):
                    return False
        elif window_mode == "final":
            pass
        else:
            return False
        source_series = Path(str(report.get("source_box_series", "")))
        if not source_series.exists():
            return False
        dependencies = [source_gro, source_series]
        outputs = [averaged_gro, report_path]
        if source_state_trr is not None:
            dependencies.append(source_state_trr)
            outputs.append(averaged_state_trr)
        tpr_path = md_dir / f"{deffnm}.tpr"
        if source_series.suffix == ".xtc" and tpr_path.exists():
            dependencies.append(tpr_path)
        newest_input = max(path.stat().st_mtime_ns for path in dependencies)
        oldest_output = min(path.stat().st_mtime_ns for path in outputs)
        return oldest_output >= newest_input
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _gromacs_stage_status_path(md_dir: Path) -> Path:
    return md_dir / "gromacs_stage_status.json"


def _gromacs_stage_history_path(md_dir: Path) -> Path:
    return md_dir / "gromacs_stage_history.jsonl"


def _write_gromacs_stage_status(
    *,
    md_dir: Path,
    stage: Mapping[str, Any],
    phase: str,
    status: str,
    deffnm: str,
    output_stem: str | None = None,
    layout: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> None:
    nsteps = int(stage.get("nsteps", 0) or 0)
    init_step = int(stage.get("init_step", 0) or 0)
    dt_ps = float(stage.get("dt_ps", 0.0) or 0.0)
    actual_output_stem = str(output_stem or deffnm)
    payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "phase": str(phase),
        "stage": str(stage.get("name", "")),
        "status": str(status),
        "deffnm": str(deffnm),
        "output_stem": actual_output_stem,
        "init_step": init_step,
        "expected_checkpoint_step": init_step + nsteps,
        "nsteps": nsteps,
        "dt_ps": dt_ps,
        "total_ps": float(nsteps) * dt_ps,
        "total_ns": float(nsteps) * dt_ps / 1000.0,
        "log": f"{actual_output_stem}.log",
        "cpt": f"{deffnm}.cpt",
        "gro": f"{actual_output_stem}.gro",
        "tpr": f"{deffnm}.tpr",
    }
    if layout is not None:
        payload["layout"] = {
            "ntmpi": int(layout.get("ntmpi", 1)),
            "ntomp": int(layout.get("ntomp", 1)),
            "source": str(layout.get("source", "")),
        }
    if error:
        payload["error"] = str(error)
    _write_json(_gromacs_stage_status_path(md_dir), payload)
    hist = _gromacs_stage_history_path(md_dir)
    hist.parent.mkdir(parents=True, exist_ok=True)
    with hist.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(
        "[gromacs-stage]",
        f"{phase}/{payload['stage']}",
        f"status={status}",
        f"nsteps={nsteps}",
        f"dt_ps={dt_ps:g}",
        f"deffnm={deffnm}",
        flush=True,
    )


def _stage_env_suffix(stage_name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in stage_name.upper()).strip("_")


def _mdrun_extra_args(stage_name: str | None = None) -> list[str]:
    if stage_name:
        stage_key = f"GROMACS_BATCH_MDRUN_{_stage_env_suffix(stage_name)}_EXTRA_ARGS"
        raw_stage = os.environ.get(stage_key, "").strip()
        if raw_stage:
            return shlex.split(raw_stage)
        if stage_name == "00_em" or "minimize" in stage_name.lower():
            raw_em = os.environ.get("GROMACS_BATCH_MDRUN_EM_EXTRA_ARGS", "").strip()
            if raw_em:
                return shlex.split(raw_em)
    raw = os.environ.get("GROMACS_BATCH_MDRUN_EXTRA_ARGS", "").strip()
    return shlex.split(raw) if raw else []


def _mdrun_extra_args_for_stage(stage: Mapping[str, Any]) -> list[str]:
    args = _mdrun_extra_args(str(stage.get("name", "")))
    if str(stage.get("kspace_compute", "")).lower() != "no":
        return args
    out = list(args)
    for i, value in enumerate(out[:-1]):
        if value == "-pme" and out[i + 1] == "gpu":
            out[i + 1] = "cpu"
    return out


def _merge_mdrun_layout_args(
    base_args: Sequence[str], layout_args: Sequence[str]
) -> list[str]:
    """Merge per-stage mdrun offload options without emitting duplicates.

    The lane defaults select CPU or GPU offload globally, while a stage layout
    can force selected soft/stochastic stages back to CPU.  GROMACS rejects a
    command line that contains the same value option twice, so a layout value
    must replace the corresponding lane default instead of being appended.
    """
    overrides = list(layout_args)
    if not overrides:
        return list(base_args)

    value_options = {"-nb", "-pme", "-bonded", "-update"}
    overridden_options = {
        token for token in overrides if token in value_options
    }
    merged: list[str] = []
    index = 0
    base = list(base_args)
    while index < len(base):
        token = base[index]
        if token in overridden_options:
            if index + 1 >= len(base):
                raise ValueError(f"Missing value for mdrun option {token}")
            index += 2
            continue
        merged.append(token)
        index += 1
    merged.extend(overrides)
    return merged


def _grompp_extra_args() -> list[str]:
    raw = os.environ.get("GROMACS_BATCH_GROMPP_EXTRA_ARGS", "").strip()
    return shlex.split(raw) if raw else []


def _apply_batch_mdrun_env(env: dict[str, str]) -> dict[str, str]:
    for item in os.environ.get("GROMACS_BATCH_MDRUN_ENV", "").split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def _infer_gro_atom_count(path: Path) -> int | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) >= 2:
            value = int(lines[1].strip())
            return value if value > 0 else None
    except Exception:
        return None
    return None


def _apply_project_atom_count_env(env: dict[str, str], md_dir: Path) -> dict[str, str]:
    if env.get("GMX_PCFF_MTTK_LAMMPS_NATOMS"):
        return env
    natoms = _infer_gro_atom_count(md_dir / "conf.gro")
    if natoms:
        env["GMX_PCFF_MTTK_LAMMPS_NATOMS"] = str(natoms)
        env.setdefault("GMX_PCFF_MTTK_LAMMPS_NATOMS_FALLBACK", str(natoms))
    return env


def _gromacs_layout_for_stage(context: Mapping[str, Any], stage_name: str) -> dict[str, Any]:
    layouts = context.get("gromacs_stage_layouts") or {}
    raw: Mapping[str, Any] = {}
    if isinstance(layouts, Mapping):
        maybe = layouts.get(stage_name) or layouts.get("default")
        if isinstance(maybe, Mapping):
            raw = maybe
    ntomp = max(1, int(raw.get("ntomp", context.get("nproc", 1))))
    ntmpi = max(1, int(raw.get("ntmpi", 1)))
    extra_args = raw.get("extra_args", [])
    if isinstance(extra_args, str):
        extra_list = shlex.split(extra_args)
    elif isinstance(extra_args, list):
        extra_list = [str(x) for x in extra_args]
    else:
        extra_list = []
    env_items = raw.get("env", {})
    env_map = {str(k): str(v) for k, v in dict(env_items).items()} if isinstance(env_items, Mapping) else {}
    return {
        "ntomp": ntomp,
        "ntmpi": ntmpi,
        "extra_args": extra_list,
        "env": env_map,
        "source": raw.get("source", "stage_layouts" if raw else "default"),
    }


def _stage_runtime_env(env: dict[str, str], stage: Mapping[str, Any]) -> dict[str, str]:
    out = dict(env)
    name = str(stage.get("name", ""))
    if stage.get("gen_seed") is not None:
        # LAMMPS `velocity create ... mom yes rot yes` removes both linear and
        # angular momentum and then rescales to the requested temperature.
        # The patched grompp path is opt-in so unrelated GROMACS jobs retain
        # their native velocity-generation behavior.
        out["GMX_PCFF_GEN_VEL_LAMMPS_MOM_ROT"] = "1"
    if str(stage.get("kind", "")) == "em" and name == "eq03_pre_2fs_minimize":
        out["GMX_PCFF_LAMMPS_CG_EM"] = "1"
    if str(stage.get("kspace_compute", "")).lower() == "no":
        out["GMX_PCFF_EWALD_REAL_ONLY"] = "1"
        out.setdefault("GMX_PCFF_EXACT_RESPA_DISABLE_NBNXM_NARROW", "0")
    ensemble = str(stage.get("ensemble", ""))
    if ensemble == "npt":
        tau_p_ps = stage.get("tau_p_ps")
        if tau_p_ps is not None:
            out["GMX_PCFF_MTTK_LAMMPS_PDAMP_PS"] = f"{float(tau_p_ps):.9g}"
        # LAMMPS uses nreset=20000 outer steps. Every PolyGen stage has four
        # GROMACS base steps per LAMMPS outer step, hence 80000 base steps.
        out.setdefault("GMX_PCFF_MTTK_NRESET_STEPS", "80000")
        p0 = stage.get("pressure_start_atm")
        p1 = stage.get("pressure_end_atm")
        if p0 is not None and p1 is not None and not math.isclose(float(p0), float(p1)):
            out["GMX_PCFF_REFP_RAMP_START_BAR"] = f"{float(p0) * ATM_TO_BAR:.9g}"
            out["GMX_PCFF_REFP_RAMP_END_BAR"] = f"{float(p1) * ATM_TO_BAR:.9g}"
            out["GMX_PCFF_REFP_RAMP_DURATION_PS"] = f"{float(stage.get('nsteps', 0)) * float(stage.get('dt_ps', 0.0)):.9g}"
        drag = float(stage.get("lammps_fix_nh_drag", 0.0) or 0.0)
        if drag > 0.0:
            out["GMX_PCFF_MTTK_LAMMPS_DRAG"] = f"{drag:.9g}"
    soft_start = stage.get("soft_start")
    if isinstance(soft_start, Mapping):
        out["GMX_PCFF_EXACT_RESPA_SOFT_START"] = "1"
        out["GMX_PCFF_EXACT_RESPA_NVE_LIMIT_XMAX_NM"] = (
            f"{float(soft_start['displacement_limit_nm']):.12g}"
        )
        out["GMX_PCFF_EXACT_RESPA_LANGEVIN_TEMP_K"] = (
            f"{float(soft_start['langevin_temp_k']):.12g}"
        )
        out["GMX_PCFF_EXACT_RESPA_LANGEVIN_TAU_PS"] = (
            f"{float(soft_start['langevin_damp_ps']):.12g}"
        )
        out["GMX_PCFF_EXACT_RESPA_LANGEVIN_SEED"] = str(
            int(soft_start["langevin_seed"])
        )
        out["GMX_PCFF_EXACT_RESPA_LANGEVIN_ZERO_RANDOM"] = (
            "1" if bool(soft_start.get("zero_net_force", True)) else "0"
        )
    suffix = _stage_env_suffix(name)
    for item in os.environ.get(f"GROMACS_BATCH_MDRUN_{suffix}_ENV", "").split(";"):
        item = item.strip()
        if item and "=" in item:
            key, value = item.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def _merge_stage_layout_env(
    env: dict[str, str],
    stage: Mapping[str, Any],
    layout_env: Mapping[str, Any],
) -> dict[str, str]:
    """Merge per-stage layout overrides for the selected GROMACS stage.

    The patched binary validates feature-specific variables.  In particular,
    ``GMX_PCFF_EWALD_BETA_INV_A`` is supported for both exact r-RESPA dynamics
    and the CG minimization used by PolyGen Eq03, so it must not be stripped
    from an EM layout here.
    """

    del stage
    out = dict(env)
    out.update({str(key): str(value) for key, value in layout_env.items()})
    return out


def _md_stage_output_names(stage_name: str) -> tuple[str, str]:
    return (f"{stage_name}.tpr", f"{stage_name}")


def _is_exact_soft_start_stage(stage: Mapping[str, Any]) -> bool:
    """Return whether ``stage`` uses the non-checkpointable soft-start path."""

    return str(stage.get("kind", "")) == "md" and isinstance(
        stage.get("soft_start"), Mapping
    )


def _archive_incomplete_gromacs_stage_outputs(
    *,
    md_dir: Path,
    stage: Mapping[str, Any],
    deffnm: str,
    original_structure: Path,
    original_state_trr: Path | None,
) -> Path:
    """Move one incomplete stage's outputs aside before a clean restart.

    Exact soft-start dynamics are stochastic and deliberately reject ``-cpi``.
    Restarting them therefore requires regenerating the TPR from the same input
    structure/state used on the first attempt.  Moving all deffnm-owned files
    out of ``md_dir`` prevents GROMACS from appending to, backing up, or otherwise
    mixing the new attempt with the partial one.  The stage MDP is an input and
    remains in place.
    """

    md_dir = md_dir.resolve()
    mdp_path = Path(str(stage["mdp_path"])).resolve()
    candidates = sorted(
        (
            path
            for path in md_dir.iterdir()
            if (path.is_file() or path.is_symlink())
            and path.resolve() != mdp_path
            and (
                path.name.startswith(deffnm)
                or path.name.startswith(f"#{deffnm}")
            )
        ),
        key=lambda path: path.name,
    )
    if not candidates:
        raise RuntimeError(
            f"Cannot archive incomplete soft-start stage {stage.get('name')}: "
            f"no {deffnm}-owned output files were found in {md_dir}"
        )

    stage_archive_root = md_dir / "incomplete_stage_restarts" / deffnm
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S.%f")
    archive_dir = stage_archive_root / timestamp
    suffix = 1
    while archive_dir.exists():
        archive_dir = stage_archive_root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    archive_dir.mkdir(parents=True, exist_ok=False)

    moved: list[tuple[Path, Path]] = []
    try:
        for source in candidates:
            destination = archive_dir / source.name
            source.replace(destination)
            moved.append((source, destination))
        _write_json(
            archive_dir / "restart_manifest.json",
            {
                "schema_name": "gromacs_incomplete_soft_stage_restart",
                "schema_version": 1,
                "timestamp": datetime.now().isoformat(timespec="microseconds"),
                "stage": str(stage.get("name", "")),
                "deffnm": str(deffnm),
                "reason": (
                    "incomplete exact soft-start checkpoints cannot be resumed; "
                    "restart from the original stage input state"
                ),
                "original_structure": str(original_structure),
                "original_state_trr": (
                    str(original_state_trr)
                    if original_state_trr is not None
                    else None
                ),
                "archived_files": [destination.name for _, destination in moved],
            },
        )
    except Exception:
        manifest_path = archive_dir / "restart_manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                destination.replace(source)
        try:
            archive_dir.rmdir()
        except OSError:
            pass
        raise
    return archive_dir


def _require_ready(context: Mapping[str, Any]) -> None:
    if str(context.get("overall_status")) != "ready_for_md":
        raise RuntimeError(f"GROMACS runtime context is blocked: {context.get('failure_reason')}")


def run_gromacs_equilibration(
    context: Mapping[str, Any],
    *,
    progress_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    _require_ready(context)
    md_dir = Path(context["md_dir"]).resolve()
    gmx = str(context["gmx_binary"])
    stages = [stage for stage in context["stages"] if stage["phase"] == "equilibration"]
    env = dict(os.environ)
    if context.get("gmxlib"):
        env["GMXLIB"] = str(context["gmxlib"])
    env = _apply_batch_mdrun_env(env)
    env = _apply_project_atom_count_env(env, md_dir)
    resume_existing_effective = bool(context.get("resume_existing_effective", False))

    current_structure = md_dir / "conf.gro"
    current_topology = md_dir / "topol.top"
    if not current_structure.exists() or not current_topology.exists():
        raise RuntimeError(
            "equilibration requires assembled conf.gro/topol.top in MD_GMX; mixed-system assembly is not implemented"
        )

    previous_stage: Mapping[str, Any] | None = None
    current_checkpoint: Path | None = None
    current_state_trr: Path | None = None
    for stage in stages:
        tpr_name, deffnm = _md_stage_output_names(stage["name"])
        tpr_path = md_dir / tpr_name
        gro_path = md_dir / f"{deffnm}.gro"
        layout = _gromacs_layout_for_stage(context, str(stage["name"]))
        stage_env = _merge_stage_layout_env(
            _stage_runtime_env(env, stage), stage, layout["env"]
        )
        stage_checkpoint = md_dir / f"{deffnm}.cpt"
        output_stem = deffnm
        completed_output = gro_path.exists()
        if (
            resume_existing_effective
            and completed_output
            and str(stage.get("kind", "")) == "md"
        ):
            if not stage_checkpoint.exists():
                raise RuntimeError(
                    f"Cannot verify existing MD output {gro_path}: missing checkpoint "
                    f"{stage_checkpoint}"
                )
            checkpoint_step = _gromacs_checkpoint_step(
                gmx=gmx,
                checkpoint=stage_checkpoint,
                cwd=md_dir,
                env=stage_env,
            )
            completed_output = checkpoint_step >= _stage_expected_checkpoint_step(stage)
        if resume_existing_effective and completed_output:
            current_structure = gro_path
            if str(stage.get("kind", "")) == "md":
                current_state_trr = _materialize_final_state_trr(
                    gmx=gmx,
                    md_dir=md_dir,
                    stage=stage,
                    checkpoint=stage_checkpoint,
                    env=stage_env,
                )
            else:
                current_state_trr = None
            if str(stage["name"]) == "eq12_npt_1200ps":
                averaged_gro = md_dir / "eq12_npt_1200ps_average_cell.gro"
                averaged_state_trr = (
                    md_dir / "eq12_npt_1200ps_average_cell_final_state.trr"
                )
                eqfactor = float(context.get("eqfactor", 1.0))
                if _eq12_average_cell_is_current(
                        md_dir=md_dir,
                        stage=stage,
                        source_gro=gro_path,
                        source_state_trr=current_state_trr,
                        eqfactor=eqfactor,
                    ):
                    current_structure = averaged_gro
                    current_state_trr = averaged_state_trr
                else:
                    current_structure = _apply_eq12_average_cell(
                        gmx=gmx,
                        md_dir=md_dir,
                        stage=stage,
                        source_gro=gro_path,
                        source_state_trr=current_state_trr,
                        env=stage_env,
                        eqfactor=eqfactor,
                    )
                    current_state_trr = averaged_state_trr
            current_checkpoint = stage_checkpoint if stage_checkpoint.exists() else None
            previous_stage = stage
            _write_gromacs_stage_status(
                md_dir=md_dir,
                stage=stage,
                phase="equilibration",
                status="resume_skip_stage",
                deffnm=deffnm,
                layout=layout,
            )
            if progress_hook is not None:
                progress_hook(stage["name"], stage)
            continue
        try:
            stage_state_trr: Path | None = None
            same_base_stage = (
                previous_stage is not None
                and stage.get("base_index") == previous_stage.get("base_index")
                and int(stage.get("segment_index", 1)) > 1
            )
            resume_incomplete_stage = (
                resume_existing_effective
                and tpr_path.exists()
                and stage_checkpoint.exists()
                and not completed_output
            )
            restart_incomplete_soft_stage = (
                resume_existing_effective
                and stage_checkpoint.exists()
                and not completed_output
                and _is_exact_soft_start_stage(stage)
            )
            if restart_incomplete_soft_stage:
                if same_base_stage:
                    raise RuntimeError(
                        f"Cannot safely restart chunked soft-start stage {stage.get('name')}; "
                        "the current protocol requires soft-start stages to be standalone"
                    )
                archive_dir = _archive_incomplete_gromacs_stage_outputs(
                    md_dir=md_dir,
                    stage=stage,
                    deffnm=deffnm,
                    original_structure=current_structure,
                    original_state_trr=current_state_trr,
                )
                _write_gromacs_stage_status(
                    md_dir=md_dir,
                    stage=stage,
                    phase="equilibration",
                    status="restart_incomplete_soft_stage",
                    deffnm=deffnm,
                    layout=layout,
                )
                print(
                    "[gromacs-stage]",
                    f"archived incomplete soft-start outputs at {archive_dir}",
                    flush=True,
                )
                # Regenerate this stage from current_structure/current_state_trr.
                # In particular, never pass its rejected partial CPT to mdrun.
                resume_incomplete_stage = False
            if resume_incomplete_stage and int(stage.get("segment_index", 1)) > 1:
                checkpoint_state = _gromacs_checkpoint_state(
                    gmx=gmx,
                    checkpoint=stage_checkpoint,
                    cwd=md_dir,
                    env=stage_env,
                )
                output_stem = (
                    f"{deffnm}.part{checkpoint_state['simulation_part'] + 1:04d}"
                )
            elif same_base_stage and current_checkpoint is not None:
                checkpoint_state = _gromacs_checkpoint_state(
                    gmx=gmx,
                    checkpoint=current_checkpoint,
                    cwd=md_dir,
                    env=stage_env,
                )
                output_stem = (
                    f"{deffnm}.part{checkpoint_state['simulation_part'] + 1:04d}"
                )
            if not resume_incomplete_stage:
                _write_gromacs_stage_status(
                    md_dir=md_dir,
                    stage=stage,
                    phase="equilibration",
                    status="grompp",
                    deffnm=deffnm,
                    layout=layout,
                )
                grompp_cmd = [
                    gmx,
                    "grompp",
                    "-f",
                    stage["mdp_path"],
                    "-c",
                    str(current_structure),
                    "-p",
                    str(current_topology),
                    "-o",
                    str(tpr_path),
                ]
                if same_base_stage:
                    if current_checkpoint is None or not current_checkpoint.exists():
                        raise RuntimeError(
                            f"Cannot hand off {previous_stage.get('name')} to {stage.get('name')}: "
                            "the previous same-base chunk checkpoint is missing"
                        )
                    grompp_cmd.extend(["-t", str(current_checkpoint)])
                elif current_state_trr is not None:
                    if not current_state_trr.exists():
                        raise RuntimeError(
                            f"Cannot hand off {previous_stage.get('name')} to "
                            f"{stage.get('name')}: full-precision state TRR is missing"
                        )
                    grompp_cmd.extend(["-t", str(current_state_trr)])
                grompp_cmd.extend(_grompp_extra_args())
                _run_cmd(grompp_cmd, cwd=md_dir, env=stage_env)
            _write_gromacs_stage_status(
                md_dir=md_dir,
                stage=stage,
                phase="equilibration",
                status="mdrun_resume" if resume_incomplete_stage else "mdrun",
                deffnm=deffnm,
                output_stem=output_stem,
                layout=layout,
            )
            merged_mdrun_args = _merge_mdrun_layout_args(
                _mdrun_extra_args_for_stage(stage), layout["extra_args"]
            )
            mdrun_cmd = [
                    gmx,
                    "mdrun",
                    "-deffnm",
                    deffnm,
                    "-ntmpi",
                    str(layout["ntmpi"]),
                    "-ntomp",
                    str(layout["ntomp"]),
                    *merged_mdrun_args,
                ]
            if resume_incomplete_stage:
                mdrun_cmd.extend(["-cpi", str(stage_checkpoint)])
                mdrun_cmd.append(
                    "-noappend"
                    if int(stage.get("segment_index", 1)) > 1
                    else "-append"
                )
            elif same_base_stage:
                if current_checkpoint is None or not current_checkpoint.exists():
                    raise RuntimeError(
                        f"Cannot start {stage.get('name')}: previous checkpoint is missing"
                    )
                mdrun_cmd.extend(["-cpi", str(current_checkpoint), "-noappend"])
            _run_cmd(mdrun_cmd, cwd=md_dir, env=stage_env)
            if str(stage.get("kind", "")) == "md":
                if int(stage.get("segment_index", 1)) > 1:
                    gro_path, output_stem = _canonicalize_cross_chunk_gro(
                        gmx=gmx,
                        md_dir=md_dir,
                        stage=stage,
                        checkpoint=stage_checkpoint,
                        env=stage_env,
                    )
                _require_completed_md_stage(
                    gmx=gmx,
                    md_dir=md_dir,
                    stage=stage,
                    gro_path=gro_path,
                    checkpoint=stage_checkpoint,
                    env=stage_env,
                )
                stage_state_trr = _materialize_final_state_trr(
                    gmx=gmx,
                    md_dir=md_dir,
                    stage=stage,
                    checkpoint=stage_checkpoint,
                    env=stage_env,
                )
            elif not gro_path.exists():
                raise RuntimeError(
                    f"GROMACS stage returned without the required final structure {gro_path}"
                )
        except Exception as exc:
            _write_gromacs_stage_status(
                md_dir=md_dir,
                stage=stage,
                phase="equilibration",
                status="failed",
                deffnm=deffnm,
                output_stem=output_stem,
                layout=layout,
                error=repr(exc),
            )
            raise
        current_structure = md_dir / f"{deffnm}.gro"
        current_state_trr = stage_state_trr
        if str(stage["name"]) == "eq12_npt_1200ps":
            current_structure = _apply_eq12_average_cell(
                gmx=gmx,
                md_dir=md_dir,
                stage=stage,
                source_gro=current_structure,
                source_state_trr=current_state_trr,
                env=stage_env,
                eqfactor=float(context.get("eqfactor", 1.0)),
            )
            current_state_trr = (
                md_dir / "eq12_npt_1200ps_average_cell_final_state.trr"
            )
        current_checkpoint = stage_checkpoint if stage_checkpoint.exists() else None
        previous_stage = stage
        _write_gromacs_stage_status(
            md_dir=md_dir,
            stage=stage,
            phase="equilibration",
            status="done",
            deffnm=deffnm,
            output_stem=output_stem,
            layout=layout,
        )
        if progress_hook is not None:
            progress_hook(stage["name"], stage)

    (md_dir / "equilibration_complete.flag").write_text("done\n", encoding="utf-8")
    return {
        "status": "ok",
        "final_structure": str(current_structure),
        "final_state_trr": (
            str(current_state_trr) if current_state_trr is not None else None
        ),
    }


def run_gromacs_production(
    context: Mapping[str, Any],
    *,
    progress_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    _require_ready(context)
    md_dir = Path(context["md_dir"]).resolve()
    gmx = str(context["gmx_binary"])
    stages = [stage for stage in context["stages"] if stage["phase"] == "production"]
    env = dict(os.environ)
    if context.get("gmxlib"):
        env["GMXLIB"] = str(context["gmxlib"])
    env = _apply_batch_mdrun_env(env)
    env = _apply_project_atom_count_env(env, md_dir)
    resume_existing_effective = bool(context.get("resume_existing_effective", False))

    current_structure = md_dir / str(context.get("production_start_structure", "05_nvt_relaxed.gro"))
    production_state_name = context.get("production_start_state_trr")
    current_state_trr = (
        md_dir / str(production_state_name) if production_state_name else None
    )
    if not current_structure.exists() and (md_dir / "equilibration_complete.flag").exists():
        equil_stages = [stage for stage in context.get("stages", []) if stage.get("phase") == "equilibration"]
        if equil_stages:
            fallback_structure = md_dir / f"{equil_stages[-1]['name']}.gro"
            if fallback_structure.exists():
                current_structure = fallback_structure
                if equil_stages[-1].get("kind") == "md":
                    current_state_trr = _final_state_trr_path(md_dir, equil_stages[-1])
    current_topology = md_dir / "topol.top"
    if not current_structure.exists() or not current_topology.exists():
        raise RuntimeError(
            "production requires relaxed GROMACS structure/topology in MD_GMX; equilibration or mixed-system assembly is incomplete"
        )

    for stage in stages:
        tpr_name, deffnm = _md_stage_output_names(stage["name"])
        tpr_path = md_dir / tpr_name
        cpt_path = md_dir / f"{deffnm}.cpt"
        gro_path = md_dir / f"{deffnm}.gro"
        layout = _gromacs_layout_for_stage(context, str(stage["name"]))
        stage_env = _merge_stage_layout_env(
            _stage_runtime_env(env, stage), stage, layout["env"]
        )
        completed_output = gro_path.exists()
        if resume_existing_effective and completed_output:
            if not cpt_path.exists():
                raise RuntimeError(
                    f"Cannot verify existing production output {gro_path}: missing checkpoint "
                    f"{cpt_path}"
                )
            checkpoint_step = _gromacs_checkpoint_step(
                gmx=gmx,
                checkpoint=cpt_path,
                cwd=md_dir,
                env=stage_env,
            )
            completed_output = checkpoint_step >= _stage_expected_checkpoint_step(stage)
        if resume_existing_effective and completed_output:
            current_structure = gro_path
            _write_gromacs_stage_status(
                md_dir=md_dir,
                stage=stage,
                phase="production",
                status="resume_skip_stage",
                deffnm=deffnm,
                layout=layout,
            )
            if progress_hook is not None:
                progress_hook(stage["name"], stage)
            continue
        resume_from_checkpoint = (
            resume_existing_effective
            and cpt_path.exists()
            and tpr_path.exists()
            and not completed_output
        )
        try:
            if resume_from_checkpoint:
                _write_gromacs_stage_status(
                    md_dir=md_dir,
                    stage=stage,
                    phase="production",
                    status="checkpoint_resume",
                    deffnm=deffnm,
                    layout=layout,
                )
            else:
                _write_gromacs_stage_status(
                    md_dir=md_dir,
                    stage=stage,
                    phase="production",
                    status="grompp",
                    deffnm=deffnm,
                    layout=layout,
                )
                grompp_cmd = [
                        gmx,
                        "grompp",
                        "-f",
                        stage["mdp_path"],
                        "-c",
                        str(current_structure),
                        "-p",
                        str(current_topology),
                        "-o",
                        str(tpr_path),
                    ]
                if current_state_trr is not None:
                    if not current_state_trr.exists():
                        raise RuntimeError(
                            "Production requires the full-precision final equilibration "
                            f"state, but it is missing: {current_state_trr}"
                        )
                    grompp_cmd.extend(["-t", str(current_state_trr)])
                grompp_cmd.extend(_grompp_extra_args())
                _run_cmd(grompp_cmd, cwd=md_dir, env=stage_env)
            _write_gromacs_stage_status(
                md_dir=md_dir,
                stage=stage,
                phase="production",
                status="mdrun",
                deffnm=deffnm,
                layout=layout,
            )
            merged_mdrun_args = _merge_mdrun_layout_args(
                _mdrun_extra_args_for_stage(stage), layout["extra_args"]
            )
            _run_cmd(
                [
                    gmx,
                    "mdrun",
                    "-deffnm",
                    deffnm,
                    "-ntmpi",
                    str(layout["ntmpi"]),
                    "-ntomp",
                    str(layout["ntomp"]),
                    *(["-cpi", str(cpt_path), "-append"] if resume_from_checkpoint else []),
                    *merged_mdrun_args,
                ],
                cwd=md_dir,
                env=stage_env,
            )
            _require_completed_md_stage(
                gmx=gmx,
                md_dir=md_dir,
                stage=stage,
                gro_path=gro_path,
                checkpoint=cpt_path,
                env=stage_env,
            )
        except Exception as exc:
            _write_gromacs_stage_status(
                md_dir=md_dir,
                stage=stage,
                phase="production",
                status="failed",
                deffnm=deffnm,
                layout=layout,
                error=repr(exc),
            )
            raise
        current_structure = md_dir / f"{deffnm}.gro"
        _write_gromacs_stage_status(
            md_dir=md_dir,
            stage=stage,
            phase="production",
            status="done",
            deffnm=deffnm,
            layout=layout,
        )
        if progress_hook is not None:
            progress_hook(stage["name"], stage)

    (md_dir / "production_complete.flag").write_text("done\n", encoding="utf-8")
    return {"status": "ok", "final_structure": str(current_structure)}


def write_gromacs_meta_json(
    proj: Path | str,
    *,
    monomer_smiles: str,
    placeholder: str,
    force_field: str,
    temperature_k: float,
    production_total_ns: float,
    prepare_report: Mapping[str, Any],
    runtime_context: Mapping[str, Any],
) -> Path:
    proj = Path(proj).resolve()
    md_dir = proj / "MD_GMX"
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "monomer_smiles": monomer_smiles,
        "placeholder": placeholder,
        "force_field": force_field,
        "temperature_k": float(temperature_k),
        "production_total_ns": float(production_total_ns),
        "prepare_report_path": runtime_context.get("prepare_report_path"),
        "prepare_status": prepare_report["workflow"]["overall_status"],
        "gmx_binary": runtime_context.get("gmx_binary"),
        "gmxlib": runtime_context.get("gmxlib"),
    }
    out_path = md_dir / "meta.json"
    _write_json(out_path, payload)
    return out_path
