# Gate A Oracle Manifest

- Status: `PASS`
- Artifact root: `.`
- Precision mode: `mixed`
- ntmpi / ntomp: `1` / `1`
- DLB: `no`
- PME rank count: `0`
- Reproducibility flags: `-reprod -dlb no -pin off -nb cpu -pme cpu -bonded cpu -update cpu`
- Simulator pin: `GMX_DISABLE_MODULAR_SIMULATOR=1`
- Rerun used: `no`
- Normal MD used: `yes`

## Systems
- `small_oligomer` event trace: `./small_oligomer/summaries/event_trace.json`
- `small_oligomer` per-level totals: `./small_oligomer/summaries/per_level_force_totals.json`
- `small_oligomer` class2 subterms: `./small_oligomer/summaries/class2_subterm_energy_trace.json`
- `small_oligomer` cpu corrections: `./small_oligomer/summaries/cpu_correction_energy_trace.json`
- `small_oligomer` energy terms: `./small_oligomer/summaries/energy_terms.json`
- `small_oligomer` restart summary: `./small_oligomer/summaries/restart_summary.json`
- `small_salt_polymer_box` event trace: `./small_salt_polymer_box/summaries/event_trace.json`
- `small_salt_polymer_box` per-level totals: `./small_salt_polymer_box/summaries/per_level_force_totals.json`
- `small_salt_polymer_box` class2 subterms: `./small_salt_polymer_box/summaries/class2_subterm_energy_trace.json`
- `small_salt_polymer_box` cpu corrections: `./small_salt_polymer_box/summaries/cpu_correction_energy_trace.json`
- `small_salt_polymer_box` energy terms: `./small_salt_polymer_box/summaries/energy_terms.json`
- `small_salt_polymer_box` restart summary: `./small_salt_polymer_box/summaries/restart_summary.json`

## Known Limitations
- The standalone exact r-RESPA path is frozen from the direct CLI mdrun path, but it still executes inside the legacy simulator container; the trace itself comes from exactrespastepper.cpp standalone exact-r-RESPA entrypoints.
- The direct CLI path has a bootstrap step-0 event pattern that differs from the older LAMMPS-style recursive test harness; Gate A freezes the actual CLI behavior rather than forcing the older wrapper-derived reference.
- Per-term virial contributors are not frozen individually; the oracle freezes EDR virial and pressure tensor components plus raw per-level merge-trace virial-related buffers.
- PCFF class2 subterm visibility is frozen from a host-side diagnostic rescan of the exact-r-RESPA level interaction lists and coordinates; it is an ownership/debug oracle, not a raw GPU accumulator dump.
- CPU reciprocal/self/exclusion electrostatics are frozen from runtime split traces written by the exact standalone path; they are later comparison oracles, not standalone user-facing EDR terms.
- No GPU-path comparison is performed here; Gate A only freezes the CPU oracle and validates its internal consistency.

## Recommended Comparison Fields
- event_trace.actual_event_trace[].base_step
- event_trace.actual_event_trace[].event
- event_trace.actual_event_trace[].level
- per_level_force_totals.entries[].relative_path
- per_level_force_totals.entries[].vector_sum
- class2_subterm_energy_trace.entries[].step
- class2_subterm_energy_trace.entries[].level
- class2_subterm_energy_trace.entries[].terms_kj_mol.*
- class2_subterm_energy_trace.entries[].interaction_counts.*
- cpu_correction_energy_trace.entries[].step
- cpu_correction_energy_trace.entries[].level
- cpu_correction_energy_trace.entries[].terms_kj_mol.*
- cpu_correction_energy_trace.entries[].interaction_counts.*
- total_force_summary.entries[].step
- total_force_summary.entries[].highest_active_level
- total_force_summary.entries[].force
- energy_terms.step0_terms_kj_mol.*
- energy_terms.derived_terms_step0_kj_mol.*
- energy_terms.frames[].terms[Potential]
- energy_terms.frames[].terms[Total Energy]
- energy_terms.frames[].terms[Vir-XX]
- energy_terms.frames[].terms[Vir-YY]
- energy_terms.frames[].terms[Vir-ZZ]
- energy_terms.frames[].terms[Pres-XX]
- energy_terms.frames[].terms[Pres-YY]
- energy_terms.frames[].terms[Pres-ZZ]
- m2p_trace/step0_force_component_trace.txt
- m2p_trace/step0_potential_ledger_trace.txt
- m2p_trace/step0_virial_pressure_ledger_trace.txt
- m2p_trace/step0_realspace_force_subcomponent_trace.txt
- restart_summary.potential_abs_delta_kj_mol
- restart_summary.total_abs_delta_kj_mol
- restart_summary.max_coordinate_abs_delta_nm
- restart_summary.max_velocity_abs_delta_nm_ps
