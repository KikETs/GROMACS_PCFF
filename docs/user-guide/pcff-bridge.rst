.. _gmx-pcff-bridge:

GROMACS-PCFF bridge
===================

This page documents the PCFF/Class2 bridge in this derived fork. It is not a
claim about upstream |Gromacs| and it is not broad PCFF support.

Current source of truth
-----------------------

Before using or citing the bridge, check these files from the repository root:

* ``README.md`` for the public scope statement.
* ``docs/current_status_note.md`` for the current claim boundary.
* ``docs/current_active_issues.md`` for unresolved work.
* ``tests/reference_results/pcff_ion_narrow_claim/support_matrix.json`` for the
  machine-readable support matrix.
* ``tests/reference_results/pcff_ion_narrow_claim/narrow_claim_summary.json`` for
  the compact supported and unsupported claim summary.

Supported scope
---------------

The checked-in evidence currently supports only the bounded scope below.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Area
     - Supported
     - Not supported
   * - Chemistry typing/export
     - The frozen PT8 SPE cases:
       ``monoglyme_litfsi_1to1``, ``diglyme_litfsi_1to1``, and
       ``triglyme_litfsi_2to2``.
     - Broad PCFF chemistry or the CSV snapshot target. The current CSV audit is
       not release-ready.
   * - M5 chemistry expansion
     - One workflow-level charged assembly with an acyclic alkane neutral
       additive: ``monoglyme_ethane_litfsi_1to1``.
     - Broad alkane, arbitrary co-solvent, dense ensemble, or transport
       readiness from that single M5 assembly.
   * - Charged mechanics
     - Frozen Class2/LJ 9-6/long-range Coulomb semantics, sixth-power mixing,
       and ``special_bonds lj/coul 0.0 0.0 1.0 angle no dihedral no`` on the
       checked fixtures.
     - Arbitrary charged cross-pair overrides or generic charged topology
       readiness.
   * - Dense charged parity
     - ``gate_h_dense_salt_polymer_2x2x2`` in the M4 strict subset, and the two
       M11.4/M11.6 strict-PCFF-qualified dense charged pairs at their declared
       pressure protocols.
     - Direct ambient ``1 bar`` equilibrium dense charged parity or generic dense
       charged ensemble readiness.
   * - Transport-facing evidence
     - GROMACS CPU/GPU 10 ns screening diagnostics in the 2026-05-10 PolyGen
       report.
     - LAMMPS-vs-GROMACS charged transport parity or charged transport readiness.

Required user stance
--------------------

Use the bridge as a validation-bounded research path. Do not treat a successful
parser/emitter run as scientific support for a new chemistry or ensemble.

For any new system, the minimum evidence chain is:

1. Typing/export evidence for the exact chemistry.
2. Frozen mechanics evidence for Class2 bonded terms, LJ 9-6 mixing, exclusions,
   and charged long-range semantics.
3. Dense ensemble evidence for the exact pressure and thermostat/barostat
   protocol being claimed.
4. Transport analysis only after the active 20 ns and LAMMPS-vs-GROMACS parity
   issues are closed.

Common entry points
-------------------

Run commands from the repository root unless the tool says otherwise.

Typing and chemistry-scope checks::

    python tools/run_typing_validation/generate.py
    python tools/run_csv_scope_audit/generate.py

Frozen short-MD and r-RESPA parity harnesses::

    python tools/pcff_short_md_parity/prepare_reference.py --lammps-cmd lmp
    python tools/pcff_respa_parity/run.py --prepare-reference --lammps-cmd lmp

PolyGen multi-system orchestration helpers::

    python tools/pcff_respa_parity/polygen_multisystem_manifest.py
    python tools/pcff_respa_parity/polygen_multisystem_worker.py --help
    python tools/pcff_respa_parity/polygen_multisystem_collect_results.py --help
    python tools/pcff_respa_parity/polygen_multisystem_transport_analysis.py --help

Use ``GMX_BIN`` when a tool should use a specific GROMACS binary, ``LMP_BIN`` for
LAMMPS, ``PCFF_BRIDGE_REPO`` for the data-bridge sibling repository, and
``CSV_SCOPE_ADAPTER_PYTHON`` for the CSV-scope adapter Python interpreter.

Current non-claims
------------------

Do not make any of these claims from the current repository state:

* Full PCFF readiness across all chemistries.
* Charged polymer-electrolyte transport readiness.
* Direct ambient ``1 bar`` equilibrium dense charged parity.
* Generic dense charged ensemble parity outside the explicit M11 subsets.
* LAMMPS-vs-GROMACS charged transport parity.
* Endpoint continuation safety from the corrected TP1 final coordinates.
* ACPYPE/GAFF2-prepared artifacts as strict PCFF parity evidence.

Validation checklist before citing results
------------------------------------------

Before writing a claim or release note, check:

* ``docs/current_status_note.md`` for the reusable public claim.
* ``docs/current_active_issues.md`` for unresolved limitations.
* ``docs/release_readiness_matrix.md`` for release-readiness blockers.
* ``tests/reference_results/pcff_ion_narrow_claim/support_matrix.json`` for the
  exact support status.
* The specific validation report and artifact paths for the system being cited.
