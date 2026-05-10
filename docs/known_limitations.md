# Known Limitations

This document lists current technical and physical limitations of the GROMACS-PCFF bridge.

Current issue source of truth:

- [Current Active Issues](current_active_issues.md)
- [Current Status Note](current_status_note.md)
- [Transport Scope Matrix](transport_scope_matrix.md)

## 1. Charged Transport Readiness

**Issue:** Diffusion, conductivity, transference, and cNE0 are not production-validated.

**Current evidence:** The latest PolyGen CPU/GPU transport screening is 10 ns. The local charged transport protocol requires at least 20 ns.

**Boundary:** NE is currently screening-only. HTP-MD-style cNE0 is diagnostic-only.

## 2. LAMMPS-vs-GROMACS Transport Parity

**Issue:** LAMMPS-vs-GROMACS charged transport parity is not closed.

**Current evidence:** The 2026-05-10 transport analysis compares GROMACS CPU and GPU. LAMMPS production dump data was used only for topology metadata in that analysis.

## 3. Chemistry Scope

**Issue:** Broad PCFF chemistry remains unsupported.

**Current evidence:** The bridge is bounded to the explicitly validated PT8/M11 subsets. The CSV-scope audit is not a broad chemistry pass.

## 4. Strict GPU Performance

**Issue:** Strict PolyGen GPU production remains below the 200 ns/day target.

**Current evidence:** Final strict GPU full-run production speed is `147.315 ns/day` mean under `-nb gpu -pme cpu -bonded gpu -update cpu`.

**Boundary:** Runtime-condition changes that alter physics are not acceptable as performance fixes.

## 5. Historical rc1 Limitations

Older `v1.0.0-rc1` density/transport wording is superseded by the current status documents. Historical reports are retained as evidence, but they are not active issue definitions unless listed in [Current Active Issues](current_active_issues.md).
