from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

from atom_typing.engine import _build_atom_contexts


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRC_PATH = REPO_ROOT / "frc_file" / "pcff.frc"


PHASE1_DIRECT_PCFF_TYPE_BY_RULE_ID = {
    "carbonate_like_carbamate_carbonyl_carbon": "c_2",
    "amide_like_carbonyl_carbon": "c_1",
    "amide_like_imide_carbonyl_carbon": "c_1",
    "amide_like_carbamate_carbonyl_carbon": "c_2",
    "amide_like_formamide_carbonyl_carbon": "c_1",
    "amide_like_amide_nitrogen": "n",
    "carbonate_like_amide_nitrogen": "n_2",
    "amide_like_carbonyl_oxygen": "o_1",
    "amide_like_carbonyl_bridging_oxygen": "o_2",
    "carbonate_like_carbonyl_bridging_oxygen": "oz",
    "carbonate_like_ether_oxygen": "o",
    "amide_like_ether_oxygen": "o",
    "amide_like_nitrile_carbon": "ct",
    "carbonate_like_nitrile_carbon": "ct",
    "amide_like_nitrile_nitrogen": "nt",
    "carbonate_like_nitrile_nitrogen": "nt",
    "amide_like_fluorine_on_sp3_carbon": "f",
    "carbonate_like_fluorine_on_sp3_carbon": "f",
    "amide_like_amine_nitrogen_primary": "na",
    "amide_like_amine_nitrogen_secondary": "na",
    "amide_like_amine_nitrogen_tertiary": "na",
    "carbonate_like_amine_nitrogen_primary": "na",
    "carbonate_like_amine_nitrogen_secondary": "na",
    "carbonate_like_amine_nitrogen_tertiary": "na",
    "amide_like_alkene_carbon_sp2": "c=",
    "carbonate_like_alkene_carbon_sp2": "c=",
    "amide_like_alkyne_carbon_sp": "ct",
    "carbonate_like_alkyne_carbon_sp": "ct",
}

PHASE1_CARBON_RULE_IDS = {
    "carbonate_like_alkyl_methyl",
    "amide_like_alkyl_methyl",
    "carbonate_like_alkyl_methyl_amino",
    "amide_like_alkyl_methyl_amino",
    "carbonate_like_alkyl_ch2_oxygen",
    "amide_like_alkyl_ch2_oxygen",
    "carbonate_like_alkyl_ch2_nitrogen",
    "amide_like_alkyl_ch2_nitrogen",
    "carbonate_like_alkyl_ch2_sulfur",
    "amide_like_alkyl_ch2_sulfur",
    "carbonate_like_alkyl_ch2_dialkyl",
    "amide_like_alkyl_ch2_dialkyl",
    "carbonate_like_alkyl_ch_oxygen",
    "amide_like_alkyl_ch_oxygen",
    "carbonate_like_alkyl_ch_nitrogen",
    "amide_like_alkyl_ch_nitrogen",
    "carbonate_like_alkyl_c_quaternary_oxygen",
    "amide_like_alkyl_c_quaternary_oxygen",
    "carbonate_like_alkyl_c_quaternary_nitrogen",
    "amide_like_alkyl_c_quaternary_nitrogen",
    "carbonate_like_alkyl_ch_sulfur",
    "amide_like_alkyl_ch_sulfur",
    "carbonate_like_alkyl_c_quaternary",
    "amide_like_alkyl_c_quaternary",
    "carbonate_like_alkyl_ch_tertiary",
    "amide_like_alkyl_ch_tertiary",
    "carbonate_like_fluorinated_sp3_carbon",
    "amide_like_fluorinated_sp3_carbon",
}

PHASE1_HYDROGEN_ON_CARBON_RULE_IDS = {
    "carbonate_like_hydrogen_on_h3n1_carbon",
    "amide_like_hydrogen_on_h3n1_carbon",
    "carbonate_like_hydrogen_on_sp3_alpha_oxygen",
    "amide_like_hydrogen_on_sp3_alpha_oxygen",
    "carbonate_like_hydrogen_on_sp3_alpha_nitrogen",
    "amide_like_hydrogen_on_sp3_alpha_nitrogen",
    "carbonate_like_hydrogen_on_sp3_alpha_sulfur",
    "amide_like_hydrogen_on_sp3_alpha_sulfur",
    "carbonate_like_hydrogen_on_sp3_methyl",
    "amide_like_hydrogen_on_sp3_methyl",
    "carbonate_like_hydrogen_on_sp3_methylene",
    "amide_like_hydrogen_on_sp3_methylene",
    "carbonate_like_hydrogen_on_sp3_tertiary_carbon",
    "amide_like_hydrogen_on_sp3_tertiary_carbon",
    "carbonate_like_hydrogen_on_sp3_alpha_oxygen_tertiary",
    "amide_like_hydrogen_on_sp3_alpha_oxygen_tertiary",
    "carbonate_like_hydrogen_on_sp3_alpha_nitrogen_tertiary",
    "amide_like_hydrogen_on_sp3_alpha_nitrogen_tertiary",
    "carbonate_like_hydrogen_on_sp3_alpha_sulfur_tertiary",
    "amide_like_hydrogen_on_sp3_alpha_sulfur_tertiary",
    "carbonate_like_hydrogen_on_fluorinated_carbon",
    "amide_like_hydrogen_on_fluorinated_carbon",
    "carbonate_like_hydrogen_on_vinylic_carbon",
    "amide_like_hydrogen_on_vinylic_carbon",
    "carbonate_like_hydrogen_on_alkynyl_carbon",
    "amide_like_hydrogen_on_alkynyl_carbon",
    "carbonate_like_hydrogen_on_carbonyl_carbon",
    "amide_like_hydrogen_on_carbonyl_carbon",
}

PHASE1_GENERIC_SULFUR_RULE_IDS = {
    "carbonate_like_thioether_sulfur_dialkyl",
    "amide_like_thioether_sulfur_dialkyl",
    "carbonate_like_thioether_sulfur_thiol",
    "amide_like_thioether_sulfur_thiol",
}

PHASE1_GENERIC_SULFUR_H_RULE_IDS = {
    "carbonate_like_hydrogen_on_thiol_sulfur",
    "amide_like_hydrogen_on_thiol_sulfur",
}

SECTIONS_BY_TERM_KIND = {
    "bond_main": ("quartic_bond", "bond"),
    "angle_main": ("quartic_angle", "angle"),
    "angle_bb": ("bond-bond", "angle"),
    "angle_ba": ("bond-angle", "angle"),
    "dihedral_main": ("torsion_3", "torsion"),
    "dihedral_bb13": ("bond-bond_1_3", "torsion"),
    "dihedral_ebt": ("end_bond-torsion_3", "torsion"),
    "dihedral_mbt": ("middle_bond-torsion_3", "torsion"),
    "dihedral_at": ("angle-torsion_3", "torsion"),
    "dihedral_aat": ("angle-angle-torsion_1", "torsion"),
    "improper_main": ("wilson_out_of_plane", "oop"),
    "improper_aa": ("angle-angle", "oop"),
    "nonbonded": ("nonbond(9-6)", "nonbond"),
    "bond_increment": ("bond_increments", "bond"),
}

ANALOG_AUTO_EQUIVALENTS = {
    "cz": "c_1",
    "c_2": "c_1",
    "oo": "o_1",
    "oz": "o_2",
    "n_2": "n",
    "hn2": "h*",
    "ho2": "h*",
    "oh": "o",
    "ho": "h*",
}


def build_phase1_pcff_atom_index(ir_component: dict, perception_component: dict, typing_component: dict) -> dict[int, dict]:
    atom_contexts = _build_atom_contexts(ir_component, perception_component)
    typed_atoms = {atom["canonical_index"]: atom for atom in typing_component["atoms"]}
    contexts = {context["canonical_index"]: context for context in atom_contexts}
    pcff_index: dict[int, dict] = {}
    for canonical_index, typed_atom in typed_atoms.items():
        if typed_atom["status"] != "assigned":
            continue
        pcff_type = _phase1_pcff_type_for_atom(typed_atom, contexts[canonical_index], contexts)
        if pcff_type is None:
            continue
        pcff_index[canonical_index] = {
            "canonical_index": canonical_index,
            "pcff_type": pcff_type,
            "assigned_family": typed_atom["assigned_family"],
            "matched_rule_id": typed_atom["matched_rule_id"],
            "bridge_source": "phase1_pcff_bridge_v1",
        }
    return pcff_index


def resolve_bonded_interaction_from_frc(
    kind: str,
    atom_indices: list[int],
    pcff_atom_index: dict[int, dict],
) -> tuple[dict | None, dict | None]:
    reference = load_pcff_frc()
    if any(index not in pcff_atom_index for index in atom_indices):
        return None, None

    atom_types = [pcff_atom_index[index]["pcff_type"] for index in atom_indices]
    if kind == "bond":
        return _resolve_bond_parameters(reference, atom_types)
    if kind == "angle":
        return _resolve_angle_parameters(reference, atom_types)
    if kind == "dihedral":
        return _resolve_dihedral_parameters(reference, atom_types)
    if kind == "improper":
        return _resolve_improper_parameters(reference, atom_types)
    raise ValueError(f"Unsupported bonded interaction kind {kind!r}")


def resolve_bonded_atom_types_from_frc(
    kind: str,
    atom_types: list[str],
) -> tuple[dict | None, dict | None]:
    reference = load_pcff_frc()
    if kind == "bond":
        return _resolve_bond_parameters(reference, atom_types)
    if kind == "angle":
        return _resolve_angle_parameters(reference, atom_types)
    if kind == "dihedral":
        return _resolve_dihedral_parameters(reference, atom_types)
    if kind == "improper":
        return _resolve_improper_parameters(reference, atom_types)
    raise ValueError(f"Unsupported bonded interaction kind {kind!r}")


def resolve_nonbonded_atom_from_frc(pcff_type: str) -> tuple[dict | None, dict | None]:
    reference = load_pcff_frc()
    return _resolve_nonbonded_self(reference, pcff_type)


def resolve_phase1_bond_increment_charges(
    ir_component: dict,
    pcff_atom_index: dict[int, dict],
) -> tuple[dict[int, dict] | None, list[dict]]:
    reference = load_pcff_frc()
    if not pcff_atom_index:
        return None, []

    charges = {atom["canonical_index"]: 0.0 for atom in ir_component["atoms"]}
    diagnostics = []
    provenance_by_atom: dict[int, list[dict]] = {atom["canonical_index"]: [] for atom in ir_component["atoms"]}
    for bond in ir_component["bonds"]:
        left, right = bond["atom_indices"]
        if left not in pcff_atom_index or right not in pcff_atom_index:
            return None, []
        left_type = pcff_atom_index[left]["pcff_type"]
        right_type = pcff_atom_index[right]["pcff_type"]
        resolved, provenance = _resolve_bond_increment(reference, left_type, right_type)
        if resolved is None or provenance is None:
            diagnostics.append(
                {
                    "scope": "charge",
                    "code": "missing_bond_increment",
                    "atom_indices": [left, right],
                    "pcff_types": [left_type, right_type],
                    "message": f"Missing bond increment for {left_type}-{right_type}",
                }
            )
            return None, diagnostics
        delta_left, delta_right = resolved
        charges[left] += delta_left
        charges[right] += delta_right
        provenance_by_atom[left].append(copy.deepcopy(provenance))
        provenance_by_atom[right].append(copy.deepcopy(provenance))

    return (
        {
            canonical_index: {
                "source": "bond_increments",
                "value": round(value, 8),
                "provenance": provenance_by_atom[canonical_index],
            }
            for canonical_index, value in charges.items()
        },
        diagnostics,
    )


@lru_cache(maxsize=1)
def load_pcff_frc(path: str | Path | None = None) -> dict:
    frc_path = DEFAULT_FRC_PATH if path is None else Path(path)
    occurrences = _parse_frc_occurrences(frc_path)
    exact = {
        "equivalence": _parse_equivalence(occurrences["equivalence"][0]),
        "quartic_bond": _parse_table(
            occurrences["quartic_bond"][0],
            arity=2,
            canonicalizer=_canonicalize_pair_types,
            fields=("r0_angstrom", "k2_kcal_mol_per_a2", "k3_kcal_mol_per_a3", "k4_kcal_mol_per_a4"),
        ),
        "quartic_angle": _parse_table(
            occurrences["quartic_angle"][0],
            arity=3,
            canonicalizer=_canonicalize_angle_types,
            fields=("theta0_deg", "k2_kcal_mol", "k3_kcal_mol", "k4_kcal_mol"),
        ),
        "torsion_3": _parse_table(
            occurrences["torsion_3"][0],
            arity=4,
            canonicalizer=_canonicalize_dihedral_types,
            fields=("k1_kcal_mol", "phi1_deg", "k2_kcal_mol", "phi2_deg", "k3_kcal_mol", "phi3_deg"),
        ),
        "bond-bond": _parse_table(
            occurrences["bond-bond"][0],
            arity=3,
            canonicalizer=_canonicalize_angle_types,
            fields=("k_kcal_mol_per_a2",),
        ),
        "bond-angle": _parse_table(
            occurrences["bond-angle"][0],
            arity=3,
            canonicalizer=_canonicalize_angle_types,
            fields=("k1_kcal_mol_per_a", "k2_kcal_mol_per_a"),
        ),
        "bond-bond_1_3": _parse_table(
            occurrences["bond-bond_1_3"][0],
            arity=4,
            canonicalizer=_canonicalize_dihedral_types,
            fields=("k_kcal_mol_per_a2",),
        ),
        "end_bond-torsion_3": _parse_table(
            occurrences["end_bond-torsion_3"][0],
            arity=4,
            canonicalizer=_canonicalize_dihedral_types,
            fields=(
                "f1_1_kcal_mol_per_a",
                "f2_1_kcal_mol_per_a",
                "f3_1_kcal_mol_per_a",
                "f1_2_kcal_mol_per_a",
                "f2_2_kcal_mol_per_a",
                "f3_2_kcal_mol_per_a",
            ),
        ),
        "middle_bond-torsion_3": _parse_table(
            occurrences["middle_bond-torsion_3"][0],
            arity=4,
            canonicalizer=_canonicalize_dihedral_types,
            fields=("f1_kcal_mol_per_a", "f2_kcal_mol_per_a", "f3_kcal_mol_per_a"),
        ),
        "angle-torsion_3": _parse_table(
            occurrences["angle-torsion_3"][0],
            arity=4,
            canonicalizer=_canonicalize_dihedral_types,
            fields=(
                "f1_1_kcal_mol",
                "f2_1_kcal_mol",
                "f3_1_kcal_mol",
                "f1_2_kcal_mol",
                "f2_2_kcal_mol",
                "f3_2_kcal_mol",
            ),
        ),
        "angle-angle-torsion_1": _parse_table(
            occurrences["angle-angle-torsion_1"][0],
            arity=4,
            canonicalizer=_canonicalize_dihedral_types,
            fields=("k_kcal_mol",),
        ),
        "wilson_out_of_plane": _merge_table_occurrences(
            occurrences["wilson_out_of_plane"],
            arity=4,
            canonicalizer=_canonicalize_frc_improper_row_types,
            fields=("k0_kcal_mol", "chi0_deg"),
        ),
        "angle-angle": _parse_table(
            occurrences["angle-angle"][0],
            arity=4,
            canonicalizer=_canonicalize_frc_improper_row_types,
            fields=("k_kcal_mol",),
        ),
        "nonbond(9-6)": _parse_table(
            occurrences["nonbond(9-6)"][0],
            arity=1,
            canonicalizer=lambda atom_types: tuple(atom_types),
            fields=("sigma_angstrom", "epsilon_kcal_mol"),
        ),
        "bond_increments": _parse_table(
            occurrences["bond_increments"][0],
            arity=2,
            canonicalizer=lambda atom_types: tuple(atom_types),
            fields=("delta_ij", "delta_ji"),
            keep_direction=True,
        ),
    }
    return {
        "frc_path": str(frc_path),
        "equivalence": exact["equivalence"],
        "tables": {name: value for name, value in exact.items() if name != "equivalence"},
    }


def _phase1_pcff_type_for_atom(typed_atom: dict, context: dict, contexts: dict[int, dict]) -> str | None:
    rule_id = typed_atom["matched_rule_id"]
    if rule_id in PHASE1_DIRECT_PCFF_TYPE_BY_RULE_ID:
        return PHASE1_DIRECT_PCFF_TYPE_BY_RULE_ID[rule_id]
    if rule_id in PHASE1_CARBON_RULE_IDS:
        return "c"
    if rule_id in PHASE1_HYDROGEN_ON_CARBON_RULE_IDS:
        return "h"
    if rule_id in PHASE1_GENERIC_SULFUR_RULE_IDS:
        return "s"
    if rule_id in PHASE1_GENERIC_SULFUR_H_RULE_IDS:
        return "h"
    if rule_id == "carbonate_like_carbonyl_carbon":
        if context["single_bond_neighbor_element_counts"].get("H", 0) == 1:
            return "c_1"
        return "cz"
    if rule_id == "carbonate_like_carbonyl_oxygen":
        attached = context["attached_atom"]
        if attached is not None and attached["single_bond_neighbor_element_counts"].get("H", 0) == 1:
            return "o_1"
        return "oo"
    if rule_id in {"carbonate_like_hydroxyl_oxygen", "amide_like_hydroxyl_oxygen"}:
        attached = context["attached_atom"]
        if attached is not None and attached["is_carbonyl_carbon"]:
            return "o_2" if typed_atom["matched_rule_id"].startswith("amide_like_") else "oz"
        return "oh"
    if rule_id in {"carbonate_like_hydrogen_on_hydroxyl", "amide_like_hydrogen_on_hydroxyl"}:
        attached_oxygen = context["attached_atom"]
        if attached_oxygen is not None and attached_oxygen["single_bond_neighbor_carbonyl_carbon_count"] == 1:
            return "ho2"
        return "ho"
    if rule_id == "amide_like_hydrogen_on_amide_n":
        return "hn"
    if rule_id == "carbonate_like_hydrogen_on_amide_n":
        return "hn2"
    if rule_id in {"amide_like_hydrogen_on_amine_n", "carbonate_like_hydrogen_on_amine_n"}:
        return "h*"
    return None


def _resolve_bond_parameters(reference: dict, atom_types: list[str]) -> tuple[dict | None, dict | None]:
    record, provenance = _lookup_term(reference, "bond_main", atom_types)
    if record is None or provenance is None:
        return None, None
    return {"main": copy.deepcopy(record["parameters"])}, provenance


def _resolve_angle_parameters(reference: dict, atom_types: list[str]) -> tuple[dict | None, dict | None]:
    main_record, main_provenance = _lookup_term(reference, "angle_main", atom_types)
    bb_record, bb_provenance = _lookup_term(reference, "angle_bb", atom_types)
    ba_record, ba_provenance = _lookup_term(reference, "angle_ba", atom_types)
    left_bond, left_bond_provenance = _lookup_term(reference, "bond_main", atom_types[:2])
    right_bond, right_bond_provenance = _lookup_term(reference, "bond_main", atom_types[1:])
    if any(
        value is None
        for value in (
            main_record,
            bb_record,
            ba_record,
            left_bond,
            right_bond,
            main_provenance,
            bb_provenance,
            ba_provenance,
            left_bond_provenance,
            right_bond_provenance,
        )
    ):
        return None, None

    return (
        {
            "main": copy.deepcopy(main_record["parameters"]),
            "bb": {
                "k_kcal_mol_per_a2": bb_record["parameters"]["k_kcal_mol_per_a2"],
                "r1_angstrom": left_bond["parameters"]["r0_angstrom"],
                "r2_angstrom": right_bond["parameters"]["r0_angstrom"],
            },
            "ba": {
                "k1_kcal_mol_per_a": ba_record["parameters"]["k1_kcal_mol_per_a"],
                "k2_kcal_mol_per_a": ba_record["parameters"]["k2_kcal_mol_per_a"],
                "r1_angstrom": left_bond["parameters"]["r0_angstrom"],
                "r2_angstrom": right_bond["parameters"]["r0_angstrom"],
            },
        },
        _combine_provenance(
            main=main_provenance,
            bb=bb_provenance,
            ba=ba_provenance,
            left_bond=left_bond_provenance,
            right_bond=right_bond_provenance,
        ),
    )


def _resolve_dihedral_parameters(reference: dict, atom_types: list[str]) -> tuple[dict | None, dict | None]:
    main_record, main_provenance = _lookup_term(reference, "dihedral_main", atom_types)
    bb13_record, bb13_provenance = _lookup_term(reference, "dihedral_bb13", atom_types)
    ebt_record, ebt_provenance = _lookup_term(reference, "dihedral_ebt", atom_types)
    mbt_record, mbt_provenance = _lookup_term(reference, "dihedral_mbt", atom_types)
    at_record, at_provenance = _lookup_term(reference, "dihedral_at", atom_types)
    aat_record, aat_provenance = _lookup_term(reference, "dihedral_aat", atom_types)

    left_bond, left_bond_provenance = _lookup_term(reference, "bond_main", atom_types[:2])
    middle_bond, middle_bond_provenance = _lookup_term(reference, "bond_main", atom_types[1:3])
    right_bond, right_bond_provenance = _lookup_term(reference, "bond_main", atom_types[2:])
    left_angle, left_angle_provenance = _lookup_term(reference, "angle_main", atom_types[:3])
    right_angle, right_angle_provenance = _lookup_term(reference, "angle_main", atom_types[1:])

    if any(
        value is None
        for value in (
            main_record,
            bb13_record,
            ebt_record,
            mbt_record,
            at_record,
            aat_record,
            left_bond,
            middle_bond,
            right_bond,
            left_angle,
            right_angle,
            main_provenance,
            bb13_provenance,
            ebt_provenance,
            mbt_provenance,
            at_provenance,
            aat_provenance,
            left_bond_provenance,
            middle_bond_provenance,
            right_bond_provenance,
            left_angle_provenance,
            right_angle_provenance,
        )
    ):
        return None, None

    return (
        {
            "main": copy.deepcopy(main_record["parameters"]),
            "bb13": {
                "k_kcal_mol_per_a2": bb13_record["parameters"]["k_kcal_mol_per_a2"],
                "r1_angstrom": left_bond["parameters"]["r0_angstrom"],
                "r3_angstrom": right_bond["parameters"]["r0_angstrom"],
            },
            "ebt": {
                **copy.deepcopy(ebt_record["parameters"]),
                "r0_1_angstrom": left_bond["parameters"]["r0_angstrom"],
                "r0_2_angstrom": right_bond["parameters"]["r0_angstrom"],
            },
            "mbt": {
                **copy.deepcopy(mbt_record["parameters"]),
                "r0_angstrom": middle_bond["parameters"]["r0_angstrom"],
            },
            "at": {
                **copy.deepcopy(at_record["parameters"]),
                "theta0_1_deg": left_angle["parameters"]["theta0_deg"],
                "theta0_2_deg": right_angle["parameters"]["theta0_deg"],
            },
            "aat": {
                "k_kcal_mol": aat_record["parameters"]["k_kcal_mol"],
                "theta0_1_deg": left_angle["parameters"]["theta0_deg"],
                "theta0_2_deg": right_angle["parameters"]["theta0_deg"],
            },
        },
        _combine_provenance(
            main=main_provenance,
            bb13=bb13_provenance,
            ebt=ebt_provenance,
            mbt=mbt_provenance,
            at=at_provenance,
            aat=aat_provenance,
            left_bond=left_bond_provenance,
            middle_bond=middle_bond_provenance,
            right_bond=right_bond_provenance,
            left_angle=left_angle_provenance,
            right_angle=right_angle_provenance,
        ),
    )


def _resolve_improper_parameters(reference: dict, atom_types: list[str]) -> tuple[dict | None, dict | None]:
    main_record, main_provenance = _lookup_term(reference, "improper_main", atom_types)
    aa_record, aa_provenance = _lookup_term(reference, "improper_aa", atom_types)
    if main_record is None or main_provenance is None or aa_record is None or aa_provenance is None:
        return None, None

    center = atom_types[0]
    neighbors = atom_types[1:]
    angle_1, angle_1_provenance = _lookup_term(reference, "angle_main", [neighbors[0], center, neighbors[1]])
    angle_2, angle_2_provenance = _lookup_term(reference, "angle_main", [neighbors[0], center, neighbors[2]])
    angle_3, angle_3_provenance = _lookup_term(reference, "angle_main", [neighbors[1], center, neighbors[2]])
    if any(
        value is None
        for value in (angle_1, angle_2, angle_3, angle_1_provenance, angle_2_provenance, angle_3_provenance)
    ):
        return None, None

    return (
        {
            "main": copy.deepcopy(main_record["parameters"]),
            "aa": {
                "k1_kcal_mol": aa_record["parameters"]["k_kcal_mol"],
                "k2_kcal_mol": aa_record["parameters"]["k_kcal_mol"],
                "k3_kcal_mol": aa_record["parameters"]["k_kcal_mol"],
                "theta0_1_deg": angle_1["parameters"]["theta0_deg"],
                "theta0_2_deg": angle_2["parameters"]["theta0_deg"],
                "theta0_3_deg": angle_3["parameters"]["theta0_deg"],
            },
        },
        _combine_provenance(
            main=main_provenance,
            aa=aa_provenance,
            angle_1=angle_1_provenance,
            angle_2=angle_2_provenance,
            angle_3=angle_3_provenance,
        ),
    )


def _resolve_nonbonded_self(reference: dict, pcff_type: str) -> tuple[dict | None, dict | None]:
    record, provenance = _lookup_term(reference, "nonbonded", [pcff_type])
    if record is None or provenance is None:
        return None, None
    return copy.deepcopy(record["parameters"]), provenance


def _resolve_bond_increment(reference: dict, left_type: str, right_type: str) -> tuple[tuple[float, float] | None, dict | None]:
    table = reference["tables"]["bond_increments"]
    exact = table.get((left_type, right_type))
    if exact is not None:
        return (
            (
                exact["parameters"]["delta_ij"],
                exact["parameters"]["delta_ji"],
            ),
            _term_provenance(exact, "exact", (left_type, right_type)),
        )
    reverse_exact = table.get((right_type, left_type))
    if reverse_exact is not None:
        return (
            (
                reverse_exact["parameters"]["delta_ji"],
                reverse_exact["parameters"]["delta_ij"],
            ),
            _term_provenance(reverse_exact, "exact", (right_type, left_type)),
        )

    equivalence = reference["equivalence"]
    left_equiv = equivalence.get(left_type, {}).get("bond", left_type)
    right_equiv = equivalence.get(right_type, {}).get("bond", right_type)
    fallback = table.get((left_equiv, right_equiv))
    if fallback is None:
        reverse_fallback = table.get((right_equiv, left_equiv))
        if reverse_fallback is None:
            analog_left = ANALOG_AUTO_EQUIVALENTS.get(left_equiv, left_equiv)
            analog_right = ANALOG_AUTO_EQUIVALENTS.get(right_equiv, right_equiv)
            analog = table.get((analog_left, analog_right))
            if analog is not None:
                return (
                    (
                        analog["parameters"]["delta_ij"],
                        analog["parameters"]["delta_ji"],
                    ),
                    _term_provenance(analog, "auto_equivalent", (analog_left, analog_right)),
                )
            reverse_analog = table.get((analog_right, analog_left))
            if reverse_analog is None:
                return None, None
            return (
                (
                    reverse_analog["parameters"]["delta_ji"],
                    reverse_analog["parameters"]["delta_ij"],
                ),
                _term_provenance(reverse_analog, "auto_equivalent", (analog_right, analog_left)),
            )
        return (
            (
                reverse_fallback["parameters"]["delta_ji"],
                reverse_fallback["parameters"]["delta_ij"],
            ),
            _term_provenance(reverse_fallback, "equivalence", (right_equiv, left_equiv)),
        )
    return (
        (
            fallback["parameters"]["delta_ij"],
            fallback["parameters"]["delta_ji"],
        ),
        _term_provenance(fallback, "equivalence", (left_equiv, right_equiv)),
    )


def _lookup_term(reference: dict, term_kind: str, atom_types: list[str]) -> tuple[dict | None, dict | None]:
    section_name, equivalence_field = SECTIONS_BY_TERM_KIND[term_kind]
    table = reference["tables"][section_name]
    canonicalizer = _canonicalizer_for_arity(len(atom_types), improper=(term_kind.startswith("improper")))
    key = canonicalizer(atom_types)
    record = table.get(key)
    if record is not None:
        return record, _term_provenance(record, "exact", key)
    wildcard_record, wildcard_key = _lookup_wildcard_record(table, key)
    if wildcard_record is not None and wildcard_key is not None:
        return wildcard_record, _term_provenance(wildcard_record, "exact", wildcard_key, used_wildcard=True)

    equivalence = reference["equivalence"]
    mapped_types = [equivalence.get(atom_type, {}).get(equivalence_field, atom_type) for atom_type in atom_types]
    mapped_key = canonicalizer(mapped_types)
    mapped_record = table.get(mapped_key)
    if mapped_record is not None:
        return mapped_record, _term_provenance(mapped_record, "equivalence", mapped_key)
    wildcard_mapped_record, wildcard_mapped_key = _lookup_wildcard_record(table, mapped_key)
    if wildcard_mapped_record is not None and wildcard_mapped_key is not None:
        return wildcard_mapped_record, _term_provenance(
            wildcard_mapped_record,
            "equivalence",
            wildcard_mapped_key,
            used_wildcard=True,
        )

    analog_types = [ANALOG_AUTO_EQUIVALENTS.get(atom_type, atom_type) for atom_type in mapped_types]
    if analog_types == mapped_types or term_kind in {"nonbonded", "bond_increment"}:
        return None, None
    analog_key = canonicalizer(analog_types)
    analog_record = table.get(analog_key)
    if analog_record is not None:
        return analog_record, _term_provenance(analog_record, "auto_equivalent", analog_key)
    wildcard_analog_record, wildcard_analog_key = _lookup_wildcard_record(table, analog_key)
    if wildcard_analog_record is not None and wildcard_analog_key is not None:
        return wildcard_analog_record, _term_provenance(
            wildcard_analog_record,
            "auto_equivalent",
            wildcard_analog_key,
            used_wildcard=True,
        )
    return None, None


def _parse_frc_occurrences(path: Path) -> dict[str, list[list[dict]]]:
    sections: dict[str, list[list[dict]]] = {}
    current_name = None
    current_rows: list[dict] | None = None
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            current_name = stripped.split()[0][1:]
            sections.setdefault(current_name, [])
            current_rows = []
            sections[current_name].append(current_rows)
            continue
        if current_rows is None:
            continue
        if stripped[0] in {"!", ">", "@"}:
            continue
        current_rows.append({"line_number": line_number, "tokens": stripped.split()})
    return sections


def _parse_equivalence(rows: list[dict]) -> dict[str, dict]:
    table: dict[str, dict] = {}
    for row in rows:
        tokens = row["tokens"]
        if len(tokens) < 8:
            continue
        version = float(tokens[0])
        reference = int(tokens[1])
        atom_type = tokens[2]
        record = {
            "nonbond": tokens[3],
            "bond": tokens[4],
            "angle": tokens[5],
            "torsion": tokens[6],
            "oop": tokens[7],
            "_meta": {
                "line_number": row["line_number"],
                "version": version,
                "reference": reference,
            },
        }
        current = table.get(atom_type)
        if current is None or _meta_key(record["_meta"]) > _meta_key(current["_meta"]):
            table[atom_type] = record
    return table


def _parse_table(
    rows: list[dict],
    *,
    arity: int,
    canonicalizer,
    fields: tuple[str, ...],
    keep_direction: bool = False,
) -> dict[tuple[str, ...], dict]:
    table: dict[tuple[str, ...], dict] = {}
    for row in rows:
        tokens = row["tokens"]
        if len(tokens) < 2 + arity + len(fields):
            continue
        version = float(tokens[0])
        reference = int(tokens[1])
        atom_types = tokens[2 : 2 + arity]
        parameter_tokens = tokens[2 + arity : 2 + arity + len(fields)]
        key = tuple(atom_types) if keep_direction else canonicalizer(atom_types)
        record = {
            "atom_types": list(atom_types),
            "parameters": {field: float(value) for field, value in zip(fields, parameter_tokens)},
            "meta": {
                "line_number": row["line_number"],
                "version": version,
                "reference": reference,
            },
        }
        current = table.get(key)
        if current is None or _meta_key(record["meta"]) > _meta_key(current["meta"]):
            table[key] = record
    return table


def _merge_table_occurrences(
    occurrences: list[list[dict]],
    *,
    arity: int,
    canonicalizer,
    fields: tuple[str, ...],
    keep_direction: bool = False,
) -> dict[tuple[str, ...], dict]:
    table: dict[tuple[str, ...], dict] = {}
    for rows in occurrences:
        parsed = _parse_table(
            rows,
            arity=arity,
            canonicalizer=canonicalizer,
            fields=fields,
            keep_direction=keep_direction,
        )
        for key, record in parsed.items():
            current = table.get(key)
            if current is None or _meta_key(record["meta"]) > _meta_key(current["meta"]):
                table[key] = record
    return table


def _canonicalize_pair_types(atom_types: list[str]) -> tuple[str, ...]:
    return _canonicalize_linear_types(atom_types)


def _canonicalize_angle_types(atom_types: list[str]) -> tuple[str, ...]:
    return _canonicalize_linear_types(atom_types)


def _canonicalize_dihedral_types(atom_types: list[str]) -> tuple[str, ...]:
    return _canonicalize_linear_types(atom_types)


def _canonicalize_improper_types(atom_types: list[str]) -> tuple[str, ...]:
    center = atom_types[0]
    neighbors = sorted(atom_types[1:])
    return tuple([center, *neighbors])


def _canonicalize_frc_improper_row_types(atom_types: list[str]) -> tuple[str, ...]:
    center = atom_types[1]
    neighbors = sorted([atom_types[0], atom_types[2], atom_types[3]])
    return tuple([center, *neighbors])


def _canonicalizer_for_arity(arity: int, *, improper: bool) -> callable:
    if improper:
        return _canonicalize_improper_types
    if arity == 1:
        return lambda atom_types: tuple(atom_types)
    if arity == 2:
        return _canonicalize_pair_types
    if arity == 3:
        return _canonicalize_angle_types
    if arity == 4:
        return _canonicalize_dihedral_types
    raise ValueError(f"Unsupported interaction arity {arity!r}")


def _term_provenance(
    record: dict,
    source_resolution: str,
    resolved_key: tuple[str, ...],
    *,
    used_wildcard: bool = False,
) -> dict:
    provenance = {
        "source_kind": "pcff_frc_exact_table",
        "source_file": "frc_file/pcff.frc",
        "source_resolution": source_resolution,
        "resolved_key": list(resolved_key),
        "line_number": record["meta"]["line_number"],
        "version": record["meta"]["version"],
        "reference": record["meta"]["reference"],
    }
    if used_wildcard:
        provenance["used_wildcard"] = True
    return provenance


def _combine_provenance(**entries: dict) -> dict:
    return {
        "source_kind": "pcff_frc_composite_resolution",
        "source_file": "frc_file/pcff.frc",
        "subterms": {name: copy.deepcopy(value) for name, value in entries.items()},
        "source_resolution": "composite",
    }


def _meta_key(meta: dict) -> tuple[float, int, int]:
    return (float(meta["version"]), int(meta["reference"]), int(meta["line_number"]))


def _canonicalize_linear_types(atom_types: list[str]) -> tuple[str, ...]:
    forward = tuple(atom_types)
    reverse = tuple(reversed(atom_types))
    return reverse if reverse < forward else forward


def _lookup_wildcard_record(
    table: dict[tuple[str, ...], dict],
    canonical_key: tuple[str, ...],
) -> tuple[dict | None, tuple[str, ...] | None]:
    best_key = None
    best_record = None
    best_score = None
    for key, record in table.items():
        if len(key) != len(canonical_key):
            continue
        if any(expected != "*" and expected != actual for expected, actual in zip(key, canonical_key)):
            continue
        score = (-sum(1 for token in key if token == "*"), _meta_key(record["meta"]))
        if best_score is None or score > best_score:
            best_score = score
            best_key = key
            best_record = record
    return best_record, best_key
