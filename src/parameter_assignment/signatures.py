from __future__ import annotations


def canonicalize_bond(atom_indices: list[int], atom_families: list[str]) -> tuple[list[int], list[str], str]:
    return _canonicalize_linear("bond", atom_indices, atom_families)


def canonicalize_angle(atom_indices: list[int], atom_families: list[str]) -> tuple[list[int], list[str], str]:
    return _canonicalize_linear("angle", atom_indices, atom_families)


def canonicalize_dihedral(atom_indices: list[int], atom_families: list[str]) -> tuple[list[int], list[str], str]:
    return _canonicalize_linear("dihedral", atom_indices, atom_families)


def canonicalize_improper(
    center_index: int,
    neighbor_indices: list[int],
    center_family: str,
    neighbor_families: list[str],
) -> tuple[list[int], list[str], str]:
    ordered = sorted(
        zip(neighbor_families, neighbor_indices),
        key=lambda item: (item[0], item[1]),
    )
    ordered_families = [center_family, *(family for family, _ in ordered)]
    ordered_indices = [center_index, *(atom_index for _, atom_index in ordered)]
    return ordered_indices, ordered_families, format_signature("improper", ordered_families)


def format_signature(kind: str, atom_families: list[str]) -> str:
    return f"{kind}({('|'.join(atom_families))})"


def _canonicalize_linear(
    kind: str,
    atom_indices: list[int],
    atom_families: list[str],
) -> tuple[list[int], list[str], str]:
    forward = (tuple(atom_families), tuple(atom_indices))
    reverse = (tuple(reversed(atom_families)), tuple(reversed(atom_indices)))
    if reverse < forward:
        chosen_indices = list(reversed(atom_indices))
        chosen_families = list(reversed(atom_families))
    else:
        chosen_indices = list(atom_indices)
        chosen_families = list(atom_families)
    return chosen_indices, chosen_families, format_signature(kind, chosen_families)
