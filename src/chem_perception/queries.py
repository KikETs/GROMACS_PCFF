from __future__ import annotations

from collections import deque


def query_neighbor_shell(
    report: dict,
    atom_index: int,
    depth: int,
    *,
    heavy_only: bool = False,
) -> dict:
    if depth < 1:
        raise ValueError("depth must be >= 1")

    component = report["components"][0]
    atoms = {atom["canonical_index"]: atom for atom in component["atoms"]}
    if atom_index not in atoms:
        raise KeyError(atom_index)

    visited = {atom_index}
    frontier = deque([(atom_index, 0)])
    matches = []
    while frontier:
        current, current_depth = frontier.popleft()
        if current_depth == depth:
            if current != atom_index:
                matches.append(current)
            continue
        for neighbor_index in atoms[current]["neighbor_indices"]:
            if neighbor_index in visited:
                continue
            visited.add(neighbor_index)
            frontier.append((neighbor_index, current_depth + 1))

    if heavy_only:
        matches = [index for index in matches if atoms[index]["element"] != "H"]

    element_counts = {}
    for neighbor_index in sorted(matches):
        element = atoms[neighbor_index]["element"]
        element_counts.setdefault(element, 0)
        element_counts[element] += 1

    return {
        "atom_index": atom_index,
        "depth": depth,
        "atom_indices": sorted(matches),
        "element_counts": dict(sorted(element_counts.items())),
    }
