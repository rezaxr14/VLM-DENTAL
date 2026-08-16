import random

def compute_slice_assignment(
    image_ids: list[int],
    total_slices: int,
    seed: int = 42,
) -> dict[int, int]:
    """Return {image_id: slice_index}. Deterministic for a given (ids, seed, total_slices)."""
    if total_slices < 1:
        raise ValueError("total_slices must be >= 1")

    sorted_ids = sorted(set(image_ids))
    if not sorted_ids:
        return {}

    rng = random.Random(seed)
    shuffled = sorted_ids.copy()
    rng.shuffle(shuffled)

    # Split into chunks as evenly as possible
    base_size = len(shuffled) // total_slices
    remainder = len(shuffled) % total_slices

    chunks = []
    start = 0
    for i in range(total_slices):
        chunk_size = base_size + (1 if i < remainder else 0)
        chunks.append(shuffled[start : start + chunk_size])
        start += chunk_size

    assignment = {img_id: chunk_idx for chunk_idx, chunk in enumerate(chunks) for img_id in chunk}

    # Self-check assertion
    assert sum(len(c) for c in chunks) == len(sorted_ids), "Chunk total size mismatch"
    assert len(assignment) == len(sorted_ids), "Assignment total size mismatch"

    return assignment

def get_slice_ids(
    image_ids: list[int],
    total_slices: int,
    slice_index: int,
    seed: int = 42,
) -> list[int]:
    """Convenience wrapper: image ids assigned to one slice, sorted ascending. (1-indexed)"""
    if not (1 <= slice_index <= total_slices):
        raise ValueError(f"slice_index must be in range 1 to {total_slices}")
    
    assignment = compute_slice_assignment(image_ids, total_slices, seed)
    internal_index = slice_index - 1
    return sorted([img_id for img_id, s_idx in assignment.items() if s_idx == internal_index])
