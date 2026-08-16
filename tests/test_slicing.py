import pytest
from dental_agent.data.slicing import compute_slice_assignment, get_slice_ids

def test_compute_slice_assignment_validation():
    with pytest.raises(ValueError):
        compute_slice_assignment([1, 2], 0)

def test_get_slice_ids_validation():
    with pytest.raises(ValueError):
        get_slice_ids([1, 2, 3], total_slices=5, slice_index=0)  # Under bounds (1-indexed)
    with pytest.raises(ValueError):
        get_slice_ids([1, 2, 3], total_slices=5, slice_index=6)  # Over bounds

@pytest.mark.parametrize("total_slices", [1, 10, 12, 14, 20])
def test_slicing_properties(total_slices):
    image_ids = list(range(100, 100 + 678))  # Simulate 678 images
    
    assignment = compute_slice_assignment(image_ids, total_slices, seed=42)
    
    # (a) union of all slices' ids equals the full id set
    assigned_ids = set(assignment.keys())
    assert assigned_ids == set(image_ids)
    
    # (b) no id appears in more than one slice
    # This is intrinsically guaranteed by the dict structure of assignment
    
    # (c) max/min slice size differ by at most 1
    slice_sizes = [0] * total_slices
    for sid in assignment.values():
        slice_sizes[sid] += 1
        
    assert max(slice_sizes) - min(slice_sizes) <= 1
    
    # (d) calling get_slice_ids twice with same arguments returns identical output
    slice_1_a = get_slice_ids(image_ids, total_slices, 1, seed=42)
    slice_1_b = get_slice_ids(image_ids, total_slices, 1, seed=42)
    assert slice_1_a == slice_1_b
    
    # (e) all slice parts reconstruct the original
    reconstructed = []
    for i in range(1, total_slices + 1):
        reconstructed.extend(get_slice_ids(image_ids, total_slices, i, seed=42))
    assert sorted(reconstructed) == sorted(image_ids)

    # (f) calling with a different seed produces a different partition
    if total_slices > 1:
        assignment_diff_seed = compute_slice_assignment(image_ids, total_slices, seed=99)
        assert assignment != assignment_diff_seed
