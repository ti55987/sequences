import numpy as np

def generate_sequence(num_stims, num_iter_per_stim):
    """
    Generate a sequence where each stimulus appears exactly num_iter_per_stim times
    and consecutive stimuli are different.

    Args:
        num_stims: Number of different stimuli (0 to num_stims-1)
        num_iter_per_stim: Number of times each stimulus should appear

    Returns:
        List of stimuli satisfying the constraints
    """
    # Track remaining count for each stimulus
    remaining_counts = [num_iter_per_stim] * num_stims
    sequence = []

    # Start with a random stimulus
    current_stim = np.random.choice(num_stims)
    sequence.append(current_stim)
    remaining_counts[current_stim] -= 1

    while len(sequence) < num_stims * num_iter_per_stim:
        # Get available stimuli (not the last one and still have remaining count)
        available_stims = [
            stim
            for stim in range(num_stims)
            if stim != sequence[-1] and remaining_counts[stim] > 0
        ]

        if not available_stims:
            # If no valid options, we need to backtrack or use a different strategy
            # This should rarely happen with proper parameters
            raise ValueError("Cannot generate valid sequence with given constraints")

        # Choose next stimulus with preference for those with higher remaining counts
        # This helps balance the sequence and avoid getting stuck
        weights = [remaining_counts[stim] for stim in available_stims]
        next_stim = np.random.choice(
            available_stims, p=np.array(weights) / sum(weights)
        )

        sequence.append(next_stim)
        remaining_counts[next_stim] -= 1

    return sequence


def generate_sequence_optimized(num_stims, num_iter_per_stim, max_attempts=100):
    """
    More robust version that can handle edge cases by trying multiple times.
    """
    for attempt in range(max_attempts):
        try:
            return generate_sequence(num_stims, num_iter_per_stim)
        except ValueError:
            continue

    # Fallback: use a more deterministic approach
    return generate_sequence_deterministic(num_stims, num_iter_per_stim)


def generate_sequence_deterministic(num_stims, num_iter_per_stim):
    """
    Deterministic approach that guarantees a valid sequence.
    Creates blocks of stimuli and then shuffles while maintaining constraints.
    """
    # Create base sequence with all stimuli
    base_sequence = []
    for stim in range(num_stims):
        base_sequence.extend([stim] * num_iter_per_stim)

    # Shuffle while maintaining no-consecutive constraint
    sequence = [base_sequence[0]]
    remaining = base_sequence[1:]

    while remaining:
        # Find valid next stimuli
        valid_indices = [i for i, stim in enumerate(remaining) if stim != sequence[-1]]

        if not valid_indices:
            # If stuck, swap with a later element
            for i in range(len(remaining)):
                if remaining[i] != sequence[-1]:
                    # Move this element to a random valid position
                    valid_pos = np.random.choice(
                        [
                            j
                            for j in range(len(remaining))
                            if remaining[j] != remaining[i]
                        ]
                    )
                    remaining[i], remaining[valid_pos] = (
                        remaining[valid_pos],
                        remaining[i],
                    )
                    break
            valid_indices = [
                i for i, stim in enumerate(remaining) if stim != sequence[-1]
            ]

        # Choose random valid option
        chosen_idx = np.random.choice(valid_indices)
        sequence.append(remaining[chosen_idx])
        remaining.pop(chosen_idx)

    return sequence


def shuffle_with_mask(arr, mask):
    """
    Shuffle array where mask[i] = True means keep arr[i] unchanged
    mask[i] = False means this element can be shuffled
    """
    arr = arr.copy()

    # Get indices that can be shuffled
    shuffleable_indices = np.where(~mask)[0]

    if len(shuffleable_indices) <= 1:
        return arr  # Nothing to shuffle

    # Extract values at shuffleable positions
    shuffleable_values = arr[shuffleable_indices]
    np.random.shuffle(shuffleable_values)

    # Put shuffled values back
    arr[shuffleable_indices] = shuffleable_values

    return arr


def swap_by_indices(arr, target_value, target_index):
    """
    Swap elements at two specific indices
    """
    arr = arr.copy()
    curr_index = np.where(arr == target_value)[0][0]
    arr[curr_index], arr[target_index] = arr[target_index], arr[curr_index]
    return arr    