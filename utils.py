from typing import List


def flatten_indices(batch_indices: List[List[int]], context_length: int) -> List[int]:
    """
    Flattens the batch_indices list into 1D
    Assumes each batch has context_length indices

    e.g [[1,3], [0, 3]]  -> [1,3,4,7]

    [0,3] -> [4,7] because it's the 2nd batch
    """
    flattened_indices: List[int] = []
    for batch_id, batch in enumerate(batch_indices):
        batch_start_idx = batch_id * context_length
        new_indices = [batch_start_idx + idx for idx in batch]
        flattened_indices += new_indices
    return flattened_indices
