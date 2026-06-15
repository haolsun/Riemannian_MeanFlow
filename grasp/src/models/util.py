import torch

from src.data.util import GraspData


def duplicate_batch_to_size(batch):
    """
    Duplicates elements in a batch until it reaches the target batch size.
    Works with tensors, dictionaries, tuples, strings, and basic types.
    """
    target_batch_size = 256
    # Handle basic types (str, int, float) by just returning them as is
    if isinstance(batch, (str, int, float)):
        return batch

    if isinstance(batch, torch.Tensor):
        # For single tensor batch
        if len(batch.shape) == 0:  # Scalar tensor
            return batch

        current_size = batch.size(0)
        if current_size >= target_batch_size:
            raise RuntimeError(
                "Current batch size is already larger than target size, this is not acceptable"
            )

        # Calculate how many full copies we need and the remainder
        num_copies = target_batch_size // current_size
        remainder = target_batch_size % current_size

        # For the SDF tensor (which should have shape [48, 48, 48]),
        if batch.shape == torch.Size([48, 48, 48]):
            # Expand to [target_batch_size, 48, 48, 48]
            duplicated = batch.unsqueeze(0).expand(target_batch_size, 48, 48, 48)
            return duplicated
        else:
            # For other tensors, use normal duplication
            duplicated = batch.repeat(
                num_copies, *(1 for _ in range(len(batch.shape) - 1))
            )
            if remainder > 0:
                duplicated = torch.cat([duplicated, batch[:remainder]], dim=0)
            return duplicated

    elif isinstance(batch, dict):
        # For dictionary of tensors
        return {k: duplicate_batch_to_size(v) for k, v in batch.items()}

    elif isinstance(batch, tuple):
        # Special handling for namedtuples
        if hasattr(batch, "_fields"):  # Check if it's a namedtuple
            field_values = [
                duplicate_batch_to_size(getattr(batch, field))
                for field in batch._fields
            ]
            return type(batch)(*field_values)
        # Regular tuple handling
        return type(batch)(duplicate_batch_to_size(x) for x in batch)

    elif isinstance(batch, list):
        # For lists
        return type(batch)(duplicate_batch_to_size(x) for x in batch)

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def get_grasp_from_batch(batch, idx=0):
    return GraspData(
        rotation=batch.rotation[idx],
        translation=batch.translation[idx],
        sdf=batch.sdf[idx],
        mesh_path=batch.mesh_path[idx],
        dataset_mesh_scale=batch.dataset_mesh_scale[idx],
        normalization_scale=batch.normalization_scale[idx],
        centroid=batch.centroid[idx],
    )
