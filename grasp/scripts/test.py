import pickle
from dataclasses import dataclass
from pathlib import Path

import torch

from data.data_manager import GraspCacheEntry


@dataclass
class GraspResult:
    """Represents a grasp result with mesh and transformation data."""

    mesh_path: Path
    rotations: torch.Tensor  # SO(3) rotations matrix
    translations: torch.Tensor  # R3 translation vector
    norm_scale: float


def find_mesh_path(object_id: str, meshes_dir: Path) -> Path:
    """Find the first matching .obj file for the given object ID."""
    matches = list(meshes_dir.glob(f"**/{object_id}.obj"))
    if not matches:
        raise FileNotFoundError(f"No .obj file found for object_id: {object_id}")
    return matches[0]


def load_grasp_results(grasp_dir: str, meshes_dir: str) -> list[GraspResult]:
    """Load grasp results from pickle files and match with their corresponding meshes."""
    grasp_path = Path(grasp_dir)
    meshes_path = Path(meshes_dir)
    results = []

    # Build mesh lookup cache
    mesh_cache = {path.stem: path for path in meshes_path.glob("**/*.obj")}

    # Process pickle files
    for pkl_path in grasp_path.glob("*.pkl"):
        # Parse filename: "object_id_scale_number.pkl"
        object_id, norm_scale = pkl_path.stem.split("_scale_")

        if object_id not in mesh_cache:
            print(f"Skipping {pkl_path}: No mesh found for {object_id}")
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            results.append(
                GraspResult(
                    mesh_path=mesh_cache[object_id],
                    rotations=data["so3_output"],
                    translations=data["r3_output"],
                    norm_scale=float(norm_scale),
                )
            )

    return results


def match_grasp_cache(
    result: GraspResult, cache: dict[str, GraspCacheEntry]
) -> GraspCacheEntry | None:
    """Find matching cache entry based on mesh name and normalization scale."""
    TOLERANCE = 1e-2
    item_name = result.mesh_path.parent.name
    item_id = result.mesh_path.stem

    for filename, entry in cache.items():
        if not filename.endswith(".h5"):
            raise ValueError(f"Invalid cache filename: {filename}")

        # Parse cache filename: "item_name_item_id_norm_params.h5"
        parts = filename[:-3].split("_")  # Remove .h5 and split
        cache_norm = float(parts[-1])
        cache_id = parts[-2]
        cache_name = "_".join(parts[:-2])  # Handle names with underscores

        if item_name == cache_name and item_id == cache_id:
            print("result.norm_scale", result.norm_scale)
            print("cache_norm", entry.normalization_scale)

            if abs(result.norm_scale - entry.normalization_scale) <= TOLERANCE:
                return entry

    return None


if __name__ == "__main__":
    # Load grasp results and cache
    results = load_grasp_results("grasp_results", "data/meshes")

    with open("data/grasp_cache/grasp_cache.pkl", "rb") as f:
        cache = pickle.load(f)

    print(f"Loaded {len(cache)} cache entries")

    # Example: Find cache match for first result
    match = match_grasp_cache(results[0], cache)
    print(f"Found matching cache entry: {match is not None}")
