import glob
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch

from core.visualize import check_collision
from data.data_manager import GraspCacheEntry
from data.util import CPU_Unpickler, denormalize_translation
from models.util import get_grasp_from_batch


@dataclass
class GraspResult:
    """Represents a grasp result with mesh and transformation data."""

    mesh_path: Path
    rotations: torch.Tensor  # SO(3) rotations matrix
    translations: torch.Tensor  # R3 translation vector
    mesh_scale: float


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
        object_id, mesh_scale = pkl_path.stem.split("_scale_")

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
                    mesh_scale=float(mesh_scale),
                )
            )

    return results


def match_grasp_cache(
    result: GraspResult, cache: dict[str, GraspCacheEntry]
) -> Tuple[str, GraspCacheEntry]:
    """Find matching cache entry based on mesh name and normalization scale.

    Raises:
        ValueError: If multiple matching cache entries are found or if invalid cache filename
    """
    TOLERANCE = 1e-9
    item_name = result.mesh_path.parent.name
    item_id = result.mesh_path.stem

    matches: list[Tuple[str, GraspCacheEntry]] = []

    for filename, entry in cache.items():
        if not filename.endswith(".h5"):
            raise ValueError(f"Invalid cache filename: {filename}")

        # Parse cache filename: "item_name_item_id_norm_params.h5"
        parts = filename[:-3].split("_")  # Remove .h5 and split
        cache_id = parts[-2]
        cache_mesh_scale = float(parts[-1])
        cache_name = "_".join(parts[:-2])  # Handle names with underscores

        if item_name == cache_name and item_id == cache_id:
            # print(f"Found matching cache entry: {filename}")
            # print("Result mesh scale:", result.mesh_scale)
            # print("Cache mesh scale:", cache_mesh_scale)

            if abs(result.mesh_scale - cache_mesh_scale) < TOLERANCE:
                # if str(entry.normalization_scale).startswith(f"{result.norm_scale:.3f}"):
                matches.append((filename, entry))

    if not matches:
        raise ValueError(f"No matching cache entry found for {item_name} {item_id}")

    # print(result.norm_scale)
    # TODO: this id is donkey. Hardcoding it for now because apparently they have VERY similar mesh_scale
    # This is the only very similar one
    if len(matches) > 1 and item_id != "b09e0a52bd3b1b4eab2bd7322386ffd":
        print([(f, e.dataset_mesh_scale, e.normalization_scale) for f, e in matches])
        raise ValueError(
            f"Found multiple matching cache entries: {[m[0] for m in matches]}"
        )

    return matches[0]


if __name__ == "__main__":
    # Load grasp results and cache
    results = load_grasp_results("grasp_results", "data/meshes")

    with open("data/grasp_cache/grasp_cache.pkl", "rb") as f:
        cache = pickle.load(f)

    print(f"Loaded {len(cache)} cache entries")

    translation_norm_param_path = "logs/checkpoints/used_norm_params.pkl"

    with open(translation_norm_param_path, "rb") as f:
        norm_params = CPU_Unpickler(f).load()

    with open("logs/checkpoints/run_20250202_233846/used_grasp_files.json", "r") as f:
        used_files = json.load(f)

    # Get all files in the grasps directory
    all_files = os.listdir("data/grasps")

    # Initialize lists to store data
    grasp_data = []

    # Process each result
    for result in results:
        filename, match = match_grasp_cache(result, cache)

        # Get training status
        is_used_in_training = match.mesh_path in used_files

        translation = result.translations
        rotation = result.rotations
        mesh_path = result.mesh_path
        dataset_mesh_scale = match.dataset_mesh_scale
        centroid = match.centroid

        # Denormalize and adjust translation with centroid
        denormalized_translation = denormalize_translation(translation, norm_params)
        final_translation = denormalized_translation + torch.tensor(
            centroid, device=denormalized_translation.device
        )

        has_collision, scene, min_distance, is_graspable = check_collision(
            rotation,
            final_translation,
            mesh_path,
            dataset_mesh_scale,
        )

        # Create entry for each grasp
        for i in range(len(rotation)):
            grasp_info = {
                "mesh_path": mesh_path,
                "is_used_in_training": is_used_in_training,
                "dataset_mesh_scale": dataset_mesh_scale,
                "normalization_scale": match.normalization_scale,
                "has_collision": bool(has_collision[i]),
                "min_distance": float(min_distance[i]),
                "is_graspable": bool(is_graspable[i]),
                "grasp_translation": final_translation[i].tolist(),
                "grasp_rotation": rotation[i].tolist(),
                "centroid": centroid,
            }
            grasp_data.append(grasp_info)

        print(f"Processed {mesh_path}: {len(rotation)} grasps")

    # Create DataFrame
    df = pd.DataFrame(grasp_data)

    # Save to CSV
    output_path = "grasp_analysis_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved analysis results to {output_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total grasps analyzed: {len(df)}")
    print(f"Unique meshes: {df['mesh_path'].nunique()}")
    print(
        f"Successful grasps: {df['is_graspable'].sum()} ({(df['is_graspable'].mean() * 100):.2f}%)"
    )
    print(f"Average minimum distance: {df['min_distance'].mean():.4f}")
