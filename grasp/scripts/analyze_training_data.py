import glob
import json
import os
import pickle
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from core.visualize import check_collision
from data.data_manager import GraspCacheEntry
from data.util import CPU_Unpickler


def get_mesh_path_from_h5(h5_filename: str) -> str:
    parts = h5_filename[:-3].split("_")
    item_id = parts[-2]
    item_name = "_".join(parts[:-2])
    return f"data/meshes/{item_name}/{item_id}.obj"


if __name__ == "__main__":
    with open("data/grasp_cache/grasp_cache.pkl", "rb") as f:
        cache: dict[str, GraspCacheEntry] = pickle.load(f)

    print(f"Loaded {len(cache)} cache entries")

    # translation_norm_param_path = (
    #     "logs/checkpoints/run_20250204_204207/used_norm_params.pkl"
    # )
    translation_norm_param_path = "logs/checkpoints/used_norm_params.pkl"

    with open(translation_norm_param_path, "rb") as f:
        norm_params = CPU_Unpickler(f).load()

    with open("logs/checkpoints/run_20250202_233846/used_grasp_files.json", "r") as f:
        used_files = json.load(f)

    # Initialize list to store data
    grasp_data = []

    # Process each result
    for filename, value in tqdm(cache.items()):
        # Handle None value case
        if value is None:
            grasp_info = {
                "mesh_path": get_mesh_path_from_h5(filename) if filename else None,
                "is_used_in_training": filename in used_files if filename else None,
                "dataset_mesh_scale": None,
                "normalization_scale": None,
                "has_collision": None,
                "min_distance": None,
                "is_graspable": None,
                "grasp_translation": None,
                "grasp_rotation": None,
                "centroid": None,
                "was_skipped": True,
            }
            grasp_data.append(grasp_info)
            continue

        # Process normal case
        mesh_path = get_mesh_path_from_h5(filename)
        is_used_in_training = filename in used_files

        translation = torch.tensor(value.transforms[..., :3, 3])
        rotation = torch.tensor(value.transforms[..., :3, :3])
        final_translation = translation + torch.tensor(
            value.centroid, device=translation.device
        )

        has_collision, scene, min_distance, is_graspable = check_collision(
            rotation,
            final_translation,
            mesh_path,
            value.dataset_mesh_scale,
        )

        # Create entry for each grasp
        for i in range(len(rotation)):
            grasp_info = {
                "mesh_path": mesh_path,
                "is_used_in_training": is_used_in_training,
                "dataset_mesh_scale": value.dataset_mesh_scale,
                "normalization_scale": value.normalization_scale,
                "has_collision": bool(has_collision[i]),
                "min_distance": float(min_distance[i]),
                "is_graspable": bool(is_graspable[i]),
                "grasp_translation": final_translation[i].tolist(),
                "grasp_rotation": rotation[i].tolist(),
                "centroid": value.centroid,
                "was_skipped": False,
            }
            grasp_data.append(grasp_info)

        print(f"Processed {mesh_path}: {len(rotation)} grasps")

    # Create DataFrame
    df = pd.DataFrame(grasp_data)

    # Save to CSV
    output_path = "training.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved analysis results to {output_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total entries: {len(df)}")
    print(f"Skipped entries: {df['was_skipped'].sum()}")
    print(f"Successfully processed: {len(df) - df['was_skipped'].sum()}")
    print(f"Unique meshes: {df['mesh_path'].nunique()}")

    # Calculate success rate for non-skipped entries
    non_skipped = df[~df["was_skipped"]]
    if len(non_skipped) > 0:
        success = (non_skipped["is_graspable"] == True) & (
            non_skipped["has_collision"] == False
        )
        print(f"Successful grasps: {success.sum()} ({(success.mean() * 100):.2f}%)")
        print(f"Average minimum distance: {non_skipped['min_distance'].mean():.4f}")
