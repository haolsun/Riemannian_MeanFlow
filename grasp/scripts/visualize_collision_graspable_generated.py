import pandas as pd
import torch

from core.visualize import check_collision

# Load the CSV
df = pd.read_csv("grasp_analysis_results_updated.csv")

# Filter for grasps that have collision but are graspable
interesting_grasps = df[
    (df["has_collision"] == True) & (df["is_graspable"] == True)
    # &
    # (~df['was_skipped'])
]

print(f"Found {len(interesting_grasps)} grasps with collision that are graspable")


# Function to convert string representation of list back to tensor
def str_to_tensor(s):
    # Remove brackets and split by comma
    values = eval(s)  # Be careful with eval - use only on trusted data
    return torch.tensor(values)


# Visualize each interesting grasp
for idx, grasp in interesting_grasps.iterrows():
    print(f"\nVisualizing grasp {idx} for mesh: {grasp['mesh_path']}")

    # Convert stored data back to tensors
    rotation = str_to_tensor(grasp["grasp_rotation"]).unsqueeze(
        0
    )  # Add batch dimension
    translation = str_to_tensor(grasp["grasp_translation"]).unsqueeze(0)

    # Call check_collision to get the scene
    has_collision, scene, min_distance, is_graspable = check_collision(
        rotation,
        translation,
        # "../"+grasp['mesh_path'],
        grasp["mesh_path"],
        grasp["dataset_mesh_scale"],
    )

    # Show the scene
    scene.show()

    break
