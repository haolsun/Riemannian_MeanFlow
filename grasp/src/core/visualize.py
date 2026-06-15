import colorsys
import random
from typing import List, Tuple, Union

import numpy as np
import torch
import trimesh
import wandb
from trimesh.collision import CollisionManager
from trimesh.ray.ray_triangle import RayMeshIntersector

from src.data.util import enforce_trimesh

def random_blue():
    # Generate completely random RGB values
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    return [red, green, blue, 255]


def create_parallel_gripper_mesh(
    color: List[int] = [0, 0, 255],
    cylinder_sections: int = 12,
    gripper_width: float = 0.082,
    gripper_height: float = 0.11217,
    finger_thickness: float = 0.002,
    base_height: float = 0.066,
) -> trimesh.Trimesh:
    """Creates a 3D mesh representing a parallel-jaw gripper."""
    half_width = gripper_width / 2

    # Create basic components
    left_finger = trimesh.creation.cylinder(
        radius=finger_thickness,
        sections=cylinder_sections,
        segment=[
            [half_width, 0, base_height],
            [half_width, 0, gripper_height],
        ],
    )

    right_finger = trimesh.creation.cylinder(
        radius=finger_thickness,
        sections=cylinder_sections,
        segment=[
            [-half_width, 0, base_height],
            [-half_width, 0, gripper_height],
        ],
    )

    base_cylinder = trimesh.creation.cylinder(
        radius=finger_thickness,
        sections=cylinder_sections,
        segment=[[0, 0, 0], [0, 0, base_height]],
    )

    connector = trimesh.creation.cylinder(
        radius=finger_thickness,
        sections=cylinder_sections,
        segment=[[-half_width, 0, base_height], [half_width, 0, base_height]],
    )

    # Combine components
    gripper_mesh = trimesh.util.concatenate(
        [base_cylinder, connector, right_finger, left_finger]
    )
    gripper_mesh.visual.face_colors = color

    return gripper_mesh


def create_grasp_volume(
    gripper_width: float = 0.082,
    gripper_height: float = 0.11217,
    base_height: float = 0.066,
) -> trimesh.Trimesh:
    """Creates a box mesh representing the volume between gripper fingers."""
    # Create a box that spans between the fingers
    grasp_box = trimesh.creation.box(
        extents=[
            gripper_width,  # Width between fingers
            0.02,  # Thickness (a bit thicker than fingers for safety)
            gripper_height - base_height,  # Height of fingers
        ]
    )

    # Move the box to the correct position (centered between fingers, at the right height)
    transform = np.eye(4)
    transform[2, 3] = base_height + (gripper_height - base_height) / 2
    grasp_box.apply_transform(transform)

    return grasp_box


def find_contact_points(
    gripper_transform: np.ndarray,
    object_mesh: trimesh.Trimesh,
    gripper_width: float = 0.082,
    gripper_height: float = 0.11217,
    base_height: float = 0.066,
    num_vertical_rays: int = 30,  # Increased from 10
    num_horizontal_rays: int = 10,  # New parameter for horizontal sampling
    num_depth_rays: int = 5,  # New parameter for depth sampling
) -> np.ndarray:
    """Find potential contact points between gripper fingers and object with dense sampling.

    Args:
        gripper_transform: 4x4 transformation matrix for gripper pose
        object_mesh: Trimesh object of the target object
        gripper_width: Width between gripper fingers
        gripper_height: Total height of gripper
        base_height: Height of gripper base
        num_vertical_rays: Number of rays to cast along finger height
        num_horizontal_rays: Number of rays to cast along finger width
        num_depth_rays: Number of rays to cast along finger depth

    Returns:
        contacts: Array of contact points from left finger
    """
    intersector = RayMeshIntersector(object_mesh)

    # Calculate sampling points
    heights = np.linspace(base_height, gripper_height, num_vertical_rays)
    widths = np.linspace(
        -0.005, 0.005, num_horizontal_rays
    )  # Sample across finger width
    depths = np.linspace(-0.005, 0.005, num_depth_rays)  # Sample across finger depth

    half_width = gripper_width / 2

    # Create grid of points
    heights_grid, widths_grid, depths_grid = np.meshgrid(heights, widths, depths)
    num_rays = num_vertical_rays * num_horizontal_rays * num_depth_rays

    # Initialize ray origins
    left_origins = np.zeros((num_rays, 3))
    right_origins = np.zeros((num_rays, 3))

    # Set ray origins with offset in all dimensions
    left_origins[:, 0] = (
        half_width + widths_grid.flatten()
    )  # X coordinate with width variation
    left_origins[:, 1] = depths_grid.flatten()  # Y coordinate with depth variation
    left_origins[:, 2] = heights_grid.flatten()  # Z coordinate with height variation

    # right_origins[:, 0] = -half_width + widths_grid.flatten()
    # right_origins[:, 1] = depths_grid.flatten()
    # right_origins[:, 2] = heights_grid.flatten()

    # Ray directions (left finger rays go right, right finger rays go left)
    left_directions = np.tile([-1, 0, 0], (num_rays, 1))
    # right_directions = np.tile([1, 0, 0], (num_rays, 1))

    # Transform ray origins and directions to world frame
    left_origins = trimesh.transform_points(left_origins, gripper_transform)
    # right_origins = trimesh.transform_points(right_origins, gripper_transform)

    rotation = gripper_transform[:3, :3]
    left_directions = np.dot(left_directions, rotation.T)
    # right_directions = np.dot(right_directions, rotation.T)

    # Find intersections
    left_locations, left_index_ray, left_index_tri = intersector.intersects_location(
        left_origins, left_directions
    )
    # right_locations, right_index_ray, right_index_tri = intersector.intersects_location(
    #     right_origins, right_directions
    # )

    return left_locations


def check_collision(
    rotation_matrix: torch.Tensor,
    translation_vector: torch.Tensor,
    object_mesh_path: str,
    mesh_scale: float,
) -> Tuple[List[bool], trimesh.Scene, List[float], List[bool]]:
    """Checks for collisions between gripper poses and object using trimesh's CollisionManager.

    Returns:
        Tuple containing:
        - List of collision flags for each grasp
        - Visualization scene
        - List of minimum distances for each grasp
        - List of graspability flags for each grasp
    """
    # Load and scale object mesh
    object_mesh = trimesh.load(object_mesh_path)
    if torch.is_tensor(mesh_scale):
        mesh_scale = mesh_scale.cpu().numpy()
    object_mesh.apply_scale(mesh_scale)
    object_mesh = enforce_trimesh(object_mesh)

    # Check if rotation matrix is SO3 (3x3) or batched (Nx3x3)
    is_so3 = rotation_matrix.shape == torch.Size([3, 3])
    if is_so3:
        rotation_matrix = rotation_matrix.unsqueeze(0)
        translation_vector = translation_vector.unsqueeze(0)

    # Validate shapes
    assert len(rotation_matrix.shape) == 3 and rotation_matrix.shape[1:] == (3, 3), (
        f"Expected rotation matrix shape (N, 3, 3), got {rotation_matrix.shape}"
    )
    assert len(translation_vector.shape) == 2 and translation_vector.shape[1] == 3, (
        f"Expected translation vector shape (N, 3), got {translation_vector.shape}"
    )
    assert rotation_matrix.shape[0] == translation_vector.shape[0], (
        f"Batch sizes don't match: {rotation_matrix.shape[0]} != {translation_vector.shape[0]}"
    )

    batch_size = rotation_matrix.shape[0]
    gripper_meshes = []
    contact_spheres = []
    collision_list = []
    min_distance_list = []
    graspable_list = []

    # Create collision manager for the object
    object_manager = CollisionManager()
    object_manager.add_object("object", object_mesh)

    # Process each grasp
    for batch_idx in range(batch_size):
        # Create transformation matrix
        gripper_transform = torch.eye(4)
        gripper_transform[:3, :3] = rotation_matrix[batch_idx]
        gripper_transform[:3, 3] = translation_vector[batch_idx]
        gripper_transform = gripper_transform.cpu().numpy()

        # Create and transform gripper mesh
        gripper_mesh = create_parallel_gripper_mesh(color=[0, 255, 0])
        gripper_mesh.apply_transform(gripper_transform)

        # Create and transform grasp volume mesh
        grasp_volume = create_grasp_volume()
        grasp_volume.apply_transform(gripper_transform)

        # Find contact points with increased density
        left_contacts = find_contact_points(
            gripper_transform,
            object_mesh,
            num_vertical_rays=10,
            num_horizontal_rays=1,
            num_depth_rays=1,
        )

        # Create smaller visual markers for contact points
        sphere_radius = 0.001  # Reduced sphere size for denser visualization
        color = random_blue()

        for contact in left_contacts:
            contact_sphere = trimesh.creation.icosphere(radius=sphere_radius)
            contact_sphere.apply_translation(contact)
            contact_sphere.visual.face_colors = color
            contact_spheres.append(contact_sphere)

        # Create collision managers
        gripper_manager = CollisionManager()
        gripper_manager.add_object("gripper", gripper_mesh)

        volume_manager = CollisionManager()
        volume_manager.add_object("grasp_volume", grasp_volume)

        # Check collisions and update visualization
        has_collision, _, _ = object_manager.in_collision_other(
            gripper_manager, return_names=True, return_data=True
        )

        is_graspable, _, _ = object_manager.in_collision_other(
            volume_manager, return_names=True, return_data=True
        )

        min_distance, _, _ = object_manager.min_distance_other(
            gripper_manager, return_names=True, return_data=True
        )

        # Store results for this grasp
        collision_list.append(has_collision)
        min_distance_list.append(min_distance)
        graspable_list.append(is_graspable)

        # Update gripper color based on grasp evaluation
        if has_collision:
            color = [255, 0, 0]  # Red
        elif is_graspable:
            color = [0, 255, 0]  # Green
        else:
            color = [255, 255, 0]  # Yellow

        gripper_mesh.visual.face_colors = color
        gripper_meshes.append(gripper_mesh)

    # Create visualization
    if isinstance(object_mesh, trimesh.Scene):
        scene = object_mesh
        for mesh in gripper_meshes + contact_spheres:
            scene.add_geometry(mesh)
    else:
        all_meshes = [object_mesh] + gripper_meshes + contact_spheres
        scene = trimesh.Scene(all_meshes)

    # print('collision_list:', collision_list)
    # print('graspable_list:', graspable_list)

    return collision_list, scene, min_distance_list, graspable_list


def scene_to_wandb_3d(scene: trimesh.Scene) -> wandb.Object3D:
    """Convert trimesh scene to wandb 3D object."""
    scene.export("logs/mesh.glb")
    return wandb.Object3D("logs/mesh.glb")
