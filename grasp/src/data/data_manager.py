import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import trimesh

from src.data.util import enforce_trimesh, process_mesh_to_sdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GraspCacheEntry:
    """Cache entry for processed grasp data."""

    sdf: np.ndarray
    transforms: np.ndarray  # Scaled transforms
    dataset_mesh_scale: float
    normalization_scale: float
    mesh_path: str
    centroid: np.ndarray


class GraspCache:
    """Cache for processed grasp data."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "grasp_cache.pkl"
        self.cache: dict[str, GraspCacheEntry] = {}
        self._load()

    def _load(self):
        """Load entire cache from pickle once. (Main process only)"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Starting empty cache.")

    def _save(self):
        """Write entire cache dict to pickle once. (Main process only)"""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    @staticmethod
    def process_one_file(
        args: Tuple[str, str, int],
    ) -> Optional[Tuple[str, Optional[GraspCacheEntry], np.ndarray, np.ndarray, int]]:
        """
        Worker function to process a single .h5 file.
        Returns (filename, entry, local_min, local_max, num_grasps)
        or None if something failed badly.

        - entry can be None if the file has 0 successful transforms.
        - local_min, local_max are 3D vectors from the transforms.
        - num_grasps = number of transforms.
        """
        filename, data_root, sdf_size = args

        try:
            grasp_file = os.path.join(data_root, "grasp", filename)
            # print(f"------- test ------- {grasp_file}")
            with h5py.File(grasp_file, "r") as h5file:
                transforms = h5file["grasps"]["transforms"][:]
                success = h5file["grasps"]["qualities"]["flex"]["object_in_gripper"][:]
                transforms = transforms[success == 1]
                if len(transforms) == 0:
                    # We'll return an entry=None to indicate no valid grasps
                    return (filename, None, None, None, 0)

                mesh_fname = h5file["object/file"][()].decode("utf-8")
                dataset_mesh_scale = h5file["object/scale"][()]
                #
                # for key in h5file.keys():
                #     print(h5file[key], key, h5file[key].name)
                #     for i in h5file[key]:
                #         print(i)

            # print('data_root:', data_root, '   mesh_fname:',mesh_fname)
            mesh_path = os.path.join(data_root, mesh_fname)
            mesh = trimesh.load(mesh_path)
            mesh.apply_scale(dataset_mesh_scale)
            mesh = enforce_trimesh(mesh)

            # Compute SDF
            sdf, normalization_scale, centroid = process_mesh_to_sdf(mesh, sdf_size)

            # Adjust transforms by centroid
            transforms[:, :3, 3] -= centroid

            # local min/max from these transforms
            local_min = np.min(transforms[:, :3, 3], axis=0)
            local_max = np.max(transforms[:, :3, 3], axis=0)
            num_grasps = len(transforms)

            entry = GraspCacheEntry(
                sdf=sdf,
                transforms=transforms,
                dataset_mesh_scale=dataset_mesh_scale,
                normalization_scale=normalization_scale,
                mesh_path=mesh_path,
                centroid=centroid,
            )
            return (filename, entry, local_min, local_max, num_grasps)

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            raise e
            return None
