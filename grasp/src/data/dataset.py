import concurrent.futures
import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler, dataset

from src.core.config import ExperimentConfig
from src.data.data_manager import GraspCache
from src.data.util import (
    CPU_Unpickler,
    GraspData,
    NormalizationParams,
    normalize_translation,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from tqdm import tqdm


# Used so we can get only unique meshes.
# So, each batch is just one sample with unique mesh.
class MeshBatchSampler(Sampler):
    def __init__(self, dataset):
        self.mesh_paths = []
        seen = set()
        print(len(dataset))
        for idx in tqdm(range(len(dataset))):
            mesh_path = dataset[idx].mesh_path
            if mesh_path not in seen:
                self.mesh_paths.append(idx)
                seen.add(mesh_path)

    def __iter__(self):
        for idx in self.mesh_paths:
            yield [idx]  # Wrap in list since DataLoader expects an iterable

    def __len__(self):
        return len(self.mesh_paths)


class SingleSDFSampler(Sampler):
    """
    Yields one batch of indices per SDF. Each batch = [start_idx, ..., end_idx-1].
    So inside predict_step(), you see *all* grasps for that single SDF.
    """

    def __init__(self, dataset, shuffle: bool = False):
        self.dataset = dataset
        self.shuffle = shuffle

        # Build a list of all SDF index ranges
        self.batches = []
        for _, start_idx, end_idx in dataset.grasp_entries:
            indices = list(range(start_idx, end_idx))
            self.batches.append(indices)

    def __iter__(self):
        for batch_indices in self.batches:
            print(len(batch_indices))
            yield batch_indices

    def __len__(self):
        return len(self.batches)


class GraspDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        grasp_files: Union[List[str], int],
        config: ExperimentConfig,
        split: str = "train",
        num_samples: Optional[int] = None,
        sdf_size: int = 32,
        device: torch.device = torch.device("cuda"),
    ):
        self.data_root = data_root
        self.sdf_size = sdf_size
        self.device = device
        self.cache = GraspCache(os.path.join(data_root, "grasp_cache"))
        self.config = config
        # Process all grasp files
        self.grasp_entries = []
        total_grasps = 0
        # TODO: Why does normalization works better check it.
        # print("----- grasp_files -----", grasp_files)
        if isinstance(grasp_files, int):
            # Perform globbing to get all .h5 files
            all_h5 = list(Path(self.data_root, "grasp").glob("*.h5"))
            random.seed(42)  # Fix the seed for reproducibility
            random.shuffle(all_h5)
            selected_files = all_h5[:grasp_files]
            # Extract only the filenames
            self.grasp_files = [f.name for f in selected_files]
        else:
            self.grasp_files = grasp_files

        # print("----- self.grasp_files -----", self.grasp_files)

        logger.info(f"Number of .h5 files to process: {len(self.grasp_files)}")
        results = [None] * len(self.grasp_files)
        # print('-------- grasp_files --------:', self.grasp_files)
        # print('-------- results --------:', results)

        # Decide which files need to be processed
        to_process = []
        for i, fname in enumerate(self.grasp_files):
            # print('------ i, fname--------:', i, fname)
            if fname in self.cache.cache:
                # Already in cache (could be valid or None)
                entry = self.cache.cache[fname]
                if entry is None:
                    # Means 0 transforms from previous run => skip
                    results[i] = (fname, None, None, None, 0)
                else:
                    # It's a GraspCacheEntry => gather local min/max
                    transforms = entry.transforms
                    local_min = np.min(transforms[:, :3, 3], axis=0)
                    local_max = np.max(transforms[:, :3, 3], axis=0)
                    num_grasps = len(transforms)
                    results[i] = (fname, entry, local_min, local_max, num_grasps)
            else:
                # Not in cache => queue for processing
                to_process.append((i, fname))

        # print('-------- to_process --------:', to_process)
        # print('-------- results --------:', results)

        if to_process:
            logger.info(f"Processing {len(to_process)} files in parallel.")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.config.data.dataset_workers
            ) as executor:
                future_to_idx = {}
                for i, fname in to_process:
                    args = (fname, self.data_root, self.sdf_size)
                    fut = executor.submit(self.cache.process_one_file, args)
                    future_to_idx[fut] = i

                for fut in concurrent.futures.as_completed(future_to_idx):
                    i = future_to_idx[fut]
                    res = (
                        fut.result()
                    )  # => (filename, entry, local_min, local_max, num_grasps) or None
                    if res is None:
                        # Something failed => skip
                        results[i] = None
                    else:
                        results[i] = res

        for i, out in enumerate(results):
            # print('-----i,out------:', i, out)
            if out is None:
                # either error or didn't exist => store as None in cache
                fname = self.grasp_files[i]
                self.cache.cache[fname] = None
            else:
                fname, entry, _, _, _ = out
                # print('----- fname -----:', fname)
                # print('----- entry -----:', entry)
                # entry might be None if 0 transforms => also store None
                self.cache.cache[fname] = entry
        self.cache._save()  # Write cache once

        # exit(0)
        self.grasp_entries = []

        total_grasps = 0
        # self.trans_min = None
        # self.trans_max = None
        for out in results:
            # print('------ out------:', out)
            if (out is None) or (out[1] is None):
                # skip
                continue
            fname, entry, local_min, local_max, num_grasps = out
            # self.grasp_entries: (filename, start_idx, end_idx)
            self.grasp_entries.append((fname, total_grasps, total_grasps + num_grasps))

            # if self.trans_min is None:
            #     self.trans_min = local_min
            #     self.trans_max = local_max
            # else:
            #     self.trans_min = np.minimum(self.trans_min, local_min)
            #     self.trans_max = np.maximum(self.trans_max, local_max)

            total_grasps += num_grasps

        self.total_grasps = total_grasps
        if total_grasps == 0:
            logger.warning("No valid grasps found in all files.")

        # if self.trans_min is not None and self.trans_max is not None:
        #     self.norm_params = NormalizationParams(
        #         min=torch.tensor(self.trans_min, dtype=torch.float32),
        #         max=torch.tensor(self.trans_max, dtype=torch.float32),
        #     )
        #     logger.info(f"Global min: {self.trans_min}, max: {self.trans_max}")
        # else:
        #     self.norm_params = NormalizationParams(
        #         min=torch.zeros(3),
        #         max=torch.ones(3),
        #     )

        self.norm_params = self._load_or_compute_norm_params(results)
        # Optionally sample a subset
        if num_samples and num_samples < self.total_grasps:
            chosen = torch.randperm(self.total_grasps)[:num_samples]
            self.selected_indices = sorted(chosen.tolist())
            self.total_grasps = num_samples
        else:
            self.selected_indices = None

    def __len__(self):
        return self.total_grasps

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, str, NormalizationParams, float, float
    ]:
        # print("getitem")

        if self.selected_indices is not None:
            idx = self.selected_indices[idx]

        # Find which grasp file contains this index
        entry_match = next(
            (entry for entry in self.grasp_entries if entry[1] <= idx < entry[2])
        )
        filename, start_idx, _ = entry_match
        entry = self.cache.cache[filename]
        grasp_idx = idx - start_idx

        rotation = torch.tensor(entry.transforms[grasp_idx][:3, :3], device=self.device)
        translation = torch.tensor(
            entry.transforms[grasp_idx][:3, 3], device=self.device
        )
        normalized_translation = normalize_translation(translation, self.norm_params)

        return GraspData(
            rotation=rotation,
            translation=normalized_translation,
            sdf=torch.tensor(entry.sdf),
            mesh_path=entry.mesh_path,
            dataset_mesh_scale=entry.dataset_mesh_scale,
            normalization_scale=entry.normalization_scale,
            centroid=entry.centroid,
        )

    def _load_or_compute_norm_params(
        self, concurrency_results: List[Tuple[str, ...]]
    ) -> NormalizationParams:
        """
        If `config.data.translation_norm_param_path` is provided, load that file.
        Otherwise, compute min/max from `concurrency_results`.
        """
        # 1. If path provided => load directly
        if self.config.data.translation_norm_param_path is not None:
            with open(self.config.data.translation_norm_param_path, "rb") as f:
                # norm_params = pickle.load(f, map_location=self.device)
                norm_params = CPU_Unpickler(f).load()

            # print('self.config.data.translation_norm_param_path: ', '/home/zhongzichen/code/RFM-Grasp-main/' + self.config.data.translation_norm_param_path)

            # norm_params = torch.load('/home/zhongzichen/code/RFM-Grasp-main/' + self.config.data.translation_norm_param_path, map_location=torch.device('cuda'))

            logger.info(
                f"Loaded normalization parameters from {self.config.data.translation_norm_param_path}"
            )
            return norm_params

        # 2. Otherwise => compute min/max from concurrency_results
        trans_min, trans_max = None, None
        for out in concurrency_results:
            if out is None or out[1] is None:
                # skip
                continue
            _, _, local_min, local_max, _ = out
            if trans_min is None:
                trans_min, trans_max = local_min, local_max
            else:
                trans_min = np.minimum(trans_min, local_min)
                trans_max = np.maximum(trans_max, local_max)

        if trans_min is None or trans_max is None:
            # Fallback if no valid grasps
            logger.warning(
                "No valid grasps found for computing normalization. Using defaults."
            )
            return NormalizationParams(
                min=torch.zeros(3).to(self.device),
                max=torch.ones(3).to(self.device),
            )

        logger.info(f"Global min: {trans_min}, max: {trans_max}")
        return NormalizationParams(
            min=torch.tensor(trans_min, dtype=torch.float32, device=self.device),
            max=torch.tensor(trans_max, dtype=torch.float32, device=self.device),
        )


class DataModule(LightningDataModule):
    def __init__(
        self,
        config: ExperimentConfig,
    ):
        super().__init__()
        self.config = config

        self.data_root = config.data.data_path
        self.grasp_files = config.data.files
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        self.num_samples = config.data.sample_limit
        self.split_ratio = config.data.split_ratio

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.split_ratio < 0.5:
            print("Warning: Split ratio is less than 0.5.")

        self.val_all_data = GraspDataset(
            self.data_root,
            self.grasp_files,
            self.config,
            num_samples=self.num_samples,
            device=self.device,
        )
        self.batch_sampler = MeshBatchSampler(self.val_all_data)

    def setup(self, stage: Optional[str] = None):
        # Create split datasets
        if stage == "fit" or stage is None:
            # Create full dataset first
            self.full_dataset = self.val_all_data
            # self.full_dataset = GraspDataset(
            #     self.data_root,
            #     self.grasp_files,
            #     self.config,
            #     num_samples=self.num_samples,
            #     device=self.device,
            # )
            # print("------ train_dataset:", self.full_dataset.__len__())

            # Save to json
            dirpath = (
                self.config.training.checkpoint_dir
                + "/"
                + self.config.training.run_name
            )
            os.makedirs(
                dirpath, exist_ok=True
            )  # Create the directory if it does not exist

            used_grasps_file_path = os.path.join(dirpath, "used_grasp_files.json")
            with open(used_grasps_file_path, "w") as file:
                json.dump(self.full_dataset.grasp_files, file, indent=4)

            used_norm_params_file_path = os.path.join(dirpath, "used_norm_params.pkl")
            with open(used_norm_params_file_path, "wb") as file:
                pickle.dump(self.full_dataset.norm_params, file)

            # Calculate split sizes
            train_size = int(len(self.full_dataset) * self.split_ratio)
            # print('train_size:', train_size)
            # print('self.full_dataset:', len(self.full_dataset), self.full_dataset[0])
            val_size = len(self.full_dataset) - train_size

            if train_size == len(self.full_dataset) or val_size == 0 or train_size == 0:
                # This should only happen if we have sample_limit=1 or split_ratio=1.0
                print("Using the same dataset for training and validation.")
                self.train_dataset = self.full_dataset
                self.val_dataset = self.full_dataset
            else:
                # TODO: When it becomes a Subset it fails.
                self.train_dataset, self.val_dataset = dataset.random_split(
                    self.full_dataset,
                    [train_size, val_size],
                    generator=torch.Generator(device=self.device),
                )

    def train_dataloader(self):
        # print("------ train_dataset:", self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            num_workers=self.num_workers,
            generator=torch.Generator(device=self.device),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_all_data,
            # batch_size=self.batch_size,
            batch_sampler=self.batch_sampler,
            shuffle=False,
            persistent_workers=True,
            num_workers=self.num_workers,
            generator=torch.Generator(device=self.device),
        )

    def test_dataloader(self):
        # self.val_all_data = GraspDataset(
        #     self.data_root,
        #     self.grasp_files,
        #     self.config,
        #     num_samples=self.num_samples,
        #     device=self.device,
        # )
        # self.batch_sampler = MeshBatchSampler(self.val_all_data)
        return DataLoader(
            dataset=self.val_all_data,
            batch_sampler=self.batch_sampler,
            shuffle=False,
            persistent_workers=True,
            num_workers=15,
            generator=torch.Generator(device=self.device),
        )

    def predict_dataloader(self):
        # dataset = self.predict_dataset()  # custom helper to get the dataset
        data = GraspDataset(
            self.data_root,
            self.grasp_files,
            self.config,
            num_samples=self.num_samples,
            device=self.device,
        )
        return DataLoader(
            dataset=data,
            batch_sampler=SingleSDFSampler(data),
            shuffle=False,
            persistent_workers=True,
            num_workers=15,
            generator=torch.Generator(device=self.device),
        )
