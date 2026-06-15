from typing import Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from src.models.sdf_encoder import VoxelSDFEncoder
from src.core.config import ExperimentConfig
from typing import List,Optional


class VelocityNetwork(nn.Module):
    """Neural network for predicting the velocity field."""

    def __init__(
        self, config: ExperimentConfig, input_dim: int = 12, hidden_dim: int = 128
    ):
        super().__init__()

        activation = config.model.activation  # nn.GELU
        input_dim = config.model.input_dim  # 12
        hidden_dim = config.model.hidden_dim  # 128
        num_hidden_layers = config.model.num_hidden_layers  # 3
        voxel_output_size = config.model.voxel_output_size  # 256
        self.sdf_encoder = VoxelSDFEncoder().float()
        # Time embedding
        self.time_proj = nn.Sequential(nn.Linear(1, hidden_dim), activation())
        self.time_proj_r = nn.Sequential(nn.Linear(1, hidden_dim), activation())

        # Input projection
        self.input_proj = nn.Linear(input_dim + voxel_output_size+1, hidden_dim)
        # Hidden layers
        layers = []
        layers.append(activation())
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
        self.hidden_layers = nn.ModuleList(layers)

        # Output projection
        self.final = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        so3_inputs: Tensor,
        r3_inputs: Tensor,
        sdf_inputs: Tensor,
        t: Tensor,
        r: Tensor,
        #dataset_mesh_scale: float,
        normalization_scale: Tensor,
        sdf_path: Optional[Tuple[str]]=None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass computing velocities for both SO3 and R3 components.

        Args:
            so3_input: SO3 input tensor [batch, 3, 3]
            r3_input: R3 input tensor [batch, 3]
            sdf_input: SDF input tensor [batch, 48, 48, 48]
            t: Time tensor [batch] or [batch, 1]

        Returns:
            Tuple of (so3_velocity [batch, 3, 3], r3_velocity [batch, 3])
        """
        # Ensure t is 2D [batch, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            r = r.unsqueeze(-1)
        elif t.dim() == 3:
            t = t.squeeze(1)
            r = r.squeeze(1)
        #print(type(sdf_path),sdf_inputs.shape)
        if sdf_path:
            sdf_features = self.efficient_sdf_forward(sdf_inputs,sdf_path)
        else:
            sdf_features = self.sdf_encoder(sdf_inputs)

        normalization_scale = torch.atleast_1d(normalization_scale)  # Ensure tensor
        normalization_scale = normalization_scale.view(-1, 1)  # [batch_size, 1]
        normalization_scale = normalization_scale.to(device=so3_inputs.device, 
                                                    dtype=so3_inputs.dtype)
        #print(sdf_features.shape)
        if sdf_features.shape[0] != so3_inputs.shape[0]:
            sdf_features = self.duplicate_to_batch_size(sdf_features,so3_inputs.shape[0])
            normalization_scale = self.duplicate_to_batch_size(normalization_scale,so3_inputs.shape[0])

        # Flatten SO3 input for processing
        so3_flat = rearrange(so3_inputs, "b c d -> b (c d)")

        # Combine inputs
        x = torch.cat([sdf_features, so3_flat, r3_inputs,normalization_scale], dim=-1)

        # Process time and state
        t_emb = self.time_proj(t)
        r_emb = self.time_proj_r(r)
        h = self.input_proj(x)

        # Combine time embedding and state
        c = t_emb + r_emb
        # h = h + t_emb + r_emb

        # Pass through hidden layers and get combined velocity
        for i, hidden_layer in enumerate(self.hidden_layers):
            h = hidden_layer(h + c)
        # h = self.hidden_layers(h)
        combined_velocity = self.final(h)

        # Split outputs
        so3_velocity_flat = combined_velocity[:, :9]
        r3_velocity = combined_velocity[:, 9:]

        # Reshape SO3 velocity back to matrix form for tangent space projection
        so3_velocity = rearrange(so3_velocity_flat, "b (c d) -> b c d", c=3, d=3)

        # Project to tangent space
        skew_symmetric_part = 0.5 * (so3_velocity - so3_velocity.permute(0, 2, 1))
        so3_velocity = so3_inputs @ skew_symmetric_part

        return so3_velocity, r3_velocity



    def duplicate_to_batch_size(self,input:Tensor,target_batch_size:int,duplicate_ratio:int = 1):
        current_size = input.size(0)
        if (current_size>=target_batch_size) and (duplicate_ratio == 1):
            return input
        elif duplicate_ratio > 1:
            duplicated = input.repeat(
            duplicate_ratio, *(1 for _ in range(len(input.shape) - 1))
        )
            return duplicated
        else: 
            num_copies = target_batch_size // current_size
            remainder = target_batch_size % current_size

            duplicated = input.repeat(
                num_copies, *(1 for _ in range(len(input.shape) - 1))
            )
            if remainder > 0:
                duplicated = torch.cat([duplicated, input[:remainder]], dim=0)
            
            return duplicated
    
    def efficient_sdf_forward(self,sdf_input: Tensor,sdf_paths: Union[str, Tuple[str]]):
        
        if isinstance(sdf_paths, str):
            return self.sdf_encoder(sdf_input)  
        unique_sdf_paths = []
        unique_indices = []  # indices into 'batch'
        filename_to_unique_idx = {}
        
        for i, sdf_path in enumerate(sdf_paths):
            if sdf_path not in filename_to_unique_idx:
                # This is the first time we see filename f
                filename_to_unique_idx[sdf_path] = len(unique_sdf_paths)  # e.g. 0, 1, 2, ...
                unique_sdf_paths.append(sdf_path)
                unique_indices.append(i)
                
        mapping = []
        for sdf_path in sdf_paths:
            mapping.append(filename_to_unique_idx[sdf_path])
        mapping = torch.tensor(mapping, dtype=torch.int)  # shape (N,)
        
        unique_batch = sdf_input[unique_indices]
        encoded_unique = self.sdf_encoder(unique_batch)
        final_output = encoded_unique[mapping]
        return final_output
