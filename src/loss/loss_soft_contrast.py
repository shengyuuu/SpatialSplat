# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional
from jaxtyping import Float
from typing import Union
from dataclasses import dataclass
import torch
from torch import Tensor
from einops import rearrange


from .loss import Loss
from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
import time

@dataclass
class LossSoftContrastCfg:
    weight: float
    temperature: float
    sample_num: int = 1
    lambda_sim: float = 0.5
    learnable_temperature: bool = False

@dataclass
class LossSoftContrastCfgWrapper:
    soft_contrast: LossSoftContrastCfg

class LossSoftContrast(Loss[LossSoftContrastCfg, LossSoftContrastCfgWrapper]):
    
    def __init__(self, cfg: LossSoftContrastCfgWrapper):
        super().__init__(cfg)
        # set a learnable temperature
        if self.cfg.learnable_temperature:
            self.temperature = torch.nn.Parameter(torch.zeros(1))
        else:
            self.temperature = torch.tensor(self.cfg.temperature)
        self.lambda_sim = self.cfg.lambda_sim
    
    def split_mask(self, 
                   masks: Float[Tensor, 'bv hw'],
                   ) -> list[list[Tensor]]:
        # import ipdb;ipdb.set_trace()
        masks = (masks * 255).int()
        splited_masks_batch = []
        for mask in masks:
            splited_mask = []
            for id in range(mask.max() + 1):
                mask_id = (mask == id)
                if id == 0:
                    # splited_mask.append(mask_id)
                    continue
                if torch.nonzero(mask_id).shape[0] < 16*16:
                    continue
                splited_mask.append(mask_id)
            if len(splited_mask) == 0:
                splited_mask.append(torch.zeros_like(mask))
            splited_masks_batch.append(splited_mask)
        return splited_masks_batch
            
    def similarity_kernal(self,
                        instance: Float[Tensor,'m c'],
                        instance_d: Float[Tensor,'m c']
                        ) -> Tensor:
        # * distance_sim
        # mean_distance = torch.norm(instance - instance_d, dim=-1).mean() # (1,) 
        # similarity = torch.exp(-mean_distance / self.temperature.exp())
        
        # * cosine_sim, instance has been normalized
        # m, c = instance.shape
        # mask = 1 - torch.eye(m).to(instance.device)
        # similarity = (instance @ instance.T + 1.) / 2.
        # similarity = similarity * mask
        # similarity = similarity.sum() / (m * (m - 1))
        similarity = ((instance * instance_d).sum(dim=-1) + 1. )/ 2.
        similarity = similarity.mean()
        # del mask
        # del instance
        # del instance_d
        
        return similarity
        
    
    def forward(
            self, 
            prediction: DecoderOutput,
            batch: BatchedExample,
            gaussians: Gaussians,
            global_step: int,
            masks: Tensor = None,
            clip_feats: Tensor = None,
        ) -> Float[Tensor, ""]:

        # init
        gt_mask_image = rearrange(batch["target"]["mask_image"],
                                  'b v 1 h w -> (b v) (h w)')
        instance_image = prediction.instance_image
        if masks is not None:
            instance_image = instance_image * masks[..., None, :,:]
        render_features = rearrange(instance_image,
                                    'b v c h w -> (b v) (h w) c')    # (B, V, C, H, W)
        assert render_features.shape[-2] == gt_mask_image.shape[-1], f"render_features.shape {render_features.shape} != gt_mask_image.shape {gt_mask_image.shape}"

        # split render feature into different instances and calculate the self similarity
        splated_masks_batch = self.split_mask(gt_mask_image) #
        total_instances = []
        self_similarities = []
        for batch, splated_masks in enumerate(splated_masks_batch):
            instance = []
            similarities = []
            for mask in splated_masks:
                # //print(batch)
                # //import ipdb; ipdb.set_trace()
                instance_ = render_features[batch][mask] # (n, c)
                instance_d =instance_.clone()[torch.randperm(instance_.shape[0])] #(n, c)
                # soft self similarity
                similarity_ = self.similarity_kernal(instance_, instance_d) # (1,)
                instance.append(instance_.mean(dim=0)) # (c,)
                similarities.append(similarity_)
            batch_instance = torch.stack(instance) # (n, c)
            total_instances.append(batch_instance)
            batch_similarity = torch.stack(similarities) # (n,)
            self_similarities.append(batch_similarity)
        # //import ipdb; ipdb.set_trace()    
        # calculate the similarity between different instances
        # //print(f"split time: {time.time() - start}")
        cross_similarities = []
        for batch, instance in enumerate(total_instances):
            # * cosin similarity
            n,_ = instance.shape
            mask = 1 - torch.eye(n).to(instance.device)
            cross_similarity = (instance @ instance.T + 1.) / 2.  # (n, n)
            cross_similarity = (1 - cross_similarity) * mask
            # import ipdb; ipdb.set_trace()
            cross_similarity = cross_similarity.sum() / (n * (n - 1))
            cross_similarities.append(cross_similarity)
            
            # * distance similarity
            # cross_distance = torch.norm(instance.unsqueeze(1) - instance.unsqueeze(0), dim=-1) # (n, n)
            # corss_similarity = torch.exp(-cross_distance / 1.) # (n, n)
            # cross_similarities.append(1 - corss_similarity)

        total_loss = []    
        # * loss = - log(/Sigma_{i} self_similarity_{i}) / /Sigma_{i,j} cross_similarity_{i,j})
        for batch, self_similarity in enumerate(self_similarities):
            n = self_similarity.shape[0]
            if n <= 2:
                total_loss.append(torch.tensor(0.).to(self_similarity.device))
                continue
            loss = - (torch.log((
                self.lambda_sim * self_similarity.mean() + (1 - self.lambda_sim) * cross_similarities[batch]
                )))
            total_loss.append(loss)
        
        return self.cfg.weight * torch.stack(total_loss).mean()