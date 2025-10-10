import gc
from typing import List

import torch
from einops import rearrange
from PIL import Image
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm
from  .clip import clip
from jaxtyping import Float
from torch import nn
from .lseg_extract import LSegExtractor

class CLIPArgs:
    # model_name: str = "ViT-L/14@336px"
    model_name: str = "ViT-B/16"
    # model_name: str = "ViT-L/14"
    skip_center_crop: bool = True
    batch_size: int = 64

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the CLIP model parameters."""
        return {
            "model_name": cls.model_name,
            "skip_center_crop": cls.skip_center_crop,
        }
        
class CLIPExtractor:
    def __init__(self, device):
        self.device = device
        self.model, self.preprocess = clip.load(CLIPArgs.model_name, device=device)
        print(f"==> Loaded CLIP model {CLIPArgs.model_name}")
        
        # Patch the preprocess if we want to skip center crop
        if CLIPArgs.skip_center_crop:
            # Check there is exactly one center crop transform
            is_center_crop = [isinstance(t, CenterCrop) for t in self.preprocess.transforms]
            assert (
                sum(is_center_crop) == 1
            ), "There should be exactly one CenterCrop transform"
            # Create new preprocess without center crop
            self.preprocess = Compose(
                [t for t in self.preprocess.transforms if not isinstance(t, CenterCrop)]
            )
            print("==> Skipping center crop")
        self.lseg = None
            
    @property
    def patch_size(self) -> int:
        """Return the patch size of the CLIP model"""
        return self.model.visual.patch_size

    @torch.no_grad()
    def extract_language_features(self,images:torch.Tensor,
                                feats_size=[16,16]) -> Float[torch.Tensor, 'batch c h w']:
        """Extract dense patch-level CLIP features for given images"""
        
        # Preprocess the images
        # images = [Image.open(path) for path in image_paths]
        # preprocessed_images = torch.stack([self.preprocess(image) for image in images])
        # preprocessed_images = preprocessed_images.to(self.device)  # (b, 3, h, w)
        preprocessed_images = images
        # print(f"Preprocessed {len(images)} images into {preprocessed_images.shape}")

        # Get CLIP embeddings for the images
        embeddings = []
        for i in range(0, len(preprocessed_images), CLIPArgs.batch_size):
            batch = preprocessed_images[i : i + CLIPArgs.batch_size]
            embeddings.append(self.model.get_patch_encodings(batch))
        embeddings = torch.cat(embeddings, dim=0)

        # Reshape embeddings from flattened patches to patch height and width
        h_in, w_in = preprocessed_images.shape[-2:]
        if CLIPArgs.model_name.startswith("ViT"):
            h_out = h_in // self.model.visual.patch_size
            w_out = w_in // self.model.visual.patch_size
        elif CLIPArgs.model_name.startswith("RN"):
            h_out = max(h_in / w_in, 1.0) * self.model.visual.attnpool.spacial_dim
            w_out = max(w_in / h_in, 1.0) * self.model.visual.attnpool.spacial_dim
            h_out, w_out = int(h_out), int(w_out)
        else:
            raise ValueError(f"Unknown CLIP model name: {CLIPArgs.model_name}")
        embeddings = rearrange(embeddings, "b (h w) c -> b h w c", h=h_out, w=w_out)
        embeddings = rearrange(embeddings, "b h w c -> b c h w")
        # print(f"Extracted CLIP embeddings of shape {embeddings.shape}")
        if embeddings.shape[2:] != tuple(feats_size):
            embeddings = nn.functional.interpolate(
                embeddings, size=feats_size, mode='bilinear', align_corners=True)
        # Delete and clear memory to be safe
        # del model
        # del preprocess
        # del preprocessed_images
        # torch.cuda.empty_cache()

        return embeddings.float()
    
    @torch.no_grad()
    def decode_feature(self, image_features, labelset=''):
        imshape = image_features.shape
        # encode text
        if labelset == '':
            text = self.text
        else:
            text = clip.tokenize(labelset)
        text = text.to(image_features.device)
        text_features = self.model.encode_text(text)
        image_features = image_features.permute(0,2,3,1).reshape(-1, 512)
        
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # logits_per_image = self.logit_scale * image_features.half() @ text_features.t()
        logits_per_image = image_features.half() @ text_features.t()
        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)
        return out
