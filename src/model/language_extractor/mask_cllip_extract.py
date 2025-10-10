import maskclip_onnx
import torch
from torch import nn
import os
from .lseg_extract import LSegExtractor

class MaskCLIPExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model, self.preprocess = maskclip_onnx.clip.load(
            "ViT-B/16",
            download_root=os.getenv('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch')),
            # device=device,
            # jit=True,
            convert_to_fp16=False,
        )
        self.model.eval()
        self.patch_size = self.model.visual.patch_size
        self.lseg = LSegExtractor()

    
    @torch.no_grad()
    def extract_language_features(self, img, feats_size=[16,16]):
        b, _, input_size_h, input_size_w = img.shape
        patch_h = input_size_h // self.patch_size
        patch_w = input_size_w // self.patch_size
        features = self.model.get_patch_encodings(img).to(torch.float32)
        feats =  features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)
        if feats_size is not None:
            feats = nn.functional.interpolate(feats, size=feats_size, mode='bilinear', align_corners=False)
        return feats
    

    @torch.no_grad()
    def decode_feature(self, image_features, labelset=''):
        return self.lseg.decode_feature(image_features, labelset)