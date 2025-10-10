from .clip_extract import CLIPExtractor, CLIPArgs
from .lseg_extract import LSegExtractor, LSegArgs
from .mask_cllip_extract import MaskCLIPExtractor

def build_language_extractor(model_name, device):
    if model_name == 'clip':
        return CLIPExtractor(device)
    elif model_name == 'lseg':
        return LSegExtractor.from_pretrained(device=device).eval()
    elif model_name == 'maskclip':
        return MaskCLIPExtractor(device)
    else:
        raise ValueError(f"Unknown language extractor model: {model_name}")