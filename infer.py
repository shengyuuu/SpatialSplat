import torch
import cv2
import argparse
import os
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf
import json
from tqdm import tqdm
from src.spatialsplat import SpatialSplat

argparser = argparse.ArgumentParser(description="Active Splat")
argparser.add_argument("--input_dir",type=str,default="demo_data/",help="Directory to load the input data",)
argparser.add_argument("--output_dir",type=str,default="output/",help="Directory to save the output data",)
argparser.add_argument("--cfg_path",type=str,default="config/spatialsplat.yaml",help="Active Splat config file")
args = argparser.parse_args()

# Load the config file
cfg = OmegaConf.load(args.cfg_path)
if torch.cuda.is_available():
    devices = 'cuda:0'
else:
    devices = 'cpu'
print(f'Using device: {devices}')

# Init model
model = SpatialSplat(cfg.spatialsplat, devices=devices, eval=True)
model.eval()
img_list = [path for path in Path(args.input_dir).glob('*.jpg')]

# Inference
model.inference(img_list, output_path=args.output_dir)