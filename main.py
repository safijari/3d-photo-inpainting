import numpy as np
import argparse
import glob
import os
from functools import partial
import vispy
import scipy.misc as misc
from tqdm import tqdm
import yaml
import time
import sys
from mesh import write_ply, read_ply, output_3d_photo
from utils import get_MiDaS_samples, read_MiDaS_depth, transform_MiDaS_depth, make_MiDaS_sample
import torch
import cv2
from skimage.transform import resize
import imageio
import copy
from networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net
from MiDaS.run import run_depth, run_depth_single_image_return
from MiDaS.monodepth_net import MonoDepthNet
import MiDaS.MiDaS_utils as MiDaS_utils
from bilateral_filtering import sparse_bilateral_filtering

config = yaml.load(open("argument.yml", "r"))
if config["offscreen_rendering"] is True:
    vispy.use(app="egl")
os.makedirs(config["mesh_folder"], exist_ok=True)
os.makedirs(config["video_folder"], exist_ok=True)
os.makedirs(config["depth_folder"], exist_ok=True)
# sample_list = get_MiDaS_samples(config["src_folder"], config["depth_folder"], config, config["specific"])
normal_canvas, all_canvas = None, None

if isinstance(config["gpu_ids"], int) and (config["gpu_ids"] >= 0):
    device = config["gpu_ids"]
else:
    device = "cpu"

print(f"running on device {device}")


# TODO
# sample = sample_list[idx]
# mesh_fi = os.path.join(config['mesh_folder'], sample['src_pair_name'] +'.ply')
# image = imageio.imread(sample['ref_img_fi'])


class ModelsHolder:
    def __init__(self):
        # load network
        self.mono_depth_model = MonoDepthNet(config["MiDaS_model_ckpt"])
        self.mono_depth_model.to(device)
        self.mono_depth_model.eval()

        torch.cuda.empty_cache()
        self.depth_edge_model = Inpaint_Edge_Net(init_weights=True)
        depth_edge_weight = torch.load(config["depth_edge_model_ckpt"])
        self.depth_edge_model.load_state_dict(depth_edge_weight)
        self.depth_edge_model = self.depth_edge_model.to(device)
        self.depth_edge_model.eval()

        self.depth_feat_model = Inpaint_Depth_Net()
        depth_feat_weight = torch.load(config["depth_feat_model_ckpt"])
        self.depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
        self.depth_feat_model = self.depth_feat_model.to(device)
        self.depth_feat_model.eval()

        self.rgb_model = Inpaint_Color_Net()
        rgb_feat_weight = torch.load(config["rgb_feat_model_ckpt"])
        self.rgb_model.load_state_dict(rgb_feat_weight)
        self.rgb_model.eval()
        self.rgb_model = self.rgb_model.to(device)


class ImageProcessor:
    def __init__(self, models_holder: ModelsHolder):
        self.mh = models_holder
        self.rt_info = None
        self.sample = None
        self.image_rgb = None
        self.depth = None

    def pre_process_image(self, image_rgb):
        self.sample = make_MiDaS_sample(image_rgb, config)
        # run depth
        print("depthing")
        depth = run_depth_single_image_return(image_rgb, self.mh.mono_depth_model, MiDaS_utils, device, target_w=640)

        # do ... things ... wut?
        config["output_h"], config["output_w"] = depth.shape[:2]
        frac = config["longer_side_len"] / max(config["output_h"], config["output_w"])
        config["output_h"], config["output_w"] = int(config["output_h"] * frac), int(config["output_w"] * frac)
        config["original_h"], config["original_w"] = config["output_h"], config["output_w"]

        if image_rgb.ndim == 2:
            image_rgb = image_rgb[..., None].repeat(3, -1)
        if np.sum(np.abs(image_rgb[..., 0] - image_rgb[..., 1])) == 0 and np.sum(np.abs(image_rgb[..., 1] - image_rgb[..., 2])) == 0:
            config["gray_image"] = True
        else:
            config["gray_image"] = False

        # main image_rgb and depth pre-processing
        image_rgb = cv2.resize(image_rgb, (config["output_w"], config["output_h"]), interpolation=cv2.INTER_AREA)
        depth = transform_MiDaS_depth(depth, 3.0, config["output_h"], config["output_w"])

        print("bilateralling")
        # bilateral filtering
        vis_photos, vis_depths = sparse_bilateral_filtering(
            depth.copy(), image_rgb.copy(), config, num_iter=config["sparse_iter"], spdb=False
        )

        depth = vis_depths[-1]

        mesh_fi = "/tmp/mesh"

        print("plying")
        self.rt_info = write_ply(
            image_rgb,
            depth,
            self.sample["int_mtx"],
            mesh_fi,
            config,
            self.mh.rgb_model,
            self.mh.depth_edge_model,
            self.mh.depth_edge_model,
            self.mh.depth_feat_model,
        )

        self.image_rgb = image_rgb
        self.depth = depth

    def run_image(self, video_basename):
        verts, colors, faces, Height, Width, hFov, vFov = self.rt_info

        videos_poses = copy.deepcopy(self.sample["tgts_poses"])
        top = config.get("original_h") // 2 - self.sample["int_mtx"][1, 2] * config["output_h"]
        left = config.get("original_w") // 2 - self.sample["int_mtx"][0, 2] * config["output_w"]
        down, right = top + config["output_h"], left + config["output_w"]
        border = [int(xx) for xx in [top, down, left, right]]
        normal_canvas, all_canvas = output_3d_photo(
            verts.copy(),
            colors.copy(),
            faces.copy(),
            copy.deepcopy(Height),
            copy.deepcopy(Width),
            copy.deepcopy(hFov),
            copy.deepcopy(vFov),
            copy.deepcopy(self.sample["tgt_pose"]),
            self.sample["video_postfix"],
            copy.deepcopy(self.sample["ref_pose"]),
            copy.deepcopy(config["video_folder"]),
            self.image_rgb.copy(),
            copy.deepcopy(self.sample["int_mtx"]),
            config,
            self.image_rgb,
            videos_poses,
            video_basename,
            config.get("original_h"),
            config.get("original_w"),
            border=border,
            depth=self.depth,
            normal_canvas=None,
            all_canvas=None,
        )

if __name__ == "__main__":
    mh = ModelsHolder()
    ip = ImageProcessor(mh)
    def run_on_image(path):
        im = cv2.imread(path)[:, :, ::-1]
        ip.pre_process_image(im)
        ip.run_image(path)
    import ipdb; ipdb.set_trace()
