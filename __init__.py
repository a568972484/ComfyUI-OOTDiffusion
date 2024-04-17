import os
import warnings
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
import torch
from .inference_ootd import OOTDiffusion
from .ootd_utils import get_mask_location
from torchvision.transforms import ToTensor
from torchvision import transforms

transform = transforms.ToTensor()

_category_get_mask_input = {
    "upperbody": "upper_body",
    "lowerbody": "lower_body",
    "dress": "dresses",
}

_category_readable = {
    "Upper body": "upperbody",
    "Lower body": "lowerbody",
    "Dress": "dress",
}


class LoadOOTDPipeline:
    display_name = "Load OOTDiffusion Local"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "type": (["Half body", "Full body"],),
                "path": ("STRING", {"default": "models/OOTDiffusion"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"

    CATEGORY = "OOTD"

    @staticmethod
    def load_impl(type, path):
        if type == "Half body":
            type = "hd"
        elif type == "Full body":
            type = "dc"
        else:
            raise ValueError(
                f"unknown input type {type} must be 'Half body' or 'Full body'"
            )
        if not os.path.isdir(path):
            raise ValueError(f"input path {path} is not a directory")
        return OOTDiffusion(path, model_type=type)

    def load(self, type, path):
        return (self.load_impl(type, path),)


class LoadOOTDPipelineHub(LoadOOTDPipeline):
    display_name = "Load OOTDiffusion from Hub🤗"

    repo_id = "levihsu/OOTDiffusion"
    repo_revision = "d33c517dc1b0718ea1136533e3720bb08fae641b"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "type": (["Half body", "Full body"],),
            }
        }

    def load(self, type):  # type: ignore
        # DiffusionPipeline.from_pretrained doesn't support subfolder
        # So we use snapshot_download to get local path first
        path = snapshot_download(
            self.repo_id,
            revision=self.repo_revision,
            resume_download=True,
        )
        if os.path.exists("models/OOTDiffusion"):
            warnings.warn(
                "You've downloaded models with huggingface_hub cache. "
                "Consider removing 'models/OOTDiffusion' directory to free your disk space."
            )
        return (LoadOOTDPipeline.load_impl(type, path),)


class OOTDGenerate:
    display_name = "OOTDiffusion Generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MODEL",),
                "cloth_image": ("IMAGE",),
                "model_image": ("IMAGE",),
                # Openpose from comfyui-controlnet-aux not work
                # "keypoints": ("POSE_KEYPOINT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "num_samples": ("INT", {"default": 1, "min": 1, "max": 10}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 14.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "category": (list(_category_readable.keys()),),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"

    CATEGORY = "OOTD"

    def generate(
        self, pipe: OOTDiffusion, cloth_image, model_image, category, seed, steps,num_samples, cfg
    ):
        # if model_image.shape != (1, 1024, 768, 3) or (
        #     cloth_image.shape != (1, 1024, 768, 3)
        # ):
        #     raise ValueError(
        #         f"Input image must be size (1, 1024, 768, 3). "
        #         f"Got model_image {model_image.shape} cloth_image {cloth_image.shape}"
        #     )
        category = _category_readable[category]
        if pipe.model_type == "hd" and category != "upperbody":
            raise ValueError(
                "Half body (hd) model type can only be used with upperbody category"
            )
        model_image_list = []
        print(f"model_image:{model_image.shape}")
        for img_index in range(len(model_image)):
            model_image_list.append(model_image[img_index])
        res = []
        #model_image:torch.Size([2, 1280, 960, 3])
        #减少一个维度
        cloth_image = cloth_image.squeeze(0)
        #torch.Size([1280, 960, 3])
        cloth_image = cloth_image.permute((2, 0, 1))
        #torch.Size([3,1280, 960])
        cloth_image = to_pil_image(cloth_image)
        if cloth_image.size != (768, 1024):
            print(f"Inconsistent cloth_image size {cloth_image.size} != (768, 1024)")
        cloth_image = cloth_image.resize((768, 1024))
        print(f'列表数量{len(model_image_list)}')
        for model_image in model_image_list:
            model_image = model_image.squeeze(0)
            model_image = model_image.permute((2, 0, 1))
            model_image = to_pil_image(model_image)
            if model_image.size != (768, 1024):
                print(f"Inconsistent model_image size {model_image.size} != (768, 1024)")
            model_image = model_image.resize((768, 1024))
            model_parse, _ = pipe.parsing_model(model_image.resize((384, 512)))
            keypoints = pipe.openpose_model(model_image.resize((384, 512)))
            mask, mask_gray = get_mask_location(
                pipe.model_type,
                _category_get_mask_input[category],
                model_parse,
                keypoints,
                width=384,
                height=512,
            )
            mask = mask.resize((768, 1024), Image.NEAREST)
            mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

            masked_vton_img = Image.composite(mask_gray, model_image, mask)
            images = pipe(
                category=category,
                image_garm=cloth_image,
                image_vton=masked_vton_img,
                mask=mask,
                image_ori=model_image,
                num_samples=num_samples,
                num_steps=steps,
                image_scale=cfg,
                seed=seed,
            )

            # pil(H,W,3) -> tensor(H,W,3)
            print(f"运行成功:images:{images}")
            for img in images:
                #  img:torch.Size([3, 1024, 768])
                output_image = to_tensor(img)
                # output_image:torch.Size([1, 3, 1024, 768])
                # print(f'output_image:{output_image.shape}')
                # output_image = output_image.squeeze(0)
                # print(f'output_image:{output_image.shape}')
                res.append(output_image)
        # tensor_batch = torch.cat(res, dim=0)
        tensor_batch = torch.stack(res)
        tensor_batch = tensor_batch.permute(0, 2, 3, 1)
        print(f'tensor_batch:{tensor_batch.shape}')
        print(tensor_batch)
        return (tensor_batch,)

def images_to_batch(images):

    # 创建ToTensor转换器实例
    to_tensor = ToTensor()

    # 将图像列表转换为张量列表
    tensors = [img.squeeze(0)  for img in images]

    # 将张量列表堆叠为一个批量张量
    batch_tensor = torch.stack(tensors)

    return batch_tensor

_export_classes = [
    LoadOOTDPipeline,
   # LoadOOTDPipelineHub,
    OOTDGenerate,
]

NODE_CLASS_MAPPINGS = {c.__name__: c for c in _export_classes}

NODE_DISPLAY_NAME_MAPPINGS = {
    c.__name__: getattr(c, "display_name", c.__name__) for c in _export_classes
}
