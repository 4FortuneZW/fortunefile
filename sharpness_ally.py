#Developed by Ally - https://www.patreon.com/theally
#https://civitai.com/user/theally

#This node provides a simple interface to adjust the sharpness of the output image prior to saving

import torch
import numpy as np
from PIL import Image, ImageFilter

class SharpnessFilter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 6.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_filter"

    CATEGORY = "Image Processing"

    def apply_filter(self, image, strength):

        # Convert the input image tensor to a PIL Image
        i = 255. * image.cpu().numpy().squeeze()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Apply sharpness adjustment using ImageFilter.UnsharpMask
        radius = 1
        percent = int(round(strength * 100))
        threshold = 0
        sharpness_filter = ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
        sharpened_image = img.filter(sharpness_filter)

        # Convert the sharpened PIL Image back to a tensor
        sharpened_image_np = np.array(sharpened_image).astype(np.float32) / 255.0
        sharpened_image_tensor = torch.from_numpy(sharpened_image_np).unsqueeze(0)

        return (sharpened_image_tensor, )

NODE_CLASS_MAPPINGS = {
    "Image Sharpening": SharpnessFilter
}
