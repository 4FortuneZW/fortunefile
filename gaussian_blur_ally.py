#Developed by Ally - https://www.patreon.com/theally
#https://civitai.com/user/theally

#This node provides a simple interface to apply a gaussian blur approximation (with box blur) to the image prior to output

import torch
import numpy as np
from PIL import Image, ImageFilter

class GaussianBlur:
    """
    This node provides a simple interface to apply Gaussian blur to the output image.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Input Types
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_filter"

    CATEGORY = "Image Processing"

    def apply_filter(self, image, strength):

        # Convert the input image tensor to a PIL Image
        i = 255. * image.cpu().numpy().squeeze()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Apply Gaussian blur using the strength value
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=strength))

        # Convert the blurred PIL Image back to a tensor
        blurred_image_np = np.array(blurred_img).astype(np.float32) / 255.0
        blurred_image_tensor = torch.from_numpy(blurred_image_np).unsqueeze(0)

        return (blurred_image_tensor,)

NODE_CLASS_MAPPINGS = {
    "Gaussian Blur": GaussianBlur
}
