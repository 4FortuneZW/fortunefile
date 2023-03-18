#Developed by Ally - https://www.patreon.com/theally
#https://civitai.com/user/theally

#This node provides a simple interface to equalize the histogram of the output image prior to saving

import torch
import numpy as np
from PIL import Image, ImageOps

class HistogramEqualization:
    """
    This node provides a simple interface to equalize the histogram of the output image.
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
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_filter"

    CATEGORY = "Image Processing"

    def apply_filter(self, image, strength):

        # Convert the input image tensor to a PIL Image
        i = 255. * image.cpu().numpy().squeeze()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Equalize the histogram of the image
        equalized_img = ImageOps.equalize(img)

        # Blend the original image with the equalized image based on the strength
        blended_img = Image.blend(img, equalized_img, alpha=strength)

        # Convert the blended PIL Image back to a tensor
        blended_image_np = np.array(blended_img).astype(np.float32) / 255.0
        blended_image_tensor = torch.from_numpy(blended_image_np).unsqueeze(0)

        return (blended_image_tensor,)

NODE_CLASS_MAPPINGS = {
    "Histogram Equalization": HistogramEqualization
}
