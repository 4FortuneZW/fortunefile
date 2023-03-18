#Developed by Ally - https://www.patreon.com/theally
#https://civitai.com/user/theally

#This node provides a simple interface to adjust the saturation of the output image prior to saving

import torch
import numpy as np
from PIL import Image, ImageEnhance

class Saturation:
    """
    This node provides a simple interface to adjust the saturation of the output image.
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
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),            
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_filter"

    CATEGORY = "Image Processing"

    def apply_filter(self, image, strength):

        # Convert the input image tensor to a PIL Image
        i = 255. * image.cpu().numpy().squeeze()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Adjust the saturation using the 'strength' value
        img = ImageEnhance.Color(img).enhance(strength)

        # Convert the adjusted PIL Image back to a tensor
        adjusted_image_np = np.array(img).astype(np.float32) / 255.0
        adjusted_image_tensor = torch.from_numpy(adjusted_image_np).unsqueeze(0)

        return (adjusted_image_tensor,)

NODE_CLASS_MAPPINGS = {
    "Saturation": Saturation
}