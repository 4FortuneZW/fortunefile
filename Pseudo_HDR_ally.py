#Developed by Ally - https://www.patreon.com/theally
#https://civitai.com/user/theally

#This node provides a simple interface to create a pseudo-HDR effect in images by boosting the alpha. 
#if OpenCV include is available, can create a much better version. Investigating!

import torch
import numpy as np
from PIL import Image, ImageEnhance

class HDRStyleImage:
    """
    This node provides a simple interface to create an HDR style image from a given input image.
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
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_filter"

    CATEGORY = "Image Processing"

    def apply_filter(self, image, intensity):

        # Convert the input image tensor to a PIL Image
        i = 255. * image.cpu().numpy().squeeze()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Enhance the brightness of the image
        enhancer = ImageEnhance.Brightness(img)
        result = enhancer.enhance(2)

        # Blend the original image with the enhanced image based on the intensity
        blended = Image.blend(img, result, alpha=intensity)

        # Convert the blended PIL Image back to a tensor
        blended_image_np = np.array(blended).astype(np.float32) / 255.0
        blended_image_tensor = torch.from_numpy(blended_image_np).unsqueeze(0)

        return (blended_image_tensor,)

NODE_CLASS_MAPPINGS = {
    "Pseudo HDR Style": HDRStyleImage
}
