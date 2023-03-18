#Developed by Ally - https://www.patreon.com/theally
#https://civitai.com/user/theally

#This node provides a simple interface to adjust the brightness/contrast of the output image prior to saving

import numpy as np

class BrightnessContrast:
    """
        
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
        Input Types

        """
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["brightness", "contrast"],),
                "strength": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.01}),           
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_filter"

    CATEGORY = "Image Processing"

    def apply_filter(self, image, mode, strength):

        # Choose a filter based on the 'mode' value
        if mode == "brightness":
            image = np.clip(image + strength, 0.0, 1.0)
        elif mode == "contrast":
            image = np.clip(image * strength, 0.0, 1.0)
        else:
            print(f"Invalid filter option: {mode}. No changes applied.")
            return (image,)

        return (image,)

NODE_CLASS_MAPPINGS = {
    "Brightness & Contrast": BrightnessContrast
}
