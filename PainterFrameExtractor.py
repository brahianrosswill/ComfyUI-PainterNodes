import torch


class PainterFrameExtractor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "image1_idx": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
                "image2_idx": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("start_image", "end_image", "image1", "image2", "image1_idx", "image2_idx")
    FUNCTION = "extract"
    CATEGORY = "image/batch"

    def extract(self, image, image1_idx, image2_idx):
        total = image.shape[0]
        
        idx1 = max(0, min(image1_idx, total - 1))
        idx2 = max(0, min(image2_idx, total - 1))
        
        start_image = image[0:1]
        end_image = image[-1:]
        image1 = image[idx1:idx1+1]
        image2 = image[idx2:idx2+1]
        
        return (start_image, end_image, image1, image2, image1_idx, image2_idx)


NODE_CLASS_MAPPINGS = {
    "PainterFrameExtractor": PainterFrameExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterFrameExtractor": "Painter Frame Extractor"
}
