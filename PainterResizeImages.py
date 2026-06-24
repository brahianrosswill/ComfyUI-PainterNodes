import torch
import torch.nn.functional as F


class PainterResizeImages:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "long_edge": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resize"
    CATEGORY = "Painter/Utils"

    def resize(self, images, long_edge):
        if images is None or images.shape[0] == 0:
            empty = torch.zeros((1, 64, 64, 3))
            return (empty, 64, 64)

        b, h, w, c = images.shape

        if h >= w:
            scale = long_edge / h
            new_h = long_edge
            new_w = max(1, int(w * scale))
        else:
            scale = long_edge / w
            new_w = long_edge
            new_h = max(1, int(h * scale))

        images_nchw = images.permute(0, 3, 1, 2)
        resized = F.interpolate(
            images_nchw,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False
        )
        resized = resized.permute(0, 2, 3, 1).clamp(0.0, 1.0)

        return (resized, new_w, new_h)


NODE_CLASS_MAPPINGS = {
    "PainterResizeImages": PainterResizeImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterResizeImages": "Painter Resize Images"
}
