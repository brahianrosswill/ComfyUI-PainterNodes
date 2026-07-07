import torch
import numpy as np
import nodes as comfy_nodes

VHSLoadFormats = {
    'None': {},
    'AnimateDiff': {'target_rate': 8, 'dim': (8, 0, 512, 512)},
    'Mochi': {'target_rate': 24, 'dim': (16, 0, 848, 480), 'frames': (6, 1)},
    'LTXV': {'target_rate': 24, 'dim': (32, 0, 768, 512), 'frames': (8, 1)},
    'Hunyuan': {'target_rate': 24, 'dim': (16, 0, 848, 480), 'frames': (4, 1)},
    'Cosmos': {'target_rate': 24, 'dim': (16, 0, 1280, 704), 'frames': (8, 1)},
    'Wan': {'target_rate': 16, 'dim': (8, 0, 832, 480), 'frames': (4, 1)},
}

if not hasattr(comfy_nodes, 'VHSLoadFormats'):
    comfy_nodes.VHSLoadFormats = {}

def get_load_formats():
    formats = {}
    formats.update(comfy_nodes.VHSLoadFormats)
    formats.update(VHSLoadFormats)
    return (list(formats.keys()), {'default': 'AnimateDiff', 'formats': formats})

def get_format(format_name):
    if format_name in VHSLoadFormats:
        return VHSLoadFormats[format_name]
    return comfy_nodes.VHSLoadFormats.get(format_name, {})

class PainterFrameRateConverter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "force_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.1}),
                "format": get_load_formats(),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "frame_count")
    FUNCTION = "convert_frame_rate"
    CATEGORY = "PainterNodes"

    def convert_frame_rate(self, image, source_fps, force_rate, format):
        if image is None or len(image.shape) < 4 or image.shape[0] == 0:
            return (image, 0)

        num_frames = image.shape[0]

        if force_rate <= 0 or abs(force_rate - source_fps) < 0.001:
            out_images = image
        else:
            duration = num_frames / source_fps
            target_num = max(1, int(round(duration * force_rate)))

            out_indices = []
            for i in range(target_num):
                t = i / force_rate
                src_idx = int(round(t * source_fps))
                if src_idx >= num_frames:
                    src_idx = num_frames - 1
                out_indices.append(src_idx)

            out_images = image[out_indices]

        fmt = get_format(format)
        if 'frames' in fmt and len(out_images) > 0:
            frames_rule = fmt['frames']
            if len(out_images) % frames_rule[0] != frames_rule[1]:
                if len(frames_rule) > 2 and frames_rule[2]:
                    raise RuntimeError(
                        "The number of frames " + str(len(out_images)) + " does not match the requirements of the selected format " + str(format) + "."
                    )
                div, mod = frames_rule[:2]
                valid_count = (len(out_images) - mod) // div * div + mod
                if valid_count < 0:
                    valid_count = 0
                out_images = out_images[:valid_count]

        return (out_images, len(out_images))

NODE_CLASS_MAPPINGS = {
    "PainterFrameRateConverter": PainterFrameRateConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterFrameRateConverter": "Painter Frame Rate Converter",
}
