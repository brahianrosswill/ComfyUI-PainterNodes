import torch
import comfy.utils


class PainterMinimaxH3LatentUpscaler:
    """
    Latent spatial upscaler dedicated to MiniMax H3.
    MiniMax H3 VisualVAE uses a spatial compression factor of 16x,
    unlike the standard 8x used by most image diffusion models.
    This node correctly scales H3 latents by dividing pixel dimensions by 16.
    """

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "upscale_method": (s.upscale_methods,),
                "width": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 16}),
                "height": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 16}),
                "crop": (s.crop_methods,),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "PainterNodes/latent"
    DESCRIPTION = (
        "Upscales MiniMax H3 latents using the correct 16x spatial compression factor. "
        "Input width/height are in pixels."
    )

    def upscale(self, samples, upscale_method, width, height, crop):
        # MiniMax H3 VisualVAE spatial compression factor is 16x
        spatial_compression = 16

        if width == 0 and height == 0:
            return (samples,)

        s = samples.copy()
        latent = samples["samples"]

        # Handle NestedTensor if present (used by some video models)
        is_nested = False
        nested_tensors = None
        if hasattr(latent, "is_nested") and latent.is_nested:
            is_nested = True
            nested_tensors = latent.unbind()
            ref_tensor = nested_tensors[0]
        else:
            ref_tensor = latent

        if width == 0:
            height = max(spatial_compression, height)
            width = max(
                spatial_compression,
                round(ref_tensor.shape[-1] * height / ref_tensor.shape[-2]),
            )
        elif height == 0:
            width = max(spatial_compression, width)
            height = max(
                spatial_compression,
                round(ref_tensor.shape[-2] * width / ref_tensor.shape[-1]),
            )
        else:
            width = max(spatial_compression, width)
            height = max(spatial_compression, height)

        # Convert pixel dimensions to latent dimensions using 16x compression
        latent_width = width // spatial_compression
        latent_height = height // spatial_compression

        if is_nested:
            upscaled = [
                comfy.utils.common_upscale(t, latent_width, latent_height, upscale_method, crop)
                for t in nested_tensors
            ]
            s["samples"] = torch.nested.nested_tensor(upscaled)
        else:
            s["samples"] = comfy.utils.common_upscale(
                latent, latent_width, latent_height, upscale_method, crop
            )

        return (s,)


NODE_CLASS_MAPPINGS = {
    "PainterMinimaxH3LatentUpscaler": PainterMinimaxH3LatentUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterMinimaxH3LatentUpscaler": "Minimax H3 Latent Upscaler",
}
