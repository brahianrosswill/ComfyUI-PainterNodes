import torch
import math
from typing_extensions import override

import comfy.model_management
import comfy.utils
import node_helpers
from comfy_api.latest import ComfyExtension, io

class PainterFluxImageEdit(io.ComfyNode):
    """
    Painter Flux Image Edit using the new ComfyUI extension framework.
    Features auto-growing image inputs (up to 6).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PainterFluxImageEdit",
            display_name="Painter Flux Image Edit",
            category="advanced/conditioning",
            description="Conditioning node for Painter Flux Image Edit with auto-growing image inputs.",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True),
                io.Int.Input("batch_size", default=1, min=1, max=64, step=1),
                io.Int.Input("width", default=1024, min=512, max=4096, step=8),
                io.Int.Input("height", default=1024, min=512, max=4096, step=8),
                io.Vae.Input("vae", optional=True),
                io.Mask.Input("image_0_mask", optional=True, tooltip="Mask applied to the first image (image_1)"),
                
                # Core magic: use Autogrow for auto-growing image inputs
                io.Autogrow.Input(
                    "images", 
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image", optional=True),
                        prefix="image_", # prefix name is image_
                        min=1,           # start from image_1
                        max=6            # up to 6 images
                    )
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ]
        )

    @classmethod
    def execute(cls, clip, prompt, batch_size, width, height, vae=None, 
                image_1_mask=None, images=None) -> io.NodeOutput:
        
        if vae is None:
            raise RuntimeError("VAE is required.")

        ref_latents = []
        vl_images = []
        noise_mask = None
        
        # Process dynamically passed images dict (e.g. {"image_1": tensor, "image_2": tensor})
        if images:
            # Sort by the number after image_1, image_2
            sorted_keys = sorted(images.keys(), key=lambda x: int(x.split('_')[-1]))

            for i, key in enumerate(sorted_keys):
                image = images[key]
                if image is None:
                    continue
                    
                samples = image.movedim(-1, 1)
                
                # Calculate visual encoding block
                vl_total = int(384 * 384)
                vl_scale_by = math.sqrt(vl_total / (samples.shape[3] * samples.shape[2]))
                s_vl = comfy.utils.common_upscale(
                    samples, 
                    round(samples.shape[3] * vl_scale_by), 
                    round(samples.shape[2] * vl_scale_by), 
                    "area", 
                    "center"
                )
                vl_images.append(s_vl.movedim(1, -1))

                # VAE encode
                vae_input = comfy.utils.common_upscale(samples, width, height, "lanczos", "center").movedim(1, -1)
                ref_latents.append(vae.encode(vae_input))

                # Mask processing: only applies to the first passed image (usually image_1)
                if i == 0 and image_1_mask is not None:
                    mask = image_1_mask
                    if mask.dim() == 2: 
                        mask = mask.unsqueeze(0).unsqueeze(0)
                    elif mask.dim() == 3: 
                        mask = mask.unsqueeze(1)
                    
                    m = comfy.utils.common_upscale(mask, width // 8, height // 8, "area", "center")
                    noise_mask = m.squeeze(1)

        # Prompt and conditioning encode
        tokens = clip.tokenize(prompt, images=vl_images)
        positive = clip.encode_from_tokens_scheduled(tokens)
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

        if ref_latents:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)

        # Build output Latent
        latent_samples = ref_latents[0] if ref_latents else vae.encode(torch.zeros(1, height, width, 3))
        
        if batch_size > 1:
            positive = positive * batch_size
            negative = negative * batch_size
            latent_samples = latent_samples.repeat(batch_size, 1, 1, 1)
            if noise_mask is not None and noise_mask.shape[0] == 1:
                noise_mask = noise_mask.repeat(batch_size, 1, 1)

        out_latent = {"samples": latent_samples}
        if noise_mask is not None:
            out_latent["noise_mask"] = noise_mask

        # Return in new framework io.NodeOutput format
        return io.NodeOutput(positive, negative, out_latent)


NODE_CLASS_MAPPINGS = {
    "PainterFluxImageEdit": PainterFluxImageEdit
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterFluxImageEdit": "Painter Flux Image Edit"
}
