import torch
import comfy.utils
import comfy.model_management
import node_helpers
from comfy_api.latest import io


def _resize_long_edge(image, max_size, stride=16):
    h, w = image.shape[1], image.shape[2]
    scale = min(max_size / max(h, w), 1.0)
    nh = max(stride, round(h * scale / stride) * stride)
    nw = max(stride, round(w * scale / stride) * stride)
    return comfy.utils.common_upscale(image[:, :, :, :3].movedim(-1, 1), nw, nh, "area", "disabled").movedim(1, -1)


class PainterBerniniUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PainterBerniniUpscale",
            display_name="Painter Bernini Upscale",
            category="PainterNodes/Bernini",
            description="Upscale low-res video/image and encode to latent with Bernini conditioning.",
            inputs=[
                io.Image.Input("low_res_video", tooltip="Low resolution video or image to be upscaled."),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae", tooltip="VAE model used for encoding."),
                io.Int.Input("width", default=1536, min=16, max=8192, step=16),
                io.Int.Input("height", default=864, min=16, max=8192, step=16),
                io.Int.Input("length", default=81, min=1, max=8192, step=4),
                io.Int.Input("ref_max_size", default=1536, min=16, max=8192, step=16, optional=True),
                io.Image.Input("source_video", optional=True, tooltip="Source video to edit or restyle. Resized to width/height."),
                io.Image.Input("reference_video", optional=True, tooltip="Video to insert into the source video."),
                io.Autogrow.Input(
                    "reference_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("reference_image", tooltip="Reference image injected as an in-context token."),
                        prefix="reference_image_",
                        min=0,
                        max=8
                    )
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, low_res_video, positive, negative, vae, width, height, length, ref_max_size=1536, source_video=None, reference_video=None, reference_images=None) -> io.NodeOutput:
        samples = low_res_video.movedim(-1, 1)
        scaled = comfy.utils.common_upscale(samples, width, height, "nearest-exact", "disabled")
        scaled = scaled.movedim(1, -1)
        
        latent = vae.encode(scaled)
        latent_out = {"samples": latent}
        
        context = []
        if source_video is not None:
            vid = comfy.utils.common_upscale(source_video[:length, :, :, :3].movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
            context.append(vae.encode(vid[:, :, :, :3]))
        
        if reference_video is not None:
            ref_vid = _resize_long_edge(reference_video[:length], ref_max_size)
            context.append(vae.encode(ref_vid[:, :, :, :3]))
        
        if reference_images:
            for name in sorted(reference_images):
                imgs = reference_images[name]
                if imgs is None:
                    continue
                for i in range(imgs.shape[0]):
                    img = _resize_long_edge(imgs[i:i + 1], ref_max_size)
                    context.append(vae.encode(img[:, :, :, :3]))
        
        if context:
            positive = node_helpers.conditioning_set_values(positive, {"context_latents": context})
            negative = node_helpers.conditioning_set_values(negative, {"context_latents": context})
        
        return io.NodeOutput(latent_out, positive, negative, width, height)


NODE_CLASS_MAPPINGS = {
    "PainterBerniniUpscale": PainterBerniniUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterBerniniUpscale": "Painter Bernini Upscale",
}
