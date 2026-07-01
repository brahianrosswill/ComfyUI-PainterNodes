import torch
import comfy.utils
import comfy.model_management
from comfy.comfy_types import ComfyNodeABC, InputTypeDict


class PainterV2AV(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "image": ("IMAGE",),
                "video": ("IMAGE",),
                "width": ("INT", {"default": 1920, "min": 64, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 1088, "min": 64, "max": 16384, "step": 1}),
                "frames_number": ("INT", {"default": 101, "min": 1, "max": 1000, "step": 1}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 1000.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("audio_latent", "video_latent")
    FUNCTION = "execute"
    CATEGORY = "Painter/LTXV"

    def execute(self, video_vae, audio_vae, image, video, width, height, frames_number, frame_rate, strength, batch_size):
        video_samples = video.movedim(-1, 1)
        scaled_video = comfy.utils.common_upscale(video_samples, width, height, "nearest-exact", "disabled")
        scaled_video = scaled_video.movedim(1, -1)

        video_latent_tensor = video_vae.encode(scaled_video)

        latent_samples = video_latent_tensor.clone()
        _, height_scale_factor, width_scale_factor = video_vae.downscale_index_formula
        _, _, _, latent_height, latent_width = latent_samples.shape
        target_width = latent_width * width_scale_factor
        target_height = latent_height * height_scale_factor

        if image.shape[1] != target_height or image.shape[2] != target_width:
            img_samples = image.movedim(-1, 1)
            img_scaled = comfy.utils.common_upscale(img_samples, target_width, target_height, "bilinear", "center")
            img_scaled = img_scaled.movedim(1, -1)
        else:
            img_scaled = image

        encode_pixels = img_scaled[:, :, :, :3]
        encoded_image = video_vae.encode(encode_pixels)

        latent_samples[:, :, :encoded_image.shape[2]] = encoded_image

        batch_size_actual = latent_samples.shape[0]
        latent_length = latent_samples.shape[2]
        noise_mask = torch.ones(
            (batch_size_actual, 1, latent_length, 1, 1),
            dtype=torch.float32,
            device=latent_samples.device,
        )
        noise_mask[:, :, :encoded_image.shape[2]] = 1.0 - strength

        video_latent = {"samples": latent_samples, "noise_mask": noise_mask}

        z_channels = audio_vae.latent_channels
        audio_freq = audio_vae.first_stage_model.latent_frequency_bins
        num_audio_latents = audio_vae.first_stage_model.num_of_latents_from_frames(frames_number, frame_rate)
        audio_latents = torch.zeros(
            (batch_size, z_channels, num_audio_latents, audio_freq),
            device=comfy.model_management.intermediate_device(),
        )
        audio_latent = {"samples": audio_latents, "type": "audio"}

        return (audio_latent, video_latent)


NODE_CLASS_MAPPINGS = {
    "PainterV2AV": PainterV2AV
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterV2AV": "Painter V2AV"
}
