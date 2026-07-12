import numpy as np
import torch
import comfy.utils
import comfy.model_management
import node_helpers

from comfy_extras.nodes_lt import LTXVAddGuide, get_noise_mask, get_keyframe_idxs, _append_guide_attention_entry


class PainterLTXomni2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "width": ("INT", {"default": 768, "min": 64, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64, "max": 16384, "step": 32}),
                "length": ("INT", {"default": 97, "min": 1, "max": 16384, "step": 8}),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 10.0, "step": 0.01}),
                "reference1_frame_idx": ("INT", {"default": 0, "min": -9999, "max": 9999}),
                "reference2_frame_idx": ("INT", {"default": 0, "min": -9999, "max": 9999}),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "source_video": ("IMAGE",),
                "reference_image1": ("IMAGE",),
                "reference_image2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "LATENT")
    RETURN_NAMES = ("positive", "negative", "video_latent", "audio_latent")
    FUNCTION = "execute"
    CATEGORY = "model/conditioning/ltxv"

    def execute(self, positive, negative, video_vae, audio_vae, width, height, length, frame_rate, strength,
                reference1_frame_idx, reference2_frame_idx,
                start_image=None, end_image=None, source_video=None,
                reference_image1=None, reference_image2=None):

        length = ((max(length, 1) - 1) // 8) * 8 + 1

        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})

        actual_length = length
        video_frames = 0
        video_latent_frame_index_start = 0

        time_scale_factor = video_vae.downscale_index_formula[0]
        if source_video is not None:
            video_frames = source_video.shape[0]
            orig_video_frames = video_frames
            if video_frames > length:
                source_video = source_video[:length]
                video_frames = length
                actual_length = length
                orig_video_frames = length
            else:
                actual_length = video_frames
                pad_count = int(length - video_frames) 

                if pad_count > 0:
                    padding_list = []
                    if video_frames > 0:
                        last_frame = source_video[-1:].clone()  

                        for _ in range(pad_count):
                            p = last_frame[0:1] + torch.randn_like(last_frame[0:1]) * 0.025
                            padding_list.append(torch.clamp(p, 0.0, 1.0))

                    if len(padding_list) > 0:
                        padding = torch.cat(padding_list, dim=0)
                        source_video = torch.cat([source_video, padding], dim=0) 
                        actual_length = length

            pixels = comfy.utils.common_upscale(
                source_video.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            encode_pixels = pixels[:, :, :, :3]
            t = video_vae.encode(encode_pixels)

            if source_video.shape[0] > orig_video_frames:
                pad_latent_start = (orig_video_frames - 1) // time_scale_factor + 1
                video_latent_frame_index_start = orig_video_frames
                if pad_latent_start < t.shape[2]:
                    pad_slice = t[:, :, pad_latent_start:, :, :]
                    b, c, f, h, w = pad_slice.shape

                    base_noise = torch.randn((b, c, 1, h, w), device=t.device, dtype=t.dtype)

                    t[:, :, pad_latent_start:, :, :] = pad_slice * 0.85 + base_noise * 0.15

            latent_length = t.shape[2]
            latent = {"samples": t }

        else:
            latent_length = ((actual_length - 1) // 8) + 1
            latent = {
                "samples": torch.zeros(
                    [1, 128, latent_length, height // 32, width // 32],
                    device=comfy.model_management.intermediate_device()
                )
            }

        assert audio_vae is not None, "Audio VAE model is required"
        z_channels = audio_vae.latent_channels
        audio_freq = audio_vae.first_stage_model.latent_frequency_bins
        num_audio_latents = audio_vae.first_stage_model.num_of_latents_from_frames(actual_length, frame_rate)
        audio_latents = torch.zeros(
            (1, z_channels, num_audio_latents, audio_freq),
            device=comfy.model_management.intermediate_device(),
        )
        audio_latent = {"samples": audio_latents, "type": "audio"}

        if start_image is not None:
            positive, negative, latent = self._add_guide(
                positive, negative, video_vae, latent, start_image, 1, strength
            )

        if end_image is not None:
            if video_latent_frame_index_start > 0:
                n = ((video_latent_frame_index_start - 1) // time_scale_factor) * time_scale_factor + 1
                positive, negative, latent = self._add_guide(
                    positive, negative, video_vae, latent, end_image, n, strength
                )
                positive, negative, latent = self._add_guide(
                    positive, negative, video_vae, latent, end_image, -1, strength * 0.15
                )
            else:
                positive, negative, latent = self._add_guide(
                    positive, negative, video_vae, latent, end_image, -1, strength
                )

        if reference_image1 is not None:
            positive, negative, latent = self._add_guide(
                positive, negative, video_vae, latent, reference_image1, reference1_frame_idx, strength
            )

        if reference_image2 is not None:
            positive, negative, latent = self._add_guide(
                positive, negative, video_vae, latent, reference_image2, reference2_frame_idx, strength
            )


        return (positive, negative, latent, audio_latent)

    def _add_guide(self, positive, negative, vae, latent, image, frame_idx, strength):
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        latent_downscale_factor = 1

        time_scale_factor = scale_factors[0]
        num_frames_to_keep = ((image.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
        resolved_frame_idx = frame_idx
        if frame_idx < 0:
            _, num_keyframes = get_keyframe_idxs(positive, latent_image.shape)
            resolved_frame_idx = max((latent_length - num_keyframes - 1) * time_scale_factor + 1 + frame_idx, 0)
        causal_fix = resolved_frame_idx == 0 or num_frames_to_keep == 1

        if not causal_fix:
            image = torch.cat([image[:1], image], dim=0)

        image, t = LTXVAddGuide.encode(vae, latent_width, latent_height, image, scale_factors, latent_downscale_factor)

        if not causal_fix:
            t = t[:, :, 1:, :, :]
            image = image[1:]

        guide_latent_shape = list(t.shape[2:])
        guide_mask = None

        frame_idx_out, latent_idx = LTXVAddGuide.get_latent_index(
            positive, latent_length, len(image), frame_idx, scale_factors, latent_shape=latent_image.shape
        )

        if latent_idx + t.shape[2] > latent_length:
            raise ValueError("Conditioning frames exceed the length of the latent sequence.")

        positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
            positive, negative, frame_idx_out, latent_image, noise_mask, t, strength, scale_factors,
            guide_mask=guide_mask, latent_downscale_factor=latent_downscale_factor, causal_fix=causal_fix
        )

        pre_filter_count = t.shape[2] * t.shape[3] * t.shape[4]
        positive, negative = _append_guide_attention_entry(
            positive, negative, pre_filter_count, guide_latent_shape, strength=strength, attention_mask=None
        )

        return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}


NODE_CLASS_MAPPINGS = {
    "PainterLTXomni2": PainterLTXomni2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterLTXomni2": "Painter LTX by Master"
}
