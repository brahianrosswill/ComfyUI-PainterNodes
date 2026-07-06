import torch
import comfy.utils
import node_helpers
from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier, latent_to_pixel_coords
from comfy_api.latest import ComfyExtension, io

ICLoRAParameters = io.Custom("IC_LORA_PARAMETERS")


def get_noise_mask(latent):
    noise_mask = latent.get("noise_mask", None)
    latent_image = latent["samples"]
    if noise_mask is None:
        batch_size, _, latent_length, _, _ = latent_image.shape
        noise_mask = torch.ones(
            (batch_size, 1, latent_length, 1, 1),
            dtype=torch.float32,
            device=latent_image.device,
        )
    else:
        noise_mask = noise_mask.clone()
    return noise_mask


def conditioning_get_any_value(conditioning, key, default=None):
    for t in conditioning:
        if key in t[1]:
            return t[1][key]
    return default


def get_keyframe_idxs(cond, latent_shape=None):
    keyframe_idxs = conditioning_get_any_value(cond, "keyframe_idxs", None)
    if keyframe_idxs is None:
        return None, 0
    if latent_shape is not None and len(latent_shape) == 5:
        tokens_per_frame = latent_shape[-2] * latent_shape[-1]
        num_keyframes = keyframe_idxs.shape[2] // tokens_per_frame
        return keyframe_idxs, num_keyframes
    entries = conditioning_get_any_value(cond, "guide_attention_entries", None)
    if entries:
        num_keyframes = sum(e["latent_shape"][0] for e in entries)
        return keyframe_idxs, num_keyframes
    num_keyframes = torch.unique(keyframe_idxs[:, 0, :, 0]).shape[0]
    return keyframe_idxs, num_keyframes


def _append_guide_attention_entry(positive, negative, pre_filter_count, latent_shape, strength=1.0, attention_mask=None):
    new_entry = {
        "pre_filter_count": pre_filter_count,
        "strength": strength,
        "pixel_mask": attention_mask.unsqueeze(0).unsqueeze(0) if attention_mask is not None else None,
        "latent_shape": latent_shape,
    }
    results = []
    for cond in (positive, negative):
        existing = []
        for t in cond:
            found = t[1].get("guide_attention_entries", None)
            if found is not None:
                existing = found
                break
        entries = [*existing, new_entry]
        results.append(node_helpers.conditioning_set_values(
            cond, {"guide_attention_entries": entries}
        ))
    return results[0], results[1]


class PainterLTXVAddGuide(io.ComfyNode):
    PATCHIFIER = SymmetricPatchifier(1, start_end=True)

    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            input=io.Image.Input(
                "reference_image",
                optional=True,
                tooltip="Reference image or video for conditioning. Auto-expands up to 8 references.",
            ),
            prefix="reference_image_",
            min=1,
            max=8,
        )
        return io.Schema(
            node_id="PainterLTXVAddGuide",
            display_name="Painter LTXV Add Guide",
            category="painter/conditioning/ltxv",
            description="Multi-guide conditioning for LTXV. Supports start frame, end frame, and up to 8 optional reference images with auto-expanding inputs.",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
                io.Image.Input(
                    "start_image",
                    optional=True,
                    tooltip="Start frame image. If connected, conditions the first frame.",
                ),
                io.Image.Input(
                    "end_image",
                    optional=True,
                    tooltip="End frame image. If connected, conditions the last frame.",
                ),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Mask.Input(
                    "attention_mask",
                    optional=True,
                    tooltip="Optional pixel-space spatial mask applied to all guides.",
                ),
                ICLoRAParameters.Input(
                    "iclora_parameters",
                    optional=True,
                    tooltip="Optional IC-LoRA parameters.",
                ),
                io.Autogrow.Input("reference_images", template=autogrow_template),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def add_keyframe_index(cls, cond, frame_idx, guiding_latent, scale_factors, latent_downscale_factor=1, causal_fix=None):
        keyframe_idxs, _ = get_keyframe_idxs(cond)
        _, latent_coords = cls.PATCHIFIER.patchify(guiding_latent)
        if causal_fix is None:
            causal_fix = frame_idx == 0 or guiding_latent.shape[2] == 1
        pixel_coords = latent_to_pixel_coords(latent_coords, scale_factors, causal_fix=causal_fix)
        pixel_coords[:, 0] += frame_idx

        spatial_end_offset = (latent_downscale_factor - 1) * torch.tensor(
            scale_factors[1:],
            device=pixel_coords.device,
        ).view(1, -1, 1, 1)
        pixel_coords[:, 1:, :, 1:] += spatial_end_offset.to(pixel_coords.dtype)

        if keyframe_idxs is None:
            keyframe_idxs = pixel_coords
        else:
            keyframe_idxs = torch.cat([keyframe_idxs, pixel_coords], dim=2)
        return node_helpers.conditioning_set_values(cond, {"keyframe_idxs": keyframe_idxs})

    @classmethod
    def _process_single_guide(cls, positive, negative, vae, latent_image, noise_mask, image, frame_idx, strength, scale_factors, latent_length, latent_height, latent_width, latent_downscale_factor, attention_mask=None):
        time_scale_factor = scale_factors[0]

        if image.dim() == 3:
            image = image.unsqueeze(0)

        num_frames_to_keep = ((image.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
        resolved_frame_idx = frame_idx
        if frame_idx < 0:
            _, num_keyframes = get_keyframe_idxs(positive, latent_image.shape)
            resolved_frame_idx = max((latent_length - num_keyframes - 1) * time_scale_factor + 1 + frame_idx, 0)
        causal_fix = resolved_frame_idx == 0 or num_frames_to_keep == 1

        if not causal_fix:
            image = torch.cat([image[:1], image], dim=0)

        time_scale_factor_enc, width_scale_factor, height_scale_factor = scale_factors
        image = image[:(image.shape[0] - 1) // time_scale_factor_enc * time_scale_factor_enc + 1]
        target_width = int(latent_width * width_scale_factor / latent_downscale_factor)
        target_height = int(latent_height * height_scale_factor / latent_downscale_factor)
        pixels = comfy.utils.common_upscale(image.movedim(-1, 1), target_width, target_height, "bilinear", crop="center").movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)

        if not causal_fix:
            t = t[:, :, 1:, :, :]
            image = image[1:]

        guide_latent_shape = list(t.shape[2:])
        guide_mask = None
        if latent_downscale_factor > 1:
            scale = int(latent_downscale_factor)
            dilated_shape = t.shape[:3] + (t.shape[3] * scale, t.shape[4] * scale)
            dilated = torch.zeros(dilated_shape, device=t.device, dtype=t.dtype)
            dilated[..., ::scale, ::scale] = t
            dilated_mask = torch.full(
                (dilated.shape[0], 1, dilated.shape[2], dilated.shape[3], dilated.shape[4]),
                -1.0, device=t.device, dtype=t.dtype,
            )
            dilated_mask[..., ::scale, ::scale] = 1.0
            t = dilated
            guide_mask = dilated_mask

        _, num_keyframes = get_keyframe_idxs(positive, latent_image.shape)
        latent_count = latent_length - num_keyframes
        resolved_frame_idx = frame_idx if frame_idx >= 0 else max((latent_count - 1) * time_scale_factor + 1 + frame_idx, 0)
        guide_length = image.shape[0]
        if guide_length > 1 and resolved_frame_idx != 0:
            resolved_frame_idx = (resolved_frame_idx - 1) // time_scale_factor * time_scale_factor + 1
        latent_idx = (resolved_frame_idx + time_scale_factor - 1) // time_scale_factor

        assert latent_idx + t.shape[2] <= latent_length, "Conditioning frames exceed the length of the latent sequence."

        positive = cls.add_keyframe_index(positive, resolved_frame_idx, t, scale_factors, latent_downscale_factor, causal_fix=causal_fix)
        negative = cls.add_keyframe_index(negative, resolved_frame_idx, t, scale_factors, latent_downscale_factor, causal_fix=causal_fix)

        if guide_mask is not None:
            target_h = max(noise_mask.shape[3], guide_mask.shape[3])
            target_w = max(noise_mask.shape[4], guide_mask.shape[4])
            if noise_mask.shape[3] == 1 or noise_mask.shape[4] == 1:
                noise_mask = noise_mask.expand(-1, -1, -1, target_h, target_w)
            if guide_mask.shape[3] == 1 or guide_mask.shape[4] == 1:
                guide_mask = guide_mask.expand(-1, -1, -1, target_h, target_w)
            mask = guide_mask - strength
        else:
            mask = torch.full(
                (noise_mask.shape[0], 1, t.shape[2], noise_mask.shape[3], noise_mask.shape[4]),
                max(0.0, 1.0 - strength),
                dtype=noise_mask.dtype,
                device=noise_mask.device,
            )

        if latent_image.shape[1] > t.shape[1]:
            pad_len = latent_image.shape[1] - t.shape[1]
            t = torch.nn.functional.pad(t, pad=(0, 0, 0, 0, 0, 0, 0, pad_len), value=0)

        latent_image = torch.cat([latent_image, t], dim=2)
        noise_mask = torch.cat([noise_mask, mask], dim=2)

        pre_filter_count = t.shape[2] * t.shape[3] * t.shape[4]
        positive, negative = _append_guide_attention_entry(
            positive, negative, pre_filter_count, guide_latent_shape, strength=strength, attention_mask=attention_mask
        )

        return positive, negative, latent_image, noise_mask

    @classmethod
    def execute(cls, positive, negative, vae, latent, start_image=None, end_image=None, strength=1.0, attention_mask=None, iclora_parameters=None, reference_images=None) -> io.NodeOutput:
        scale_factors = vae.downscale_index_formula
        time_scale_factor = scale_factors[0]
        latent_image = latent["samples"].clone()
        noise_mask = get_noise_mask(latent)
        _, _, latent_length, latent_height, latent_width = latent_image.shape

        latent_downscale_factor = 1
        if iclora_parameters is not None:
            try:
                latent_downscale_factor = max(1, round(float(iclora_parameters.get("reference_downscale_factor", 1))))
            except (TypeError, ValueError):
                latent_downscale_factor = 1

        if latent_downscale_factor > 1:
            if latent_width % latent_downscale_factor != 0 or latent_height % latent_downscale_factor != 0:
                raise ValueError(
                    f"Latent spatial size {latent_width}x{latent_height} must be divisible by "
                    f"reference_downscale_factor {latent_downscale_factor}."
                )

        ref_list = []
        if reference_images is not None:
            sorted_items = sorted(reference_images.items(), key=lambda x: x[0])
            ref_list = [v for k, v in sorted_items if v is not None]

        guides = []
        total_pixels = (latent_length - 1) * time_scale_factor + 1

        if start_image is not None:
            guides.append((start_image, 0))

        if ref_list:
            lo = 0
            if start_image is not None:
                lo = time_scale_factor

            hi = total_pixels - 1
            if end_image is not None:
                hi = max(total_pixels - 1 - time_scale_factor, lo)

            n = len(ref_list)
            if n == 1:
                if hi > lo:
                    frame_idx = (lo + hi) // 2
                else:
                    frame_idx = lo
                if ref_list[0].shape[0] > 1 and frame_idx != 0:
                    frame_idx = (frame_idx - 1) // time_scale_factor * time_scale_factor + 1
                guides.append((ref_list[0], frame_idx))
            else:
                if hi > lo:
                    for i, img in enumerate(ref_list):
                        frame_idx = int(lo + (hi - lo) * (i + 1) / (n + 1))
                        if img.shape[0] > 1 and frame_idx != 0:
                            frame_idx = (frame_idx - 1) // time_scale_factor * time_scale_factor + 1
                        guides.append((img, frame_idx))
                else:
                    for img in ref_list:
                        frame_idx = lo
                        if img.shape[0] > 1 and frame_idx != 0:
                            frame_idx = (frame_idx - 1) // time_scale_factor * time_scale_factor + 1
                        guides.append((img, frame_idx))

        if end_image is not None:
            guides.append((end_image, -1))

        for img, frame_idx in guides:
            positive, negative, latent_image, noise_mask = cls._process_single_guide(
                positive, negative, vae, latent_image, noise_mask, img, frame_idx, strength,
                scale_factors, latent_length, latent_height, latent_width, latent_downscale_factor,
                attention_mask,
            )

        return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})


class PainterLTXVAddGuideExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PainterLTXVAddGuide]


async def comfy_entrypoint() -> PainterLTXVAddGuideExtension:
    return PainterLTXVAddGuideExtension()


NODE_CLASS_MAPPINGS = {
    "PainterLTXVAddGuide": PainterLTXVAddGuide
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterLTXVAddGuide": "Painter LTXV Add Guide"
}
