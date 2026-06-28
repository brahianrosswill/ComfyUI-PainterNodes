import torch
import comfy.utils
from comfy_api.latest import ComfyExtension, io


class PainterImageConcat(io.ComfyNode):
    """Concatenate multiple images or image sequences (video) along a chosen direction.
    All inputs are resized to match the first image (image_0) before concatenation.
    Short sequences are padded with the last frame frozen to match the longest sequence."""

    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            input=io.Image.Input("image", tooltip=("Image or image sequence (video) to concatenate.")),
            prefix="image_",
            min=1,
            max=5,
        )
        return io.Schema(
            node_id="PainterImageConcat",
            display_name="Painter Image Concat",
            category="painter/image",
            description="Concatenate multiple images or image sequences along a chosen direction. All inputs are resized to match image_0 before concatenation. Short sequences are padded with the last frame frozen to match the longest sequence.",
            inputs=[
                io.Autogrow.Input("images", template=autogrow_template),
                io.Combo.Input("direction", options=["right", "left", "down", "up"], default="down"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, images: io.Autogrow.Type, direction) -> io.NodeOutput:
        sorted_items = sorted(images.items(), key=lambda x: x[0])
        image_list = [v for k, v in sorted_items if v is not None]

        if len(image_list) == 0:
            raise ValueError("At least one image input must be connected.")

        ref = image_list[0]
        if ref.dim() == 3:
            ref = ref.unsqueeze(0)
        ref_h = ref.shape[1]
        ref_w = ref.shape[2]
        ref_c = ref.shape[3]

        max_frames = 0
        resized_list = []
        for img in image_list:
            if img is None:
                continue
            if img.dim() == 3:
                img = img.unsqueeze(0)
            if img.shape[1] != ref_h or img.shape[2] != ref_w:
                img = img.movedim(-1, 1)
                img = comfy.utils.common_upscale(img, ref_w, ref_h, "area", "disabled")
                img = img.movedim(1, -1)
            if img.shape[0] > max_frames:
                max_frames = img.shape[0]
            resized_list.append(img)

        if len(resized_list) == 0:
            raise ValueError("No valid images after filtering.")

        padded_list = []
        for img in resized_list:
            current_frames = img.shape[0]
            if current_frames < max_frames:
                last_frame = img[-1:]
                freeze_pad = last_frame.repeat(max_frames - current_frames, 1, 1, 1)
                img = torch.cat([img, freeze_pad], dim=0)
            padded_list.append(img)

        if direction in ("right", "left"):
            if direction == "left":
                padded_list = list(reversed(padded_list))
            result = torch.cat(padded_list, dim=2)
        else:
            if direction == "up":
                padded_list = list(reversed(padded_list))
            result = torch.cat(padded_list, dim=1)

        return io.NodeOutput(result)


class PainterImageConcatExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PainterImageConcat]


async def comfy_entrypoint() -> PainterImageConcatExtension:
    return PainterImageConcatExtension()


NODE_CLASS_MAPPINGS = {
    "PainterImageConcat": PainterImageConcat
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterImageConcat": "Painter Image Concat"
}
