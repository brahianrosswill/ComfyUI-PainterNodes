import torch
import comfy.utils
from comfy_api.latest import ComfyExtension, io


class PainterImageConcat(io.ComfyNode):
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
            description="Concatenate multiple images or image sequences along a chosen direction. All inputs are resized to match image_0 with aspect ratio preserved. Short sequences are padded with the last frame frozen to match the longest sequence. Final output can be constrained by max long edge.",
            inputs=[
                io.Autogrow.Input("images", template=autogrow_template),
                io.Combo.Input("direction", options=["right", "left", "down", "up"], default="down"),
                io.Int.Input("max_long_edge", default=0, min=0, max=8192, tooltip="If > 0, scale down the final concatenated result so that its longest edge does not exceed this value. Aspect ratio is preserved. 0 means no constraint."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, images: io.Autogrow.Type, direction, max_long_edge) -> io.NodeOutput:
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

            original_h = img.shape[1]
            original_w = img.shape[2]

            if direction in ("right", "left"):
                target_h = ref_h
                target_w = int(original_w * target_h / original_h)
            else:
                target_w = ref_w
                target_h = int(original_h * target_w / original_w)

            if img.shape[1] != target_h or img.shape[2] != target_w:
                img = img.movedim(-1, 1)
                img = comfy.utils.common_upscale(img, target_w, target_h, "area", "disabled")
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

        if max_long_edge > 0:
            result_h = result.shape[1]
            result_w = result.shape[2]
            current_long_edge = max(result_h, result_w)
            if current_long_edge > max_long_edge:
                scale = max_long_edge / current_long_edge
                new_h = int(result_h * scale)
                new_w = int(result_w * scale)
                result = result.movedim(-1, 1)
                result = comfy.utils.common_upscale(result, new_w, new_h, "area", "disabled")
                result = result.movedim(1, -1)

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
