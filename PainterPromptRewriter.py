import torch

import comfy.utils
from comfy_api.latest import io


REWRITE_INSTRUCTION = (
    "You are a professional prompt rewriter. Rewrite the [USER PROMPT] strictly according to "
    "the structure and rules of the [TEMPLATE] below. Follow every section and formatting rule "
    "defined in the template. Output ONLY the rewritten prompt body: no explanations, no "
    "prefaces, no markdown fences."
)

IMAGE_HINT = (
    "{n} reference image(s) are attached. Observe them carefully and describe the locked "
    "appearance details (identity, face, hair, costume, props, environment) faithfully in "
    "your rewrite."
)

CHATML_BLOCK = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class PainterPromptRewriter(io.ComfyNode):
    """Rewrites a short user prompt into a fully structured prompt based on a user-provided
    template, optionally guided by up to 9 reference images. Generation uses the same
    interface as the core TextGenerate node (clip.tokenize / clip.generate / clip.decode).
    """

    @classmethod
    def define_schema(cls):
        sampling_options = [
            io.DynamicCombo.Option(
                key="on",
                inputs=[
                    io.Float.Input("temperature", default=0.7, min=0.01, max=2.0, step=0.000001),
                    io.Int.Input("top_k", default=64, min=0, max=1000),
                    io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("repetition_penalty", default=1.05, min=0.0, max=5.0, step=0.01),
                    io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                    io.Float.Input("presence_penalty", optional=True, default=0.0, min=0.0, max=5.0, step=0.01),
                ],
            ),
            io.DynamicCombo.Option(key="off", inputs=[]),
        ]

        return io.Schema(
            node_id="PainterPromptRewriter",
            display_name="Painter Prompt Rewriter",
            category="text",
            description="Rewrite a simple prompt according to a template, optionally guided by up to 9 reference images.",
            search_aliases=["LLM", "prompt enhance", "prompt rewrite"],
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default="",
                                tooltip="The simple prompt to be rewritten."),
                io.String.Input("template", multiline=True, dynamic_prompts=True, default="",
                                tooltip="The target format / instruction template the rewrite must follow."),
                io.Int.Input("max_length", default=2048, min=1, max=32768),
                io.DynamicCombo.Input("sampling_mode", options=sampling_options, display_name="Sampling Mode"),
                io.Int.Input("max_image_long_side", default=960, min=256, max=4096, step=8, optional=True,
                             tooltip="Downscale each reference image so its long side is at most this value before it is sent to the model."),
                io.Boolean.Input("thinking", optional=True, default=False,
                                 tooltip="Operate in thinking mode if the model supports it."),
                io.Boolean.Input("use_default_template", optional=True, default=True,
                                 tooltip="Apply the built-in chat template of the model.", advanced=True),
                io.Boolean.Input("use_chatml_system", optional=True, default=False,
                                 tooltip="Experimental: inject the template as a ChatML system message instead of user text.", advanced=True),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image", optional=True),
                        prefix="image_",
                        min=1,
                        max=9,
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="generated_text"),
            ],
        )

    @staticmethod
    def _downscale_long_side(samples_bchw, max_long_side):
        h, w = samples_bchw.shape[2], samples_bchw.shape[3]
        long_side = max(h, w)
        if long_side <= max_long_side:
            return samples_bchw
        scale = max_long_side / float(long_side)
        new_w = max(8, int(round(w * scale)))
        new_h = max(8, int(round(h * scale)))
        return comfy.utils.common_upscale(samples_bchw, new_w, new_h, "area", "center")

    @staticmethod
    def _resize_to(samples_bchw, target_w, target_h):
        if samples_bchw.shape[2] == target_h and samples_bchw.shape[3] == target_w:
            return samples_bchw
        return comfy.utils.common_upscale(samples_bchw, target_w, target_h, "area", "center")

    @classmethod
    def execute(cls, clip, prompt, template, max_length, sampling_mode,
                max_image_long_side=960, thinking=False, use_default_template=True,
                use_chatml_system=False, images=None) -> io.NodeOutput:
        prompt = (prompt or "").strip()
        template = (template or "").strip()

        # Collect reference images (image_1 ... image_9) and downscale the long side.
        vl_images = []
        if images:
            sorted_keys = sorted(images.keys(), key=lambda x: int(x.split("_")[-1]))
            for key in sorted_keys:
                image = images[key]
                if image is None:
                    continue
                samples = image.movedim(-1, 1)
                samples = cls._downscale_long_side(samples, max_image_long_side)
                vl_images.append(samples.movedim(1, -1))

        image_hint = IMAGE_HINT.format(n=len(vl_images)) if vl_images else ""

        # Assemble the final input text. Works with zero images as well.
        if template:
            if use_chatml_system:
                system_text = f"{REWRITE_INSTRUCTION}\n\n[TEMPLATE]\n{template}"
                if image_hint:
                    system_text = f"{system_text}\n\n{image_hint}"
                full_prompt = CHATML_BLOCK.format(system=system_text, user=prompt)
                skip_template = True
            else:
                parts = [REWRITE_INSTRUCTION, "", "[TEMPLATE]", template, "", "[USER PROMPT]", prompt]
                if image_hint:
                    parts += ["", image_hint]
                full_prompt = "\n".join(parts)
                skip_template = not use_default_template
        else:
            full_prompt = prompt
            skip_template = not use_default_template

        if not full_prompt:
            return io.NodeOutput("")

        # Batch fallback: every frame of an IMAGE batch must share the same H/W.
        image_batch = None
        if vl_images:
            base = vl_images[0].movedim(-1, 1)
            merged = [base]
            for im in vl_images[1:]:
                im_bchw = im.movedim(-1, 1)
                merged.append(cls._resize_to(im_bchw, base.shape[3], base.shape[2]))
            image_batch = torch.cat(merged, dim=0).movedim(1, -1)

        # Prefer the multi-image list interface. Fall back to a single merged batch.
        tokens = None
        if vl_images:
            try:
                tokens = clip.tokenize(full_prompt, images=vl_images, skip_template=skip_template,
                                       min_length=1, thinking=thinking)
            except TypeError:
                tokens = None
        if tokens is None:
            tokens = clip.tokenize(full_prompt, image=image_batch, skip_template=skip_template,
                                   min_length=1, thinking=thinking)

        do_sample = sampling_mode.get("sampling_mode") == "on"
        generated_ids = clip.generate(
            tokens,
            do_sample=do_sample,
            max_length=max_length,
            temperature=sampling_mode.get("temperature", 1.0),
            top_k=sampling_mode.get("top_k", 50),
            top_p=sampling_mode.get("top_p", 1.0),
            min_p=sampling_mode.get("min_p", 0.0),
            repetition_penalty=sampling_mode.get("repetition_penalty", 1.0),
            presence_penalty=sampling_mode.get("presence_penalty", 0.0),
            seed=sampling_mode.get("seed", None),
        )
        generated_text = clip.decode(generated_ids)
        return io.NodeOutput(generated_text)


NODE_CLASS_MAPPINGS = {
    "PainterPromptRewriter": PainterPromptRewriter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterPromptRewriter": "Painter Prompt Rewriter",
}
