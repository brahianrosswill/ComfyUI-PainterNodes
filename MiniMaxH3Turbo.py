import math

import torch

import comfy.samplers


SHIFT_V = 12.0
SHIFT_A = 3.0


def _time_shift_sigma(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return to * base / (1.0 + (to - 1.0) * base)


def _time_shift_slope(sigma, fr, to):
    base = sigma / (fr + sigma * (1.0 - fr))
    return (to * (1.0 + (fr - 1.0) * base) ** 2) / (fr * (1.0 + (to - 1.0) * base) ** 2)


def _audio_sigma(sv):
    return _time_shift_sigma(sv, SHIFT_V, SHIFT_A)


def _audio_slope(sv):
    return _time_shift_slope(sv, SHIFT_V, SHIFT_A)


def _latent_shapes(model):
    guider = getattr(model, "inner_model", model)
    conds = getattr(guider, "conds", None)
    if conds:
        for cond_list in conds.values():
            for c in (cond_list or []):
                mc = c.get("model_conds", {}) if isinstance(c, dict) else {}
                if "latent_shapes" in mc:
                    return mc["latent_shapes"].cond
    return None


@torch.no_grad()
def _turbo_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
    extra_args = {} if extra_args is None else extra_args
    shapes = _latent_shapes(model)
    if not shapes or len(shapes) < 2:
        raise RuntimeError(
            "MiniMaxH3Turbo expects the MiniMax-H3 video+audio latent "
            "(the EmptyMiniMaxH3LatentAV / MiniMaxH3ImageToVideo output)."
        )
    v_numel = math.prod(shapes[0][1:])
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sv = float(sigmas[i])
        sv_n = float(sigmas[i + 1])
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        out = (x - denoised) / sigmas[i]
        xv = x[..., :v_numel]
        ov = out[..., :v_numel]
        xa = x[..., v_numel:]
        oa = out[..., v_numel:]
        xv = xv + (sv_n - sv) * ov
        sl = _audio_slope(max(sv, 1e-6))
        xa = xa + (_audio_sigma(sv_n) - _audio_sigma(sv)) * (oa / sl)
        x = torch.cat([xv, xa], dim=-1)
        if callback is not None:
            callback({
                "i": i,
                "denoised": denoised,
                "x": x,
                "sigma": sigmas[i],
                "sigma_hat": sigmas[i],
            })
    return x


class MiniMaxH3Turbo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling"
    DESCRIPTION = "4-step sampler for the MiniMax-H3 Turbo LoRA. Feed into SamplerCustomAdvanced and set the scheduler to 4 steps."

    def get_sampler(self):
        return (comfy.samplers.KSAMPLER(_turbo_sampler),)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Turbo": MiniMaxH3Turbo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Turbo": "MiniMax-H3 Turbo",
}
