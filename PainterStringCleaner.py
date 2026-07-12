import re

class PainterStringCleaner:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "eliminate_pattern": ("STRING", {"default": "[]"}),
                "replace_with": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("original_text", "cleaned_text")
    FUNCTION = "clean_string"
    CATEGORY = "PainterNodes"

    def clean_string(self, text, eliminate_pattern, replace_with):
        if not eliminate_pattern or len(eliminate_pattern) < 2:
            return (text, text)

        left = eliminate_pattern[0]
        right = eliminate_pattern[-1]

        left_escaped = re.escape(left)
        right_escaped = re.escape(right)

        pattern = f"{left_escaped}[^{right_escaped}]*{right_escaped}"
        cleaned = re.sub(pattern, replace_with, text)

        return (text, cleaned)

NODE_CLASS_MAPPINGS = {
    "PainterStringCleaner": PainterStringCleaner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterStringCleaner": "Painter String Cleaner",
}
