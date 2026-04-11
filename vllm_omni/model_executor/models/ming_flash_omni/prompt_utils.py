"""Shared prompt-building helpers for Ming-flash-omni standalone talker.

Both the offline example
(``examples/offline_inference/ming_flash_omni_tts/end2end.py``) and the
online speech endpoint (``vllm_omni/entrypoints/openai/serving_speech.py``)
build the talker's ``instruction`` field from the same caption-JSON
template, so the template lives here to keep them in sync.
"""

import copy
import json
from typing import Any

DEFAULT_PROMPT = "Please generate speech based on the following description.\n"

BASE_CAPTION_TEMPLATE: dict[str, Any] = {
    "audio_sequence": [
        {
            "序号": 1,
            "说话人": "speaker_1",
            "方言": None,
            "风格": None,
            "语速": None,
            "基频": None,
            "音量": None,
            "情感": None,
            "BGM": {
                "Genre": None,
                "Mood": None,
                "Instrument": None,
                "Theme": None,
                "ENV": None,
                "SNR": None,
            },
            "IP": None,
        }
    ]
}


def create_instruction(user_input: dict[str, Any]) -> str:
    """Return a JSON caption string for ``audio_sequence[0]``.

    Only keys already present on the base template are merged in; unknown
    keys are silently ignored to keep the output schema stable.
    """
    caption = copy.deepcopy(BASE_CAPTION_TEMPLATE)
    item = caption["audio_sequence"][0]
    for key, value in user_input.items():
        if key in item:
            item[key] = value
    return json.dumps(caption, ensure_ascii=False)
