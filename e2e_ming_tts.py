"""Single-stage TTS example for Ming-flash-omni-2.0 Talker."""

import os

import soundfile as sf

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.ming_flash_omni.spk_embedding import SpkembExtractor


def main():
    model_path = "Jonathan1909/Ming-flash-omni-2.0"  # or local path
    stage_config = "vllm_omni/model_executor/stage_configs/ming_flash_omni_tts.yaml"
    reference_wav = os.getenv("MING_REFERENCE_WAV", "").strip()

    # Initialize Omni with single-stage talker config
    omni = Omni(
        model=model_path,
        stage_configs_path=stage_config,
        trust_remote_code=True,
        log_stats=True,
    )

    # --- Optional speaker embedding from reference wav ---
    # Prefer reference speaker conditioning for higher quality.
    speaker_info = {"use_zero_spk_emb": True}
    if reference_wav:
        campplus_path = os.path.join(model_path, "talker", "campplus.onnx")
        if os.path.exists(campplus_path) and os.path.exists(reference_wav):
            extractor = SpkembExtractor(campplus_path)
            spk_emb = extractor._extract_spk_embedding(reference_wav)
            speaker_info = {"spk_emb": spk_emb}
            print(f"Using reference speaker embedding from: {reference_wav}")
        else:
            print("Reference wav or campplus.onnx not found; fallback to zero speaker embedding.")

    # --- Basic TTS ---
    inputs = OmniTokensPrompt(
        prompt_token_ids=[0],  # dummy; talker builds its own embeddings
        additional_information={
            "text": "Hello, welcome to Ming flash omni two point zero.",
            "prompt": "Please generate speech based on the following description.\n",
            "max_steps": 20,
            **speaker_info,
        },
    )

    outputs = omni.generate(inputs)
    result = outputs[0]
    multimodal_output = result.outputs[0].multimodal_output

    if multimodal_output and "audio" in multimodal_output:
        waveform = multimodal_output["audio"]
        sample_rate = int(multimodal_output["sr"])  # 44100
        # waveform shape: (1, 1, T) — squeeze to 1D
        audio = waveform.squeeze().numpy()
        sf.write("output_basic.wav", audio, sample_rate)
        print(f"Saved output_basic.wav ({len(audio) / sample_rate:.2f}s, {sample_rate}Hz)")

    # --- TTS with Speaker Embedding (Zero-Shot Voice Cloning) ---
    # Extract speaker embedding from a reference audio file using CAMPPlus.
    # The SpkembExtractor runs on CPU and is loaded automatically by the talker.
    # You can pre-extract embeddings and pass them directly:
    #
    #   from vllm_omni.model_executor.models.ming_flash_omni.spk_embedding import SpkembExtractor
    #   campplus_path = "<model_path>/talker/campplus.onnx"
    #   extractor = SpkembExtractor(campplus_path)
    #   spk_emb = extractor._extract_spk_embedding("reference.wav")  # -> (1, 192)
    #
    # Then pass it in additional_information:
    #
    #   inputs = OmniTokensPrompt(
    #       prompt_token_ids=[0],
    #       additional_information={
    #           "text": "This is voice cloning with a reference speaker.",
    #           "prompt": "Please generate speech based on the following description.\n",
    #           "spk_emb": spk_emb,  # (1, 192) tensor or list of tensors
    #       },
    #   )

    # --- TTS with Custom CFM Parameters ---
    inputs_custom = OmniTokensPrompt(
        prompt_token_ids=[0],
        additional_information={
            "text": "This sentence uses custom generation parameters.",
            "prompt": "Please generate speech based on the following description.\n",
            "max_steps": 20,
            "cfg": 2.0,  # Classifier-free guidance strength (default: 2.0)
            "sigma": 0.25,  # SDE noise level (default: 0.25)
            "temperature": 0.0,  # Sampling temperature (default: 0.0)
            **speaker_info,
        },
    )

    outputs = omni.generate(inputs_custom)
    result = outputs[0]
    multimodal_output = result.outputs[0].multimodal_output
    if multimodal_output and "audio" in multimodal_output:
        audio = multimodal_output["audio"].squeeze().numpy()
        sf.write("output_custom.wav", audio, int(multimodal_output["sr"]))
        print("Saved output_custom.wav")


if __name__ == "__main__":
    main()
