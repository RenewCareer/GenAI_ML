"""
Intermediate example: text generation with a real, local transformer model.

Requires: transformers, torch (see requirements.txt).
First run will download GPT-2 (~500MB) from Hugging Face and cache it locally.

GPT-2 is a genuine generative language model -- the same family of
architecture (the "Transformer") that powers modern LLMs like Claude, just
far smaller (124M parameters vs. hundreds of billions) and trained on far
less data. Running it locally, with no API key and no network call at
generation time, makes two things concrete:

  1. Generative text models are just software you can run yourself --
     there is no magic happening on someone else's server that couldn't
     happen on your laptop, just at a much smaller scale.
  2. Model SIZE and TRAINING DATA matter enormously for quality. GPT-2
     will produce fluent-sounding but often incoherent or factually wrong
     text. That gap is exactly what frontier models (see example 03)
     are built to close.

Run it with:
    python 02_intermediate_local_transformer_generation.py
"""

from transformers import pipeline, set_seed


def main() -> None:
    print("Loading GPT-2 (small)... this may take a moment on first run.\n")

    # `pipeline` is a high-level Hugging Face helper: give it a task name
    # ("text-generation") and a model name, and it downloads, loads, and
    # wraps everything needed to run inference.
    generator = pipeline("text-generation", model="gpt2")

    # Setting a seed makes the "random" sampling reproducible for this demo.
    set_seed(42)

    prompt = "The future of artificial intelligence is"

    print(f"Prompt: {prompt!r}\n")
    print("Generating 3 different completions (same prompt, different samples):\n")

    outputs = generator(
        prompt,
        max_length=40,       # stop after ~40 tokens total (prompt + generated)
        num_return_sequences=3,  # ask for 3 independent generations
        truncation=True,
    )

    for i, output in enumerate(outputs, start=1):
        print(f"  Completion {i}: {output['generated_text']!r}\n")

    print("-" * 70)
    print("Observe:")
    print("  - Each completion is DIFFERENT, even though the prompt is identical.")
    print("    This is the model SAMPLING from a probability distribution over")
    print("    the next token, not looking up a fixed answer.")
    print("  - The text is grammatically fluent but often not factually")
    print("    grounded or fully coherent -- GPT-2 is a small model with no")
    print("    mechanism for fact-checking itself. Fluency and correctness")
    print("    are two separate things a generative model has to earn.")


if __name__ == "__main__":
    main()
