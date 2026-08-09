"""
Advanced example: generating text with a frontier model via the Anthropic API,
and reasoning about the quality gap versus the local GPT-2 model from
example 02.

Requires: anthropic (see requirements.txt) and an ANTHROPIC_API_KEY
environment variable. Get a key at https://console.anthropic.com

Run it with:
    python 03_advanced_api_generation_and_comparison.py
"""

import os
import sys

import anthropic

MODEL = "claude-opus-4-8"


def generate(client: anthropic.Anthropic, prompt: str) -> str:
    """Send one prompt to Claude and return the text of the reply.

    Same underlying mechanism as examples 01 and 02 -- predict a
    distribution over the next token and sample -- but running on a
    frontier-scale model with training data, alignment, and safety work
    behind it that a hobby project could never replicate.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    # response.content is a list of content blocks; for a plain text reply
    # the first (and only) block has type "text".
    return next(block.text for block in response.content if block.type == "text")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable first.")
        print('  macOS/Linux:  export ANTHROPIC_API_KEY="your-key-here"')
        print('  PowerShell:   $env:ANTHROPIC_API_KEY = "your-key-here"')
        sys.exit(1)

    client = anthropic.Anthropic()

    prompt = "The future of artificial intelligence is"

    print(f"Prompt: {prompt!r}\n")
    print(f"Asking {MODEL} to complete this exact prompt used in example 02:\n")

    completion = generate(client, prompt)
    print(f"  Claude's completion:\n  {completion}\n")

    print("-" * 70)
    print("Now a task GPT-2 (example 02) cannot meaningfully do: follow an")
    print("instruction and reason about a real-world question.\n")

    instruction_prompt = (
        "In exactly 3 bullet points, explain why a 124-million-parameter "
        "model like GPT-2 produces less coherent and less factually reliable "
        "text than a modern frontier LLM. Be concise."
    )
    print(f"Prompt: {instruction_prompt!r}\n")
    answer = generate(client, instruction_prompt)
    print(f"  Claude's answer:\n  {answer}\n")

    print("=" * 70)
    print("WHY THIS COMPARISON MATTERS")
    print("=" * 70)
    print(
        "Both GPT-2 (example 02) and Claude (this example) do the same basic\n"
        "thing mechanically: predict a probability distribution over the next\n"
        "token and sample from it, repeatedly. The dramatic quality difference\n"
        "you just saw comes from THREE factors that scale independently of\n"
        "the core mechanism:\n"
        "  1. Model size (parameters) -- GPT-2 small: 124M. Frontier models:\n"
        "     orders of magnitude larger.\n"
        "  2. Training data quality and volume -- far larger, more curated,\n"
        "     and more recent training corpora.\n"
        "  3. Post-training alignment -- techniques like RLHF/instruction\n"
        "     tuning that teach a model to actually follow instructions and\n"
        "     be helpful/honest/harmless, rather than just continue text\n"
        "     plausibly.\n\n"
        "Track 3 (GenAI Application Developer) is where you'll learn to build\n"
        "real applications on top of frontier models like this one."
    )


if __name__ == "__main__":
    main()
