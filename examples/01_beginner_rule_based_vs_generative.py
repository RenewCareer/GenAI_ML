"""
Beginner example: Rule-based ("classic AI") vs. a simple generative model.

No third-party dependencies -- pure Python standard library.

This script builds two tiny systems side by side:
  1. A rule-based chatbot: hard-coded if/elif logic. This is "AI" in the
     broad sense (a program that responds intelligently to input) but it
     is NOT generative -- it can only ever say things a programmer wrote.
  2. A Markov-chain text generator: it "learns" which words tend to follow
     which other words from a small training text, then GENERATES new
     sentences it has never seen before by repeatedly predicting "what
     word comes next" and sampling one. This is the same core idea behind
     GPT-2 and Claude, just wildly smaller and simpler.

Run it with:
    python 01_beginner_rule_based_vs_generative.py
"""

import random


# ---------------------------------------------------------------------------
# 1. Rule-based chatbot -- NOT generative. Every possible reply is written
#    by a human ahead of time. It can never say anything novel.
# ---------------------------------------------------------------------------
def rule_based_chatbot(user_input: str) -> str:
    text = user_input.lower().strip()

    if "hello" in text or "hi" in text:
        return "Hello! How can I help you today?"
    if "weather" in text:
        return "I don't have live weather data, but I hope it's nice out!"
    if "your name" in text:
        return "I'm a simple rule-based bot with no understanding of language."
    if "bye" in text:
        return "Goodbye!"
    return "Sorry, I only know how to respond to a few fixed phrases."


# ---------------------------------------------------------------------------
# 2. A minimal generative model: a Markov chain over words.
#
#    How it works:
#    - Look at a training text and record, for every pair of consecutive
#      words (w1, w2), which word came next.
#    - To generate text: start from a seed pair of words, look up all the
#      words that have followed that pair before, pick one at RANDOM
#      (weighted by how often it occurred), append it, slide the window
#      forward, and repeat.
#
#    This is exactly the same *shape* of process an LLM uses --
#    predict a probability distribution over "next token" and sample --
#    just with word-pair counts instead of a neural network.
# ---------------------------------------------------------------------------
TRAINING_TEXT = """
the quick brown fox jumps over the lazy dog. the dog barks at the fox.
the fox runs into the forest. the forest is dark and quiet. the quick
fox is not afraid of the dark forest. the lazy dog sleeps all day.
"""


def build_markov_chain(text: str) -> dict:
    words = text.split()
    chain: dict[tuple[str, str], list[str]] = {}
    for i in range(len(words) - 2):
        key = (words[i], words[i + 1])
        next_word = words[i + 2]
        chain.setdefault(key, []).append(next_word)
    return chain


def generate_markov_text(chain: dict, num_words: int = 20) -> str:
    # Start from a random key (word pair) that actually exists in the chain.
    current_key = random.choice(list(chain.keys()))
    result = list(current_key)

    for _ in range(num_words - 2):
        possible_next = chain.get(current_key)
        if not possible_next:
            break
        next_word = random.choice(possible_next)  # <-- the "generation" step
        result.append(next_word)
        current_key = (current_key[1], next_word)

    return " ".join(result)


def main() -> None:
    print("=" * 70)
    print("PART 1: Rule-based chatbot (NOT generative)")
    print("=" * 70)
    test_inputs = ["Hello there", "What's your name?", "How's the weather?", "bye"]
    for user_msg in test_inputs:
        print(f"  You: {user_msg}")
        print(f"  Bot: {rule_based_chatbot(user_msg)}\n")
    print("Notice: every reply above was written by a human in advance.")
    print("The bot cannot say anything that wasn't explicitly programmed.\n")

    print("=" * 70)
    print("PART 2: Markov chain (a minimal GENERATIVE model)")
    print("=" * 70)
    chain = build_markov_chain(TRAINING_TEXT)
    print("Learned word-pair -> next-word statistics from a short training text.")
    print("Generating 3 new sentences the model has never seen verbatim:\n")
    for i in range(3):
        print(f"  Generated {i + 1}: {generate_markov_text(chain)}")

    print("\nNotice: this text is NEW -- it wasn't copy-pasted from the training")
    print("text. It was produced by repeatedly predicting 'what word is likely")
    print("to come next' and sampling from that prediction. That is the same")
    print("core mechanism (predict-and-sample) that GPT-2 and Claude use --")
    print("just with word-pair counts instead of a neural network.")


if __name__ == "__main__":
    main()
