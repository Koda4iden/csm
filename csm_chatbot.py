import os
from dataclasses import dataclass
from typing import List

import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM

from generator import load_csm_1b, Segment


@dataclass
class ChatTurn:
    speaker: str
    text: str


def load_text_model(device: str = "cuda"):
    """Load a small LLM for text generation."""
    model_id = "meta-llama/Llama-3.2-1B"  # Requires HF access
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device=device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    return tokenizer, model


def generate_text(tokenizer, model, prompt: str, device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt):].strip()


def main() -> None:
    os.environ["NO_TORCH_COMPILE"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models
    generator = load_csm_1b(device=device)
    tokenizer, text_model = load_text_model(device=device)

    conversation: List[ChatTurn] = []
    generated_segments: List[Segment] = []

    print("Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        conversation.append(ChatTurn("User", user_input))

        # Build prompt from conversation
        prompt = "\n".join(f"{t.speaker}: {t.text}" for t in conversation)
        prompt += "\nBot:"

        bot_text = generate_text(tokenizer, text_model, prompt, device)
        print(f"Bot: {bot_text}")

        # Generate audio for the bot response
        audio = generator.generate(
            text=bot_text,
            speaker=0,
            context=generated_segments,
            max_audio_length_ms=10_000,
        )
        torchaudio.save("bot_response.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
        print("Saved audio to bot_response.wav")

        conversation.append(ChatTurn("Bot", bot_text))
        generated_segments.append(Segment(text=bot_text, speaker=0, audio=audio))


if __name__ == "__main__":
    main()
