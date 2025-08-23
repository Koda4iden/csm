#!/usr/bin/env python3
"""
Simple CSM Example - Basic Text-to-Speech
This demonstrates the simplest way to use CSM to convert text to speech.
"""

import torch
import torchaudio
from generator import load_csm_1b

def simple_tts():
    """Generate speech from text using CSM."""
    
    # Check available devices
    if torch.cuda.is_available():
        device = "cuda"
        print("🚀 Using CUDA GPU for faster generation")
    else:
        device = "cpu"
        print("🐌 Using CPU (slower but works everywhere)")
    
    print(f"Loading CSM model on {device}...")
    
    # Load the CSM model
    generator = load_csm_1b(device=device)
    
    # Your text to convert to speech
    text = "Hello! This is CSM, a powerful text-to-speech model. It can generate natural-sounding speech from any text input."
    
    print(f"Generating speech for: '{text}'")
    
    # Generate audio (this will take a few seconds)
    audio = generator.generate(
        text=text,
        speaker=0,  # Speaker ID (0 for first speaker)
        context=[],  # No context for simple generation
        max_audio_length_ms=10000,  # Max 10 seconds
    )
    
    # Save the audio file
    output_file = "simple_example.wav"
    torchaudio.save(
        output_file,
        audio.unsqueeze(0).cpu(),  # Add batch dimension and move to CPU
        generator.sample_rate
    )
    
    print(f"✅ Audio saved to: {output_file}")
    print(f"📊 Audio length: {len(audio) / generator.sample_rate:.2f} seconds")
    print(f"🎵 Sample rate: {generator.sample_rate} Hz")
    
    return output_file

if __name__ == "__main__":
    try:
        simple_tts()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have access to the CSM model on Hugging Face")