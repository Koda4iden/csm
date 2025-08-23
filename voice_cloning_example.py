#!/usr/bin/env python3
"""
Voice Cloning Example with CSM
This demonstrates how to use CSM to clone a specific voice using audio prompts.
"""

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment

def voice_cloning_example():
    """Demonstrate voice cloning using CSM with audio prompts."""
    
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
    
    try:
        # Download voice prompts from Hugging Face
        print("📥 Downloading voice prompts...")
        prompt_filepath = hf_hub_download(
            repo_id="sesame/csm-1b",
            filename="prompts/conversational_a.wav"
        )
        print(f"✅ Downloaded prompt: {prompt_filepath}")
        
        # Load and prepare the voice prompt
        prompt_audio, prompt_sample_rate = torchaudio.load(prompt_filepath)
        prompt_audio = prompt_audio.squeeze(0)  # Remove batch dimension
        
        # Resample to match generator's sample rate
        prompt_audio = torchaudio.functional.resample(
            prompt_audio, 
            orig_freq=prompt_sample_rate, 
            new_freq=generator.sample_rate
        )
        
        # Create a Segment with the voice prompt
        voice_prompt = Segment(
            text="This is my voice prompt that CSM will use to clone my speaking style.",
            speaker=0,
            audio=prompt_audio
        )
        
        # Text to generate with the cloned voice
        new_text = "Hello! I'm speaking with a cloned voice. CSM has learned my speaking patterns from the audio prompt and can now generate speech that sounds like me."
        
        print(f"🎭 Generating speech with cloned voice for: '{new_text}'")
        
        # Generate audio using the voice prompt as context
        audio = generator.generate(
            text=new_text,
            speaker=0,  # Same speaker as the prompt
            context=[voice_prompt],  # Use the voice prompt as context
            max_audio_length_ms=15000,  # Max 15 seconds
        )
        
        # Save the audio file
        output_file = "cloned_voice_example.wav"
        torchaudio.save(
            output_file,
            audio.unsqueeze(0).cpu(),
            generator.sample_rate
        )
        
        print(f"✅ Cloned voice audio saved to: {output_file}")
        print(f"📊 Audio length: {len(audio) / generator.sample_rate:.2f} seconds")
        print(f"🎵 Sample rate: {generator.sample_rate} Hz")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error downloading prompts: {e}")
        print("💡 You may need to login to Hugging Face first:")
        print("   huggingface-cli login")
        return None

if __name__ == "__main__":
    voice_cloning_example()