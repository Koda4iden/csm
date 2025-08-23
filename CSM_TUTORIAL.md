# 🎤 CSM Tutorial: How to Use Conversational Speech Model

## 🚀 **What is CSM?**

CSM (Conversational Speech Model) is an AI model that can:
- **Convert text to speech** with natural-sounding voices
- **Clone specific voices** using audio prompts
- **Generate conversational audio** between multiple speakers
- **Maintain voice consistency** across conversations

## 📋 **Prerequisites**

Before you start, you need:
1. **Hugging Face Account** - to access the models
2. **Python 3.10+** (you have 3.13.3 ✅)
3. **Dependencies** (already installed ✅)

## 🔑 **Step 1: Hugging Face Login**

You need to login to access the CSM models:

```bash
huggingface-cli login
```

This will prompt you for your Hugging Face token. Get it from: https://huggingface.co/settings/tokens

## 🎯 **Step 2: Try the Examples**

### **Option A: Simple Text-to-Speech**
```bash
python3 simple_example.py
```
This creates `simple_example.wav` - basic text-to-speech without voice cloning.

### **Option B: Voice Cloning**
```bash
python3 voice_cloning_example.py
```
This creates `cloned_voice_example.wav` - speech that mimics a specific voice.

### **Option C: Full Conversation**
```bash
python3 run_csm.py
```
This creates `full_conversation.wav` - a conversation between two speakers.

### **Option D: Interactive Chatbot**
```bash
python3 csm_chatbot.py
```
This creates an interactive chatbot that generates speech responses.

## 🔧 **How CSM Works**

### **Basic Generation**
```python
from generator import load_csm_1b

# Load the model
generator = load_csm_1b(device="cuda")  # or "cpu"

# Generate speech
audio = generator.generate(
    text="Your text here",
    speaker=0,  # Speaker ID
    context=[],  # No context for simple generation
    max_audio_length_ms=10000,  # Max 10 seconds
)
```

### **Voice Cloning**
```python
from generator import Segment

# Create a voice prompt
voice_prompt = Segment(
    text="This is my voice prompt",
    speaker=0,
    audio=your_audio_tensor
)

# Generate with cloned voice
audio = generator.generate(
    text="New text with cloned voice",
    speaker=0,
    context=[voice_prompt],  # Use voice prompt as context
    max_audio_length_ms=10000,
)
```

### **Multi-Speaker Conversations**
```python
# Each speaker gets a different ID
speaker_0_audio = generator.generate(text="Hello!", speaker=0, context=[])
speaker_1_audio = generator.generate(text="Hi there!", speaker=1, context=[])

# Combine into conversation
conversation = [speaker_0_audio, speaker_1_audio]
```

## 📁 **Output Files**

- `simple_example.wav` - Basic text-to-speech
- `cloned_voice_example.wav` - Voice cloning example
- `full_conversation.wav` - Multi-speaker conversation
- `bot_response.wav` - Chatbot responses

## ⚠️ **Important Notes**

1. **First Run**: The first time you run CSM, it will download the model (~2GB)
2. **GPU vs CPU**: GPU is much faster, but CPU works everywhere
3. **Voice Prompts**: For best voice cloning, use clear, high-quality audio prompts
4. **Text Length**: Longer text = longer generation time
5. **Context**: More context = better voice consistency but slower generation

## 🎨 **Customization Tips**

### **Change the Voice**
- Modify `speaker` parameter (0, 1, 2, etc.)
- Use different audio prompts for different voices
- Adjust `max_audio_length_ms` for longer/shorter audio

### **Improve Quality**
- Use longer audio prompts (10+ seconds)
- Provide consistent context across generations
- Use the same speaker ID for the same voice

### **Batch Processing**
```python
texts = ["Hello", "How are you?", "Nice to meet you"]
for i, text in enumerate(texts):
    audio = generator.generate(text=text, speaker=0, context=[])
    torchaudio.save(f"output_{i}.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

## 🐛 **Troubleshooting**

### **"Model not found" Error**
- Make sure you're logged into Hugging Face
- Check your internet connection
- Verify you have access to the CSM model

### **"CUDA out of memory" Error**
- Reduce `max_audio_length_ms`
- Use CPU instead of GPU
- Close other applications using GPU

### **"Audio quality is poor"**
- Use longer voice prompts
- Provide more context
- Check your input text quality

## 🌟 **Advanced Features**

### **Custom Voice Prompts**
You can use your own audio files as voice prompts:
```python
import torchaudio

# Load your own audio
audio, sample_rate = torchaudio.load("my_voice.wav")
audio = torchaudio.functional.resample(audio, sample_rate, generator.sample_rate)

# Create custom voice prompt
custom_prompt = Segment(text="My voice prompt", speaker=0, audio=audio)
```

### **Real-time Generation**
For real-time applications, you can generate shorter audio segments:
```python
# Generate short segments for real-time use
audio = generator.generate(
    text="Short phrase",
    speaker=0,
    context=[],
    max_audio_length_ms=2000,  # 2 seconds
)
```

## 🎉 **You're Ready!**

Now you know how to use CSM! Start with the simple examples and gradually explore more advanced features. The key is experimenting with different voice prompts and contexts to get the best results for your use case.

Happy speech generating! 🎤✨