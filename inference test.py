import torch
from transformers import GPT2Tokenizer
from model import MultimodalGPT2
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import sounddevice as sd

# Load your checkpoint and model
checkpoint_path = "trained_models/multimodal_gpt2_medium.pth"
gpt2_model_name = "gpt2-medium"
wav2vec_model_name = "facebook/wav2vec2-base-960h"

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load Wav2Vec2 processor and model for live audio
wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)
wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
wav2vec_model.eval()

# Load your multimodal model
hidden_size = 1024  # GPT-2 Medium's hidden size
num_heads = 16  # GPT-2 Medium's number of heads
num_layers = 24  # GPT-2 Medium's number of layers

multimodal_model = MultimodalGPT2(gpt2_model=None, hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers)
multimodal_model.load_state_dict(torch.load(checkpoint_path))
multimodal_model.eval()

# Function to record your voice and convert it to embeddings
def get_audio_embedding():
    duration = 3  # 3 seconds of audio for "Hello there"
    sample_rate = 16000

    print("Recording audio... Say something!")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished

    # Process the audio
    inputs = wav2vec_processor(recording, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        audio_embeddings = wav2vec_model(**inputs).last_hidden_state

    return audio_embeddings

# Perform inference
def perform_inference():
    audio_embeddings = get_audio_embedding()  # Get the audio embeddings
    input_ids = torch.tensor([tokenizer.eos_token_id]).unsqueeze(0)  # Start with EOS token as input
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = multimodal_model(input_ids=input_ids, attention_mask=attention_mask, audio_embeddings=audio_embeddings)
    
    generated_ids = torch.argmax(output, dim=-1)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Model Response: {generated_text}")

if __name__ == "__main__":
    perform_inference()
  