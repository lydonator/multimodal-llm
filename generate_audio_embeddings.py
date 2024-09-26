# generate_audio_embeddings.py

import os
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import soundfile as sf
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import argparse
import logging
import sys

class AudioDataset(Dataset):
    def __init__(self, audio_dir, transcripts_dir):
        """
        Initializes the dataset by collecting audio files that have corresponding non-empty transcripts.

        Args:
            audio_dir (str): Directory containing audio segment files.
            transcripts_dir (str): Directory containing transcript segment files.
        """
        self.audio_dir = audio_dir
        self.transcripts_dir = transcripts_dir

        # Collect audio files with non-empty transcripts
        all_audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')]
        valid_audio_files = []
        for f in all_audio_files:
            segment_name = os.path.splitext(f)[0]
            transcript_file = f"{segment_name}.txt"
            transcript_path = os.path.join(transcripts_dir, transcript_file)
            if os.path.exists(transcript_path) and os.path.getsize(transcript_path) > 0:
                valid_audio_files.append(f)
            else:
                logging.warning(f"Transcript file {transcript_file} is missing or empty for audio file {f}. Skipping.")
        
        self.audio_files = valid_audio_files
        logging.info(f"Total audio files to process after filtering: {len(self.audio_files)}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        segment_name = os.path.splitext(audio_file)[0]
        return audio_file, segment_name

def setup_logging(verbose):
    """
    Configures the logging settings.

    Args:
        verbose (bool): If True, set logging level to INFO. Else, WARNING.
    """
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def generate_audio_embeddings(audio_dir, transcripts_dir, output_file, model_name='facebook/wav2vec2-base-960h', batch_size=16, device='cuda', verbose=False):
    """
    Generates audio embeddings using batch processing on the GPU.

    Args:
        audio_dir (str): Directory containing audio segment files.
        transcripts_dir (str): Directory containing transcript segment files.
        output_file (str): Path to save the embeddings dictionary.
        model_name (str, optional): Name of the pre-trained Wav2Vec2 model. Defaults to 'facebook/wav2vec2-base-960h'.
        batch_size (int, optional): Number of audio files to process in a batch. Defaults to 16.
        device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.
        verbose (bool, optional): If True, enable detailed logging. Defaults to False.
    """
    setup_logging(verbose)
    
    # Initialize model and processor
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)
        model.to(device)
        model.eval()
    except Exception as e:
        logging.error(f"Error initializing model or processor: {e}")
        sys.exit(1)
    
    # Create dataset and dataloader
    dataset = AudioDataset(audio_dir, transcripts_dir)
    if len(dataset) == 0:
        logging.error("No valid audio-transcript pairs found. Exiting.")
        return
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    audio_embeddings = {}
    
    for batch in tqdm(dataloader, desc='Processing audio files'):
        audio_files_batch, segment_names_batch = batch
        
        # Load and process audio data
        audio_inputs = []
        for audio_file in audio_files_batch:
            audio_path = os.path.join(audio_dir, audio_file)
            try:
                audio_input, sample_rate = sf.read(audio_path)
                if audio_input.ndim > 1:
                    audio_input = audio_input[:, 0]  # Take first channel if stereo
                audio_inputs.append(audio_input)
            except Exception as e:
                logging.error(f"Error reading {audio_file}: {e}")
                continue  # Skip this audio file
        
        if not audio_inputs:
            continue  # All audio files in this batch failed to load
        
        # Tokenize audio inputs without truncation
        try:
            inputs = processor(audio_inputs, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)
        except Exception as e:
            logging.error(f"Error during tokenization: {e}")
            continue  # Skip this batch
        
        # Generate embeddings
        try:
            with torch.no_grad():
                outputs = model(input_values)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # Take the first token's embedding
        except Exception as e:
            logging.error(f"Error during embedding generation: {e}")
            continue  # Skip this batch
        
        # Store embeddings
        for segment_name, embedding in zip(segment_names_batch, embeddings):
            audio_embeddings[segment_name] = embedding
    
    # Save embeddings to file
    try:
        torch.save(audio_embeddings, output_file)
        logging.info(f"Saved audio embeddings to {output_file}")
    except Exception as e:
        logging.error(f"Error saving embeddings: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Audio Embeddings for Segmented Audio Files")
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio segment files."
    )
    parser.add_argument(
        "--transcripts_dir",
        type=str,
        required=True,
        help="Directory containing transcript segment files."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file to save the embeddings dictionary."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='facebook/wav2vec2-base-960h',
        help="Name of the pre-trained Wav2Vec2 model to use."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of audio files to process in a batch."
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device to run the model on ('cuda' or 'cpu')."
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Increase output verbosity."
    )

    args = parser.parse_args()
    
    generate_audio_embeddings(
        audio_dir=args.audio_dir,
        transcripts_dir=args.transcripts_dir,
        output_file=args.output_file,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        verbose=args.verbose
    )

if __name__ == '__main__':
    main()
