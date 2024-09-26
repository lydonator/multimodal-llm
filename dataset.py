# dataset.py

import torch
from torch.utils.data import Dataset
import os
from typing import Dict, Any, List
import logging
from torch.nn.utils.rnn import pad_sequence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MultimodalDataset(Dataset):
    def __init__(self, audio_embeddings_path: str, transcriptions_dir: str,
                 tokenizer: Any, max_length: int = 512):
        """
        Initializes the MultimodalDataset.

        Args:
            audio_embeddings_path (str): Path to the audio embeddings .pt file.
            transcriptions_dir (str): Path to the directory containing transcription .txt files.
            tokenizer (Any): Tokenizer object from transformers.
            max_length (int, optional): Maximum sequence length for tokenization. Defaults to 512.
        """
        self.audio_embeddings = self._load_embeddings(audio_embeddings_path)
        self.transcriptions = self._load_transcriptions(transcriptions_dir)
        self.keys = self._verify_keys(self.audio_embeddings, self.transcriptions)
        self.tokenizer = tokenizer
        self.max_length = max_length
        logging.info(f"Loaded {len(self.keys)} samples from the dataset.")

    def _load_embeddings(self, file_path: str) -> Dict[str, torch.Tensor]:
        """
        Loads audio embeddings from a .pt file.

        Args:
            file_path (str): Path to the audio embeddings .pt file.

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping filenames to audio embedding tensors.
        """
        if not os.path.exists(file_path):
            logging.error(f"Audio embeddings file not found: {file_path}")
            raise FileNotFoundError(f"Audio embeddings file not found: {file_path}")
        try:
            embeddings = torch.load(file_path, map_location='cpu', weights_only=True)  # Load on CPU first
            if not isinstance(embeddings, dict):
                logging.error(f"Audio embeddings file {file_path} is not a dictionary.")
                raise ValueError(f"Audio embeddings file {file_path} is not a dictionary.")
            # Ensure all values are torch.Tensor
            for key, value in embeddings.items():
                if not isinstance(value, torch.Tensor):
                    logging.error(f"Value for key {key} is not a torch.Tensor.")
                    raise ValueError(f"Value for key {key} is not a torch.Tensor.")
            logging.info(f"Loaded audio embeddings from {file_path}.")
            return embeddings
        except Exception as e:
            logging.error(f"Failed to load audio embeddings from {file_path}: {str(e)}")
            raise

    def _load_transcriptions(self, transcriptions_dir: str) -> Dict[str, str]:
        """
        Loads transcriptions from individual .txt files in a directory.

        Args:
            transcriptions_dir (str): Path to the directory containing transcription .txt files.

        Returns:
            Dict[str, str]: Dictionary mapping filenames to transcription strings.
        """
        if not os.path.exists(transcriptions_dir):
            logging.error(f"Transcriptions directory not found: {transcriptions_dir}")
            raise FileNotFoundError(f"Transcriptions directory not found: {transcriptions_dir}")
        transcriptions = {}
        for filename in os.listdir(transcriptions_dir):
            if filename.endswith('.txt'):
                key = os.path.splitext(filename)[0]
                file_path = os.path.join(transcriptions_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        transcriptions[key] = f.read().strip()
                except Exception as e:
                    logging.error(f"Failed to read transcription file {file_path}: {str(e)}")
                    continue
        logging.info(f"Loaded transcriptions from {transcriptions_dir}.")
        return transcriptions

    def _verify_keys(self, audio_emb: Dict[str, torch.Tensor],
                    transcriptions: Dict[str, str]) -> List[str]:
        """
        Verifies that both audio embeddings and transcriptions have matching keys.

        Args:
            audio_emb (Dict[str, torch.Tensor]): Audio embeddings dictionary.
            transcriptions (Dict[str, str]): Transcriptions dictionary.

        Returns:
            List[str]: List of keys present in both dictionaries.
        """
        audio_keys = set(audio_emb.keys())
        transcription_keys = set(transcriptions.keys())
        common_keys = audio_keys.intersection(transcription_keys)
        missing_in_transcriptions = audio_keys - transcription_keys
        missing_in_audio = transcription_keys - audio_keys

        if missing_in_transcriptions:
            logging.warning(f"{len(missing_in_transcriptions)} audio files have no matching transcription.")
        if missing_in_audio:
            logging.warning(f"{len(missing_in_audio)} transcription files have no matching audio embedding.")

        if not common_keys:
            logging.error("No matching keys found between audio embeddings and transcriptions.")
            raise ValueError("No matching keys found between audio embeddings and transcriptions.")

        return sorted(list(common_keys))  # Sort for consistent ordering

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves the item at the specified index.

        Args:
            idx (int): Index of the item.

        Returns:
            Dict[str, Any]: Dictionary containing 'input_ids', 'attention_mask', 'audio_embeddings'.
        """
        key = self.keys[idx]
        try:
            audio_emb = self.audio_embeddings[key]  # (audio_embedding_dim,)
            transcription = self.transcriptions[key]  # String
        except KeyError:
            logging.error(f"Key {key} not found in audio_embeddings or transcriptions.")
            raise

        # Tokenize transcription
        try:
            encoding = self.tokenizer(
                transcription,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
        except Exception as e:
            logging.error(f"Error during tokenization for key {key}: {str(e)}")
            raise

        input_ids = encoding['input_ids'].squeeze(0)  # (max_length)
        attention_mask = encoding['attention_mask'].squeeze(0)  # (max_length)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'audio_embeddings': audio_emb,  # (audio_embedding_dim,)
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to handle batching of data.

        Args:
            batch (List[Dict[str, Any]]): List of data items.

        Returns:
            Dict[str, torch.Tensor]: Batched input_ids, attention_mask, audio_embeddings.
        """
        input_ids = torch.stack([item['input_ids'] for item in batch])  # (batch_size, max_length)
        attention_mask = torch.stack([item['attention_mask'] for item in batch])  # (batch_size, max_length)
        audio_embeddings = torch.stack([item['audio_embeddings'] for item in batch])  # (batch_size, audio_embedding_dim)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'audio_embeddings': audio_embeddings,
        }

if __name__ == "__main__":
    # Example usage and testing of the dataset
    from transformers import GPT2Tokenizer

    # Define paths to your embedding files
    audio_embeddings_path = "embeddings/audio_embeddings.pt"
    transcriptions_dir = "E:\\Ted\\Text Chunks"  # Directory containing .txt files

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize dataset
    try:
        dataset = MultimodalDataset(
            audio_embeddings_path=audio_embeddings_path,
            transcriptions_dir=transcriptions_dir,
            tokenizer=tokenizer,
            max_length=512
        )
    except Exception as e:
        logging.error(f"Failed to initialize dataset: {str(e)}")
        exit(1)

    # Print dataset size
    print(f"Dataset size: {len(dataset)}")

    # Print a sample item
    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample input_ids:", sample['input_ids'])
        print("Sample attention_mask:", sample['attention_mask'])
        print("Sample audio_embeddings shape:", sample['audio_embeddings'].shape)
    else:
        print("Dataset is empty.")
