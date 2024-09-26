import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Wav2Vec2Processor, Wav2Vec2Model
from model import MultimodalGPT2
import librosa
import os
import argparse
import json
import logging
from typing import Dict, List
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str = "config.yaml") -> Dict:
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

def get_audio_embeddings(audio_path: str, processor: Wav2Vec2Processor, wav2vec_model: Wav2Vec2Model, device: torch.device) -> torch.Tensor:
    try:
        speech, sr = librosa.load(audio_path, sr=16000)
        input_values = processor(speech, sampling_rate=sr, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            embeddings = wav2vec_model(input_values).last_hidden_state  # (1, seq_len, hidden_size)
        return embeddings.squeeze(0).cpu()
    except Exception as e:
        logging.error(f"Error processing audio file {audio_path}: {str(e)}")
        raise

def load_test_cases(test_cases_file: str) -> List[Dict]:
    try:
        with open(test_cases_file, 'r') as f:
            return json.load(f)   
    except Exception as e:
        logging.error(f"Error loading test cases from {test_cases_file}: {str(e)}")
        raise

def save_test_results(results: List[Dict], output_file: str):
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Test results saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving test results to {output_file}: {str(e)}")

def evaluate_response(generated: str, expected: str) -> Dict[str, float]:
    rouge = Rouge()
    rouge_scores = rouge.get_scores(generated, expected)[0]
    
    bleu_score = sentence_bleu([expected.split()], generated.split())
    
    return {
        "bleu": bleu_score,
        "rouge1_f": rouge_scores['rouge-1']['f'],
        "rouge2_f": rouge_scores['rouge-2']['f'],
        "rougeL_f": rouge_scores['rouge-l']['f']
    }

def test_model(test_cases_file: str, output_file: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        # Load tokenizer and GPT-2 model (Medium)
        tokenizer = GPT2Tokenizer.from_pretrained(CONFIG['model_name'])
        gpt2_model = GPT2LMHeadModel.from_pretrained(CONFIG['model_name'])
        gpt2_model.to(device)

        # Initialize Multimodal GPT-2
        hidden_size = gpt2_model.config.hidden_size
        num_heads = gpt2_model.config.n_head
        num_layers = gpt2_model.config.n_layer
        multimodal_model = MultimodalGPT2(gpt2_model, hidden_size, num_heads, num_layers)
        model_path = os.path.join(CONFIG['model_save_dir'], "multimodal_gpt2_medium.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Please train the model first.")
        multimodal_model.load_state_dict(torch.load(model_path, map_location=device))
        multimodal_model.to(device)
        multimodal_model.eval()

        # Load Wav2Vec2 processor and model
        processor = Wav2Vec2Processor.from_pretrained(CONFIG['wav2vec_model'])
        wav2vec_model = Wav2Vec2Model.from_pretrained(CONFIG['wav2vec_model']).to(device)
        wav2vec_model.eval()

        # Load test cases
        test_cases = load_test_cases(test_cases_file)

        results = []
        for idx, test in enumerate(test_cases):
            audio_path = test['audio_path']
            description = test['description']
            expected = test['expected']
            logging.info(f"\nTest Case {idx + 1}: {description}")

            if not os.path.exists(audio_path):
                logging.warning(f"Audio file {audio_path} not found, skipping.")
                continue

            # Get audio embeddings
            audio_embeddings = get_audio_embeddings(audio_path, processor, wav2vec_model, device).unsqueeze(0).to(device)

            # Define a dummy text prompt (start token)
            input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
            attention_mask = torch.tensor([[1]]).to(device)

            # Forward pass
            with torch.no_grad():
                logits = multimodal_model(input_ids, attention_mask, audio_embeddings)

            # Generate text by selecting the highest probability token at each step
            generated_ids = torch.argmax(logits, dim=-1)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            logging.info(f"Generated Response: {generated_text}")
            logging.info(f"Expected Response: {expected}")

            # Evaluate the response
            evaluation = evaluate_response(generated_text, expected)
            logging.info(f"Evaluation: {evaluation}")

            results.append({
                "test_case": idx + 1,
                "description": description,
                "generated": generated_text,
                "expected": expected,
                "evaluation": evaluation
            })

        # Save test results
        save_test_results(results, output_file)

    except Exception as e:
        logging.error(f"An error occurred during testing: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Multimodal GPT-2 model.")
    parser.add_argument("--test_cases", type=str, default="test_cases.json", help="Path to the test cases JSON file.")
    parser.add_argument("--output", type=str, default="test_results.json", help="Path to save the test results.")
    args = parser.parse_args()

    test_model(args.test_cases, args.output)
