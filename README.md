# Multimodal GPT-2 for Audio-Text Processing (WARNING: Experimental Project - No confirmation of working multimodal model as yet)

This project implements a multimodal GPT-2 model that combines audio embeddings with text input for enhanced language modeling. The model is designed to process audio files alongside their transcriptions, potentially improving the quality of language generation tasks.
The goal of this project was to understand the underlying techniques that expanded the capability of text only models to other modes of language input, in this case, Audio. Upon verifying any successes, perhaps these techniques and others could be used to expand the 
Modalities of other Open Source models like Llama 3 etc. The reason for starting with GPT2-Medium, was to allow for both faster training and the use of a single GPU (RTX 2060 Super 8GB). In the scripts' current form, it does indeed train fairly quickly on said hardware and seems to converge in a rather stable fashion. 

![image](https://github.com/user-attachments/assets/07bd9008-221b-43f8-acc6-de9a99197706)



## Project Structure

The project consists of the following main components:  

1. `model.py`: Defines the MultimodalGPT2 model architecture.
2. `train.py`: Contains the training loop and logic for the model.
3. `dataset.py`: Implements the MultimodalDataset for handling audio embeddings and transcriptions.
4. `generate_audio_embeddings.py`: Generates audio embeddings from WAV files using Wav2Vec2.
5. `inference_test.py`: (Not provided, but assumed to exist for testing the trained model)
6. `test.py`: (Not provided, but assumed to exist for unit tests)

## Setup

1. Clone the repository:
   ```
   git clone [repository_url]
   cd [repository_name]
   ```

2. Install the required dependencies:
   ```
   pip install torch transformers wandb soundfile tqdm
   ```

3. Prepare your data:
   - Place your audio files (WAV format) in a directory. 
   - Place corresponding transcription files (TXT format) in another directory.
   - Ensure that the audio files and transcription files have matching names (e.g., `audio1.wav` and `audio1.txt`).
    NB: Depending on your VRAM availability, you may need to chunk your WAV files
     and corresponding Text files (see chunk_transcript_files.py and chunk_audio_files.py script) and manage your batch and 
     gradient accumulation settings in train.py according to your evironment
     
## Usage

### 1. Generate Audio Embeddings

Use the `generate_audio_embeddings.py` script to create embeddings for your audio files:

```
python generate_audio_embeddings.py --audio_dir /path/to/audio/files --transcripts_dir /path/to/transcript/files --output_file embeddings/audio_embeddings.pt --verbose
```

This script will process the audio files, generate embeddings using the Wav2Vec2 model, and save them to the specified output file.

### 2. Train the Model

To train the multimodal GPT-2 model, use the `train.py` script:

```
python train.py
```

The script will:
- Initialize the model and dataset
- Train the model using the specified hyperparameters
- Log training progress using Weights & Biases (wandb)
- Save model checkpoints after each epoch

You can modify the hyperparameters in the `train.py` file or by using command-line arguments (if implemented).

### 3. Inference

To test the trained model, you can use the `inference_test.py` script (implementation not provided):

```
python inference_test.py --model_path /path/to/trained/model --input_text "Your input text" --audio_file /path/to/audio/file.wav
```

## Model Architecture

The MultimodalGPT2 model extends the base GPT-2 architecture by incorporating audio embeddings:

- It uses a pre-trained GPT-2 model as the base.
- Audio embeddings are projected to match GPT-2's hidden size.
- A cross-attention mechanism allows the model to attend to audio embeddings while processing text.
- The model can generate text conditioned on both textual input and audio context.

## Dataset

The MultimodalDataset class handles the loading and preprocessing of data:

- It loads audio embeddings from a PyTorch file.
- It reads transcriptions from individual text files.
- It tokenizes the text data using the GPT-2 tokenizer.
- It ensures that audio embeddings and transcriptions are properly aligned.

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

MIT

## Acknowledgments

- This project uses the Hugging Face Transformers library for the GPT-2 and Wav2Vec2 models.
- Weights & Biases (wandb) is used for experiment tracking and visualization.

## Contact

Log Issue/Bug
