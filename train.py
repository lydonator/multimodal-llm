import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from dataset import MultimodalDataset
from model import MultimodalGPT2
import os
import wandb
import sys

# Flag to handle interrupt signals
interrupted = False

def save_model(multimodal_model, epoch, optimizer, scheduler, scaler):
    os.makedirs("trained_models", exist_ok=True)
    model_save_path = f"trained_models/multimodal_gpt2_medium_epoch{epoch + 1}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': multimodal_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict()
    }, model_save_path)
    print(f"Model saved at {model_save_path}")
    wandb.save(model_save_path)


def train():
    global interrupted
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Weights & Biases
    wandb.init(project='multimodal_gpt2_project', config={
        'learning_rate': 5e-5,
        'batch_size': 1,  # Adjust based on your GPU capacity
        'epochs': 3,
        'max_length': 512,
        'gradient_accumulation_steps': 16,
    })
    config = wandb.config

    # Load tokenizer and GPT-2 model (Medium)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    gpt2_model.resize_token_embeddings(len(tokenizer))  # Adjust embeddings if needed

    # Initialize Multimodal GPT-2
    audio_embedding_dim = next(iter(torch.load("embeddings/audio_embeddings.pt", weights_only=True).values())).shape[0]
    multimodal_model = MultimodalGPT2(gpt2_model, audio_embedding_dim)
    multimodal_model.to(device)

    # Load dataset
    dataset = MultimodalDataset(
        audio_embeddings_path="embeddings/audio_embeddings.pt",
        transcriptions_dir="E:\\Ted\\Text Chunks",  # Directory containing .txt files
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(multimodal_model.parameters(), lr=config.learning_rate, weight_decay=1e-2)
    total_steps = len(dataloader) * config.epochs // config.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Training parameters
    num_epochs = config.epochs
    accumulation_steps = config.gradient_accumulation_steps
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision

    multimodal_model.train()
    try:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(device)  # (batch_size, max_length)
                attention_mask = batch['attention_mask'].to(device)  # (batch_size, max_length)
                audio_embeddings = batch['audio_embeddings'].to(device)  # (batch_size, audio_embedding_dim)

                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    outputs = multimodal_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        audio_embeddings=audio_embeddings,
                    )
                    logits = outputs['logits']  # (batch_size, seq_len, vocab_size)
                    loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    loss = loss / accumulation_steps  # Normalize loss

                scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(multimodal_model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                    # Log loss
                    current_loss = loss.item() * accumulation_steps
                    total_loss += current_loss
                    global_step = epoch * len(dataloader) + batch_idx
                    wandb.log({'train_loss': current_loss, 'learning_rate': scheduler.get_last_lr()[0]}, step=global_step)
                    print(f"  Batch {batch_idx + 1} Loss: {current_loss:.4f}")

                if interrupted:
                    save_model(multimodal_model, epoch, optimizer, scheduler, scaler)
                    sys.exit(0)

            avg_loss = total_loss / (len(dataloader) / accumulation_steps)
            print(f"Average Loss for Epoch {epoch + 1}: {avg_loss:.4f}")
            wandb.log({'epoch_loss': avg_loss}, step=epoch)

            # Save model checkpoint after each epoch
            save_model(multimodal_model, epoch, optimizer, scheduler, scaler)

    except KeyboardInterrupt:
        print("Training interrupted. Saving model checkpoint...")
        interrupted = True
        save_model(multimodal_model, epoch, optimizer, scheduler, scaler)

    print("Training completed.")
    wandb.finish()

if __name__ == "__main__":
    train()
 