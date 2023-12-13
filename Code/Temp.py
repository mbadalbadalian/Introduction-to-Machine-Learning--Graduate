import os
import torch
from transformers import GPT2LMHeadModel
from torch.utils.data import DataLoader
from tqdm import tqdm

def CreateTrainedNLPModel(train_loader,test_loader,NLP_model_filepath,average_train_loss_filename,average_test_loss_filename):
    num_epochs = 3

    NLP_model = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NLP_model.to(device)
    optimizer = torch.optim.AdamW(NLP_model.parameters(),lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()
    average_train_loss_list = []
    average_test_loss_list = []

    # Directory to save/load checkpoints
    checkpoint_dir = "path/to/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check if there are existing checkpoints in the directory
    existing_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
    if existing_checkpoints:
        # Sort checkpoints based on epoch number
        existing_checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

        # Load the last checkpoint
        last_checkpoint = existing_checkpoints[-1]
        checkpoint_filename = os.path.join(checkpoint_dir, last_checkpoint)
        checkpoint = torch.load(checkpoint_filename)

        # Load model and optimizer state
        NLP_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['val_loss']
        print(f"Resuming training from epoch {start_epoch + 1}, best validation loss: {best_val_loss:.4f}")
    else:
        start_epoch = 0
        best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        NLP_model.train()
        total_train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # Training steps...
            # ...

        # Validation loop
        NLP_model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Validation"):
                # Validation steps...
                # ...

        # Save checkpoint at the end of each epoch
        checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': NLP_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': total_test_loss / len(test_loader),  # Adjust this based on your validation loss calculation
        }, checkpoint_filename)

        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {total_test_loss / len(test_loader):.4f}, Checkpoint saved at {checkpoint_filename}")

        # Update best validation loss and early stopping criteria if needed
        if total_test_loss < best_val_loss:
            best_val_loss = total_test_loss
            # Save the best model separately if needed
            torch.save(NLP_model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))

        average_train_loss_list.append(total_train_loss / len(train_loader))
        average_test_loss_list.append(total_test_loss / len(test_loader))

    NLP_model.save_pretrained(NLP_model_filepath)
    SavePKL(average_train_loss_list, average_train_loss_filename)
    SavePKL(average_test_loss_list, average_test_loss_filename)

    return NLP_model, average_train_loss_list, average_test_loss_list
