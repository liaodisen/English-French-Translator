from transformer import Transformer
from utils import english_tokenizer_load, french_tokenizer_load
from data import TextDataset
import torch
import numpy as np
import tqdm as tqdm





class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, max_sequence_length, lr, device=None):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.max_sequence_length = max_sequence_length
        self.lr = lr
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0

            # Initialize tqdm progress bar
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Epoch {epoch + 1}', leave=False)

            for batch_num, batch in progress_bar:
                eng_batch, kn_batch = batch
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = self.create_masks(eng_batch, kn_batch)

                self.optimizer.zero_grad()
                kn_predictions = self.model(
                    eng_batch,
                    kn_batch,
                    encoder_self_attention_mask.to(self.device),
                    decoder_self_attention_mask.to(self.device),
                    decoder_cross_attention_mask.to(self.device),
                    enc_start_token=False,
                    enc_end_token=False,
                    dec_start_token=True,
                    dec_end_token=True
                )

                labels = self.model.decoder.sentence_embedding.tokenize(kn_batch, start_token=False, end_token=True)
                loss = self.criterion(
                    kn_predictions.view(-1, kn_predictions.size(-1)).to(self.device),
                    labels.view(-1).to(self.device)
                )
                
                valid_indices = torch.where(labels.view(-1) == 0, False, True)
                loss = loss.sum() / valid_indices.sum()
                loss.backward()
                self.optimizer.step()

                # Update the epoch loss and tqdm progress bar
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=epoch_loss / (batch_num + 1))

            print(f"Epoch {epoch + 1} finished with average loss: {epoch_loss / len(self.train_loader):.4f}")