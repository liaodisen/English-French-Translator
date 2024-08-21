from transformer import Transformer
from utils import english_tokenizer_load, french_tokenizer_load, is_valid_length
from data import TextDataset
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import os

NEG_INFTY = -1e9

class Trainer:

    def __init__(self, model, train_loader, criterion, optimizer, max_sequence_length, lr, device=None, save_dir='models'):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.max_sequence_length = max_sequence_length
        self.lr = lr
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.model.to(self.device)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def create_mask(self, eng_batch, fr_batch):
        num_sentences = len(eng_batch)
        look_ahead_mask = torch.full([self.max_sequence_length, self.max_sequence_length] , True)
        look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
        encoder_padding_mask = torch.full([num_sentences, self.max_sequence_length, self.max_sequence_length] , False)
        decoder_padding_mask_self_attention = torch.full([num_sentences, self.max_sequence_length, self.max_sequence_length] , False)
        decoder_padding_mask_cross_attention = torch.full([num_sentences, self.max_sequence_length, self.max_sequence_length] , False)

        for idx in range(num_sentences):
            eng_sentence_length, fr_sentence_length = len(eng_batch[idx]), len(fr_batch[idx])
            eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, self.max_sequence_length)
            fr_chars_to_padding_mask = np.arange(fr_sentence_length + 1, self.max_sequence_length)
            encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
            encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
            decoder_padding_mask_self_attention[idx, :, fr_chars_to_padding_mask] = True
            decoder_padding_mask_self_attention[idx, fr_chars_to_padding_mask, :] = True
            decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
            decoder_padding_mask_cross_attention[idx, fr_chars_to_padding_mask, :] = True

        encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
        decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
        decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
        return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask
    
    def save_model(self, epoch):
        model_path = os.path.join(self.save_dir, f"transformer_final_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Final model saved at {model_path}")
        
    def train(self, num_epochs):
        fr_tokenizer = self.model.french_tokenizer
        eng_tokenizer = self.model.english_tokenizer
        self.model.train()
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0

            # Initialize tqdm progress bar
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Epoch {epoch + 1}', leave=False)

            for batch_num, batch in progress_bar:
                eng_batch, fr_batch = batch
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = self.create_mask(eng_batch, fr_batch)

                self.optimizer.zero_grad()
                fr_predictions = self.model(
                    eng_batch,
                    fr_batch,
                    encoder_self_attention_mask.to(self.device),
                    decoder_self_attention_mask.to(self.device),
                    decoder_cross_attention_mask.to(self.device),
                    enc_start_token=False,
                    enc_end_token=False,
                    dec_start_token=True,
                    dec_end_token=True
                )

                labels = self.model.decoder.sentence_embedding.tokenize(fr_batch, start_token=False, end_token=True)
                loss = self.criterion(
                    fr_predictions.view(-1, fr_predictions.size(-1)).to(self.device),
                    labels.view(-1).to(self.device)
                )
                
                valid_indices = torch.where(labels.view(-1) == 0, False, True)
                loss = loss.sum() / valid_indices.sum()
                loss.backward()
                self.optimizer.step()
                if batch_num % 500 == 0:
                    print(f"Iteration {batch_num} : {loss.item()}")
                    print(f"English: {eng_batch[0]}")
                    print(f"french Translation: {fr_batch[0]}")
                    fr_sentence_predicted = torch.argmax(fr_predictions[0], axis=1)
                    predicted_sentence = ""
                    for idx in fr_sentence_predicted:
                        if idx == fr_tokenizer.eos_id():
                            break
                        if len(predicted_sentence) > 0:
                            predicted_sentence += " "
                        predicted_sentence += fr_tokenizer.decode(idx.item())
                    print(f"french Prediction: {predicted_sentence}")


                    self.model.eval()
                    fr_sentence = [""]
                    eng_sentence = ["my favourite fruit is grape ."]
                    for word_counter in range(self.max_sequence_length):
                        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= self.create_mask(eng_sentence, fr_sentence)
                        predictions = self.model(eng_sentence,
                                                fr_sentence,
                                                encoder_self_attention_mask.to(self.device), 
                                                decoder_self_attention_mask.to(self.device), 
                                                decoder_cross_attention_mask.to(self.device),
                                                enc_start_token=False,
                                                enc_end_token=True,
                                                dec_start_token=True,
                                                dec_end_token=False)
                        next_token_prob_distribution = predictions[0][word_counter] # not actual probs
                        next_token_index = torch.argmax(next_token_prob_distribution).item()
                        if next_token_index == fr_tokenizer.eos_id():
                            break
                        next_token = fr_tokenizer.decode(next_token_index)
                        if len(fr_sentence[0]) == 0:
                            fr_sentence = [next_token]
                        else:
                            fr_sentence = [fr_sentence[0] + ' ' + next_token]
                    
                    print(f"Evaluation translation (my favourite fruit is grape .) : {fr_sentence[0]}")
                    print("-------------------------------------------")

                # Update the epoch loss and tqdm progress bar
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=epoch_loss / (batch_num + 1))

            print(f"Epoch {epoch + 1} finished with average loss: {epoch_loss / len(self.train_loader):.4f}")
        
        self.save_model(num_epochs)


def main():
    # Parameters
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    max_sequence_length = 100
    d_model = 512
    num_heads = 8
    num_layers = 6
    ffn_hidden = 2048
    drop_prob = 0.1
    num_epochs = 10
    batch_size = 64
    lr = 1e-4
    
    # Load tokenizers
    english_tokenizer = english_tokenizer_load()
    french_tokenizer = french_tokenizer_load()
    
    # Load dataset
    english_file = './data/small_vocab_en.txt'
    french_file = './data/small_vocab_fr.txt'
    with open(english_file, 'r') as file:
        english_sentences = file.readlines()
    with open(french_file, 'r') as file:
        french_sentences = file.readlines()
        TOTAL_SENTENCE = 200000
    french_sentences = french_sentences[:TOTAL_SENTENCE]
    english_sentences = english_sentences[:TOTAL_SENTENCE]
    english_sentences = [sentence.rstrip('\n') for sentence in english_sentences]
    french_sentences = [sentence.rstrip('\n') for sentence in french_sentences]
    valid_sentence_indicies = []
    for index in range(len(french_sentences)):
        french_sentence, english_sentence = french_sentences[index], english_sentences[index]
        if is_valid_length(french_sentence, max_sequence_length) \
        and is_valid_length(english_sentence, max_sequence_length):
            valid_sentence_indicies.append(index)
    english_sentences = [english_sentences[i] for i in valid_sentence_indicies]
    french_sentences = [french_sentences[i] for i in valid_sentence_indicies]
    dataset = TextDataset(english_sentences, french_sentences)
    print("length of the dataset:", len(dataset))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the model
    transformer = Transformer(
        d_model=d_model,
        ffn_hidden=ffn_hidden,
        num_heads=num_heads,
        drop_prob=drop_prob,
        num_layers=num_layers,
        max_sequence_length=max_sequence_length,
        kn_vocab_size=french_tokenizer.vocab_size(),
        english_tokenizer=english_tokenizer,
        french_tokenizer=french_tokenizer,
        START_TOKEN=english_tokenizer.bos_id(),
        END_TOKEN=english_tokenizer.eos_id(),
        PADDING_TOKEN=english_tokenizer.pad_id()
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    optim = torch.optim.Adam(transformer.parameters(), lr=lr)
    
    # Initialize the Trainer
    trainer = Trainer(
        model=transformer,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optim,
        max_sequence_length=max_sequence_length,
        lr=lr,
        device=device
    )
    
    # Start training
    trainer.train(num_epochs)

if __name__ == "__main__":
    main()
