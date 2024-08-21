import torch
from transformer import Transformer
from utils import english_tokenizer_load, french_tokenizer_load

def load_model(model_path, device, max_sequence_length=100, d_model=512, num_heads=8, num_layers=6, ffn_hidden=2048, drop_prob=0.1):
    # Load tokenizers
    english_tokenizer = english_tokenizer_load()
    french_tokenizer = french_tokenizer_load()

    # Initialize the Transformer model
    transformer = Transformer(
        d_model=d_model,
        ffn_hidden=ffn_hidden,
        num_heads=num_heads,
        drop_prob=drop_prob,
        num_layers=num_layers,
        max_sequence_length=max_sequence_length,
        kn_vocab_size=french_tokenizer.vocab_size(),
        english_tokenizer=english_tokenizer,
        chinese_tokenizer=french_tokenizer,  # Assuming 'kn' refers to the target language here
        START_TOKEN='<start>',
        END_TOKEN='<end>'
    )

    # Load the model weights
    transformer.load_state_dict(torch.load(model_path, map_location=device))
    transformer.to(device)
    transformer.eval()

    return transformer, english_tokenizer, french_tokenizer

def translate_sentence(sentence, model, english_tokenizer, french_tokenizer, max_sequence_length=100, device='cpu'):
    # Tokenize the input sentence
    eng_tokens = english_tokenizer.encode(sentence)

    # Initialize the decoder input with the start token
    fr_tokens = [french_tokenizer.bos_id()]

    for _ in range(max_sequence_length):
        # Prepare input tensors
        eng_tensor = torch.tensor([eng_tokens], dtype=torch.long).to(device)
        fr_tensor = torch.tensor([fr_tokens], dtype=torch.long).to(device)

        # Create masks (assuming no padding)
        encoder_self_attention_mask = torch.zeros((1, max_sequence_length, max_sequence_length), device=device)
        decoder_self_attention_mask = torch.triu(torch.ones((1, max_sequence_length, max_sequence_length), device=device), diagonal=1) * -1e9
        decoder_cross_attention_mask = torch.zeros((1, max_sequence_length, max_sequence_length), device=device)

        # Forward pass through the model
        with torch.no_grad():
            predictions = model(
                eng_tensor,
                fr_tensor,
                encoder_self_attention_mask,
                decoder_self_attention_mask,
                decoder_cross_attention_mask,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=True,
                dec_end_token=False
            )

        # Get the predicted token for the next word
        next_token = torch.argmax(predictions[0, -1, :]).item()
        fr_tokens.append(next_token)

        # Stop if the end token is generated
        if next_token == french_tokenizer.eos_id():
            break

    # Convert tokens back to words
    translation = french_tokenizer.decode(fr_tokens)
    return translation

if __name__ == "__main__":
    # Example usage
    model_path = 'models/transformer_final_epoch_10.pth'  # Path to your saved model
    sentence = "How are you?"  # Sentence to translate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model, english_tokenizer, french_tokenizer = load_model(model_path, device)

    # Perform translation
    translation = translate_sentence(sentence, model, english_tokenizer, french_tokenizer, device=device)
    print(f"English: {sentence}")
    print(f"Translation: {translation}")
