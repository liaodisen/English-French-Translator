import sentencepiece as spm
import numpy as np

def english_tokenizer_load():
    sp_eng = spm.SentencePieceProcessor()
    sp_eng.Load('./eng.model')
    return sp_eng

def french_tokenizer_load():
    sp_fr = spm.SentencePieceProcessor()
    sp_fr.Load('./fr.model')
    return sp_fr

def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1) # need to re-add the end token so leaving 1 space