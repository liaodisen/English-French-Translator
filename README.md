# English to French Translator

This is a repo consists of the each part of the implementation of the transformer, as well as building a English to French translator using the transformer we built. I followed the tutorial from youtube video [Transformers from scratch](https://www.youtube.com/watch?v=1tgZo2tpK44&list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4&index=12) by CodeEmporium.

## Tokenizer

The translator used a small corups of english and french. For the tokenizer, it used [SentencePiece](https://github.com/google/sentencepiece) by Google. SentencePiece is an unsupervised text tokenizer and detokenizer mainly. The translator used byte-pair-encoding (BPE) to train the tokenizer from raw sentences.

## Train


