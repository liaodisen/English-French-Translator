from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, english_sentences, french_sentences):
        self.english_sentences = english_sentences
        self.french_sentences = french_sentences

    def __len__(self):
        assert len(self.english_sentences) == len(self.french_sentences), "different length"
        return len(self.english_sentences)
    
    def __getitem__(self, index):
        return self.english_sentences[index], self.french_sentences[index]
    
    