import torch
import torch.utils.data as data


class IMDBDataset(data.Dataset):
    def __init__(self, data, vocab, max_len, device):
        super(IMDBDataset, self).__init__()
        self.data = data
        self.text = self.data['review']
        self.label = self.data['sentiment']
        self.vocab = vocab
        self.ch_label = lambda x: 0 if x == 'negative' else 1
        self.max_len = max_len
        self.device = device

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        tmp_text = self.vocab.tokenize(self.text[idx])
        if len(tmp_text) > self.max_len:
            tmp_text = tmp_text[:self.max_len]
        else:
            tmp_text = tmp_text + ["<PAD>"]*(self.max_len - len(tmp_text))

        return {'text': torch.Tensor(self.vocab.word2num(tmp_text)).to(device=self.device, dtype=torch.int32),
                'label': self.ch_label(self.label[idx])}
