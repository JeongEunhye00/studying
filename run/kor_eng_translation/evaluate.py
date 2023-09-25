import torch
import numpy as np
from tqdm.auto import tqdm
from nltk.translate.bleu_score import sentence_bleu as bleu


def evaluate(model, test_loader, device):
    model.eval()
    test_bleu = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            X = data['kor'].to(device)
            Y = data['eng'].to(device)
            output = model(X, Y)

            _, pred = torch.max(output, dim=-1)
            test_bleu.append(bleu(Y, pred))

    return np.mean(test_bleu)
