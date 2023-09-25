import torch
import math
import numpy as np
from tqdm.auto import tqdm
from nltk.translate.bleu_score import sentence_bleu as bleu


def validation(model, loss_fn, val_loader, device):
    model.eval()
    val_loss = []
    val_ppl = []
    val_bleu = []

    with torch.no_grad():
        for data in tqdm(iter(val_loader)):
            X = data['kor'].to(device)
            Y = data['eng'].to(device)

            output = model(X, Y)

            loss = loss_fn(output.view(-1, output.size(-1)), Y.view(-1))
            val_loss.append(loss.item())
            val_ppl.append(math.exp(loss.item()))

            _, pred = torch.max(output, dim=-1)
            val_bleu.append(bleu(Y, pred))

    return np.mean(val_loss), np.mean(val_ppl), np.mean(val_bleu)
