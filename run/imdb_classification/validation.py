import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score


def validation(model, loss_fn, val_loader, device):
    model.eval()
    val_loss = []
    f1, acc = 0, 0

    with torch.no_grad():
        for data in tqdm(iter(val_loader)):
            X = data['text'].to(device)
            Y = data['label'].to(device)

            output = model(X)
            loss = loss_fn(output, Y)
            val_loss.append(loss.item())

            _, pred = torch.max(output, 1)
            pred = pred.cpu().numpy()
            Y = Y.cpu().numpy()
            f1 += f1_score(Y, pred)
            acc += accuracy_score(Y, pred)

    return np.mean(val_loss), f1/len(val_loader), acc/len(val_loader)
