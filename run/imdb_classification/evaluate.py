import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score


def evaluate(model, test_loader, device):
    model.eval()
    f1, acc = 0, 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            X = data['text'].to(device)
            Y = data['label'].cpu().numpy()
            output = model(X)

            _, pred = torch.max(output, 1)
            pred = pred.cpu().numpy()

            f1 += f1_score(Y, pred)
            acc += accuracy_score(Y, pred)

    return f1/len(test_loader), acc/len(test_loader)
