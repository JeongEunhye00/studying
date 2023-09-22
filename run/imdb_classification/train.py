from tqdm.auto import tqdm


def train(model, loss_fn, optimizer, train_loader, device):
    model.train()

    cur_loss = 0

    for data in tqdm(train_loader):
        model.zero_grad()
        optimizer.zero_grad()

        X = data['text'].to(device)
        Y = data['label'].to(device)

        outputs = model(X)

        loss = loss_fn(outputs, Y)
        loss.backward()
        optimizer.step()
        cur_loss += loss.item()

    return outputs, cur_loss/(len(train_loader))
