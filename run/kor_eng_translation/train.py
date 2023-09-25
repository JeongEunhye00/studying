from tqdm.auto import tqdm


def train(model, loss_fn, optimizer, train_loader, device):
    model.train()

    cur_loss = 0

    for data in tqdm(train_loader):
        model.zero_grad()
        optimizer.zero_grad()

        X = data['kor'].to(device)
        Y = data['eng'].to(device)

        output = model(X, Y)

        loss = loss_fn(output.view(-1, output.size(-1)), Y.view(-1))
        loss.backward()
        optimizer.step()
        cur_loss += loss.item()

    return output, cur_loss/(len(train_loader))
