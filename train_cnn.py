import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from models import CNN
import numpy as np
HYPER_PARAM = {
    'lr': 5e-2,
    'epochs': 50,
    'current_path': ''
}



def train(model: nn.Module, train_loader: DataLoader):
    optimizer = torch.optim.SGD(model.parameters(), lr=HYPER_PARAM['lr'])
    loss_func = nn.CrossEntropyLoss(weight=torch.FloatTensor([1 / 75, 1 / 125]).to('cuda:0'))
    model.train()
    best_loss = 1e10

    for epochs in range(HYPER_PARAM['epochs']):
        epochs_loss = 0.0
        matrix = np.array([[0, 0], [0, 0]])

        for index, (data, label) in enumerate(train_loader):
            (data, label) = data.to('cuda:0'), label.to('cuda:0')

            optimizer.zero_grad()
            pred = model(data)
            loss_iter = loss_func(pred, label)
            loss_iter.backward()
            optimizer.step()

            epochs_loss += loss_iter.item()
            # print(pred)
            pred_soft = torch.argmax(pred, 1)
            # print(pred_soft)

            for label_, pred_ in zip(label, pred_soft):
                # print(label_, '\n', pred_)
                matrix[label_.item(), pred_.item()] += 1


        if best_loss > epochs_loss:
            torch.save(model.state_dict(), f'static/weights/{HYPER_PARAM["current_path"]}/best.pt')
            best_loss = epochs_loss

        print(f'Epochs: {epochs + 1}/{HYPER_PARAM["epochs"]}train loss: {epochs_loss: .4f}, acc: {matrix.diagonal().sum() / matrix.sum()}')
        print('matrix: \n', matrix)
    return model

def main():
    t = transforms.Compose([
        transforms.ToTensor(),
    ])
    folder = ImageFolder('./static/image', transform=t)
    loader = DataLoader(folder, batch_size=8, shuffle=True)
    model = CNN.PlainCNN()
    model.to('cuda:0')
    model = train(model, loader)
    torch.save(model.state_dict(), f'./static/weights/{HYPER_PARAM["current_path"]}/last.pt')


if __name__ == '__main__':
    import os
    num = len(os.listdir('./static/weights'))
    os.mkdir(f'./static/weights/exp{num + 1}')
    HYPER_PARAM['current_path'] = f'exp{num + 1}'
    main()