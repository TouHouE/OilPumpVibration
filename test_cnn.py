import torch
import cv2
import os
from torchvision.transforms import transforms
import numpy as np
from models.CNN import PlainCNN


def main():
    model = PlainCNN()
    model.load_state_dict(torch.load('./static/weights/exp3/best.pt'))
    matrix = np.array([[0, 0], [0, 0]])
    error_root = './static/test/error'
    normal_root = './static/test/normal'
    t = transforms.Compose([
        transforms.ToTensor()
    ])
    error = [np.ascontiguousarray(cv2.imread(f'{error_root}/{name}')[..., ::-1]) for name in os.listdir('./static/test/error')]
    normal = [np.ascontiguousarray(cv2.imread(f'{normal_root}/{name}')[..., ::-1]) for name in os.listdir(normal_root)]

    for img in error:
        img = t(img).unsqueeze(0)
        print(img.size())
        pred = model(img)
        pred_arg = torch.argmax(pred)
        matrix[0, pred_arg.item()] += 1


    for img in normal:
        img = t(img).unsqueeze(0)
        pred = model(img)
        pred_arg = torch.argmax(pred)
        matrix[1, pred_arg.item()] += 1
    print(matrix)

main()