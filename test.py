import torch
import csv
import numpy as np
import os
from option import args
from PIL import Image
from torchvision import transforms

def preprocess(img):
    img = Image.open(img).convert('RGB')
    imgs = [img, img.transpose(Image.FLIP_LEFT_RIGHT)]
    transform_list = [
        transforms.Resize((args.height, args.width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_list)
    imgs = [transform(i) for i in imgs]
    imgs = [torch.unsqueeze(i, dim=0) for i in imgs]

    return imgs

def test():
    model = torch.load('./pretrained/{}.pt'.format(args.model_name))
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    csv_reader = csv.reader(open(args.val_label, 'r'))
    root_path = args.val_img
    rank = torch.Tensor([i for i in range(101)]).cuda()
    error = 0
    count = 0
    for i in csv_reader:
        name, age, _ = i
        age = int(age)
        img_path = os.path.join(root_path, name)
        imgs = preprocess(img_path)
        predict_age = 0
        for img in imgs:
            img = img.to(device)
            output = model(img)
            predict_age += torch.sum(output*rank, dim=1).item()/2
        print('label:{} \tage:{:.2f}'.format(age, predict_age))
        error += abs(predict_age-age)
        count += 1
    print('MAE:{:.4f}'.format(error/count))


if __name__ == '__main__':
    test()
