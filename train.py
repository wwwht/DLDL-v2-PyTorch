import os
import torch
import data
import loss
import utils
import time
from option import args
from model import ThinAge, TinyAge
from test import test

models = {'ThinAge':ThinAge, 'TinyAge':TinyAge}

def get_model(pretrained=False):
    model = args.model_name
	assert model in models
	if pretrained:
		path = os.path.join('./pretrained/{}.pt'.format(model))
		assert os.path.exists(path)
		return torch.load(path)
    model = models[model]()

    return model

def main():
    model = get_model()
    device = torch.device('cuda')
    model = model.to(device)
    loader = data.Data(args).train_loader
    rank = torch.Tensor([i for i in range(101)]).cuda()
    for i in range(args.epochs):
        lr = 0.001 if i < 30 else 0.0001
        optimizer = utils.make_optimizer(args, model, lr)
        model.train()
        print('Learning rate:{}'.format(lr))
        start_time = time.time()
        for j, inputs in enumerate(loader):
            img, label, age = inputs
            img = img.to(device)
            label = label.to(device)
            age = age.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            ages = torch.sum(outputs*rank, dim=1)
            loss1 = loss.kl_loss(outputs, label)
            loss2 = loss.L1_loss(ages, age)
            total_loss = loss1 + loss2
            total_loss.backward()
            optimizer.step()
            current_time = time.time()
            print('[Epoch:{}] \t[batch:{}]\t[loss={:.4f}]'.format(i, j, total_loss.item()))
            start_time = time.time()
        torch.save(model, './pretrained/{}.pt'.format(args.model_name))
        torch.save(model.state_dict(), './pretrained/{}_dict.pt'.format(args.model_name))
        print('Test: Epoch=[{}]'.format(i))
        if (i+1) % 2 == 0:
            test()


if __name__ == '__main__':
    main()
