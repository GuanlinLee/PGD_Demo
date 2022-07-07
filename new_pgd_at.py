import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch
import torch.utils.data
import resnet
from torch import nn
from tqdm import tqdm
import random
import os,sys
import numpy as np
import argparse
import loss_functions
from torch.nn import functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--aug', type=int,
					default=0, help='data aug type')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

torch.backends.cudnn.benchmark = True


transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor() ])

transform_2=transforms.Compose([transforms.ToTensor()])
def data_aug(image):
	image = transforms.RandomCrop(32, 4)(image)
	image = transforms.RandomHorizontalFlip()(image)
	return image

transform_test=transforms.Compose([torchvision.transforms.Resize((32,32)),
								   transforms.ToTensor(),
								   ])



if args.aug == 0:
	trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_2)
else:
	trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,
											   num_workers=0, pin_memory=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0,
							 pin_memory=True)


n = resnet.resnet18('cifar10').cuda()

optimizer = torch.optim.SGD(n.parameters(),momentum=0.9,
							lr=0.1,weight_decay=5e-4)

for epoch in range(100):
	loadertrain = tqdm(train_loader, desc='{} E{:03d}'.format('train', epoch), ncols=0)
	total=0.0
	clean_acc = 0.0
	adv_acc = 0.0
	for input, target in loadertrain:
		n.eval()
		if args.aug == 0:
			x_train, y_train = data_aug(input).cuda(), target.cuda()
		else:
			x_train, y_train = input.cuda(), target.cuda()
		y_pre = n(x_train)
		loss_clean = F.cross_entropy(y_pre, y_train)
		logits_adv, loss = loss_functions.PGD(n, x_train, y_train, optimizer, args)
		loss.backward()
		optimizer.step()
		_, predicted = torch.max(y_pre.data, 1)
		_, predictedadv = torch.max(logits_adv.data, 1)
		total += y_train.size(0)
		clean_acc += predicted.eq(y_train.data).cuda().sum()
		adv_acc += predictedadv.eq(y_train.data).cuda().sum()
		fmt = '{:.4f}'.format
		loadertrain.set_postfix(loss=fmt(loss.data.item()),
								acc_cl=fmt(clean_acc.item() / total * 100),
								acc_adv=fmt(adv_acc.item() / total * 100))
