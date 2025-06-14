from __future__ import print_function

import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import os
import argparse
import time
from models import *
from utils import progress_bar
from randomaug import RandAugment
import timm
from timm.scheduler import CosineLRScheduler
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--opt', default="adam", help="Optimizer (adam or sgd)")
parser.add_argument('--resume', '-r', action='store_false', help='Resume from checkpoint')
parser.add_argument('--noamp', action='store_true', help='Disable AMP (mixed precision training)')
parser.add_argument('--net', default='vit_small', help="Network architecture (e.g., vit_small)")
parser.add_argument('--bs', default=32, type=int, help="Batch size")
parser.add_argument('--size', default=224, type=int, help="Input image size")
parser.add_argument('--n_epochs', type=int, default=5, help="Number of epochs")
parser.add_argument('--data_dir', default='./dataset/real-vs-fake/', type=str, help="Dataset directory")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_amp = not args.noamp
bs = args.bs
imsize = args.size
best_acc = 0
start_epoch = 0

print('==> Preparing data...')
transform_train = transforms.Compose([
    transforms.Resize(imsize),
    RandAugment(2, 9),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5210, 0.4261, 0.3808), std=(0.2769, 0.2514, 0.2524)),
])

transform_test = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5198, 0.4254, 0.3805), std=(0.2772, 0.2514, 0.2527)),
])

trainset = ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4)

testset = ImageFolder(root=os.path.join(args.data_dir, 'test'), transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=4)

classes = trainset.classes

print('==> Building model...')
net = timm.create_model("vit_small_patch16_224", pretrained=True)
net.head = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(net.head.in_features, len(classes))
)
net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=args.lr) if args.opt == "adam" else optim.SGD(net.parameters(), lr=args.lr)
class_counts = [trainset.targets.count(c) for c in range(len(classes))]
class_weights = [1.0 / count for count in class_counts]
class_weights = torch.tensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

scheduler = CosineLRScheduler(
    optimizer,
    t_initial=args.n_epochs,
    warmup_t=5,
    warmup_lr_init=1e-6,
    lr_min=1e-6,
)

scaler = GradScaler(enabled=use_amp)

def log_to_csv(epoch, train_acc, train_loss, test_acc, test_loss, train_f1, test_f1, log_file='training_logs.csv'):
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Epoch', 'Train Accuracy', 'Train Loss', 'Test Accuracy', 'Test Loss', 'Train F1 Score', 'Test F1 Score'])
        writer.writerow([epoch, train_acc, train_loss, test_acc, test_loss, train_f1, test_f1])

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss, correct, total = 0, 0, 0
    all_preds = []
    all_targets = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % 
                     (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    train_acc = 100. * correct / total
    train_loss = train_loss / (batch_idx + 1)
    train_f1 = f1_score(all_targets, all_preds, average='weighted')
    print(f'Train F1 Score: {train_f1:.4f}')
    return train_acc, train_loss, train_f1

def test(epoch):
    global best_acc
    net.eval()
    test_loss, correct, total = 0, 0, 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast(device_type='cuda', enabled=use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            progress_bar(batch_idx, len(testloader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)' %
                         (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    test_acc = 100. * correct / total
    test_loss = test_loss / (batch_idx + 1)
    test_f1 = f1_score(all_targets, all_preds, average='weighted')
    print(f'Test F1 Score: {test_f1:.4f}')
    log_to_csv(epoch, train_acc, train_loss, test_acc, test_loss, train_f1, test_f1)
    if test_acc > best_acc:
        print('Saving best model...')
        state = {'net': net.state_dict(), 'acc': test_acc, 'epoch': epoch}
        torch.save(state, './checkpoint/{}-best-ckpt.t7'.format(args.net))
        best_acc = test_acc

if __name__ == '__main__':
    log_file = 'training_logs.csv'
    if not os.path.isfile(log_file):
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Accuracy', 'Train Loss', 'Test Accuracy', 'Test Loss', 'Train F1 Score', 'Test F1 Score'])
    for epoch in range(start_epoch, args.n_epochs):
        train_acc, train_loss, train_f1 = train(epoch)
        test(epoch)
        scheduler.step(epoch + 1)
    print('Best accuracy: %.3f' % best_acc)
