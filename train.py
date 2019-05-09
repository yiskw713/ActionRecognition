import adabound
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import yaml

from addict import Dict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomCrop, Normalize

from utils.checkpoint import save_checkpoint, resume
from utils.class_weight import get_class_weight
from utils.dataset import Kinetics
from utils.mean import get_mean, get_std
from model import resnet
from model import slowfast
from model.metric import L2ConstrainedLinear
from model.msc import SpatialMSC, TemporalMSC, SpatioTemporalMSC


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='train a network for action recognition')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument('--resume', type=bool, default=False,
                        help='if you start training from checkpoint')

    return parser.parse_args()


def train(model, train_loader, criterion, optimizer, config, device):
    model.train()

    epoch_loss = 0.0
    for sample in tqdm.tqdm(train_loader, total=len(train_loader)):
        x = sample['clip']
        t = sample['cls_id']
        x = x.to(device)
        t = t.to(device)

        h = model(x)
        if config.msc:
            loss = 0.0
            for scaled_h in h:
                loss += criterion(scaled_h, t)
        else:
            loss = criterion(h, t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    N = output.shape[0]
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return N, res


def validation(model, val_loader, criterion, config, device):
    model.eval()
    val_loss = 0.0
    n_samples = 0.0
    top1 = 0.0
    top5 = 0.0

    with torch.no_grad():
        for sample in tqdm.tqdm(val_loader, total=len(val_loader)):
            # temporal size is input_frames(default 16) * interger
            x = sample['clip']
            x = x.to(device)
            t = sample['cls_id']
            t = t.to(device)

            h = model(x)
            val_loss += criterion(h, t).item()
            n, topk = accuracy(h, t, topk=(1, 5))
            n_samples += n
            top1 += topk[0].item()
            top5 += topk[1].item()

        val_loss /= len(val_loader)
        top1 /= n_samples
        top5 /= n_samples

    return val_loss, top1, top5


def main():
    args = get_arguments()

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))

    # writer
    if CONFIG.writer_flag:
        writer = SummaryWriter(CONFIG.result_path)
    else:
        writer = None

    # DataLoaders
    normalize = Normalize(mean=get_mean(), std=get_std())

    train_data = Kinetics(
        CONFIG,
        transform=Compose([
            RandomCrop((CONFIG.height, CONFIG.width)),
            ToTensor(),
            normalize,
        ])
    )

    val_data = Kinetics(
        CONFIG,
        transform=Compose([
            RandomCrop((CONFIG.height, CONFIG.width)),
            ToTensor(),
            normalize,
        ]),
        mode='validation'
    )

    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=CONFIG.num_workers
    )

    # load model
    print('\n------------------------Loading Model------------------------\n')

    if CONFIG.model == 'resnet18':
        print(CONFIG.model + ' will be used as a model.')
        model = resnet.resnet18(num_classes=CONFIG.n_classes)
    elif CONFIG.model == 'slowfast18':
        print(CONFIG.model + ' will be used as a model.')
        model = slowfast.resnet18(class_num=CONFIG.n_classes)
    else:
        print('resnet18 will be used as a model.')
        model = resnet.resnet18(num_classes=CONFIG.n_classes)

    # metric
    if CONFIG.metric == 'L2constrain':
        print('L2constrain metric will be used.')
        model.fc = L2ConstrainedLinear(
            model.fc.in_features, model.fc.out_features)

    # multi-scale input
    if CONFIG.msc == 'Temporal':
        print('Temporal multi-scale input will be used')
        model = TemporalMSC(model)
    elif CONFIG.msc == 'Spatial':
        print('Spatial multi-scale input will be used')
        model = SpatialMSC(model)
    elif CONFIG.msc == 'SpatioTemporal':
        print('SpatioTemporal multi-scale input will be used')
        model = SpatioTemporalMSC(model)

    # set optimizer, lr_scheduler
    if CONFIG.optimizer == 'Adam':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
    elif CONFIG.optimizer == 'SGD':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = optim.SGD(
            model.parameters(),
            lr=CONFIG.learning_rate,
            momentum=CONFIG.momentum,
            dampening=CONFIG.dampening,
            weight_decay=CONFIG.weight_decay,
            nesterov=CONFIG.nesterov)
    elif CONFIG.optimizer == 'AdaBound':
        print(CONFIG.optimizer + ' will be used as an optimizer.')
        optimizer = adabound.AdaBound(
            model.parameters(), lr=CONFIG.learning_rate, final_lr=CONFIG.final_lr, weight_decay=CONFIG.weight_decay)
    else:
        print('There is no optimizer which suits to your option. \
            Instead, SGD will be used as an optimizer.')
        optimizer = optim.SGD(
            model.parameters(),
            lr=CONFIG.learning_rate,
            momentum=CONFIG.momentum,
            dampening=CONFIG.dampening,
            weight_decay=CONFIG.weight_decay,
            nesterov=CONFIG.nesterov)

    # learning rate scheduler
    if CONFIG.optimizer == 'SGD':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=CONFIG.lr_patience)
    else:
        scheduler = None

    # resume if you want
    begin_epoch = 0
    if args.resume:
        begin_epoch, model, optimizer, scheduler = \
            resume(CONFIG, model, optimizer, scheduler)

    # send the model to cuda/cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)  # make parallel
        torch.backends.cudnn.benchmark = True

    # criterion for loss
    if CONFIG.class_weight:
        criterion = nn.CrossEntropyLoss(
            weight=get_class_weight().to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # train and validate model
    print('\n------------------------Start training------------------------\n')
    losses_train = []
    losses_val = []
    top1_accuracy = []
    top5_accuracy = []
    best_top1_accuracy = 0.0
    best_top5_accuracy = 0.0

    for epoch in range(begin_epoch, CONFIG.max_epoch):
        # training
        loss_train = train(
            model, train_loader, criterion, optimizer, CONFIG, device)
        losses_train.append(loss_train)

        # validation
        loss_val, top1, top5 = validation(
            model, val_loader, criterion, CONFIG, device)

        if CONFIG.optimizer == 'SGD':
            scheduler.step(loss_val)

        losses_val.append(loss_val)
        top1_accuracy.append(top1)
        top5_accuracy.append(top5)

        # save a model if topk accuracy is higher than ever
        if best_top1_accuracy < top1_accuracy[-1]:
            best_top1_accuracy = top1_accuracy[-1]
            torch.save(
                model.state_dict(), os.path.join(CONFIG.result_path, 'best_top1_accuracy_model.prm'))

        if best_top5_accuracy < top5_accuracy[-1]:
            best_top5_accuracy = top5_accuracy[-1]
            torch.save(
                model.state_dict(), os.path.join(CONFIG.result_path, 'best_top5_accuracy_model.prm'))

        # save checkpoint every 5 epoch
        if epoch % 5 == 0 and epoch != 0:
            save_checkpoint(CONFIG, epoch, model, optimizer, scheduler)

        # save a model every 10 epoch
        if epoch % 10 == 0 and epoch != 0:
            torch.save(
                model.state_dict(), os.path.join(CONFIG.result_path, 'epoch_{}_model.prm'.format(epoch)))

        # tensorboardx
        if writer is not None:
            writer.add_scalar("loss_train", losses_train[-1], epoch)

            writer.add_scalar('loss_val', losses_val[-1], epoch)
            writer.add_scalars("iou", {
                'top1_accuracy': top1_accuracy[-1],
                'top5_accuracy': top5_accuracy[-1]}, epoch)

        print(
            'epoch: {}\tloss train: {:.5f}\tloss val: {:.5f}\ttop1_accuracy: {:.5f}\ttop5_accuracy: {:.5f}'
            .format(epoch, losses_train[-1], losses_val[-1], top1_accuracy[-1], top5_accuracy[-1])
        )

    torch.save(
        model.state_dict(), os.path.join(CONFIG.result_path, 'final_model.prm'))


if __name__ == '__main__':
    main()
