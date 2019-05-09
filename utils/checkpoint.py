import torch
import os


def save_checkpoint(config, epoch, model, optimizer, scheduler):
    save_states = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    if scheduler is not None:
        save_states['scheduler'] = scheduler.state_dict()

    torch.save(save_states, os.path.join(config.result_path, 'checkpoint.pth'))


def resume(config, model, optimizer, scheduler=None):

    resume_path = os.path.join(config.result_path, 'checkpoint.pth')
    print('loading checkpoint {}'.format(resume_path))
    checkpoint = torch.load(resume_path)

    begin_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, model, optimizer, scheduler
