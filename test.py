import torch
import tqdm
from train import accuracy


def test(model, test_loader, criterion, config, device):
    model.eval()
    test_loss = 0.0
    n_samples = 0.0
    top1 = 0.0
    top5 = 0.0
    with torch.no_grad():
        for sample in tqdm.tqdm(test_loader, total=len(test_loader)):
            # temporal size is input_frames(default 16) * interger
            x = sample['clip']
            x = x.to(device)
            n, _, t, _, _ = x.shape

            t = sample['cls_id']
            t = t.to(device)

            h = torch.zeros(n, config.n_classes).float().to(device)
            loss = 0.0
            for i in range(t // 16):
                hh = model(x[:, :, 8 * i + 8 * (i + 1) - 1, :, :])
                h += hh
                loss += criterion(hh, t).item()

            test_loss += loss / (t // 16)
            n, topk = accuracy(h, t, topk=(1, 5))
            n_samples += n
            top1 += topk[0]
            top5 += topk[1]

        test_loss /= len(test_loader)
        top1 /= n_samples
        top5 /= n_samples

    return test_loss, top1, top5
