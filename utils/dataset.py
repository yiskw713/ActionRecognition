import torch
import numpy as np
import pandas as pd
import torchvision
import os
from torch.utils.data import Dataset
from PIL import Image


def pil_loader(path):
    return Image.open(path)


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except ModuleNotFoundError:
        # Potentially a decoding problem, fall back to PIL.Image
        torchvision.set_image_backend('PIL')
        return pil_loader(path)


def get_default_image_loader():
    torchvision.set_image_backend('accimage')

    return accimage_loader


def train_video_loader(loader, video_path, n_frames, input_frames=16, transform=None):
    """
    Return sequential 16 frames in video clips.
    A initial frame is randomly decided.
    Args:
        video_path: path for the video.
        n_frames: the number of frames of the video
        input_frames: the number of frames you want to input to the model. (default 16)
    """

    start_frame = np.random.randint(1, n_frames - input_frames + 1)
    clip = []
    for i in range(start_frame, start_frame + input_frames):
        img_path = os.path.join(video_path, 'image_{:05d}.jpg'.format(i))

        img = loader(img_path)

        if transform is not None:
            img = transform(img)

        clip.append(img)
    return clip


def test_video_loader(loader, video_path, n_frames, input_frames=16, transform=None):
    """
    Return 16 * (n_frames // 16) frames in video clips.
    Ignore the last frames which are indivisible by n_frames.
    Args:
        video_path: path for the video.
        n_frames: the number of frames of the video.
        input_frames: the number of frames you want to input to the model. (default 16)
    """

    n_frames = 16 * (n_frames // input_frames)
    clip = []
    for i in range(1, n_frames + 1):
        img_path = os.path.join(video_path, 'image_{:05d}.jpg'.format(i))
        img = loader(img_path)

        if transform is not None:
            img = transform(img)

        clip.append(img)
    return clip


class Kinetics(Dataset):
    """
    Dataset class for Kinetics
    """

    def __init__(self, config, transform=None, mode='training'):
        super().__init__()

        self.config = config

        if mode == 'validation':
            self.df = pd.read_csv(self.config.val_csv)
        elif mode == 'test':
            self.df = pd.read_csv(self.config.test_csv)
        else:
            self.df = pd.read_csv(self.config.train_csv)

        self.transform = transform
        self.mode = mode
        self.loader = get_default_image_loader()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_path = os.path.join(
            self.config.dataset_dir, self.df.iloc[idx]['video'])
        cls_id = torch.tensor(int(self.df.iloc[idx]['class_id'])).long()
        n_frames = int(self.df.iloc[idx]['n_frames'])

        if self.mode == 'test':
            clip = test_video_loader(
                self.loader, video_path, n_frames, self.config.input_frames, self.transform)
        else:
            clip = train_video_loader(
                self.loader, video_path, n_frames, self.config.input_frames, self.transform)

        # clip.shape => (C, T, H, W)
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        sample = {
            'clip': clip,
            'cls_id': cls_id,
        }

        return sample
