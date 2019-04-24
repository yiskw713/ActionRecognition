import glob
import pandas as pd
import argparse


def get_arguments():
    """
    Parse all the arguments from Command Line Interface.
    Return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description='make csv files for training and test')
    parser.add_argument(
        'dataset_dir', type=str, help='path of the dataset directory')
    parser.add_argument(
        '--div', type=bool, default=False, help='whether train dir and test dir are divided or not.')
    parser.add_argument(
        '--csv_path', type=str, default='./', help='pash where you want to save csv files')

    return parser.parse_args()


# TODO: programm when args.div == True.

args = get_arguments()

dataset_dir = args.dataset_dir
class_dir = glob.glob(dataset_dir + '/*')

id_to_cls = {}
cls_to_id = {}

for i, cls in enumerate(class_dir):
    id_to_cls[i] = cls[18:]

for key, val in id_to_cls.items():
    cls_to_id[val] = key


video_list = []
class_list = []
id_list = []
n_frames_list = []

for key, val in id_to_cls.items():
    # key => id
    # val => class

    video = glob.glob(dataset_dir + '/' + val + '/*')
    video_list += video
    id_list += [key for i in range(len(video))]
    class_list += [id_to_cls[key] for i in range(len(video))]

    for v in video:
        with open(v + '/n_frames') as f:
            n_frames = int(f.read())
            n_frames_list.append(n_frames)


video_list_train = []
class_list_train = []
id_list_train = []
n_frames_list_train = []
video_list_test = []
class_list_test = []
id_list_test = []
n_frames_list_test = []

for i, (m, c, idx, f) in enumerate(zip(video_list, class_list, id_list, n_frames_list)):
    if i % 5 != 0:
        video_list_train.append(m)
        class_list_train.append(c)
        id_list_train.append(idx)
        n_frames_list_train.append(f)
    else:
        video_list_test.append(m)
        class_list_test.append(c)
        id_list_test.append(idx)
        n_frames_list_test.append(f)


df_train = pd.DataFrame({
    'video': video_list_train,
    'n_frames': n_frames_list_train,
    'class': class_list_train,
    'class_id': id_list_train},
    columns=['video', 'class', 'class_id', 'n_frames']
)


df_test = pd.DataFrame({
    'video': video_list_test,
    'n_frames': n_frames_list_test,
    'class': class_list_test,
    'class_id': id_list_test},
    columns=['video', 'class', 'class_id', 'n_frames']
)

df_train.to_csv(args.csv_path + 'train.csv', index=None)
df_test.to_csv(args.csv_path + 'test.csv', index=None)
