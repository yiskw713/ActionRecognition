import glob
import os
import pandas as pd
import argparse
from class_label_map import get_class_label_map


def get_arguments():
    """
    Parse all the arguments from Command Line Interface.
    Return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description='make csv files for training and validation')
    parser.add_argument(
        'dataset_dir', type=str, help='path of the dataset directory')
    parser.add_argument(
        '--orig_train_csv', type=str, default='./dataset/original/kinetics-400_train.csv', help='path to the original kinetics dataset train csv')
    parser.add_argument(
        '--orig_val_csv', type=str, default='./dataset/original/kinetics-400_val.csv', help='path to the original kinetics dataset val csv')
    parser.add_argument(
        '--save_path', type=str, default='./dataset', help='path where you want to save csv files')

    return parser.parse_args()


def main():
    args = get_arguments()

    df_train = pd.read_csv(args.orig_train_csv)
    df_val = pd.read_csv(args.orig_val_csv)

    class_label_map = get_class_label_map()

    for df in [df_train, df_val]:
        path = []
        cls_id = []

        for i in range(len(df)):
            path.append(
                df.iloc[i]['label'] + '/' + df.iloc[i]['youtube_id'] + '_' + str(df.iloc[i]['time_start']).zfill(6) + '_' + str(df.iloc[i]['time_end']).zfill(6))
            cls_id.append(class_label_map[df.iloc[i]['label']])

        df['class_id'] = cls_id
        df['video'] = path

        # delete useless columns
        del df['youtube_id']
        del df['time_start']
        del df['time_end']
        del df['split']
        del df['is_cc']

        frames = []
        for i in range(len(df)):
            video_dir = os.path.join(args.dataset_dir, df.iloc[i]['video'])

            # confirm if video directory and n_frames file exist or not.
            if os.path.exists(video_dir):
                videos = glob.glob(os.path.join(video_dir, '*.jpg'))
                frames.append(len(videos))
            else:
                # Videos which have few or no frames will be removed afterwards
                frames.append(0)

        df['n_frames'] = frames

    # remove videos which have only few frames or no frames
    df_train = df_train[df_train['n_frames'] >= 150]
    df_val = df_val[df_val['n_frames'] >= 150]

    df_train.to_csv(
        os.path.join(args.save_path, 'kinetics_400_train.csv'), index=None)
    df_val.to_csv(
        os.path.join(args.save_path, 'kinetics_400_val.csv'), index=None)


if __name__ == '__main__':
    main()
