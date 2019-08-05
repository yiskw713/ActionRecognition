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
        'n_classes', type=int, default=400, help='the number of classes in kinetics dataset')
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

    class_label_map = get_class_label_map(n_classes=args.n_classes)

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
        if 'is_cc' in df.columns:
            del df['is_cc']

        for i in range(len(df)):
            video_dir = os.path.join(args.dataset_dir, df.iloc[i]['video'])

            # confirm if video directory and n_frames file exist or not.
            if os.path.exists(video_dir):
                n_f = glob.glob(os.path.join(video_dir, '*'))
                if len(n_f) > 128:
                    continue
                else:
                    # remove videos which have relatively few frames
                    df.drop(i)
            elif os.path.exists(video_dir + '.hdf5'):
                continue
            else:
                # remove videos which do not exist
                df.drop(i)

    df_train.to_csv(
        os.path.join(args.save_path, 'kinetics_{}_train.csv'.format(args.n_classes)), index=None)
    df_val.to_csv(
        os.path.join(args.save_path, 'kinetics_{}_val.csv'.format(args.n_classes)), index=None)


if __name__ == '__main__':
    main()
