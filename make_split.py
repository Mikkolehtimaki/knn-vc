import argparse
import os

import pandas as pd

def main(prematch_file: str, out_dir: str, train_portion: float=0.9):

    df = pd.read_csv(prematch_file)
    # Rename columns to hifigan format
    df = df.rename(columns={'path': 'audio_path', 'targ_path': 'feat_path'})

    # Sample without replacement to get train and validation sets
    train_df = df.sample(frac=train_portion, replace=False)
    val_df = df.drop(train_df.index)

    # Store in out_dir
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, 'hifigan_train.csv'), index=False)
    val_df.to_csv(os.path.join(out_dir, 'hifigan_val.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split data to train and validation")
    parser.add_argument('-p', '--prematch_file', required=True, type=str)
    parser.add_argument('-o', '--out_dir', required=True, type=str)
    parser.add_argument('-s', '--split_train_portion', default=0.9, type=float)

    args = parser.parse_args()
    main(args.prematch_file, args.out_dir, args.split_train_portion)
