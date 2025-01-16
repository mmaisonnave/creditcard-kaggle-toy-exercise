import os
import pandas as pd
import sys
sys.path.append('..')

from src import utils
import numpy as np


def generate_crossval_heldout_splits()->None:
    config = utils.get_config()
    rng = np.random.default_rng(seed=321318855)
    df = get_data(split='full')

    crossval_index = rng.choice(
        list(range(df.shape[0])), 
        replace=False,
        size=int(df.shape[0]*.70),
        )

    # Checking results is reproducible (always the same crossval/heldout split)
    assert list(crossval_index[:5]) == [14651, 37148, 41879, 164653, 226022]
    assert list(crossval_index[-5:]) == [68927, 121272, 41861, 83205, 227074]

    crossval_index_set = set(crossval_index)
    heldout_index = [ix for ix in range(df.shape[0]) if not ix in crossval_index_set]
    rng.shuffle(heldout_index)

    # Checking results is reproducible (always the same crossval/heldout split)
    assert list(heldout_index[:5]) == [62294, 144854, 227581, 10009, 153726]
    assert list(heldout_index[-5:]) == [12463, 124228, 269675, 149515, 179145]

    crossval = df.iloc[crossval_index,:]
    heldout = df.iloc[heldout_index,:]

    utils.io.info(f'crossval.shape={crossval.shape}')
    utils.io.info(f'heldout.shape= {heldout.shape}')


    utils.io.info(f"Number of frauds {np.sum(crossval['Class'])}/{len(crossval['Class']):,} ({100 * (np.sum(crossval['Class']) / len(crossval['Class'])):.2f}%)")
    utils.io.info(f"Number of frauds {np.sum(heldout['Class'])}/{len(heldout['Class']):,} ({100 * (np.sum(heldout['Class']) / len(heldout['Class'])):.2f}%)")

    assert len(set(crossval_index).intersection(set(heldout_index))) == 0

    crossval.to_csv(os.path.join(
        config['repository_path'], 
        config['credit_card_crossval_split']
        ), index=False)
    heldout.to_csv(os.path.join(
        config['repository_path'], 
        config['credit_card_heldout_split']), index=False)

    utils.io.ok('Cross validation and heldout splits stored in disk...')


def get_data(split='full') -> pd.DataFrame :
    if split == 'full':
        split_name='credit_card_full'
    elif split=='crossval':
        split_name='credit_card_crossval_split'
    elif split=='heldout':
        split_name='credit_card_heldout_split'

    config = utils.get_config()
    return pd.read_csv(os.path.join(
        config['repository_path'], 
        config[split_name])
        )
