"""Rank MHSA heads by how well attention separates annotated voxels from the rest."""
import argparse, os
import numpy as np, nibabel as nib
import torch
import torch.nn.functional as F
from scipy.stats import rankdata


def fit_mask(mask, shape):
    if mask.shape == tuple(shape):
        return mask
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.interpolate(t, size=tuple(shape), mode='nearest')[0, 0].numpy() > 0


def load_ds(path, step=4):
    # attention is patch-wise (16 voxels), so striding by 4 loses nothing
    return np.asarray(nib.load(path).dataobj, dtype=np.float32)[::step, ::step, ::step].copy()

def auc(score, mask, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = score[mask], score[~mask]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    pos = rng.choice(pos, min(n, len(pos)), replace=False)
    neg = rng.choice(neg, min(n, len(neg)), replace=False)
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vis-base', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--item', action='append', required=True, help='name=label_path')
    a = p.parse_args()
    rows = []
    for it in a.item:
        name, lbl = it.split('=', 1)
        mask = load_ds(lbl) > 0
        d = os.path.join(a.vis_base, a.model, name, 'mhsa')
        scores, i = [], 0
        while os.path.exists(f'{d}/nifti_{i}.nii.gz'):
            h = load_ds(f'{d}/nifti_{i}.nii.gz')
            m = fit_mask(mask, h.shape)
            scores.append(auc(h, m))
            i += 1
        print(name, ' '.join(f'{s:.3f}' for s in scores), flush=True)
        rows.append(scores)
    m = np.nanmean(np.array(rows), axis=0)
    print('MEAN AUC:', ' '.join(f'H{i}={v:.3f}' for i, v in enumerate(m)))
    print('BEST (top 3):', np.argsort(-m)[:3].tolist(), '-> use --head-idx', int(np.nanargmax(m)))

if __name__ == '__main__':
    main()
