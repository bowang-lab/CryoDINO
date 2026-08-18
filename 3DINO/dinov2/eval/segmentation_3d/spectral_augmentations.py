# Phase-preserving Fourier-domain intensity augmentations for cryo-ET fine-tuning.
#
# Both transforms multiply the magnitude of a crop's 3D FFT by a random, purely
# radial (isotropic), nonnegative gain field and leave the phase untouched, then
# inverse-transform back to a real volume. Phase carries spatial layout, so this
# only reshapes the spectral "texture" of the crop (contrast/SNR-profile-like
# changes) without moving structures around — labels are left untouched by the
# caller since these only touch the image key(s).
#
# #1 RandFourierRadialPerturbd  — a free-form random smooth radial gain curve.
# #2 RandCTFShapedSpectrald     — a gain curve shaped like a real (but nominal,
#                                  unmeasured) defocus CTF envelope, reusing the
#                                  CTF formula already vendored in
#                                  preprocessing/deconv_utils.py.

import os
import sys

import numpy as np
import torch
from monai.transforms import MapTransform

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from preprocessing.deconv_utils import CTF  # noqa: E402


def _radial_grid(shape, device):
    """Isotropic frequency radius for an rfftn output of a volume with `shape`,
    normalized to [0, 1] (1.0 = the largest Nyquist-corner radius)."""
    freqs = [torch.fft.fftfreq(s, device=device) for s in shape[:-1]]
    freqs.append(torch.fft.rfftfreq(shape[-1], device=device))
    grids = torch.meshgrid(*freqs, indexing="ij")
    r = torch.sqrt(sum(g * g for g in grids))
    return r / r.max().clamp_min(1e-8)


def _rms_match(vol_new, vol_orig):
    """Rescale vol_new around its mean so its std matches vol_orig's std."""
    mean_new = vol_new.mean()
    std_orig = vol_orig.std()
    std_new = vol_new.std().clamp_min(1e-8)
    return mean_new + (vol_new - mean_new) * (std_orig / std_new)


class RandFourierRadialPerturbd(MapTransform):
    """#1 — random smooth radial Fourier-magnitude perturbation (FACT-style,
    phase-preserving), with safeguards: DC pinned to gain 1, smooth curve in
    log-frequency/log-gain space, conservative default gain range, zero-mean
    log-gain normalization, real-guaranteed reconstruction via rfftn/irfftn,
    and RMS-matching after inverse transform.

    Args:
        keys: image key(s) only (never apply to label).
        prob: probability of applying the transform.
        gain_range: (low, high) multiplicative gain bounds, e.g. (0.7, 1.4).
        num_control_points: number of random control points (besides the
            pinned DC point) defining the smooth radial curve.
    """

    def __init__(self, keys, prob=1.0, gain_range=(0.7, 1.4), num_control_points=4,
                 allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.prob = prob
        self.gain_range = gain_range
        self.num_control_points = num_control_points

    def _make_gain(self, r):
        log_lo, log_hi = np.log(self.gain_range[0]), np.log(self.gain_range[1])
        eps = 1e-3
        cp_r = np.concatenate([[eps], np.geomspace(eps, 1.0, self.num_control_points)])
        cp_log_gain = np.concatenate([[0.0], np.random.uniform(log_lo, log_hi, self.num_control_points)])
        order = np.argsort(cp_r)
        cp_r, cp_log_gain = cp_r[order], cp_log_gain[order]

        r_np = r.detach().cpu().numpy()
        log_r = np.log(np.clip(r_np, eps, None))
        log_cp_r = np.log(cp_r)
        log_gain = np.interp(log_r, log_cp_r, cp_log_gain)

        # zero-mean-log-gain normalization (reshape spectral shape, not overall energy)
        log_gain = log_gain - log_gain.mean()
        gain = np.exp(log_gain).astype(np.float32)
        return torch.from_numpy(gain)

    def __call__(self, data):
        d = dict(data)
        if np.random.random() >= self.prob:
            return d
        for key in self.keys:
            img = d[key].float()          # (C, H, W, D)
            vol = img[0]
            F = torch.fft.rfftn(vol)
            r = _radial_grid(vol.shape, vol.device)
            gain = self._make_gain(r).to(vol.device)
            F_new = F * gain
            F_new[(0,) * vol.dim()] = F[(0,) * vol.dim()]   # explicit DC preservation
            vol_new = torch.fft.irfftn(F_new, s=vol.shape)
            vol_new = _rms_match(vol_new, vol)
            d[key] = vol_new[None].to(d[key].dtype)
        return d


class RandCTFShapedSpectrald(MapTransform):
    """#2 — CTF-shaped spectral augmentation (not real CTF simulation): reuses the
    physically-parameterized CTF envelope from `preprocessing/deconv_utils.py`
    (Mindell & Grigorieff formulation) with a randomly sampled, nominal
    (unmeasured) defocus, applied as an unsigned, floor-softened magnitude
    envelope — never the signed CTF, which would flip phase by pi.

    Args:
        keys: image key(s) only.
        prob: probability of applying the transform.
        df_range: (low, high) nominal defocus range in Angstroms to sample from.
        floor: minimum envelope value (avoids exact CTF zero-crossings erasing bands).
        apix: voxel size in Angstroms (should match the dataset's actual spacing).
        Cs: spherical aberration (mm). kV: acceleration voltage (kV).
    """

    def __init__(self, keys, prob=1.0, df_range=(10000.0, 60000.0), floor=0.25,
                 apix=13.4, Cs=2.7, kV=300.0, ampcon=0.07, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.prob = prob
        self.df_range = df_range
        self.floor = floor
        self.apix = apix
        self.Cs = Cs
        self.kV = kV
        self.ampcon = ampcon

    def _make_envelope(self, shape):
        df = np.random.uniform(self.df_range[0], self.df_range[1])
        ctf_im = CTF(imsize=shape, df1=df, df2=df, ast=0.0, ampcon=self.ampcon,
                     Cs=self.Cs, kV=self.kV, apix=self.apix, B=0.0, rfft=True)
        envelope = np.clip(np.abs(ctf_im), self.floor, 1.0).astype(np.float32)
        envelope[(0,) * len(shape)] = 1.0   # explicit DC preservation
        return torch.from_numpy(envelope)

    def __call__(self, data):
        d = dict(data)
        if np.random.random() >= self.prob:
            return d
        for key in self.keys:
            img = d[key].float()          # (C, H, W, D)
            vol = img[0]
            F = torch.fft.rfftn(vol)
            envelope = self._make_envelope(tuple(vol.shape)).to(vol.device)
            F_new = F * envelope
            vol_new = torch.fft.irfftn(F_new, s=vol.shape)
            vol_new = _rms_match(vol_new, vol)
            d[key] = vol_new[None].to(d[key].dtype)
        return d
