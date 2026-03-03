import torch
import numpy as np
from monai.transforms import Transform, Randomizable
from deconv_utils import (
    AdhocSSNR,
    CorrectCTF,
)
# deconvolve(
#                 tomo=tomo,
#                 mrcout=out_path.replace(extension, '_deconv.nii.gz'),
#                 df=50000.0,
#                 ampcon=0.07,
#                 Cs=2.7,
#                 kV=300.0,
#                 apix=new_spacing[0],
#                 strength=1.0,
#                 falloff=1.0,
#                 hp_frac=0.02,
#                 skip_lowpass=True,
#             )
class RandDeconvolution3D(Transform, Randomizable):
    def __init__(
        self,
        prob=0.3,
        df=50000.0,
        ampcon=0.07,
        Cs=2.7,
        kV=300.0,
        strength=1.0,
        falloff=1.0,
        hp_frac=0.02,
        skip_lowpass=True,
        apix=1.0,
    ):
        self.prob = prob
        self.df = df
        self.ampcon = ampcon
        self.Cs = Cs
        self.kV = kV
        self.strength = strength
        self.falloff = falloff
        self.hp_frac = hp_frac
        self.skip_lowpass = skip_lowpass
        self.apix = apix

        self._do = False

    def randomize(self):
        self._do = self.R.rand() < self.prob

    def __call__(self, img):
        self.randomize()
        if not self._do:
            return img

        device = img.device
        dtype = img.dtype

        # img: (C, H, W, D) → (H, W, D)
        vol = img[0].detach().cpu().numpy()

        ssnr = AdhocSSNR(
            imsize=vol.shape,
            apix=self.apix,
            df=self.df,
            ampcon=self.ampcon,
            Cs=self.Cs,
            kV=self.kV,
            S=self.strength,
            F=self.falloff,
            hp_frac=self.hp_frac,
            lp=not self.skip_lowpass,
        )

        wiener_constant = 1.0 / ssnr

        deconv = CorrectCTF(
            vol,
            df1=self.df,
            ast=0.0,
            ampcon=self.ampcon,
            invert_contrast=False,
            Cs=self.Cs,
            kV=self.kV,
            apix=self.apix,
            phase_flip=False,
            ctf_multiply=False,
            wiener_filter=True,
            C=wiener_constant,
            return_ctf=False,
        )

        out = torch.from_numpy(deconv).to(device=device, dtype=dtype)
        return out.unsqueeze(0)

if __name__ == "__main__":
    import SimpleITK as sitk
    import os
    in_dir = "/home/sumin/Downloads/data_aug_3DINO/imgs"
    out_dir = "/home/sumin/Downloads/data_aug_3DINO/out"
    files = os.listdir(in_dir)
    for f in files:
        print(f'Processing {f}...')
        if not f.endswith(".nii.gz"):
            continue
        in_nii = os.path.join(in_dir, f)
        os.makedirs(out_dir, exist_ok=True)
        out_nii = os.path.join(out_dir, os.path.basename(in_nii))

        img = sitk.ReadImage(in_nii)
        spacing = img.GetSpacing()  # (X, Y, Z)
        arr = sitk.GetArrayFromImage(img)  # (D, H, W)

        arr = np.transpose(arr, (1, 2, 0))  # -> (H, W, D)
        x = torch.from_numpy(arr).unsqueeze(0).float()  # (1, H, W, D)


        t = RandDeconvolution3D(prob=1.0, apix=spacing[0])
        t.set_random_state(seed=0)

        y = t(x)

        out_arr = y[0].detach().cpu().numpy()
        out_arr = np.transpose(out_arr, (2, 0, 1))  # back to (D, H, W)

        out_img = sitk.GetImageFromArray(out_arr)
        out_img.CopyInformation(img)

        os.makedirs(os.path.dirname(out_nii), exist_ok=True)
        sitk.WriteImage(out_img, out_nii)
