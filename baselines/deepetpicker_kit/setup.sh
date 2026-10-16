#!/bin/bash
# One-time setup: build the venv and patch the DeepETPicker checkout so its
# pytorch_lightning==1.1.0-era code runs on the stack this cluster actually has
# (python 3.11 / torch 2.13 / PL 2.6 / numpy 2).
#
# Idempotent: the patches are applied to a pristine checkout (`git checkout -- .` first), so
# re-running gives the same result rather than layering edits.  Any hand edits you made inside
# $DEP will be discarded -- keep your changes here in the kit instead.
#
# Usage:  [DEP=/scratch/cyyu/cryodino/DeepETPicker VENV=/scratch/cyyu/venvs/deepetpicker] bash setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DEP:=/scratch/cyyu/cryodino/DeepETPicker}"
: "${VENV:=/scratch/cyyu/venvs/deepetpicker}"

[ -d "$DEP" ] || git clone https://github.com/cbmi-group/DeepETPicker "$DEP"

# ---------------------------------------------------------------- 1) venv
module load python/3.11 2>/dev/null || true
[ -f "$VENV/bin/activate" ] || python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip

# torch 2.13.0 is currently the ONLY wheelhouse build with GPU kernels for this cluster's H100s:
#   torch 2.13.0 -> get_arch_list() == ['sm_80','sm_90','sm_100','compute_100']
#   torch 2.9.1 / 2.5.1 / 2.1.2 -> get_arch_list() == []  (they fail at runtime with
#   "CUDA error: no kernel image is available for execution on the device")
# That forces pytorch_lightning 2.x, hence the hook ports in patches 4 and 6 below.
pip install --no-index \
    'torch==2.13.0' 'torchvision==0.28.0' \
    'pytorch_lightning==2.6.5' \
    numpy pandas mrcfile scikit_image scikit_learn scipy matplotlib tensorboard \
    'batchgenerators==0.25.1'

# PyQt5 and opencv are deliberately NOT installed: they are only needed by the GUI
# (MainWindow_3D_PP.py, main.py, utils/utils.py) and by utils/colors.py's plot_legend(), none
# of which the bin/ command-line path touches.  On Alliance clusters opencv is a module, not a
# pip wheel, so avoiding it keeps the venv self-contained.  Patch 8 makes that import optional.
# pycm + scikit-plot are likewise skipped -- they are imported by utils/misc.py only for the
# built-in SHREC-style scorer, which the CLI never reaches (see README); patch 7 makes those
# imports optional.  This kit scores with its own evaluate.py.

# ---------------------------------------------------------------- 2) source patches
git -C "$DEP" checkout -- . 2>/dev/null || true   # start from pristine sources
python - "$DEP" <<'PY'
import pathlib, re, sys

dep = pathlib.Path(sys.argv[1])
changed = []

def edit(rel, fn, strict=True):
    """Apply fn to a file.  strict=True means a no-op is a bug (upstream text moved)."""
    f = dep / rel
    t = f.read_text()
    t2 = fn(t)
    if t2 == t:
        if strict:
            raise SystemExit(f"patch for {rel} matched nothing -- upstream changed?")
        return
    f.write_text(t2)
    changed.append(rel)

# 1) numpy>=1.24 removed np.float / np.int.
#    np.float MUST become np.float32, not float: several of these arrays are handed straight to
#    mrcfile, which rejects float64 ("dtype 'float64' cannot be converted to an MRC file mode").
#    Every site sits in a try/except whose fallback was already np.float32, so float32 is the
#    behaviour upstream actually gets on its own pinned numpy==1.24.
#    \b keeps np.float32 / np.int64 untouched.
for rel in ("train.py", "test.py", "dataset/dataloader_DynamicLoad.py",
            "utils/misc.py", "utils/coords2labels.py", "utils/normalization.py",
            "utils/coord_gen.py"):
    edit(rel, lambda t: re.sub(r"np\.int\b", "int", re.sub(r"np\.float\b", "np.float32", t)),
         strict=False)   # not every file has one

# 2) train.py Trainer: PL>=1.5 dropped checkpoint_callback=, >=2.0 dropped gpus=, the 'dp'
#    strategy and resume_from_checkpoint=; profiler=True is no longer a valid value.
OLD_TRAINER = """    runner = Trainer(min_epochs=min(50, args.max_epoch),
                     max_epochs=args.max_epoch,
                     logger=tb_logger,
                     gpus=args.gpu_id,
                     checkpoint_callback=checkpoint_callback,
                     callbacks=[lr_monitor],
                     accelerator='dp',
                     precision=32,
                     profiler=True,
                     sync_batchnorm=True,
                     resume_from_checkpoint=args.checkpoints)"""
NEW_TRAINER = """    multi_gpu = len(args.gpu_id) > 1
    runner = Trainer(min_epochs=min(50, args.max_epoch),
                     max_epochs=args.max_epoch,
                     logger=tb_logger,
                     accelerator='gpu',
                     devices=args.gpu_id,
                     strategy='ddp' if multi_gpu else 'auto',
                     callbacks=[lr_monitor, checkpoint_callback],
                     precision=32,
                     profiler='simple',
                     sync_batchnorm=multi_gpu)"""
edit("train.py", lambda t: t.replace(OLD_TRAINER, NEW_TRAINER))

# 3) resume_from_checkpoint moved onto fit() in PL 2.0.
edit("train.py", lambda t: t.replace("        runner.fit(model)",
                                     "        runner.fit(model, ckpt_path=args.checkpoints)"))

# 4) PL 2.0 REMOVED validation_epoch_end and raises if the module still defines it; step outputs
#    must be accumulated by hand.  (validation_step_end is merely never called -- left alone.)
edit("train.py", lambda t: t.replace(
    """        self.partical_volume = 4 / 3 * np.pi * (self.val_cfg["label_diameter"] / 2) ** 3
        self.args = args""",
    """        self.partical_volume = 4 / 3 * np.pi * (self.val_cfg["label_diameter"] / 2) ** 3
        self.args = args
        self.validation_step_outputs = []"""))
edit("train.py", lambda t: t.replace(
    """            if args.num_classes > 1:
                return self._nms_v2(self.seg_output[:, 1:], kernel=args.meanPool_kernel, mp_num=6, positions=index)
            else:
                return self._nms_v2(self.seg_output[:, :], kernel=args.meanPool_kernel, mp_num=6, positions=index)""",
    """            if args.num_classes > 1:
                out = self._nms_v2(self.seg_output[:, 1:], kernel=args.meanPool_kernel, mp_num=6, positions=index)
            else:
                out = self._nms_v2(self.seg_output[:, :], kernel=args.meanPool_kernel, mp_num=6, positions=index)
            self.validation_step_outputs.append(out)
            return out"""))
edit("train.py", lambda t: t.replace(
    """    def validation_epoch_end(self, epoch_output):
        args = self.args""",
    """    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        args = self.args
        epoch_output = self.validation_step_outputs"""))

# 5) test.py: the checkpoint was written by UNetExperiment, which calls save_hyperparameters(),
#    so load_from_checkpoint would forward `args` into UNetTest.__init__(self) -- which takes
#    none.  Load the state dict explicitly instead.  Trainer gets the same PL 2.x treatment.
edit("test.py", lambda t: t.replace(
    "                model = UNetTest.load_from_checkpoint(args.checkpoints)",
    "                model = UNetTest()\n"
    "                model.load_state_dict(torch.load(args.checkpoints,\n"
    "                                                 map_location='cpu',\n"
    "                                                 weights_only=False)['state_dict'])"))
edit("test.py", lambda t: t.replace(
    """                runner = Trainer(gpus=args.gpu_id, #
                                 accelerator='dp'
                                 )""",
    """                runner = Trainer(accelerator='gpu',
                                 devices=1,
                                 logger=False)"""))

# 6) same epoch-end port for test.py.
edit("test.py", lambda t: t.replace(
    """                        self.num_classes = args.num_classes""",
    """                        self.num_classes = args.num_classes
                        self.test_step_outputs = []"""))
edit("test.py", lambda t: t.replace(
    """                                if args.num_classes > 1:
                                    return self._nms_v2(seg_output[:, 1:], kernel=args.meanPool_kernel,
                                                        mp_num=mp_num, positions=index)
                                else:
                                    return self._nms_v2(seg_output[:, :], kernel=args.meanPool_kernel,
                                                        mp_num=mp_num, positions=index)""",
    """                                if args.num_classes > 1:
                                    out = self._nms_v2(seg_output[:, 1:], kernel=args.meanPool_kernel,
                                                       mp_num=mp_num, positions=index)
                                else:
                                    out = self._nms_v2(seg_output[:, :], kernel=args.meanPool_kernel,
                                                       mp_num=mp_num, positions=index)
                                self.test_step_outputs.append(out)
                                return out"""))
edit("test.py", lambda t: t.replace(
    """                    def test_epoch_end(self, epoch_output):
                        with torch.no_grad():""",
    """                    def on_test_epoch_start(self):
                        self.test_step_outputs.clear()

                    def on_test_epoch_end(self):
                        epoch_output = self.test_step_outputs
                        with torch.no_grad():"""))

# 7) utils/misc.py imports pycm/scikitplot at module scope for the built-in SHREC scorer.
#    That scorer is unreachable from bin/ (train_bash uses test_mode='val', test_bash uses
#    'test_only'), so make the imports optional rather than pinning two fragile packages.
edit("utils/misc.py", lambda t: t.replace(
    """from pycm import ConfusionMatrix
from pycm.pycm_output import table_print, stat_print
from pycm.pycm_param import SUMMARY_CLASS, SUMMARY_OVERALL
import scikitplot as skplt""",
    """try:  # optional: only the built-in SHREC-style scorer needs these
    from pycm import ConfusionMatrix
    from pycm.pycm_output import table_print, stat_print
    from pycm.pycm_param import SUMMARY_CLASS, SUMMARY_OVERALL
    import scikitplot as skplt
except ImportError:  # pragma: no cover
    ConfusionMatrix = table_print = stat_print = skplt = None
    SUMMARY_CLASS = SUMMARY_OVERALL = None"""))

# 8) utils/colors.py imports cv2 at module scope (train.py pulls COLORS from it) but only uses
#    it inside plot_legend(), which nothing in bin/ calls.  On Alliance clusters opencv is a
#    module rather than a pip wheel, so keep it out of the venv.
edit("utils/colors.py", lambda t: t.replace(
    "import cv2\n",
    "try:  # optional: only plot_legend() needs it\n    import cv2\n"
    "except ImportError:  # pragma: no cover\n    cv2 = None\n", 1))

# 10) model/residual_unet_att.py's residual blocks do `out += residual`, where `out` is the
#    output of a ReLU inside SingleConv.  Autograd needs that ReLU output intact for its backward,
#    so on modern torch the first backward pass dies with "one of the variables needed for gradient
#    computation has been modified by an inplace operation ... output 0 of Relu, is at version 1".
#    Out-of-place add.  (All three ExtResNetBlock variants share the line; only one is reachable
#    with our args, but they are identical bugs.)
edit("model/residual_unet_att.py",
     lambda t: t.replace("        out += residual", "        out = out + residual"))

# 9) train.py swallows every training exception (`except:` that only reports when the GUI's
#    stdout proxy is set), so a CLI failure would surface as a missing checkpoint with no
#    traceback.  Print it.
edit("train.py", lambda t: t.replace(
    """    except:
        torch.cuda.empty_cache()
        if stdout is not None:""",
    """    except:
        torch.cuda.empty_cache()
        import traceback; traceback.print_exc()
        if stdout is not None:"""))

# 11) train.py builds OneCycleLR(total_steps=max_epoch, pct_start=0.1).  Its warmup phase ends at
#    pct_start*total_steps - 1, and when that is exactly 0 the phase has zero length and get_lr()
#    divides by zero.  With pct_start=0.1 that is precisely max_epoch == 10 -- 1 and 60 both work,
#    which is why neither the smoke nor the full run exposed it.  Nudge pct_start only in that
#    degenerate case so every other epoch count keeps upstream's schedule exactly.
OLD_SCHED = """        if args.scheduler == 'OneCycleLR':
            sched = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=args.learning_rate,
                                                        total_steps=args.max_epoch,
                                                        pct_start=0.1,"""
NEW_SCHED = """        if args.scheduler == 'OneCycleLR':
            pct_start = 0.1
            if pct_start * args.max_epoch == 1.0:  # zero-length warmup -> ZeroDivisionError
                pct_start = 1.5 / args.max_epoch
            sched = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=args.learning_rate,
                                                        total_steps=args.max_epoch,
                                                        pct_start=pct_start,"""
edit("train.py", lambda t: t.replace(OLD_SCHED, NEW_SCHED))

print("patched:", ", ".join(sorted(set(changed))))
PY

# ---------------------------------------------------------------- 3) verify
echo "Verify:"
! grep -rn "np\.float\b" "$DEP/train.py" "$DEP/dataset/dataloader_DynamicLoad.py" \
        "$DEP/utils/coords2labels.py" >/dev/null && echo "  [ok] np.float -> np.float32"
grep -q "devices=args.gpu_id"        "$DEP/train.py" && echo "  [ok] train.py Trainer"
grep -q "ckpt_path=args.checkpoints" "$DEP/train.py" && echo "  [ok] train.py resume -> fit(ckpt_path=)"
grep -q "on_validation_epoch_end"    "$DEP/train.py" && echo "  [ok] train.py epoch-end hook"
! grep -q "def validation_epoch_end" "$DEP/train.py" && echo "  [ok] train.py old hook gone"
grep -q "model.load_state_dict"      "$DEP/test.py"  && echo "  [ok] test.py checkpoint load"
grep -q "on_test_epoch_end"          "$DEP/test.py"  && echo "  [ok] test.py epoch-end hook"
! grep -q "def test_epoch_end"       "$DEP/test.py"  && echo "  [ok] test.py old hook gone"
grep -q "except ImportError" "$DEP/utils/misc.py"    && echo "  [ok] misc.py optional pycm"
grep -q "except ImportError" "$DEP/utils/colors.py"  && echo "  [ok] colors.py optional cv2"
grep -q "traceback.print_exc" "$DEP/train.py"        && echo "  [ok] train.py traceback"
! grep -q "out += residual" "$DEP/model/residual_unet_att.py" && echo "  [ok] residual add out-of-place"
grep -q "zero-length warmup" "$DEP/train.py"          && echo "  [ok] OneCycleLR pct_start guard"
(cd "$DEP" && python -c "
import sys; sys.path.insert(0, '.')
import torch, pytorch_lightning as pl, numpy, train, test
from dataset.dataloader_DynamicLoad import Dataset_ClsBased
arch = torch.cuda.get_arch_list()
assert 'sm_90' in arch, f'torch has no H100 kernels: {arch}'
print(f'  [ok] torch {torch.__version__} {arch}')
print(f'  [ok] PL {pl.__version__}  numpy {numpy.__version__}  DeepETPicker imports')
")
echo "venv: $VENV"
echo "DeepETPicker: $DEP"
