#!/usr/bin/env python
"""Run a trained DeepETPicker checkpoint over the datalist's full `validation` / `test` splits.

The CZI datalist's validation (Dataset445, 121) and test (Dataset446, 364) entries are whole
.nii.gz tomograms with point labels -- the same entries CryoDINO's detection3d.py evaluates on.
This script converts them the way the deepict/membrain kits do (datalist path used verbatim, NIfTI
-> MRC), but in chunks, and runs DeepETPicker's own inference (bin/test_bash.py) on each chunk:

  for each chunk of tomograms:
    <work>/data_std/<name>.mrc   <- nib.get_fdata() (X,Y,Z), DeepETPicker standardisation,
                                    transposed to MRC (z,y,x) exactly like prepare_from_datalist.py
    bin/test_bash.py             <- picks every tomogram in <work>/data_std
    <out>/predictions/<ckpt name>/<split>/<name>.coords   <- moved out of the checkpoint's
                                                             PredictedLabels/ (one folder per
                                                             checkpoint; _checkpoint.txt guards it)

Why not put them in raw_data/ and let preprocess.py handle them: preprocess.py normalises with
Pool(len(tomograms)) -- one process per tomogram, ~290 MB each, so 485 tomograms at once -- and
also builds label/occupancy volumes that inference does not need. Writing data_std/ directly with
the same formula (utils/normalization.py: x -= mean; x /= std, float32) avoids both.

Resumable: a tomogram whose prediction already exists in this checkpoint's folder is skipped
(unless --force),
and that file only appears after test_bash.py succeeded on its chunk.

Timing: every run prints per-tomogram NIfTI read / standardise+write / DeepETPicker inference
times and extrapolates them to full validation/test and to scoring 20 checkpoints, with and
without caching the converted tomograms (<pred-root>/_timing.json). --profile additionally runs
test_bash.py under cProfile to split inference into network+peak finding vs duplicate removal.

Prediction rows are DeepETPicker's Coords_withArea format:  cls+1  x  y  z  score
with (x, y, z) = datalist (p0, p1, p2) and `score` = the mean-pooled density of the thresholded
class mask at the peak (test.py _nms_v2's h_val, in (0.1, 1]); evaluate_czi.py uses it as the
per-pick confidence for CZIDetectionMetrics' threshold sweep.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from datalist_utils import (check_points_in_volume, load_datalist, load_nifti_xyz, resolve_path,
                            save_mrc, standardize, strip_ext)

PAD_PREFIX = "_pad_"


def ckpt_tag(ckpt):
    """version_N_<checkpoint name>: unique across checkpoints and across re-trainings in one out dir.
    run_benchmark.sh and select_checkpoint.py build the same name."""
    ver = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(ckpt))))
    return f"{ver}_{os.path.splitext(os.path.basename(ckpt))[0]}"


def read_cfg(path):
    with open(path) as f:
        return json.loads("".join(f.readlines()).lstrip("train_configs="))


def write_mrc_for(entry, name, data_std, nii_root):
    path = resolve_path(entry["image"], nii_root)
    if not os.path.exists(path):
        raise SystemExit(f"{name}: tomogram not found: {path} (set --nii-root?)")
    t0 = time.time()
    vol = load_nifti_xyz(path)                                   # (X, Y, Z)
    t1 = time.time()
    n_gt = check_points_in_volume(entry.get("points", []), vol.shape, name)
    mrc = os.path.join(data_std, name + ".mrc")
    save_mrc(standardize(vol).transpose(2, 1, 0), mrc)
    t2 = time.time()
    return vol.shape, n_gt, {"read_nii": t1 - t0, "std_write_mrc": t2 - t1,
                             "mrc_bytes": os.path.getsize(mrc)}


# DeepETPicker functions whose cumulative time --profile reports. test_func = all of inference
# after imports; test_step = network + peak finding per batch (GPU work is waited on inside
# _nms_v2's .nonzero()); _nms_v2 = peak finding; de_dup = duplicate removal (pure Python).
PROFILE_FUNCS = ("test_func", "test_step", "_nms_v2", "de_dup")


def profile_summary(prof_path):
    """Cumulative seconds per PROFILE_FUNCS entry from a cProfile dump of test_bash.py."""
    import pstats
    st = pstats.Stats(prof_path)
    out = {f: 0.0 for f in PROFILE_FUNCS}
    for (fname, _, func), (_, _, _, ct, _) in st.stats.items():
        if func in out and fname.endswith(("test.py", "misc.py")):
            out[func] += ct
    out["process_total"] = st.total_tt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", required=True)
    ap.add_argument("--out", required=True, help="the training run's base_path")
    ap.add_argument("--train-config", required=True, help="<out>/<DSET>_train.py")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dep", required=True, help="DeepETPicker checkout")
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--chunk", type=int, default=16, help="tomograms per test_bash.py call")
    ap.add_argument("--limit", type=int, default=None, help="only the first N of each split (smoke)")
    ap.add_argument("--nii-root", default=None,
                    help="fallback dir for entries whose datalist path does not exist")
    ap.add_argument("--force", action="store_true", help="re-run tomograms already predicted")
    ap.add_argument("--pred-root", default=None,
                    help="default: <out>/predictions/<checkpoint name> (one folder per checkpoint)")
    ap.add_argument("--profile", action="store_true",
                    help="run test_bash.py under cProfile and report where inference time goes")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    ckpt = os.path.abspath(args.ckpt)
    d = load_datalist(args.datalist)
    pred_root = os.path.abspath(args.pred_root or os.path.join(out, "predictions", ckpt_tag(ckpt)))
    os.makedirs(pred_root, exist_ok=True)
    # Resume must never mix checkpoints: the folder records which checkpoint filled it.
    stamp = os.path.join(pred_root, "_checkpoint.txt")
    if os.path.exists(stamp) and open(stamp).read().strip() != ckpt:
        raise SystemExit(f"{pred_root} holds predictions from {open(stamp).read().strip()}, "
                         f"not {ckpt} -- use another --pred-root")
    with open(stamp, "w") as f:
        f.write(ckpt + "\n")
    timing, timing_chunks, profs = [], [], []

    # Private work dir per checkpoint, so several checkpoints can be scored in parallel jobs.
    work = os.path.join(out, "infer_work", os.path.basename(pred_root))
    data_std = os.path.join(work, "data_std")
    os.makedirs(data_std, exist_ok=True)
    cfg = read_cfg(args.train_config)
    cfg.update({"base_path": work, "tomo_path": data_std})
    cfg_path = os.path.join(work, "infer_config.py")
    with open(cfg_path, "w") as f:
        f.write("train_configs=")
        json.dump(cfg, f, indent=2)
    # test.py writes to <ckpt>/../../PredictedLabels/, so checkpoints of one run would share that
    # folder. Give each its own by pointing test.py at a symlink in the private work dir.
    link = os.path.join(work, "ckpt", "run", "checkpoints", os.path.basename(ckpt))
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(ckpt, link)
    ckpt_pred = os.path.join(work, "ckpt", "run", "PredictedLabels")

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        entries = d[split][: args.limit] if args.limit else d[split]
        if not entries:
            raise SystemExit(f"datalist split '{split}' is empty")
        names = [strip_ext(e["image"]) for e in entries]
        if len(set(names)) != len(names):
            raise SystemExit(f"'{split}' has duplicate tomogram basenames")
        pred_dir = os.path.join(pred_root, split)
        os.makedirs(pred_dir, exist_ok=True)
        todo = [(e, n) for e, n in zip(entries, names)
                if args.force or not os.path.exists(os.path.join(pred_dir, n + ".coords"))]
        print(f"=== {split}: {len(entries)} tomograms, {len(entries) - len(todo)} already "
              f"predicted, {len(todo)} to do (chunks of {args.chunk})", flush=True)

        for c0 in range(0, len(todo), args.chunk):
            chunk = todo[c0:c0 + args.chunk]
            t0 = time.time()
            for fn in os.listdir(data_std):                      # start each chunk clean
                os.remove(os.path.join(data_std, fn))
            for e, n in chunk:
                shape, n_gt, tm = write_mrc_for(e, n, data_std, args.nii_root)
                timing.append(tm)
                print(f"  {split} {n}: vol(X,Y,Z)={tuple(shape)} gt={n_gt} "
                      f"voxel_size={e.get('voxel_size')}  read_nii={tm['read_nii']:.1f}s "
                      f"std+write_mrc={tm['std_write_mrc']:.1f}s", flush=True)
            t_conv = time.time() - t0
            run_names = [n for _, n in chunk]
            if len(chunk) == 1:
                # bin/test_bash.py reads data_std/num_name.csv with np.loadtxt, which returns a
                # 1-D array for a single row, so len() == 2 and it indexes a missing tomogram.
                pad = PAD_PREFIX + run_names[0]
                shutil.copy(os.path.join(data_std, run_names[0] + ".mrc"),
                            os.path.join(data_std, pad + ".mrc"))
                run_names.append(pad)
            for sub in ("Coords_withArea", "Coords_All"):            # no stale picks
                for n in run_names:
                    p = os.path.join(ckpt_pred, sub, n + ".coords")
                    if os.path.exists(p):
                        os.remove(p)

            prof = os.path.join(work, f"profile_{len(profs)}.out")
            cmd = ([sys.executable, "-u"] + (["-m", "cProfile", "-o", prof] if args.profile else [])
                   + [os.path.join(args.dep, "bin", "test_bash.py"),
                      "--train_configs", cfg_path, "--checkpoints", link])
            t1 = time.time()
            r = subprocess.run(cmd, cwd=work)    # test.py writes result/ relative to cwd
            t_test = time.time() - t1
            if r.returncode != 0:
                raise SystemExit(f"test_bash.py failed (exit {r.returncode}) on chunk "
                                 f"{[n for _, n in chunk]}")
            if args.profile:
                ps = profile_summary(prof)
                ps.update({"n": len(run_names), "wall": t_test})
                profs.append(ps)
                print("  profile: " + "  ".join(f"{k}={v:.1f}s" for k, v in ps.items()
                                                if k != "n"), flush=True)
            print(f"  timing: convert {t_conv:.0f}s, test_bash.py {t_test:.0f}s for "
                  f"{len(run_names)} tomograms", flush=True)
            timing_chunks.append({"n": len(run_names), "convert": t_conv, "test": t_test})

            for n in run_names:
                src = os.path.join(ckpt_pred, "Coords_withArea", n + ".coords")
                if not os.path.exists(src):
                    raise SystemExit(f"test_bash.py wrote no prediction for {n}: {src}")
                if n.startswith(PAD_PREFIX):
                    os.remove(src)
                    continue
                shutil.move(src, os.path.join(pred_dir, n + ".coords"))
            done = min(c0 + args.chunk, len(todo))
            print(f"  chunk done: {done}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)

    for fn in os.listdir(data_std):
        os.remove(os.path.join(data_std, fn))
    print(f"predictions -> {pred_root}")
    if timing:
        summarize(timing, timing_chunks, profs, len(d["validation"]), len(d["test"]),
                  os.path.join(pred_root, "_timing.json"))


def summarize(timing, chunks, profs, n_val, n_test, path, n_ckpt=20):
    """Per-tomogram means and what they imply for scoring n_ckpt checkpoints on validation."""
    n = len(timing)
    read = sum(t["read_nii"] for t in timing) / n
    wr = sum(t["std_write_mrc"] for t in timing) / n
    gb = sum(t["mrc_bytes"] for t in timing) / n / 1e9
    test = sum(c["test"] for c in chunks) / sum(c["n"] for c in chunks)  # incl. pad copies
    conv = read + wr
    h = 3600.0
    s = {"tomograms_measured": n, "read_nii_s": read, "std_write_mrc_s": wr, "convert_s": conv,
         "test_bash_s_per_tomogram": test, "mrc_gb_per_tomogram": gb,
         "val_cache_gb": n_val * gb,
         "one_val_scoring_h": n_val * (conv + test) / h,
         "test_split_h": n_test * (conv + test) / h,
         f"{n_ckpt}_ckpts_convert_each_time_h": n_ckpt * n_val * (conv + test) / h,
         f"{n_ckpt}_ckpts_convert_once_h": n_val * (conv + n_ckpt * test) / h}
    print("\n=== timing summary (per tomogram, mean over "
          f"{n}; test_bash.py time includes its ~start-up per chunk)")
    print(f"  NIfTI read          {read:6.1f} s")
    print(f"  standardise + .mrc  {wr:6.1f} s     ({gb:.2f} GB per .mrc)")
    print(f"  DeepETPicker infer  {test:6.1f} s")
    if profs:
        tot = {k: sum(p[k] for p in profs) for k in PROFILE_FUNCS}
        m = sum(p["n"] for p in profs)
        s["profile_s_per_tomogram"] = {k: v / m for k, v in tot.items()}
        print("  of which (cProfile, inflates pure-Python loops):")
        print(f"    network + peak finding (test_step)  {tot['test_step'] / m:6.1f} s")
        print(f"    duplicate removal (de_dup)          {tot['de_dup'] / m:6.1f} s")
        print(f"    rest of test_func (load/pad tomo, Trainer, model load, write)  "
              f"{(tot['test_func'] - tot['test_step'] - tot['de_dup']) / m:6.1f} s")
    print(f"\n  one scoring of all {n_val} validation tomograms: {s['one_val_scoring_h']:.2f} h")
    print(f"  test split ({n_test} tomograms), once:            {s['test_split_h']:.2f} h")
    print(f"  {n_ckpt} checkpoints x {n_val} val, convert every time: "
          f"{s[f'{n_ckpt}_ckpts_convert_each_time_h']:.1f} GPU-h")
    print(f"  {n_ckpt} checkpoints x {n_val} val, convert once:       "
          f"{s[f'{n_ckpt}_ckpts_convert_once_h']:.1f} GPU-h   (needs {s['val_cache_gb']:.0f} GB cache)")
    with open(path, "w") as f:
        json.dump(s, f, indent=2)
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
