# Construction smoke tests for the detection heads.
#
# Both bugs that made this code path unrunnable were CONSTRUCTION failures that any test
# which merely instantiated the head would have caught:
#
#   1. detection3d.py passed deep_supervision= to ViTAdapterUNETRHead, whose __init__ takes
#      five params and no **kwargs  ->  TypeError.
#   2. ViTAdapterUNETRHead built ViTAdapter without pretrain_size, so it defaulted to 112
#      (a 7^3=343 pos_embed grid) against a 128-crop checkpoint (8^3=512 tokens)
#      ->  RuntimeError: shape '[1,7,7,7,-1]' is invalid for input of size 524288.
#
# Bug 2 had already been fixed for the segmentation twin (segmentation_heads.py, commit
# ef2cdfd) and simply never propagated here, which is exactly the kind of drift a shared
# test catches.
#
# These tests are deliberately CPU-only and build no ViT: they check the wiring, not the math.
#
# Run: pytest dinov2/eval/detection_3d/test_heads.py

import inspect

import pytest
import torch

from dinov2.eval.detection_3d import detection_heads
from dinov2.eval.detection_3d.detection_heads import _pretrain_size_from


class _FakeViT:
    """Just enough of a ViT for _pretrain_size_from: a pos_embed and a patch size."""

    def __init__(self, n_tokens, patch=16, dim=8):
        self.pos_embed = torch.zeros(1, n_tokens, dim)
        self.patch_embed = type("PE", (), {"patch_size": (patch, patch, patch)})()


# ---------------------------------------------------------------------------
# bug 1 — the constructor contract
# ---------------------------------------------------------------------------

def test_vitadapter_head_signature_has_no_deep_supervision():
    """ViTAdapterUNETRHead dropped deep supervision (decoder1 and the out_ds* heads are gone).

    If someone re-adds a deep_supervision parameter here, the removed auxiliary outputs must
    come back too — otherwise the flag silently does nothing.
    """
    params = inspect.signature(detection_heads.ViTAdapterUNETRHead.__init__).parameters
    assert "deep_supervision" not in params
    assert list(params) == [
        "self", "feature_model", "input_channels", "image_size", "num_classes", "autocast_ctx",
    ]


def test_vitadapter_head_rejects_unexpected_kwargs():
    """The call site used to pass deep_supervision=; make that failure explicit and intentional."""
    assert "kwargs" not in inspect.signature(detection_heads.ViTAdapterUNETRHead.__init__).parameters
    with pytest.raises(TypeError):
        detection_heads.ViTAdapterUNETRHead(
            None, 1, 128, 6, None, deep_supervision=True,
        )


def test_unetr_head_still_accepts_deep_supervision():
    """The sibling head legitimately keeps the kwarg — that asymmetry is what caused the bug."""
    assert "deep_supervision" in inspect.signature(detection_heads.UNETRHead.__init__).parameters


def test_detection3d_call_sites_match_head_signatures():
    """The real bug was at the CALL SITE, not in the head, so check the callers too.

    Binds every `<Head>(...)` call in detection3d.py against that head's actual signature.
    This is what would have caught `deep_supervision=` being passed to a head that does not
    take it — a TypeError that only surfaces when someone runs that one --segmentation-head.
    """
    import ast
    import pathlib

    src = pathlib.Path(detection_heads.__file__).parent.parent / "detection3d.py"
    tree = ast.parse(src.read_text())
    heads = {"UNETRHead", "LinearDecoderHead", "ViTAdapterUNETRHead"}

    seen = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in heads):
            continue
        name = node.func.id
        seen.add(name)
        sig = inspect.signature(getattr(detection_heads, name).__init__)
        # 'self' is bound at call time; everything else must bind cleanly.
        params = [p for p in sig.parameters.values() if p.name != "self"]
        try:
            inspect.Signature(params).bind(
                *[object() for _ in node.args],
                **{kw.arg: object() for kw in node.keywords if kw.arg is not None},
            )
        except TypeError as exc:
            pytest.fail(f"detection3d.py line {node.lineno}: {name}(...) does not match its "
                        f"signature {sig}: {exc}")

    assert seen == heads, f"expected a call site for each head, only found {sorted(seen)}"


# ---------------------------------------------------------------------------
# bug 2 — pos_embed grid must follow the checkpoint, not a hardcoded default
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grid,patch,expected", [
    (8, 16, 128),   # this repo's b200 128-crop checkpoint: 8^3 = 512 tokens
    (7, 16, 112),   # the old default that was assumed unconditionally
    (4, 16, 64),
    (8, 14, 112),   # a non-16 patch size must not be hardcoded away
])
def test_pretrain_size_derived_from_pos_embed(grid, patch, expected):
    vit = _FakeViT(n_tokens=grid ** 3 + 1, patch=patch)   # +1 for the cls token
    assert _pretrain_size_from(vit, fallback=-1) == expected


def test_pretrain_size_falls_back_when_not_a_cube():
    assert _pretrain_size_from(_FakeViT(n_tokens=500 + 1), fallback=112) == 112


def test_pretrain_size_falls_back_on_missing_attributes():
    assert _pretrain_size_from(object(), fallback=112) == 112


def test_pretrain_size_ignores_image_size_disagreement():
    """The segmentation twin passes image_size, which is only right when the finetune size
    happens to equal the SSL crop size. Deriving from the weights stays correct when they differ.
    """
    vit = _FakeViT(n_tokens=8 ** 3 + 1, patch=16)          # checkpoint pretrained at 128
    assert _pretrain_size_from(vit, fallback=256) == 128    # finetuning at 256 must not win
