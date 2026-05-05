"""Stage 1: download GSM8K and Ho et al. teacher CoTs.

GSM8K: HuggingFace `gsm8k`/`main` config.
Teacher CoTs: itsnamgyu/reasoning-teacher release. The shared Dropbox/Drive folder
contains `teacher_completion_data.tar.gz`, which holds Zero-shot-CoT outputs from
text-davinci-002 across many datasets. We extract only the GSM8K file.

Resumable: each step skips work if its output exists.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
GSM8K_DIR = RAW_DIR / "gsm8k"
HO_DIR = RAW_DIR / "ho_et_al_cots"

DROPBOX_ZIP_URL = (
    "https://www.dropbox.com/sh/hwcncpyomx87h20/"
    "AACqgVdd-ZzBQ3ncJcKqw0cVa?dl=1"
)
# Within the unzipped folder, the per-archive path. Confirmed by inspecting the
# Dropbox listing in the project README.
TEACHER_TARBALL_NAME = "teacher_completion_data.tar.gz"
# Within `teacher_completion_data.tar.gz`, the GSM8K file lives at this path.
# The schema is documented in the reasoning-teacher README under
# data.completion.CompletionDataset.
GSM8K_COMPLETION_PATH_IN_TAR = (
    "completion_data/B_text-davinci-002__C_zs_cot/D_gsm8k.json"
)


def _save_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def download_gsm8k() -> None:
    train_path = GSM8K_DIR / "train.jsonl"
    test_path = GSM8K_DIR / "test.jsonl"
    if train_path.exists() and test_path.exists():
        print(f"[gsm8k] already downloaded at {GSM8K_DIR}")
        return
    print("[gsm8k] downloading via datasets.load_dataset('gsm8k', 'main')")
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main")
    GSM8K_DIR.mkdir(parents=True, exist_ok=True)
    _save_jsonl(ds["train"], train_path)
    _save_jsonl(ds["test"], test_path)
    print(f"[gsm8k] saved {len(ds['train'])} train, {len(ds['test'])} test -> {GSM8K_DIR}")


def _curl_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # -L follow redirects, -f fail on http errors, --retry for transient failures
    cmd = ["curl", "-fSL", "--retry", "3", "-o", str(dest), url]
    print("[curl]", " ".join(cmd))
    subprocess.check_call(cmd)


def download_ho_et_al() -> None:
    out_json = HO_DIR / "gsm8k_zs_cot_text-davinci-002.json"
    if out_json.exists():
        print(f"[ho] already extracted at {out_json}")
        return

    HO_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "release.zip"
        print(f"[ho] downloading shared folder zip (~924 MB) to {zip_path}")
        _curl_download(DROPBOX_ZIP_URL, zip_path)

        # Extract only the teacher tarball to save disk
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            tar_member = next((n for n in names if n.endswith(TEACHER_TARBALL_NAME)), None)
            if tar_member is None:
                raise RuntimeError(
                    f"{TEACHER_TARBALL_NAME} not found in zip. Members: {names[:20]}..."
                )
            print(f"[ho] extracting {tar_member} from zip")
            tar_path = tmp_path / TEACHER_TARBALL_NAME
            with zf.open(tar_member) as src, tar_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

        # Extract only the GSM8K JSON from the tarball
        print(f"[ho] opening tarball {tar_path}")
        with tarfile.open(tar_path, "r:gz") as tf:
            members = tf.getnames()
            gsm_member = next(
                (m for m in members if m.endswith("D_gsm8k.json") and "zs_cot" in m and "text-davinci-002" in m),
                None,
            )
            if gsm_member is None:
                raise RuntimeError(
                    f"GSM8K zs_cot file not found in tarball. Sample members: {members[:20]}..."
                )
            print(f"[ho] extracting {gsm_member}")
            ti = tf.getmember(gsm_member)
            f = tf.extractfile(ti)
            assert f is not None
            data_bytes = f.read()
        out_json.write_bytes(data_bytes)
        print(f"[ho] saved {out_json} ({out_json.stat().st_size:,} bytes)")


def smoke_print() -> None:
    """Print 5 GSM8K examples and 5 teacher CoTs side by side, plus schema notes."""
    train_path = GSM8K_DIR / "train.jsonl"
    test_path = GSM8K_DIR / "test.jsonl"
    teacher_path = HO_DIR / "gsm8k_zs_cot_text-davinci-002.json"
    if not train_path.exists() or not teacher_path.exists():
        print("[smoke] data missing; nothing to print")
        return

    gsm_train = [json.loads(l) for l in train_path.open()]
    gsm_test = [json.loads(l) for l in test_path.open()]
    teacher = json.loads(teacher_path.read_text())
    teacher_data = teacher["data"]

    print("=" * 60)
    print("HO ET AL. SCHEMA REPORT")
    print("=" * 60)
    print("[meta]", json.dumps(teacher["metadata"], indent=2))
    n_keys = len(teacher_data)
    print(f"[counts] teacher records: {n_keys} (GSM8K train={len(gsm_train)}, test={len(gsm_test)}, sum={len(gsm_train)+len(gsm_test)})")
    print("[note] sample_index ranges over train+test concatenated; train=[0..{}], test=[{}..{}]".format(
        len(gsm_train) - 1, len(gsm_train), len(gsm_train) + len(gsm_test) - 1))
    one = teacher_data[next(iter(teacher_data))][0]
    print(f"[fields] {list(one.keys())}")
    print("[interpretation]")
    print("  reasoning_prompt    -> Q + 'Let's think step by step.' (input to step 1)")
    print("  reasoning_completion-> teacher's CoT (USE THIS as teacher_cot)")
    print("  prompt              -> reasoning_prompt + reasoning_completion + ' Therefore, the answer is'")
    print("  completion          -> teacher's final-answer extraction (parse to get teacher_predicted_answer)")
    print("  answer              -> GSM8K gold answer as a bare numeric string")
    print("  finish_reason       -> OpenAI completion finish_reason")
    print("=" * 60)

    print("\nFIVE GSM8K TRAIN EXAMPLES + TEACHER CoT")
    print("=" * 60)
    for i in range(5):
        ex = gsm_train[i]
        t = teacher_data[str(i)][0]
        print(f"\n--- train[{i}] ---")
        print("Q:", ex["question"])
        print("Gold (raw):", ex["answer"])
        print("Teacher reasoning_completion:", (t.get("reasoning_completion") or "").strip())
        print("Teacher final completion:", (t.get("completion") or "").strip())
        print("Teacher 'answer' field:", t.get("answer"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-ho", action="store_true", help="Skip Ho et al. download (debug)")
    p.add_argument("--smoke-only", action="store_true", help="Only print smoke output")
    args = p.parse_args()

    if not args.smoke_only:
        download_gsm8k()
        if not args.skip_ho:
            download_ho_et_al()
    smoke_print()


if __name__ == "__main__":
    main()
