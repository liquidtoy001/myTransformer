"""下载评测集的验证/测试划分，整理成统一格式，供去污染使用（P2-3）。

    python scripts/fetch_eval_sets.py

输出 data/eval_sets/<评测集>.jsonl，每行一道题：

    {"bench": "hellaswag", "split": "validation", "text": "题面 + 正确答案"}

**为什么是"题面 + 正确答案"**：训练语料里如果出现了题面，模型见过题；如果还连着答案，模型可能直接背下答案。
两者都算污染。测试集没有公开答案的（HellaSwag、PIQA、WinoGrande、C-Eval 的 test）只放题面。
不下载训练划分：评测不用它，训练语料里出现它也不算作弊。
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

RAW = Path("data/eval_sets/raw")
OUT = Path("data/eval_sets")


def get(repo: str, path: str) -> Path:
    return Path(hf_hub_download(repo, path, repo_type="dataset", local_dir=RAW / repo.replace("/", "__")))


def rows(repo: str, path: str) -> list[dict]:
    p = get(repo, path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.open(encoding="utf-8")]
    return pq.read_table(p).to_pylist()


def hellaswag():
    for split in ["validation", "test"]:
        for r in rows("Rowan/hellaswag", f"data/{split}-00000-of-00001.parquet"):
            ans = r["endings"][int(r["label"])] if r["label"] not in ("", None) else ""
            yield split, f"{r['ctx']} {ans}"


def piqa():
    for split in ["validation", "test"]:
        for r in rows("baber/piqa", f"piqa_{split}.parquet"):
            label = r.get("label")
            ans = r["sol1"] if label == 0 else r["sol2"] if label == 1 else ""
            yield split, f"{r['goal']} {ans}"


def arc():
    for sub in ["ARC-Easy", "ARC-Challenge"]:
        for split in ["validation", "test"]:
            for r in rows("allenai/ai2_arc", f"{sub}/{split}-00000-of-00001.parquet"):
                ch = r["choices"]
                ans = dict(zip(ch["label"], ch["text"])).get(r["answerKey"], "")
                yield f"{sub}/{split}", f"{r['question']} {ans}"


def winogrande():
    for split in ["validation", "test"]:
        for r in rows("allenai/winogrande", f"winogrande_xl/{split}-00000-of-00001.parquet"):
            opt = r.get(f"option{r['answer']}") if r.get("answer") in ("1", "2") else None
            yield split, r["sentence"].replace("_", opt) if opt else r["sentence"]


def lambada():
    for r in rows("EleutherAI/lambada_openai", "data/lambada_test_en.jsonl"):
        yield "test", r["text"]


def ceval():
    files = HfApi().list_repo_files("ceval/ceval-exam", repo_type="dataset")
    for f in sorted(files):
        if f.endswith(".parquet") and ("/val-" in f or "/test-" in f):
            for r in rows("ceval/ceval-exam", f):
                ans = r.get(r.get("answer") or "", "") if r.get("answer") else ""
                yield f.split("-")[0], f"{r['question']} {ans}"


def cmmlu():
    z = zipfile.ZipFile(get("lmlmcat/cmmlu", "cmmlu_v1_0_1.zip"))
    for name in sorted(z.namelist()):
        if name.endswith(".csv") and name.startswith(("test/", "dev/")):  # dev 是少样本示例
            for r in csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8")):
                yield name, f"{r['Question']} {r.get(r['Answer'], '')}"


def gsm8k():
    for r in rows("openai/gsm8k", "main/test-00000-of-00001.parquet"):
        yield "test", f"{r['question']} {r['answer']}"


def mmlu():
    for split in ["dev", "validation", "test"]:
        for r in rows("cais/mmlu", f"all/{split}-00000-of-00001.parquet"):
            yield split, f"{r['question']} {r['choices'][r['answer']]}"


def humaneval():
    for r in rows("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet"):
        yield "test", r["prompt"] + r["canonical_solution"]


def mbpp():
    for split in ["test", "validation", "prompt"]:
        for r in rows("google-research-datasets/mbpp", f"full/{split}-00000-of-00001.parquet"):
            yield split, f"{r['text']}\n{r['code']}"


BENCHES = {f.__name__: f for f in
           [hellaswag, piqa, arc, winogrande, lambada, ceval, cmmlu, gsm8k, mmlu, humaneval, mbpp]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in BENCHES.items():
        n = 0
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as fo:
            for split, text in fn():
                fo.write(json.dumps({"bench": name, "split": split, "text": text.strip()}, ensure_ascii=False) + "\n")
                n += 1
        print(f"[{name}] {n} 道题", flush=True)


if __name__ == "__main__":
    main()
