"""Reproducible experiments for CMPSC 448: CNN, LSTM, RQ1-RQ4.

Outputs are actual measurements from the supplied *real* model-response dataset.
Never use the synthetic test fixture as final project evidence.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import TextCNN, TextLSTM

TOKENS = re.compile(r"[A-Za-zÀ-ÿ]+(?:'[A-Za-zÀ-ÿ]+)?|\d+|[^\w\s]", re.UNICODE)
FIELDS = {"input": "LLM_Input", "output": "LLM_output"}
FAMILIES = ["ChatGPT", "Claude", "Gemini"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def norm(s):
    return " ".join(str(s).lower().split())


def load_data(path):
    d = pd.read_csv(path).fillna("")
    needed = {"LLM_name", "LLM_Input", "LLM_output", "category"}
    if needed - set(d.columns):
        raise ValueError(f"Dataset needs columns {sorted(needed)}, missing {needed-set(d.columns)}")
    d = d[d.LLM_name.isin(FAMILIES)].copy()
    d["prompt_group"] = d.LLM_Input.map(norm)
    d = d[(d.LLM_Input.str.len() >= 3) & (d.LLM_output.str.len() >= 15)].copy()
    d = d.drop_duplicates(["prompt_group", "LLM_name"], keep="first")
    counts = d.groupby("prompt_group").LLM_name.nunique()
    full = counts[counts == len(FAMILIES)].index
    d = d[d.prompt_group.isin(full)].copy().reset_index(drop=True)
    if d.prompt_group.nunique() < 18:
        raise ValueError("At least 18 complete prompts with all 3 real model responses are needed for this split protocol.")
    if d.groupby("prompt_group").category.nunique().max() > 1:
        raise ValueError("Inconsistent categories within a shared prompt.")
    d["category"] = d.category.astype(str).str.strip().str.lower()
    print(f"Dataset ready: {len(d)} responses across {d.prompt_group.nunique()} shared prompts.")
    print("Class counts:", d.LLM_name.value_counts().to_dict())
    return d


def partition(d, seed):
    groups = d.prompt_group.to_numpy()
    split1 = GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed)
    rem, test = next(split1.split(d, groups=groups))
    sub = d.iloc[rem]
    split2 = GroupShuffleSplit(n_splits=1, test_size=.1765, random_state=seed + 1)
    tr, val = next(split2.split(sub, groups=sub.prompt_group.to_numpy()))
    a, b, c = sub.iloc[tr].copy(), sub.iloc[val].copy(), d.iloc[test].copy()
    assert not (set(a.prompt_group) & set(b.prompt_group) or set(a.prompt_group) & set(c.prompt_group) or set(b.prompt_group) & set(c.prompt_group))
    for s in (a, b, c):
        assert set(s.LLM_name) == set(FAMILIES), "Each subset should include each family."
    print("Split shared prompts: train", a.prompt_group.nunique(), "/ val", b.prompt_group.nunique(), "/ test", c.prompt_group.nunique())
    return a, b, c


def tokenize(s):
    return TOKENS.findall(str(s).lower())


def format_text(frame, mode, stripped=False):
    def clean(s):
        if not stripped:
            return s
        s = re.sub(r"(?m)^\s{0,3}(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s+)", " ", s)
        s = re.sub(r"[`*_~|]", " ", s)
        return re.sub(r"\s+", " ", s).strip()
    if mode == "input":
        return frame.LLM_Input.astype(str).tolist()
    if mode == "output":
        return [clean(x) for x in frame.LLM_output.astype(str)]
    return [a + " <sep> " + clean(b) for a, b in zip(frame.LLM_Input.astype(str), frame.LLM_output.astype(str))]


def vocab_from(texts, cap):
    c = Counter(tok for text in texts for tok in tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({v: i+2 for i, (v, _) in enumerate(c.most_common(cap - 2))})
    return vocab


def encode(texts, vocab, seq_len):
    ids = np.zeros((len(texts), seq_len), dtype=np.int64)
    lengths = np.empty(len(texts), dtype=np.int64)
    for i, sentence in enumerate(texts):
        tokens = tokenize(sentence)[:seq_len]
        lengths[i] = max(len(tokens), 1)
        ids[i, :len(tokens)] = [vocab.get(t, 1) for t in tokens]
    return torch.from_numpy(ids), torch.from_numpy(lengths)


def loader(d, mode, vocab, seq_len, bs, shuffle, stripped=False):
    text = format_text(d, mode, stripped)
    ids, lengths = encode(text, vocab, seq_len)
    y = torch.tensor([FAMILIES.index(x) for x in d.LLM_name], dtype=torch.long)
    return DataLoader(TensorDataset(ids, lengths, y), batch_size=bs, shuffle=shuffle)


@torch.no_grad()
def predict(model, dataloader):
    model.eval()
    ys, preds = [], []
    loss = 0
    n = 0
    crit = nn.CrossEntropyLoss()
    for ids, lens, y in dataloader:
        p = model(ids, lens)
        loss += crit(p, y).item() * len(y)
        n += len(y)
        ys.extend(y.tolist())
        preds.extend(p.argmax(1).tolist())
    return np.array(ys), np.array(preds), loss / max(n, 1)


def metrics(y, p):
    return {"accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, labels=list(range(3)), average="macro", zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y, p, labels=list(range(3))).tolist(),
            "per_class": classification_report(y, p, labels=list(range(3)), target_names=FAMILIES, zero_division=0, output_dict=True)}


def one_experiment(model_name, mode, train, val, test, args, stripped=False):
    v = vocab_from(format_text(train, mode, stripped), args.vocab_size)
    tr = loader(train, mode, v, args.seq_len, args.batch_size, True, stripped)
    vl = loader(val, mode, v, args.seq_len, args.batch_size, False, stripped)
    te = loader(test, mode, v, args.seq_len, args.batch_size, False, stripped)
    model_cls = TextCNN if model_name == "CNN" else TextLSTM
    model = model_cls(len(v), len(FAMILIES), embed=args.embed_dim, **({"channels":args.hidden_dim} if model_name=="CNN" else {"hidden":args.hidden_dim}))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best = float("inf")
    best_state = None
    patience = 0
    epochs_ran = 0
    for epoch in range(args.epochs):
        model.train()
        for ids, lengths, y in tr:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ids, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        _, _, val_loss = predict(model, vl)
        epochs_ran = epoch + 1
        if val_loss < best - 1e-4:
            best = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    model.load_state_dict(best_state)
    y, pred, _ = predict(model, te)
    r = metrics(y, pred)
    r.update({"epochs_ran": epochs_ran, "train_rows":len(train), "val_rows":len(val), "test_rows":len(test), "vocab_size":len(v)})
    print(f"{model_name:4} {mode:7} => accuracy {r['accuracy']:.3f}, macro-F1 {r['macro_f1']:.3f} in {epochs_ran} epochs")
    return r, model, v


def feature_row(response):
    words = tokenize(response)
    words_alpha = [w for w in words if any(ch.isalpha() for ch in w)]
    sentences = [s for s in re.split(r"[.!?]+", response) if s.strip()]
    return {"response_words":len(words_alpha),
            "avg_word_length":sum(map(len, words_alpha)) / max(len(words_alpha), 1),
            "lexical_diversity":len(set(words_alpha)) / max(len(words_alpha), 1),
            "average_sentence_words":len(words_alpha) / max(len(sentences), 1),
            "markdown_symbols":sum(response.count(ch) for ch in ("*", "#", "`")),
            "bullet_lines":len(re.findall(r"(?m)^\s*[-*+]\s+", response)),
            "newlines":response.count("\n")}


def save_plots(r, d, results_dir):
    modes = ["input", "output", "combined"]
    plt.figure(figsize=(8,4.5))
    x = np.arange(3)
    for off, name in [(-.2, "CNN"),(.2,"LSTM")]:
        values = [r["rq2"][f"{name}_{mode}"]["accuracy"] for mode in modes]
        plt.bar(x + off, values, .36, label=name)
    plt.axhline(1/3, color="gray", ls="--", label="Chance (33.3%)")
    plt.xticks(x, ["Prompt only", "Response only", "Prompt + response"])
    plt.ylabel("Held-out accuracy")
    plt.ylim(0, 1)
    plt.title("RQ1 / RQ2: Which text predicts the source LLM?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir/"rq2_comparison.png", dpi=175)
    plt.close()
    df = pd.DataFrame([{"LLM_name":name, **feature_row(output)} for name, output in zip(d.LLM_name, d.LLM_output)])
    df.to_csv(results_dir / "linguistic_features.csv", index=False)
    sums = df.groupby("LLM_name").mean(numeric_only=True).reindex(FAMILIES)
    sums.round(3).to_csv(results_dir / "linguistic_summary.csv")
    # Plot features with separate y-scales so counts of bullets and Markdown marks
    # remain visible next to the much larger word-count numbers.
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 3.4))
    features = [("response_words", "Words"),
                ("markdown_symbols", "Markdown marks"),
                ("bullet_lines", "Bullet lines")]
    for ax, (column, label) in zip(axs, features):
        ax.bar(sums.index, sums[column].values)
        ax.set_title(label)
        ax.set_ylabel("Mean per answer")
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax.set_ylim(bottom=0)
    fig.suptitle("RQ4: Differences in response structure", fontsize=12)
    fig.tight_layout()
    fig.savefig(results_dir/"linguistic_patterns.png", dpi=175)
    plt.close(fig)
    m = np.asarray(r["rq2"]["CNN_output"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(3), FAMILIES)
    ax.set_yticks(range(3), FAMILIES)
    ax.set_xlabel("Predicted LLM")
    ax.set_ylabel("True LLM")
    for (i,j), v in np.ndenumerate(m):
        ax.text(j, i, str(v), ha="center", va="center", color="black")
    ax.set_title("CNN output-only: test confusion matrix")
    fig.tight_layout()
    fig.savefig(results_dir/"cnn_confusion.png", dpi=175)
    plt.close(fig)
    return sums


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/llm_responses.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--vocab-size", type=int, default=12000)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=.002)
    ap.add_argument("--skip-extras", action="store_true")
    args=ap.parse_args()
    if "fixture" in str(args.data).lower() and args.out.resolve() == Path("results").resolve():
        raise SystemExit("Fixture data are synthetic. To test code, explicitly choose --out test_results. Never put fixture metrics in final results.")
    seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    d=load_data(args.data)
    train, val, test=partition(d,args.seed)
    splitmap={v: k for k, subset in [("train",train),("val",val),("test",test)] for v in subset.prompt_group.unique()}
    d.assign(prompt_hash=d.prompt_group.map(lambda text: hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]), split=d.prompt_group.map(splitmap))[['prompt_hash','category','LLM_name','split']].to_csv(args.out/"split_audit.csv",index=False)
    res={"metadata":{"dataset_rows":len(d),"unique_prompts":d.prompt_group.nunique(),"family_counts":d.LLM_name.value_counts().to_dict(),
                     "category_counts":d.groupby('category').prompt_group.nunique().to_dict(),"train_prompts":train.prompt_group.nunique(),
                     "validation_prompts":val.prompt_group.nunique(),"test_prompts":test.prompt_group.nunique(),
                     "seed":args.seed,"sequence_length":args.seq_len,"epochs_max":args.epochs,"source_type":("SYNTHETIC TEST FIXTURE - NOT RESEARCH RESULTS" if "fixture" in str(args.data).lower() else "INPUT DATA: verify source provenance in README")},"rq2":{},"rq3":{},"rq4":{}}
    saved_models={}
    for model in ("CNN", "LSTM"):
        for mode in ("input", "output", "combined"):
            seed_all(args.seed)
            met, trained, vocab=one_experiment(model,mode,train,val,test,args)
            res["rq2"][f"{model}_{mode}"]=met
            if mode=="output":
                saved_models[model]=(trained,vocab)
                torch.save({"model_state":trained.state_dict(),"vocabulary":vocab,"classes":FAMILIES,
                            "embed_dim":args.embed_dim,"hidden_dim":args.hidden_dim,"model_type":model},args.out/f"{model.lower()}_output_model.pt")
    if not args.skip_extras:
        writing = d.category.str.contains("writing",case=False,na=False)
        held=d[writing].copy()
        before=d[~writing].copy()
        if held.prompt_group.nunique()>=3 and before.prompt_group.nunique()>=12:
            gs=GroupShuffleSplit(n_splits=1,test_size=.18,random_state=args.seed)
            tr_ind,v_ind=next(gs.split(before,groups=before.prompt_group.to_numpy()))
            domain_train, domain_val=before.iloc[tr_ind],before.iloc[v_ind]
            for model in ("CNN","LSTM"):
                seed_all(args.seed)
                met,_,_=one_experiment(model,"output",domain_train,domain_val,held,args)
                res["rq3"][model+"_unseen_writing"]=met
            res["rq3"]["held_out_category"]="writing"
            res["rq3"]["held_out_prompts"]=held.prompt_group.nunique()
        for name, (trained,vocab) in saved_models.items():
            te=loader(test,"output",vocab,args.seq_len,args.batch_size,False,stripped=True)
            y,p,_=predict(trained,te)
            res["rq4"][name+"_format_stripped_test"]=metrics(y,p)
        res["rq4"]["ablation_method"]="Keep original trained model fixed; strip markdown/list formatting from only the held-out test responses."
    sums=save_plots(res,d,args.out)
    res["rq4"]["style_feature_means"]=sums.round(3).to_dict(orient="index")
    (args.out/"experiment_results.json").write_text(json.dumps(res,indent=2,default=float),encoding="utf-8")
    if "fixture" in str(args.data).lower():
        print("\nSaved SYNTHETIC software-test results (NOT research results) to", args.out)
    else:
        print("\nSaved measured dataset results and figures to", args.out)
    print("Use: python -m src.make_report --results",args.out)

if __name__=="__main__":
    main()
