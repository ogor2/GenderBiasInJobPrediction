"""
Run BERT fairness post-processing on the bias-in-bios test set.
Outputs a CSV with baseline vs post-processed metrics.

Usage (from repo root, venv active):
    python 483-BertvsLR/bert_postprocessing.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = Path("BERT Classifier/bert_finetuned")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    data_root = Path("original_cleaned_dataset")
    X_test = pd.read_csv(data_root / "X_test.csv")["clean_text"].astype(str).tolist()
    y_test = pd.read_csv(data_root / "y_test.csv")["profession"].to_numpy()
    s_test = pd.read_csv(data_root / "s_test.csv")["gender"].to_numpy()

    batch_size = 64
    max_len = 128

    def infer(texts):
        all_logits = []
        loader = DataLoader(texts, batch_size=batch_size)
        with torch.no_grad():
            for batch in loader:
                enc = tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                ).to(device)
                out = model(**enc)
                all_logits.append(out.logits.cpu())
        return torch.cat(all_logits, dim=0)

    logits = infer(X_test)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)

    male_mask = s_test == 0
    female_mask = s_test == 1

    def metrics(y_true, y_pred):
        return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")

    acc, f1 = metrics(y_test, preds)
    acc_m = accuracy_score(y_test[male_mask], preds[male_mask])
    acc_f = accuracy_score(y_test[female_mask], preds[female_mask])
    gap = acc_m - acc_f

    # Post-processing: per-gender confidence thresholds (quantile-based)
    q = 0.05
    conf = probs.max(axis=1)
    sorted_top2 = np.argsort(-probs, axis=1)
    pred_top1 = sorted_top2[:, 0]
    pred_top2 = sorted_top2[:, 1]
    male_conf_correct = conf[male_mask & (pred_top1 == y_test)]
    female_conf_correct = conf[female_mask & (pred_top1 == y_test)]
    thr_m = np.quantile(male_conf_correct, q) if len(male_conf_correct) > 0 else 1.0
    thr_f = np.quantile(female_conf_correct, q) if len(female_conf_correct) > 0 else 1.0

    pp = pred_top1.copy()
    for i in range(len(pp)):
        thr = thr_m if s_test[i] == 0 else thr_f
        if conf[i] < thr:
            pp[i] = pred_top2[i]

    pp_acc, pp_f1 = metrics(y_test, pp)
    pp_m = accuracy_score(y_test[male_mask], pp[male_mask])
    pp_f = accuracy_score(y_test[female_mask], pp[female_mask])
    pp_gap = pp_m - pp_f

    out = Path("483-BertvsLR/fairness_summary_bert_mitigated_josh.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ["BERT-baseline", acc, f1, gap, acc_m, acc_f],
        ["BERT-postproc_q05", pp_acc, pp_f1, pp_gap, pp_m, pp_f],
    ]
    pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Overall Accuracy",
            "Overall F1 (macro)",
            "Accuracy Gap (male-female)",
            "Male Accuracy",
            "Female Accuracy",
        ],
    ).to_csv(out, index=False)
    print("Saved", out)
    print("Baseline:", acc, f1, gap)
    print("Post-proc:", pp_acc, pp_f1, pp_gap)


if __name__ == "__main__":
    main()
