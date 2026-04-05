# fm_svm_train.py
import os
import glob
import argparse
import joblib
import numpy as np
from functools import partial

import torch
import torch.nn as nn
from tqdm import tqdm

import pickle

# === re-use your existing modules ===
from model_utils import ModelWithIntermediateLayers
from setup import setup_and_build_model, get_args_parser
from data import make_data_loader, make_dataset
from data.transforms import make_classification_eval_transform

# ---------------- utils (same as infer) ----------------
def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    inter = x_tokens_list[-use_n_blocks:]
    cls_concat = torch.cat([cls for _, cls in inter], dim=-1)
    if use_avgpool:
        avg = torch.mean(inter[-1][0], dim=1)
        out = torch.cat((cls_concat, avg), dim=-1).reshape(cls_concat.shape[0], -1)
    else:
        out = cls_concat
    return out.float()

class LinearClassifier(nn.Module):
    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(out_dim, num_classes)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool

    def forward(self, x_tokens_list):
        x = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(x)

def _infer_linear_prefix(state_dict):
    for k in state_dict.keys():
        if k.endswith('linear.weight') and 'classifier' in k:
            return k[: -len('linear.')]
    return None

def load_linear_from_ckpt(classifier: LinearClassifier, ckpt_path: str, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    full = ckpt["model"] if "model" in ckpt else ckpt
    auto_prefix = _infer_linear_prefix(full)
    fallback_prefixes = [
        "classifiers_dict.classifier_4_blocks_avgpool_True_lr_0_00003.",
        "classifiers_dict.classifier_4_blocks_avgpool_True_lr_0_0001.",
        "classifiers_dict.classifier_4_blocks_avgpool_True_lr_0_00003",
    ]
    candidates = []
    if auto_prefix is not None:
        candidates.append(auto_prefix)
    for p in fallback_prefixes:
        if p not in candidates:
            candidates.append(p)

    tried = []
    for pref in candidates:
        tried.append(pref)
        state = {k.replace(pref, ''): v for k, v in full.items() if k.startswith(pref)}
        if "linear.weight" in state and "linear.bias" in state:
            classifier.load_state_dict(state, strict=True)
            print(f"[LinearLoad] Loaded with prefix: '{pref}' from {os.path.basename(ckpt_path)}")
            return classifier

    ex_keys = [k for k in full.keys() if "linear" in k][:12]
    raise ValueError(
        f"[LinearLoad][FAIL] No matching prefix in: {ckpt_path}\n"
        f"  Tried prefixes: {tried}\n"
        f"  Example keys containing 'linear': {ex_keys}"
    )

def setup_linear_classifier(sample_output, num_classes=3, n_blocks=4, avg_pool=True, device="cuda"):
    out_dim = create_linear_input(sample_output, n_blocks, avg_pool).shape[1]
    return LinearClassifier(out_dim, n_blocks, avg_pool, num_classes).to(device)

def setup_loader(dataset_str, batch_size=1, num_workers=8):
    dataset = make_dataset(dataset_str=dataset_str, transform=make_classification_eval_transform())
    loader = make_data_loader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers,
        drop_last=False, shuffle=False, persistent_workers=False,
    )
    return loader, dataset

class DINOv2MultiExtractor(nn.Module):
    def __init__(self, args, n_blocks=4, avg_pool=True, linear_ckpts=None, device="cuda"):
        super().__init__()
        self.n_blocks = n_blocks
        self.avg_pool = avg_pool
        self.num_classes = getattr(args, "training_num_classes", 3)
        self.device = device

        base_model, autocast_dtype = setup_and_build_model(args)
        base_model = base_model.to(device).eval()
        self.autocast_ctx = partial(torch.cuda.amp.autocast, enabled=(device == "cuda"), dtype=autocast_dtype)
        self.feature_model = ModelWithIntermediateLayers(base_model, self.n_blocks, self.autocast_ctx).eval()

        with torch.no_grad():
            dummy = torch.randn(1, 3, 1024, 1024, device=device)
            sample_output = self.feature_model(dummy)

        self.classifiers = nn.ModuleList()
        if linear_ckpts:
            for p in linear_ckpts:
                clf = setup_linear_classifier(sample_output, num_classes=self.num_classes,
                                              n_blocks=self.n_blocks, avg_pool=self.avg_pool, device=device)
                load_linear_from_ckpt(clf, p, device=device)
                self.classifiers.append(clf)

        self.num_models = len(self.classifiers)
        print(f"[Extractor] Loaded {self.num_models} linear heads (each outputs {self.num_classes} classes).")

    @torch.no_grad()
    def _tokens(self, x):
        return self.feature_model(x)

    @torch.no_grad()
    def extract_concat_probs(self, x):
        x_tokens_list = self._tokens(x)
        probs_all = []
        for clf in self.classifiers:
            logits = clf(x_tokens_list)
            probs = torch.softmax(logits, dim=-1)
            probs_all.append(probs)
        if not probs_all:
            raise ValueError("[Extractor] No classifiers loaded.")
        return torch.cat(probs_all, dim=-1)   # (B, K*3)

def _parse_list_arg(comma_list: str):
    if not comma_list:
        return []
    return [s.strip() for s in comma_list.split(",") if s.strip()]

def _parse_glob_arg(glob_str: str):
    if not glob_str:
        return []
    return sorted(glob.glob(glob_str))

def set_seed(seed: int | None):
    if seed is None:
        return
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_binary_label_mapper(dataset):
    normal_idx = None; abnormal_idx = set()
    class_to_idx = getattr(dataset, "class_to_idx", None)
    if isinstance(class_to_idx, dict) and len(class_to_idx) > 0:
        for name, idx in class_to_idx.items():
            low = name.lower()
            if low == "normal":
                normal_idx = idx
            elif low in ("target", "others"):
                abnormal_idx.add(idx)

    def to_binary(lbl: int) -> int:
        if class_to_idx is None or normal_idx is None:
            return 0 if int(lbl) == 0 else 1
        return 0 if int(lbl) == int(normal_idx) else 1
    return to_binary

# ---------------- train SVM ----------------
def run_train(args):
    set_seed(getattr(args, "seed", 42))
    device = "cuda" if (getattr(args, "device", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[Device] {device}")

    batch_size = getattr(args, "batch_size", 1)
    num_workers = getattr(args, "num_workers", 8)

    # Loader
    loader, dataset = setup_loader(args.train_dataset, batch_size=batch_size, num_workers=num_workers)
    to_binary = build_binary_label_mapper(dataset)

    # Linear heads (order preserved!)
    linear_list = []
    linear_list += _parse_list_arg(getattr(args, "pretrained_linear_list", None))
    linear_list += _parse_glob_arg(getattr(args, "pretrained_linear_glob", None))
    linear_list = [p for p in linear_list if os.path.exists(p)]
    if len(linear_list) == 0 and getattr(args, "pretrained_linear", None):
        linear_list = [args.pretrained_linear]

    print(f"[Linear] {len(linear_list)} ckpts found (order preserved):")
    for i, p in enumerate(linear_list):
        print(f"  {i+1:02d}. {p}")

    # ---- Fallback for missing args (핵심 변경) ----
    n_blocks = int(getattr(args, "n_blocks", 4))
    avg_pool = bool(getattr(args, "avg_pool", True))
    req_k    = getattr(args, "require_exact_num_models", 13)  # 음수면 체크 비활성
    print(f"[ExtractorCfg] n_blocks={n_blocks}, avg_pool={avg_pool}")

    extractor = DINOv2MultiExtractor(
        args=args,
        n_blocks=n_blocks,
        avg_pool=avg_pool,
        linear_ckpts=linear_list,
        device=device,
    ).eval()

    # 모델 개수 안전 체크
    if isinstance(req_k, int) and req_k >= 0 and extractor.num_models != req_k:
        raise ValueError(f"[Safety] Expected {req_k} linear heads, got {extractor.num_models}.")

    # ---------- collect features ----------
    feats_list = []
    labels_bin = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract"):
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                raise RuntimeError("Unexpected batch; expected (image, label, ...).")
            image, label = batch[0].to(device, non_blocking=True), batch[1]
            probs_concat = extractor.extract_concat_probs(image)  # (B, K*3)
            feats_list.append(probs_concat.cpu().numpy())

            if torch.is_tensor(label):
                label_np = label.cpu().numpy().astype(int)
            else:
                label_np = np.asarray(label, dtype=int)
            labels_bin.extend([to_binary(int(v)) for v in label_np])

    X_feat = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0, extractor.num_models*3), dtype=np.float32)
    y_bin = np.asarray(labels_bin, dtype=int)

    print(f"[Shape] Features: {X_feat.shape} (N, {extractor.num_models*3}), Labels: {y_bin.shape}")
    expected_dim = extractor.num_models * 3
    if X_feat.shape[1] != expected_dim:
        raise ValueError(f"[DimMismatch] expected {expected_dim}, got {X_feat.shape[1]}")

    # ---------- train SVM ----------
    from sklearn.svm import LinearSVC, SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score

    svm_type = getattr(args, "svm_type", "linear")  # 'linear' or 'rbf'
    if svm_type == "linear":
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LinearSVC(C=getattr(args, "C", 1.0), class_weight="balanced", max_iter=5000, dual=True)
        )
    elif svm_type == "rbf":
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            SVC(kernel="rbf", C=getattr(args, "C", 1.0), gamma=getattr(args, "gamma", "scale"), class_weight="balanced")
        )
    else:
        raise ValueError("--svm-type must be 'linear' or 'rbf'")

    clf.fit(X_feat, y_bin)
    train_pred = clf.predict(X_feat)
    train_acc = accuracy_score(y_bin, train_pred)
    print(f"[Train] Resub Accuracy: {train_acc*100:.2f}% (for sanity check)")

    # ---------- save ----------
    #os.makedirs(os.path.dirname(getattr(args, "svm_out")), exist_ok=True)
    #joblib.dump(clf, getattr(args, "svm_out"))
    #print(f"[SVM] Saved to: {getattr(args, 'svm_out')}")

    os.makedirs(os.path.dirname(getattr(args, "svm_out")), exist_ok=True)
    with open(getattr(args, "svm_out"), "wb") as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SVM] Saved (pickle) to: {getattr(args, 'svm_out')}")

    # optional saves
    if getattr(args, "save_features", True):
        out_dir = os.path.dirname(getattr(args, "svm_out"))
        np.save(os.path.join(out_dir, "train_features.npy"), X_feat)
        np.save(os.path.join(out_dir, "train_labels_binary.npy"), y_bin)

    return {"svm_path": getattr(args, "svm_out"), "train_acc": float(train_acc), "n": int(X_feat.shape[0])}

def build_argparser():
    # 공통 옵션은 setup.get_args_parser가 모두 넣어줍니다.
    ap = get_args_parser(description="Train SVM on concatenated probs (K×3) from DINOv2 heads")

    # === SVM 학습 전용 옵션만 추가 ===
    ap.add_argument("--train-dataset", type=str, required=True,
                    help="e.g., normal-triage:root=/workspace/dataset/v3/valid")

    ap.add_argument("--svm-out", type=str, required=True,
                    help="Path to save joblib SVM, e.g., /workspace/inference/svm_weight.pickle")

    ap.add_argument("--svm-type", type=str, choices=["linear", "rbf"], default="linear")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--gamma", type=str, default="scale")  # rbf 전용

    # 선형 head ckpt 목록 (순서 고정)
    ap.add_argument("--pretrained-linear-list", type=str, default=None)
    ap.add_argument("--pretrained-linear-glob", type=str, default=None)
    ap.add_argument("--require-exact-num-models", type=int, default=13)

    # 선택 저장
    ap.add_argument("--save-features", action="store_true", default=True)

    return ap

def main():
    parser = build_argparser()
    args = parser.parse_args()
    out = run_train(args)
    print(f"\n[Training completed] SVM saved to: {out['svm_path']} | N={out['n']} | resub acc={out['train_acc']:.4f}")

if __name__ == "__main__":
    main()

