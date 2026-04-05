# fm_HGB_train_jh.py
import os, glob, argparse, pickle, joblib, numpy as np
import torch, torch.nn as nn
from functools import partial
from tqdm import tqdm

from model_utils import ModelWithIntermediateLayers
from setup import setup_and_build_model, get_args_parser
from data import make_data_loader, make_dataset
from data.transforms import make_classification_eval_transform


# ---------- utils ----------
def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    # x_tokens_list: list of tuples (tokens, cls_token) for intermediate layers
    inter = x_tokens_list[-use_n_blocks:]
    cls_concat = torch.cat([cls for _, cls in inter], dim=-1)
    if use_avgpool:
        avg = torch.mean(inter[-1][0], dim=1)  # avg over tokens from last block
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
    fall = [
        "classifiers_dict.classifier_4_blocks_avgpool_True_lr_0_00003.",
        "classifiers_dict.classifier_4_blocks_avgpool_True_lr_0_0001.",
        "classifiers_dict.classifier_4_blocks_avgpool_True_lr_0_00003",
    ]
    cand = ([auto_prefix] if auto_prefix else []) + [p for p in fall if p != auto_prefix]
    tried = []
    for pref in cand:
        tried.append(pref)
        state = {k.replace(pref, ''): v for k, v in full.items() if k.startswith(pref)}
        if "linear.weight" in state and "linear.bias" in state:
            classifier.load_state_dict(state, strict=True)
            print(f"[LinearLoad] prefix='{pref}' from {os.path.basename(ckpt_path)}")
            return classifier
    ex_keys = [k for k in full.keys() if "linear" in k][:12]
    raise ValueError(f"[LinearLoad][FAIL] {ckpt_path}\nTried: {tried}\nKeys: {ex_keys}")


def setup_linear_classifier(sample_output, num_classes=3, n_blocks=4, avg_pool=True, device="cuda"):
    out_dim = create_linear_input(sample_output, n_blocks, avg_pool).shape[1]
    return LinearClassifier(out_dim, n_blocks, avg_pool, num_classes).to(device)


def setup_loader(dataset_str, batch_size=1, num_workers=8):
    dataset = make_dataset(dataset_str=dataset_str, transform=make_classification_eval_transform())
    loader = make_data_loader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers,
        drop_last=False, shuffle=False, persistent_workers=False
    )
    return loader, dataset


class DINOv2MultiExtractor(nn.Module):
    """
    13개의 linear head(각 3-class)의 softmax 확률을 이어붙여 (B, 13*3) 반환
    """
    def __init__(self, args, n_blocks=4, avg_pool=True, linear_ckpts=None, device="cuda"):
        super().__init__()
        self.n_blocks, self.avg_pool = n_blocks, avg_pool
        self.num_classes = getattr(args, "training_num_classes", 3)

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
        print(f"[Extractor] {self.num_models} linear heads loaded (each {self.num_classes} classes)")

    @torch.no_grad()
    def _tokens(self, x):
        return self.feature_model(x)

    @torch.no_grad()
    def extract_concat_probs(self, x):
        # -------- 입력 보정: (C,H,W)->(1,C,H,W), 1채널이면 3채널로 복제 --------
        if x.dim() == 3:
            x = x.unsqueeze(0)              # (1,C,H,W)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)        # (B,3,H,W)
        # ---------------------------------------------------------------------
        x_tokens_list = self._tokens(x)
        probs_all = []
        for clf in self.classifiers:
            logits = clf(x_tokens_list)               # (B, 3)
            probs = torch.softmax(logits, dim=-1)     # (B, 3)
            probs_all.append(probs)
        if not probs_all:
            raise ValueError("[Extractor] No classifiers loaded.")
        return torch.cat(probs_all, dim=-1)           # (B, K*3)


def _parse_list_arg(comma_list: str):
    return [] if not comma_list else [s.strip() for s in comma_list.split(",") if s.strip()]


def _parse_glob_arg(glob_str: str):
    return [] if not glob_str else sorted(glob.glob(glob_str))


def set_seed(seed: int | None):
    if seed is None:
        return
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_binary_label_mapper(dataset):
    """
    dataset.class_to_idx를 사용해 normal→0, 그 외(others/target 포함)→1 로 매핑
    """
    normal_idx = None
    class_to_idx = getattr(dataset, "class_to_idx", None)
    if isinstance(class_to_idx, dict) and len(class_to_idx) > 0:
        for name, idx in class_to_idx.items():
            if name.lower() == "normal":
                normal_idx = idx

    def to_binary(lbl: int) -> int:
        if class_to_idx is None or normal_idx is None:
            return 0 if int(lbl) == 0 else 1
        return 0 if int(lbl) == int(normal_idx) else 1

    return to_binary


# ---------- training ----------
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

    print(f"[Linear] {len(linear_list)} ckpts (order preserved):")
    for i, p in enumerate(linear_list):
        print(f"  {i+1:02d}. {p}")

    # 안전한 기본값 (parser에 없을 때 대비)
    n_blocks = int(getattr(args, "n_blocks", 4))
    avg_pool = bool(getattr(args, "avg_pool", True))
    req_k = getattr(args, "require_exact_num_models", 13)
    print(f"[ExtractorCfg] n_blocks={n_blocks}, avg_pool={avg_pool}")

    extractor = DINOv2MultiExtractor(args=args, n_blocks=n_blocks, avg_pool=avg_pool,
                                     linear_ckpts=linear_list, device=device).eval()

    if isinstance(req_k, int) and req_k >= 0 and extractor.num_models != req_k:
        raise ValueError(f"[Safety] Expected {req_k} heads, got {extractor.num_models}.")

    # 특징 수집
    feats_list, labels_bin = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract"):
            assert isinstance(batch, (list, tuple)) and len(batch) >= 2, "Expected (image,label,...)"
            image, label = batch[0].to(device, non_blocking=True), batch[1]

            probs_concat = extractor.extract_concat_probs(image)   # (B, K*3)
            feats_list.append(probs_concat.cpu().numpy())

            label_np = label.cpu().numpy().astype(int) if torch.is_tensor(label) else np.asarray(label, dtype=int)
            labels_bin.extend([to_binary(int(v)) for v in label_np])

    X_feat = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0, extractor.num_models*3), dtype=np.float32)
    y_bin = np.asarray(labels_bin, dtype=int)
    Dexp = extractor.num_models * 3
    print(f"[Shape] Features: {X_feat.shape} (N,{Dexp}), Labels: {y_bin.shape}")
    if X_feat.shape[1] != Dexp:
        raise ValueError(f"[DimMismatch] expected {Dexp}, got {X_feat.shape[1]}")

    # 분류기 선택: 'logreg' 또는 'hgb'
    clf_type = getattr(args, "clf_type", "hgb")   # 기본값을 hgb로
    print(f"[Classifier] {clf_type}")

    if clf_type == "logreg":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(
                solver="liblinear", C=getattr(args, "C", 1.0),
                class_weight="balanced", max_iter=1000, n_jobs=None
            )
        )
        clf.fit(X_feat, y_bin)
    elif clf_type == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_bin)
        clf = HistGradientBoostingClassifier(
            max_iter=getattr(args, "hgb_iters", 300),
            learning_rate=getattr(args, "hgb_lr", 0.1),
            max_depth=None, l2_regularization=0.0,
            early_stopping=True, validation_fraction=0.1,
            random_state=getattr(args, "seed", 42)
        )
        clf.fit(X_feat, y_bin, sample_weight=sample_weight)
    else:
        raise ValueError("--clf-type must be 'logreg' or 'hgb'")

    # 간단 지표(Resubstitution)
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    pred = clf.predict(X_feat)
    acc = accuracy_score(y_bin, pred)
    f1 = f1_score(y_bin, pred)

    try:
        if hasattr(clf, "predict_proba"):
            score = clf.predict_proba(X_feat)[:, 1]
        else:
            # Pipeline인 경우 마지막 스텝에 따라 다름(없으면 AUC 계산 생략)
            score = clf.predict_proba(X_feat)[:, 1]  # 있을 때만 동작
        auc = roc_auc_score(y_bin, score)
    except Exception:
        auc = float("nan")

    print(f"[Train] Resub Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    # pickle 저장
    os.makedirs(os.path.dirname(getattr(args, "clf_out")), exist_ok=True)
    with open(getattr(args, "clf_out"), "wb") as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[CLF] Saved (pickle) to: {getattr(args, 'clf_out')}")

    # 옵션: 특징/라벨 저장
    if getattr(args, "save_features", True):
        out_dir = os.path.dirname(getattr(args, "clf_out"))
        np.save(os.path.join(out_dir, "train_features.npy"), X_feat)
        np.save(os.path.join(out_dir, "train_labels_binary.npy"), y_bin)

    return {"clf_path": getattr(args, "clf_out"), "acc": float(acc), "f1": float(f1), "auc": float(auc), "n": int(X_feat.shape[0])}


def build_argparser():
    # 공통 옵션은 setup.get_args_parser가 넣어줌(중복 정의 피함)
    ap = get_args_parser(description="Train classifier (no SVM) on concatenated probs (K×3)")
    ap.add_argument("--train-dataset", type=str, required=True,
                    help="e.g., normal-triage:root=/workspace/dataset/v3/train")
    # linear heads
    ap.add_argument("--pretrained-linear-list", type=str, default=None)
    ap.add_argument("--pretrained-linear-glob", type=str, default=None)
    ap.add_argument("--require-exact-num-models", type=int, default=13)
    # classifier
    ap.add_argument("--clf-type", type=str, choices=["logreg", "hgb"], default="hgb")
    ap.add_argument("--C", type=float, default=1.0)           # for logreg
    ap.add_argument("--hgb-iters", type=int, default=300)     # for HGB
    ap.add_argument("--hgb-lr", type=float, default=0.1)      # for HGB
    # outputs
    ap.add_argument("--clf-out", type=str, required=True,
                    help="Path to save pickle classifier, e.g., /workspace/inference/clf_weight.pickle")
    ap.add_argument("--save-features", action="store_true", default=True)
    return ap


def main():
    parser = build_argparser()
    args = parser.parse_args()
    out = run_train(args)
    print(f"\n[Training completed] CLF saved to: {out['clf_path']} | N={out['n']} | Acc={out['acc']:.4f} | F1={out['f1']:.4f} | AUC={out['auc']:.4f}")


if __name__ == "__main__":
    main()

