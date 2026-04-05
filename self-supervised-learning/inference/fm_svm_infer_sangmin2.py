# fm_svm_infer_sangmin2.py

import os
import glob
import argparse
import joblib
import numpy as np
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import SimpleITK as sitk

from model_utils import ModelWithIntermediateLayers
from setup import setup_and_build_model, get_args_parser
from data import make_data_loader
from data.transforms import make_classification_eval_transform


# ---------- utils: linear input ----------
def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    # x_tokens_list element: (tokens, class_token)
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
    # find "*classifier*.linear.weight" and return prefix up to 'linear.'
    for k in state_dict.keys():
        if k.endswith('linear.weight') and 'classifier' in k:
            return k[: -len('linear.')]
    return None


# ---------- (1) 고도화된 로더: prefix fallback + 로드 검증(strict=True) ----------
def load_linear_from_ckpt(classifier: LinearClassifier, ckpt_path: str, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    full = ckpt["model"] if "model" in ckpt else ckpt

    # 자동 추정
    auto_prefix = _infer_linear_prefix(full)

    # single.py에서 사용하던 prefix 등 합리적 후보들
    fallback_prefixes = [
        "classifiers_dict.classifier_4_blocks_avgpool_True_lr_0_00003.",
        "classifiers_dict.classifier_4_blocks_avgpool_True_lr_0_0001.",
        "classifiers_dict.classifier_4_blocks_avgpool_True_lr_0_00003",  # 끝에 '.' 없는 버전
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
        # 최소 키 검증
        if "linear.weight" in state and "linear.bias" in state:
            classifier.load_state_dict(state, strict=True)
            print(f"[LinearLoad] Loaded with prefix: '{pref}' from {os.path.basename(ckpt_path)}")
            return classifier

    # 실패 시 유용한 키 힌트 제공
    ex_keys = [k for k in full.keys() if "linear" in k][:12]
    raise ValueError(
        f"[LinearLoad][FAIL] No matching prefix in: {ckpt_path}\n"
        f"  Tried prefixes: {tried}\n"
        f"  Example keys containing 'linear': {ex_keys}"
    )


def setup_linear_classifier(sample_output, num_classes=3, n_blocks=4, avg_pool=True, device="cuda"):
    out_dim = create_linear_input(sample_output, n_blocks, avg_pool).shape[1]
    return LinearClassifier(out_dim, n_blocks, avg_pool, num_classes).to(device)


# ---------- custom Dataset: DICOM 모두 normal로 가정 ----------
class AllNormalDicomDataset(Dataset):
    """
    root 이하의 모든 .dcm 파일을 읽어서 이미지로 변환하고,
    라벨은 전부 0(normal)로 고정하는 Dataset.
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # 재귀적으로 모든 .dcm 파일 수집
        self.paths = sorted(
            glob.glob(os.path.join(root, "**", "*.dcm"), recursive=True)
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"No DICOM files found under: {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        # SimpleITK로 DICOM 읽기
        img_itk = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img_itk)  # 보통 (1, H, W) 또는 (H, W)

        if arr.ndim == 3:
            # (1, H, W) → (H, W)
            arr = arr[0]

        arr = np.asarray(arr, dtype=np.float32)
        if arr.max() > 0:
            arr = arr / arr.max()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

        # PIL 이미지로 변환 (RGB)
        img = Image.fromarray(arr).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = 0  # 모든 샘플을 normal로 간주
        return img, label

    # FP/FN 경로 저장용으로 paths 속성 유지


# ---------- dataset / loader ----------
def setup_external_loader(data_path, batch_size=1, num_workers=8):
    """
    이 스크립트에서는 외부셋은 전부 'AllNormalDicomDataset'으로만 처리한다.
    data_path는 보통
      - 순수 경로: /mnt/...
      - 또는 dataset_str 형식: "normal-triage:root=/mnt/.../dicom"
    두 케이스를 모두 처리해서 실제 root 디렉토리만 추출한다.
    """
    raw = data_path

    # dataset_str 형식일 경우 root= 뒤만 잘라낸다.
    if "root=" in raw:
        root = raw.split("root=", 1)[1]
        # 혹시 뒤에 , 다른 옵션이 붙어있으면 첫 토큰만 사용
        root = root.split(",")[0]
    else:
        root = raw

    root = root.strip()
    print(f"[Loader] Using AllNormalDicomDataset (all labels=normal) for root: {root} (from '{raw}')")

    transform = make_classification_eval_transform()
    dataset = AllNormalDicomDataset(root=root, transform=transform)

    loader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
    )
    return loader, dataset

# ---------- feature extractor ----------
class DINOv2MultiExtractor(nn.Module):
    """
    K개의 linear head(각 3-class)를 순서대로 적용하여 softmax 확률을 이어붙임 (K×3)
    """
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

        # size with dummy to build linear heads
        with torch.no_grad():
            dummy = torch.randn(1, 3, 1024, 1024, device=device)
            sample_output = self.feature_model(dummy)

        self.classifiers = nn.ModuleList()
        if linear_ckpts:
            for p in linear_ckpts:  # **주어진 리스트 순서를 그대로 유지**
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
        for clf in self.classifiers:                      # **리스트 순서대로 적용**
            logits = clf(x_tokens_list)                   # (B, 3)
            probs = torch.softmax(logits, dim=-1)         # (B, 3)
            probs_all.append(probs)
        if not probs_all:
            raise ValueError("[Extractor] No classifiers loaded.")
        return torch.cat(probs_all, dim=-1)               # (B, K*3)


# ---------- helpers ----------
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- metrics ----------
def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    assert y_true.shape == y_pred.shape
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec_pos = tp / max(1, (tp + fp))
    rec_pos = tp / max(1, (tp + fn))
    f1_pos = (2 * prec_pos * rec_pos) / max(1e-12, (prec_pos + rec_pos))

    prec_neg = tn / max(1, (tn + fn))
    rec_neg = tn / max(1, (tn + fp))
    f1_neg = (2 * prec_neg * rec_neg) / max(1e-12, (prec_neg + rec_neg))

    return {
        "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "overall": {"accuracy": acc},
        "normal(0)": {"precision": prec_neg, "recall": rec_neg, "f1": f1_neg},
        "abnormal(1)": {"precision": prec_pos, "recall": rec_pos, "f1": f1_pos},
    }


def compute_binary_summary(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    N = tp + tn + fp + fn

    def div(a, b):
        return (a / b) if b != 0 else float("nan")

    accuracy    = div(tp + tn, N)
    ppv         = div(tp, tp + fp)            # Positive Predictive Value (Precision+)
    sensitivity = div(tp, tp + fn)            # Recall / TPR
    specificity = div(tn, tn + fp)            # TNR
    npv         = div(tn, tn + fn)            # Negative Predictive Value
    prevalence  = div(tp + fn, N)
    precision   = ppv                          # 동의어

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": accuracy,
        "ppv": ppv,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "npv": npv,
        "prevalence": prevalence,
    }


# ---------- main pipeline ----------
def run(args):
    set_seed(getattr(args, "seed", 42))

    device = "cuda" if (getattr(args, "device", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[Device] {device}")

    batch_size = getattr(args, "batch_size", 1)
    num_workers = getattr(args, "num_workers", 8)

    # 여기서는 항상 AllNormalDicomDataset 사용
    loader, dataset = setup_external_loader(args.test_dataset, batch_size=batch_size, num_workers=num_workers)
    all_normal = True

    # linear list (순서 유지)
    linear_list = []
    linear_list += _parse_list_arg(getattr(args, "pretrained_linear_list", None))
    linear_list += _parse_glob_arg(getattr(args, "pretrained_linear_glob", None))
    linear_list = [p for p in linear_list if os.path.exists(p)]
    if len(linear_list) == 0 and getattr(args, "pretrained_linear", None):
        linear_list = [args.pretrained_linear]

    print(f"[Linear] {len(linear_list)} ckpts found (order preserved):")
    for i, p in enumerate(linear_list):
        print(f"  {i+1:02d}. {p}")

    extractor = DINOv2MultiExtractor(
        args=args,
        n_blocks=args.n_blocks,
        avg_pool=args.avg_pool,
        linear_ckpts=linear_list,
        device=device,
    ).eval()

    if args.require_exact_num_models is not None and args.require_exact_num_models >= 0:
        if extractor.num_models != args.require_exact_num_models:
            raise ValueError(f"[Safety] Expected {args.require_exact_num_models} linear heads, got {extractor.num_models}.")

    if not os.path.exists(args.svm_model_path):
        raise FileNotFoundError(f"[SVM] Not found: {args.svm_model_path}")
    svm = joblib.load(args.svm_model_path)
    print(f"[SVM] Loaded from: {args.svm_model_path}")

    feats_list = []
    labels_bin = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Infer"):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                image, label = batch[0], batch[1]
            else:
                raise RuntimeError("Unexpected batch; expected (image, label, ...).")

            image = image.to(device, non_blocking=True)
            probs_concat = extractor.extract_concat_probs(image)  # (B, K*3)
            feats_list.append(probs_concat.cpu().numpy())

            # 이 스크립트에서는 항상 모든 샘플을 normal(0)로 간주
            batch_size_curr = image.shape[0]
            labels_bin.extend([0] * batch_size_curr)

    features = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0, len(linear_list)*3), dtype=np.float32)
    labels_bin = np.asarray(labels_bin, dtype=int)
    print(f"[Shape] Features: {features.shape} (N, {len(linear_list)*3}), Labels: {labels_bin.shape}")

    expected_dim = len(linear_list) * 3
    if features.shape[1] != expected_dim:
        raise ValueError(f"[DimMismatch] expected {expected_dim}, got {features.shape[1]}")

    #preds = svm.predict(features).astype(int)
    decision_scores = svm.decision_function(features)
    threshold = 0.3  # 원하는 threshold 값으로 변경 (기본값 0.0)
    preds = (decision_scores >= threshold).astype(int)

    # metrics (binary only)
    if labels_bin.size > 0:
        report = compute_binary_summary(labels_bin, preds)
        print("\n=== Binary Report (normal=0, abnormal=1) ===")
        print(f"Accuracy: {report['accuracy']*100:.2f}%")
        print(f"PPV: {report['ppv']*100:.2f}%")
        print(f"TP: {report['TP']}")
        print(f"TN: {report['TN']}")
        print(f"FP: {report['FP']}")
        print(f"FN: {report['FN']}")
        print(f"Prevalence: {report['prevalence']*100:.2f}%")
        print(f"Precision: {report['precision']*100:.2f}%")
        print(f"Sensitivity:{report['sensitivity']*100:.2f}%")
        print(f"Specificity:{report['specificity']*100:.2f}%")
        print(f"NPV: {report['npv']*100:.2f}%")

        # FP (False Positive) 파일 리스트 저장 - 실제 normal(0)인데 abnormal(1)로 예측
        fp_indices = [i for i, (true_label, pred_label) in enumerate(zip(labels_bin, preds))
                      if true_label == 0 and pred_label == 1]

        # FN (False Negative) 파일 리스트 저장 - 실제 abnormal(1)인데 normal(0)로 예측
        fn_indices = [i for i, (true_label, pred_label) in enumerate(zip(labels_bin, preds))
                      if true_label == 1 and pred_label == 0]

        # 데이터셋에서 실제 DICOM 파일 절대경로 가져오기
        all_paths = []

        # AllNormalDicomDataset의 paths 속성 사용
        if isinstance(dataset, AllNormalDicomDataset):
            all_paths = [os.path.abspath(p) for p in dataset.paths]
            print(f"[Info] Found {len(all_paths)} paths from AllNormalDicomDataset.paths")

        elif hasattr(dataset, 'samples') and dataset.samples:
            all_paths = [os.path.abspath(sample[0]) for sample in dataset.samples]
            print(f"[Info] Found {len(all_paths)} paths from dataset.samples")

        elif hasattr(dataset, 'imgs') and dataset.imgs:
            all_paths = [os.path.abspath(img[0]) for img in dataset.imgs]
            print(f"[Info] Found {len(all_paths)} paths from dataset.imgs")

        elif hasattr(dataset, 'image_paths'):
            all_paths = [os.path.abspath(path) for path in dataset.image_paths]
            print(f"[Info] Found {len(all_paths)} paths from dataset.image_paths")

        elif hasattr(dataset, 'file_paths'):
            all_paths = [os.path.abspath(path) for path in dataset.file_paths]
            print(f"[Info] Found {len(all_paths)} paths from dataset.file_paths")

        elif hasattr(dataset, 'paths'):
            all_paths = [os.path.abspath(path) for path in dataset.paths]
            print(f"[Info] Found {len(all_paths)} paths from dataset.paths")

        elif hasattr(dataset, 'root'):
            print(f"[Info] Scanning DICOM files from dataset root: {dataset.root}")
            all_paths = []
            for root, dirs, files in os.walk(dataset.root):
                for file in files:
                    if file.lower().endswith('.dcm'):
                        all_paths.append(os.path.abspath(os.path.join(root, file)))
            all_paths = sorted(all_paths)
            print(f"[Info] Found {len(all_paths)} DICOM files by scanning")

        else:
            print(f"[Debug] Dataset type: {type(dataset)}")
            print(f"[Debug] Dataset attributes: {[attr for attr in dir(dataset) if not attr.startswith('_')]}")

            all_paths = [f"sample_{i}" for i in range(len(labels_bin))]
            print(f"[Warning] Using fallback sample names. Please check dataset implementation.")

        print(f"[Info] Total extracted paths: {len(all_paths)}")
        if len(all_paths) > 0 and not all_paths[0].startswith('sample_'):
            print(f"[Info] Example path: {all_paths[0]}")

        # FP 파일 저장
        fp_paths = [all_paths[i] for i in fp_indices if i < len(all_paths)]
        with open("FP.txt", "w") as f:
            for path in fp_paths:
                f.write(path + "\n")
        print(f"Saved {len(fp_paths)} FP paths to FP.txt")

        # FN 파일 저장
        fn_paths = [all_paths[i] for i in fn_indices if i < len(all_paths)]
        with open("FN.txt", "w") as f:
            for path in fn_paths:
                f.write(path + "\n")
        print(f"Saved {len(fn_paths)} FN paths to FN.txt")

    else:
        print("\n[Info] No labels found; only predictions available.")

    # optional saves
    os.makedirs(args.outdir, exist_ok=True)
    if args.save_features:
        np.save(os.path.join(args.outdir, "features.npy"), features)
        np.save(os.path.join(args.outdir, "labels_binary.npy"), labels_bin)
    if args.save_preds:
        np.save(os.path.join(args.outdir, "svm_preds.npy"), preds)

    # small CSV for quick glance
    try:
        import pandas as pd
        df = pd.DataFrame({"index": np.arange(len(preds)), "label_bin": labels_bin, "svm_pred": preds})
        df.to_csv(os.path.join(args.outdir, "per_sample_binary.csv"), index=False)
    except Exception as e:
        print(f"[Warn] CSV save failed: {e}")

    return {"svm_predictions": preds, "svm_features": features, "labels_binary": labels_bin}


def build_argparser():
    ap = get_args_parser(description="DINOv2 K-Model + SVM (Binary, external all-normal DICOM)")
    # SVM
    ap.add_argument("--svm-model-path", type=str, required=True, help="Path to joblib SVM pickle")
    # linear heads
    ap.add_argument("--pretrained-linear-list", type=str, default=None, help="Comma-separated ckpt paths (order preserved)")
    ap.add_argument("--pretrained-linear-glob", type=str, default=None, help="Glob for ckpts")
    ap.add_argument("--require-exact-num-models", type=int, default=13, help="Require exact K heads (set -1 to disable)")
    # general
    ap.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    # extractor
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--avg-pool", action="store_true", default=True)
    # outputs
    ap.add_argument("--outdir", type=str, default="./svm_out")
    ap.add_argument("--save-features", action="store_true", default=True)
    ap.add_argument("--save-preds", action="store_true", default=True)
    return ap


def main():
    parser = build_argparser()
    args = parser.parse_args()
    if not getattr(args, "test_dataset", None):
        raise ValueError("--test-dataset is required")
    _ = run(args)
    print("\n[Inference completed] 13×3 probs stacked in given order → SVM → binary metrics printed.")


if __name__ == "__main__":
    main()

