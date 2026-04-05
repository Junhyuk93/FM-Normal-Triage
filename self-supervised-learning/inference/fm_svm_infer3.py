# fm_svm_infer3.py
import os
import glob
import argparse
import joblib
import numpy as np
from functools import partial

import torch
import torch.nn as nn
from tqdm import tqdm

from model_utils import ModelWithIntermediateLayers
from setup import setup_and_build_model, get_args_parser
from data import make_data_loader, make_dataset
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

import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def gray_trunc(dcm_img):
    img = sitk.GetArrayFromImage(sitk.ReadImage(dcm_img)).squeeze()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    ori = img.copy()

    # 중심부 1/4~3/4 범위 추출
    y, x = ori.shape
    x_1, x_3 = np.round(x*1/4).astype('int'), np.round(x*3/4).astype('int')
    y_1, y_3 = np.round(y*1/4).astype('int'), np.round(y*3/4).astype('int')

    img_ = ori[y_1:y_3, x_1:x_3]
    v_max, v_min = img_.max(), img_.min()

    img = np.where(ori < v_min, v_min, ori)
    img = np.where(img > v_max, v_max, img)

    # [0,255] 스케일링
    img = img - np.min(img)
    img = img / np.max(img)
    img = (img * 255).astype(np.uint8)

    return ori, img  # ori: [0,1], trunc: [0,255] uint8

def default_eval_transform(img_size=1024):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

class DICOMFolderDataset(Dataset):
    """
    normal / target / others 구조의 폴더를 읽어서 DICOM(.dcm) 파일을 gray_trunc 처리 후 Tensor 변환
    """
    def __init__(self, root, transform=None):
        self.samples = []  # (path, class_idx)
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform or default_eval_transform()

        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = os.path.join(root, cls_name)
            for dp, _, fns in os.walk(cls_dir):
                for fn in fns:
                    if fn.lower().endswith(".dcm"):
                        self.samples.append((os.path.join(dp, fn), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        _, trunc = gray_trunc(path)  # [0,255] uint8

        # 단일 채널 → RGB 복제
        img = np.stack([trunc, trunc, trunc], axis=-1)  # H×W×3
        img = Image.fromarray(img)  # PIL

        img = self.transform(img)   # ToTensor + Normalize
        return img, label


def _infer_linear_prefix(state_dict):
    # find "*classifier*.linear.weight" and return prefix up to 'linear.'
    for k in state_dict.keys():
        if k.endswith('linear.weight') and 'classifier' in k:
            return k[: -len('linear.')]
    return None


def load_linear_from_ckpt(classifier: LinearClassifier, ckpt_path: str, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    full = ckpt["model"] if "model" in ckpt else ckpt
    prefix = _infer_linear_prefix(full)
    if prefix is None:
        raise ValueError(f"[load_linear_from_ckpt] Cannot infer classifier prefix in: {ckpt_path}")
    state = {k.replace(prefix, ''): v for k, v in full.items() if k.startswith(prefix)}
    classifier.load_state_dict(state, strict=False)
    return classifier


def setup_linear_classifier(sample_output, num_classes=3, n_blocks=4, avg_pool=True, device="cuda"):
    out_dim = create_linear_input(sample_output, n_blocks, avg_pool).shape[1]
    return LinearClassifier(out_dim, n_blocks, avg_pool, num_classes).to(device)


# ---------- dataset / loader ----------
#def setup_external_loader(data_path, batch_size=1, num_workers=8):
#    dataset = make_dataset(dataset_str=data_path, transform=make_classification_eval_transform())
#    loader = make_data_loader(
#        dataset=dataset,
#        batch_size=batch_size,
#        num_workers=num_workers,
#        drop_last=False,
#        shuffle=False,
#        persistent_workers=False,
#    )
#    return loader, dataset

def setup_external_loader(data_path, batch_size=1, num_workers=8):
    if isinstance(data_path, str) and data_path.startswith("normal-triage:root="):
        root = data_path.split("normal-triage:root=")[-1]
        dataset = DICOMFolderDataset(root=root, transform=default_eval_transform())
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        return loader, dataset

    # 기존 fallback
    dataset = make_dataset(dataset_str=data_path, transform=make_classification_eval_transform())
    loader = make_data_loader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                              drop_last=False, shuffle=False, persistent_workers=False)
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


def build_binary_label_mapper(dataset):
    """
    폴더명(class_to_idx)을 사용해 이진 라벨 매퍼를 만든다.
    normal -> 0, target/others -> 1 (others 하위 케이스는 dataset이 others로 묶어줌)
    """
    normal_idx = None
    abnormal_idx = set()

    class_to_idx = getattr(dataset, "class_to_idx", None)
    if isinstance(class_to_idx, dict) and len(class_to_idx) > 0:
        for name, idx in class_to_idx.items():
            low = name.lower()
            if low == "normal":
                normal_idx = idx
            elif low in ("target", "others"):
                abnormal_idx.add(idx)

    def to_binary(lbl: int) -> int:
        # class_to_idx가 없거나 매칭 실패 시: 0이면 normal, 그 외는 abnormal로 처리(보수적)
        if class_to_idx is None or normal_idx is None:
            return 0 if int(lbl) == 0 else 1
        return 0 if int(lbl) == int(normal_idx) else 1

    return to_binary


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


# ---------- main pipeline ----------
def run(args):
    set_seed(getattr(args, "seed", 42))

    device = "cuda" if (getattr(args, "device", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[Device] {device}")

    batch_size = getattr(args, "batch_size", 1)
    num_workers = getattr(args, "num_workers", 8)  # <-- 없으면 8로 기본값


    loader, dataset = setup_external_loader(args.test_dataset, batch_size=batch_size, num_workers=num_workers)

    # ✅ 여기서 binary label mapper를 정의
    to_binary = build_binary_label_mapper(dataset)

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

            # label -> binary via folder-name mapping
            if torch.is_tensor(label):
                label_np = label.cpu().numpy().astype(int)
            else:
                label_np = np.asarray(label, dtype=int)
            labels_bin.extend([to_binary(int(v)) for v in label_np])

    features = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0, len(linear_list)*3), dtype=np.float32)
    labels_bin = np.asarray(labels_bin, dtype=int)
    print(f"[Shape] Features: {features.shape} (N, {len(linear_list)*3}), Labels: {labels_bin.shape}")

    expected_dim = len(linear_list) * 3
    if features.shape[1] != expected_dim:
        raise ValueError(f"[DimMismatch] expected {expected_dim}, got {features.shape[1]}")

    preds = svm.predict(features).astype(int)

    # metrics
    if labels_bin.size > 0:
        m = binary_metrics(labels_bin, preds)
        print("\n=== Binary Metrics (normal=0, abnormal=1) ===")
        print(f"Accuracy: {m['overall']['accuracy']*100:.2f}%")
        print("Confusion Matrix: TP={TP} TN={TN} FP={FP} FN={FN}".format(**m["confusion"]))
        for k in ("normal(0)", "abnormal(1)"):
            pr = m[k]["precision"] * 100
            rc = m[k]["recall"] * 100
            f1 = m[k]["f1"] * 100
            print(f"{k:>11}: Precision {pr:.2f}% | Recall {rc:.2f}% | F1 {f1:.2f}%")
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
    ap = get_args_parser(description="DINOv2 K-Model + SVM (Binary via folder names)")
    # SVM
    ap.add_argument("--svm-model-path", type=str, required=True, help="Path to joblib SVM pickle")
    # linear heads
    ap.add_argument("--pretrained-linear-list", type=str, default=None, help="Comma-separated ckpt paths (order preserved)")
    ap.add_argument("--pretrained-linear-glob", type=str, default=None, help="Glob for ckpts")
    ap.add_argument("--require-exact-num-models", type=int, default=13, help="Require exact K heads (set -1 to disable)")
    # general (⚠️ batch-size, training-num-classes 등은 get_args_parser에 이미 있음 → 중복 정의하지 말기)
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

