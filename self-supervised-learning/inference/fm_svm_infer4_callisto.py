#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from functools import partial

import numpy as np
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from scipy.special import expit, softmax

import pandas as pd
from PIL import Image
import SimpleITK as sitk  # pydicom 대신 SimpleITK 사용

from model_utils import ModelWithIntermediateLayers
from setup import setup_and_build_model, get_args_parser
from data.transforms import make_classification_eval_transform


# ============================================================
# 1. 공통 유틸: DINOv2 중간 토큰 → Linear 입력
# ============================================================

def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    """
    x_tokens_list: List[Tuple[tokens, cls_token]]
      - tokens: (B, N, C)
      - cls_token: (B, C)
    """
    inter = x_tokens_list[-use_n_blocks:]
    cls_concat = torch.cat([cls for _, cls in inter], dim=-1)
    if use_avgpool:
        avg = torch.mean(inter[-1][0], dim=1)  # 마지막 블록의 patch 토큰 avg
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
        if k.endswith("linear.weight") and "classifier" in k:
            return k[: -len("linear.")]
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
        state = {k.replace(pref, ""): v for k, v in full.items() if k.startswith(pref)}
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


def set_seed(seed: int | None):
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. DINOv2 + 다중 Linear 헤드 → 특징 추출기
# ============================================================

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

        # dummy 입력으로 Linear 입력 차원 파악
        with torch.no_grad():
            dummy = torch.randn(1, 3, 1024, 1024, device=device)
            sample_output = self.feature_model(dummy)

        self.classifiers = nn.ModuleList()
        if linear_ckpts:
            for p in linear_ckpts:
                clf = setup_linear_classifier(
                    sample_output,
                    num_classes=self.num_classes,
                    n_blocks=self.n_blocks,
                    avg_pool=self.avg_pool,
                    device=device,
                )
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
        return torch.cat(probs_all, dim=-1)


# ============================================================
# 3. 레이블 없는 DICOM Dataset (폴더 구조 무시, .dcm만 수집)
# ============================================================

class UnlabeledDicomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.paths = []

        for r, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(".dcm"):
                    self.paths.append(os.path.join(r, f))

        if len(self.paths) == 0:
            raise RuntimeError(f"[UnlabeledDataset] No DICOM files found under: {root}")

        print(f"[UnlabeledDataset] Found {len(self.paths)} DICOM files under {root}")

    def __len__(self):
        return len(self.paths)

    def _load_dicom_with_sitk(self, path):
        import SimpleITK as sitk
        from PIL import Image

        img_itk = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img_itk)  # (Z,H,W) or (H,W)

        if arr.ndim == 3:
            arr = arr[0]  # 첫 슬라이스 사용

        # float 변환 및 0~255 스케일링
        arr = arr.astype(np.float32)
        arr = arr - np.min(arr)
        max_val = np.max(arr)
        if max_val > 0:
            arr = arr / max_val * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        # --- 여기서 PhotometricInterpretation 확인 후 반전 처리 ---
        try:
            if img_itk.HasMetaDataKey("0028|0004"):
                photometric = img_itk.GetMetaData("0028|0004").upper().strip()
                # MONOCHROME1: high values = dark → 우리가 쓰는 0~255 기준에선 뒤집어야 함
                if "MONOCHROME1" in photometric:
                    arr = 255 - arr
        except Exception as e:
            # 태그 없거나 이상하면 그냥 패스
            pass
    # ---------------------------------------------------------

        pil_img = Image.fromarray(arr).convert("L").convert("RGB")
        return pil_img



    def __getitem__(self, idx):
        import torch
        from torchvision import transforms as T

        path = self.paths[idx]
        img = self._load_dicom_with_sitk(path)  # PIL RGB

        # 1) repo의 eval transform이 PIL을 입력으로 받는 경우
        if self.transform is not None:
            out = self.transform(img)
        else:
            out = img

        # 2) transform 결과를 torch.Tensor (C,H,W)로 강제 변환
        # 여러 케이스를 방어적으로 처리
        if isinstance(out, dict) and "image" in out:
            # albumentations 스타일: {"image": np.ndarray, ...}
            out = out["image"]

        if isinstance(out, np.ndarray):
            arr = out
            if arr.ndim == 2:
                # (H,W) → (1,H,W) → 3채널로 복제
                arr = np.stack([arr, arr, arr], axis=0)  # (3,H,W)
            elif arr.ndim == 3:
                # (H,W,C) or (C,H,W)
                if arr.shape[0] in (1, 3):
                    # 이미 CHW
                    if arr.shape[0] == 1:
                        # 1채널이면 3채널로 반복
                        arr = np.repeat(arr, 3, axis=0)
                elif arr.shape[-1] in (1, 3):
                    # HWC → CHW
                    arr = np.transpose(arr, (2, 0, 1))
                    if arr.shape[0] == 1:
                        arr = np.repeat(arr, 3, axis=0)
                else:
                    # 애매하면 마지막 축이 채널이라고 보고 변환
                    arr = np.transpose(arr, (2, 0, 1))
            else:
                raise RuntimeError(f"[UnlabeledDataset] Unexpected ndarray shape: {arr.shape}")
            img_t = torch.from_numpy(arr.astype(np.float32)) / 255.0

        elif isinstance(out, Image.Image):
            # PIL → Tensor (C,H,W)
            to_tensor = T.ToTensor()  # [0,1], (C,H,W)
            img_t = to_tensor(out)
            if img_t.shape[0] == 1:
                img_t = img_t.repeat(3, 1, 1)

        elif torch.is_tensor(out):
            img_t = out
            if img_t.ndim == 2:
                # (H,W) → (1,H,W) → 3채널 복제
                img_t = img_t.unsqueeze(0)
                img_t = img_t.repeat(3, 1, 1)
            elif img_t.ndim == 3:
                # (C,H,W) 또는 (H,W,C)
                if img_t.shape[0] not in (1, 3) and img_t.shape[-1] in (1, 3):
                    # HWC → CHW
                    img_t = img_t.permute(2, 0, 1)
                if img_t.shape[0] == 1:
                    img_t = img_t.repeat(3, 1, 1)
            else:
                raise RuntimeError(f"[UnlabeledDataset] Unexpected tensor shape: {tuple(img_t.shape)}")

        else:
            raise RuntimeError(f"[UnlabeledDataset] Unsupported transform output type: {type(out)}")

        # 최종: img_t는 (3, H, W)
        dummy_label = 0
        return img_t, dummy_label


def setup_unlabeled_loader(root_dir, batch_size=1, num_workers=8):
    transform = make_classification_eval_transform()
    dataset = UnlabeledDicomDataset(root=root_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    return loader, dataset


# ============================================================
# 4. Unlabeled 추론 + CSV 저장
# ============================================================

def _parse_list_arg(comma_list: str):
    if not comma_list:
        return []
    return [s.strip() for s in comma_list.split(",") if s.strip()]


def _parse_glob_arg(glob_str: str):
    if not glob_str:
        return []
    return sorted(glob.glob(glob_str))

def run_unlabeled(args):
    set_seed(getattr(args, "seed", 42))

    # ------------------ Device & Loader ------------------
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[Device] {device}")

    batch_size = getattr(args, "batch_size", 1)
    num_workers = getattr(args, "num_workers", 8)

    loader, dataset = setup_unlabeled_loader(args.infer_root, batch_size=batch_size, num_workers=num_workers)

    # ------------------ Linear ckpt 리스트 구성 ------------------
    linear_list = []
    linear_list += _parse_list_arg(getattr(args, "pretrained_linear_list", None))
    linear_list += _parse_glob_arg(getattr(args, "pretrained_linear_glob", None))
    linear_list = [p for p in linear_list if os.path.exists(p)]

    if len(linear_list) == 0 and getattr(args, "pretrained_linear", None):
        linear_list = [args.pretrained_linear]

    # 중복 제거
    linear_list = list(dict.fromkeys(linear_list))

    print(f"[Linear] {len(linear_list)} ckpts found (order preserved):")
    for i, p in enumerate(linear_list):
        print(f"  {i+1:02d}. {p}")

    # ------------------ Backbone + Linear Extractor ------------------
    extractor = DINOv2MultiExtractor(
        args=args,
        n_blocks=args.n_blocks,
        avg_pool=args.avg_pool,
        linear_ckpts=linear_list,
        device=device,
    ).eval()

    # ------------------ SVM 로드 ------------------
    if not os.path.exists(args.svm_model_path):
        raise FileNotFoundError(f"[SVM] Not found: {args.svm_model_path}")
    svm = joblib.load(args.svm_model_path)
    print(f"[SVM] Loaded from: {args.svm_model_path}")

    # ------------------ Inference ------------------
    results = []   # ← CSV에 저장될 정보 저장하는 리스트

    with torch.no_grad():
        for idx, (image, _) in enumerate(tqdm(loader, desc="Infer (unlabeled DICOM)")):
            # image shape 보정
            if image.ndim == 3:
                image = image.unsqueeze(0)
            elif image.ndim == 2:
                image = image.unsqueeze(0).unsqueeze(0)
                image = image.repeat(1, 3, 1, 1)

            image = image.to(device, non_blocking=True)

            # 13개의 linear head concat probabilities (B, 39)
            probs_concat = extractor.extract_concat_probs(image)
            probs_np = probs_concat.cpu().numpy()  # shape (B, 39)

            # SVM classification
            decision_scores = svm.decision_function(probs_np)

            if decision_scores.ndim == 1:
                prob_class_1 = expit(decision_scores)
                prob_class_0 = 1 - prob_class_1
                probs_svm = np.stack([prob_class_0, prob_class_1], axis=1)
                preds = (decision_scores >= args.svm_threshold).astype(int)
            else:
                probs_svm = softmax(decision_scores, axis=1)
                preds = np.argmax(probs_svm, axis=1)

            # ---- 결과 저장 ----
            # dataset.paths 는 loader index와 동일한 순서
            path = dataset.paths[idx]
            filename = os.path.basename(path)

            results.append({
                "filepath": path,
                "filename": filename,
                "predicted_label": int(preds[0]),
                "prob_normal": float(probs_svm[0, 0]),
                "prob_abnormal": float(probs_svm[0, 1]),
                "confidence": float(probs_svm[0, preds[0]]),
            })

    # ------------------ Save CSV ------------------
    df = pd.DataFrame(results)
    os.makedirs(args.outdir, exist_ok=True)
    out_csv_path = os.path.join(args.outdir, "unlabeled_predictions.csv")
    df.to_csv(out_csv_path, index=False)

    print(f"[CSV] Saved to: {out_csv_path}  (Rows: {len(df)})")

    return df



# ============================================================
# 5. ArgParser & main
# ============================================================

def build_argparser_unlabeled():
    # 기존 설정 (config-file, pretrained-weights, test-dataset, batch-size 등 포함)
    ap = get_args_parser(description="DINOv2 K-Model + SVM (Unlabeled DICOM Inference)")

    # 여기부터는 get_args_parser 에서 정의하지 않은 추가 옵션만 넣기
    ap.add_argument("--svm-model-path", type=str, required=True,
                    help="Path to joblib SVM pickle")
    ap.add_argument("--pretrained-linear-list", type=str, default=None,
                    help="Comma-separated ckpt paths (order preserved)")
    ap.add_argument("--pretrained-linear-glob", type=str, default=None,
                    help="Glob for ckpts")
    ap.add_argument("--require-exact-num-models", type=int, default=13,
                    help="Require exact K heads (set -1 to disable)")
    ap.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--avg-pool", action="store_true", default=True)
    ap.add_argument("--outdir", type=str, default="./svm_out_unlabeled")
    ap.add_argument("--save-features", action="store_true", default=False)
    ap.add_argument("--save-preds", action="store_true", default=False)
    ap.add_argument("--svm-threshold", type=float, default=0.29,
                    help="Threshold on decision_function for binary prediction")

    # ✅ 레이블 없는 DICOM 루트 경로
    ap.add_argument("--infer-root", type=str, required=True,
                    help="Root directory of unlabeled DICOM files")

    return ap


def main():
    parser = build_argparser_unlabeled()
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = run_unlabeled(args)

    # 옵션: feature / preds 저장
    # (위 run_unlabeled에서 반환 값에 features/preds를 포함시키고 싶으면 구조를 약간 바꾸면 됨)


if __name__ == "__main__":
    main()

