# fm_svm_infer_multi_format.py
import os
import glob
import argparse
import joblib
import numpy as np
from functools import partial
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from model_utils import ModelWithIntermediateLayers
from setup import setup_and_build_model, get_args_parser
from data import make_data_loader, make_dataset
from data.transforms import make_classification_eval_transform


# ---------- utils: linear input ----------
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

    for pref in candidates:
        state = {k.replace(pref, ''): v for k, v in full.items() if k.startswith(pref)}
        if "linear.weight" in state and "linear.bias" in state:
            classifier.load_state_dict(state, strict=True)
            print(f"[LinearLoad] Loaded with prefix: '{pref}' from {os.path.basename(ckpt_path)}")
            return classifier

    ex_keys = [k for k in full.keys() if "linear" in k][:12]
    raise ValueError(f"[LinearLoad][FAIL] No matching prefix in: {ckpt_path}\nExample keys: {ex_keys}")


def setup_linear_classifier(sample_output, num_classes=3, n_blocks=4, avg_pool=True, device="cuda"):
    out_dim = create_linear_input(sample_output, n_blocks, avg_pool).shape[1]
    return LinearClassifier(out_dim, n_blocks, avg_pool, num_classes).to(device)


# ---------- 다중 포맷 이미지 데이터셋 ----------
class MultiFormatImageDataset(Dataset):
    """DICOM, JPG, PNG 등 다양한 이미지 형식을 지원하는 데이터셋"""
    
    def __init__(self, image_paths, labels=None, img_size=(1024, 1024), transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # 파일 확장자에 따라 다른 로딩 방법 적용
            ext = os.path.splitext(image_path)[1].lower()
            
            if ext in ['.dcm', '.dicom']:
                # DICOM 파일 처리
                image = self._load_dicom(image_path)
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                # 일반 이미지 파일 처리
                image = self._load_regular_image(image_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            # 전처리 적용
            if self.transform:
                image = self.transform(image)
            else:
                # 기본 전처리: PIL Image -> Tensor
                if isinstance(image, Image.Image):
                    image = transforms.ToTensor()(image)
                elif isinstance(image, np.ndarray):
                    image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            
            # 라벨이 있는 경우
            if self.labels is not None:
                label = self.labels[idx]
                return image, label
            else:
                return image
                
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # 에러 시 빈 이미지 반환
            if self.labels is not None:
                return torch.zeros(3, *self.img_size), 0
            else:
                return torch.zeros(3, *self.img_size)
    
    def _load_dicom(self, path):
        """DICOM 파일 로딩"""
        try:
            # chest_dcm_to_png 함수 사용 (기존 로직)
            from chest_dcm_to_png import dicom_to_png
            image_array = dicom_to_png(path)
            
            # 그레이스케일을 RGB로 변환
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif len(image_array.shape) == 3 and image_array.shape[2] == 1:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                
        except ImportError:
            # chest_dcm_to_png가 없는 경우 pydicom 사용
            try:
                import pydicom
                import SimpleITK as sitk
                
                dcm = sitk.ReadImage(path)
                image_array = sitk.GetArrayFromImage(dcm)
                
                # 정규화
                image_array = image_array.astype(np.float32)
                if image_array.max() > image_array.min():
                    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
                image_array = (image_array * 255).astype(np.uint8)
                
                # 차원 조정
                if len(image_array.shape) == 3:
                    image_array = image_array[0]  # 첫 번째 슬라이스 사용
                
                # RGB로 변환
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                
            except Exception as e:
                print(f"Failed to load DICOM with both methods: {e}")
                return Image.new('RGB', self.img_size, color=0)
        
        # PIL Image로 변환 및 리사이즈
        image = Image.fromarray(image_array)
        image = image.resize(self.img_size, Image.LANCZOS)
        
        return image
    
    def _load_regular_image(self, path):
        """일반 이미지 파일 로딩 (JPG, PNG 등)"""
        # PIL로 이미지 로드
        image = Image.open(path)
        
        # RGBA나 P 모드를 RGB로 변환
        if image.mode in ('RGBA', 'LA'):
            # 투명도가 있는 경우 흰 배경으로 합성
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[-1])  # 알파 채널을 마스크로 사용
            else:
                background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode == 'P':
            image = image.convert('RGB')
        elif image.mode == 'L':
            # 그레이스케일을 RGB로 변환
            image = image.convert('RGB')
        
        # 리사이즈
        image = image.resize(self.img_size, Image.LANCZOS)
        
        return image


def collect_images_from_path(data_path):
    """경로에서 다양한 형식의 이미지 파일들을 수집 (디버그 추가)"""
    supported_extensions = ['.dcm','.dicom', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    print(f"[DEBUG] Searching in: {data_path}")
    print(f"[DEBUG] Path exists: {os.path.exists(data_path)}")
    print(f"[DEBUG] Is directory: {os.path.isdir(data_path)}")
    
    all_images = []
    all_labels = []
    
    if os.path.isfile(data_path):
        return [data_path], [0]
    
    # 디렉토리인 경우
    for root, dirs, files in os.walk(data_path):
        print(f"[DEBUG] Checking directory: {root}")
        print(f"[DEBUG] Found {len(files)} files in {root}")
        
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in supported_extensions:
                file_path = os.path.join(root, file)
                all_images.append(file_path)
                
                # 폴더명으로 라벨링
                folder_name = os.path.basename(root).lower()
                if 'normal' in folder_name:
                    label = 0
                else:
                    label = 1
                all_labels.append(label)
                
                # 처음 몇 개만 출력
                if len(all_images) <= 5:
                    print(f"[DEBUG] Found image: {file_path}, label: {label}")
    
    print(f"[DEBUG] Total images found: {len(all_images)}")
    return all_images, all_labels

def setup_multi_format_loader(data_path, batch_size=1, num_workers=8, img_size=(1024, 1024)):
    """다중 형식 이미지를 위한 데이터로더 설정"""
    
    # 이미지 파일들 수집
    image_paths, labels = collect_images_from_path(data_path)
    
    print(f"Found {len(image_paths)} images")
    
    # 파일 형식별 통계
    format_counts = {}
    for path in image_paths:
        ext = os.path.splitext(path)[1].lower()
        format_counts[ext] = format_counts.get(ext, 0) + 1
    
    print("File format distribution:")
    for ext, count in format_counts.items():
        print(f"  {ext}: {count} files")
    
    # 기본 transform 설정
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 및 데이터로더 생성
    dataset = MultiFormatImageDataset(
        image_paths=image_paths,
        labels=labels,
        img_size=img_size,
        transform=transform
    )
    
    # 간단한 라벨 매퍼 (normal=0, abnormal=1)
    class_to_idx = {'normal': 0, 'abnormal': 1}
    dataset.class_to_idx = class_to_idx
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=False,
    )
    
    return loader, dataset


# ---------- 기존 setup_external_loader 함수 수정 ----------
def setup_external_loader(data_path, batch_size=1, num_workers=8):
    """
    데이터셋 문자열을 확인하여 기존 방식 또는 다중 형식 방식 선택
    """
    # 데이터셋 문자열 형식 확인 (예: "normal-triage:root=/path")
    if ':' in data_path and 'root=' in data_path:
        # 기존 DINOv2 데이터셋 로더 사용
        try:
            dataset = make_dataset(dataset_str=data_path, transform=make_classification_eval_transform())
            loader = make_data_loader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=False,
                shuffle=False,
                persistent_workers=False,
            )
            return loader, dataset
        except Exception as e:
            print(f"Failed to load with original method: {e}")
            print("Falling back to multi-format loader...")
            # 경로 추출
            if 'root=' in data_path:
                path_part = data_path.split('root=')[1].split(':')[0]
                return setup_multi_format_loader(path_part, batch_size, num_workers)
    else:
        # 직접 경로가 주어진 경우 다중 형식 로더 사용
        return setup_multi_format_loader(data_path, batch_size, num_workers)


# ---------- feature extractor (기존과 동일) ----------
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
    def extract_concat_probs(self, x):
        x_tokens_list = self.feature_model(x)
        probs_all = []
        for clf in self.classifiers:
            logits = clf(x_tokens_list)
            probs = torch.softmax(logits, dim=-1)
            probs_all.append(probs)
        if not probs_all:
            raise ValueError("[Extractor] No classifiers loaded.")
        return torch.cat(probs_all, dim=-1)


# ---------- 나머지 함수들 (기존과 동일) ----------
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
    normal_idx = None
    
    class_to_idx = getattr(dataset, "class_to_idx", None)
    if isinstance(class_to_idx, dict) and len(class_to_idx) > 0:
        for name, idx in class_to_idx.items():
            low = name.lower()
            if low == "normal":
                normal_idx = idx
                break

    def to_binary(lbl: int) -> int:
        if class_to_idx is None or normal_idx is None:
            return 0 if int(lbl) == 0 else 1
        return 0 if int(lbl) == int(normal_idx) else 1

    return to_binary


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

    accuracy = div(tp + tn, N)
    ppv = div(tp, tp + fp)
    sensitivity = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    npv = div(tn, tn + fn)
    prevalence = div(tp + fn, N)
    precision = ppv

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

    loader, dataset = setup_external_loader(args.test_dataset, batch_size=batch_size, num_workers=num_workers)

    to_binary = build_binary_label_mapper(dataset)

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
            probs_concat = extractor.extract_concat_probs(image)
            feats_list.append(probs_concat.cpu().numpy())

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

    if labels_bin.size > 0:
        report = compute_binary_summary(labels_bin, preds)
        print("\n=== Binary Report (normal=0, abnormal=1) ===")
        print(f"Accuracy:   {report['accuracy']*100:.2f}%")
        print(f"PPV:        {report['ppv']*100:.2f}%")
        print(f"TP:         {report['TP']}")
        print(f"TN:         {report['TN']}")
        print(f"FP:         {report['FP']}")
        print(f"FN:         {report['FN']}")
        print(f"Prevalence: {report['prevalence']*100:.2f}%")
        print(f"Precision:  {report['precision']*100:.2f}%")
        print(f"Sensitivity:{report['sensitivity']*100:.2f}%")
        print(f"Specificity:{report['specificity']*100:.2f}%")
        print(f"NPV:        {report['npv']*100:.2f}%")
    else:
        print("\n[Info] No labels found; only predictions available.")

    os.makedirs(args.outdir, exist_ok=True)
    if args.save_features:
        np.save(os.path.join(args.outdir, "features.npy"), features)
        np.save(os.path.join(args.outdir, "labels_binary.npy"), labels_bin)
    if args.save_preds:
        np.save(os.path.join(args.outdir, "svm_preds.npy"), preds)

    try:
        import pandas as pd
        df = pd.DataFrame({"index": np.arange(len(preds)), "label_bin": labels_bin, "svm_pred": preds})
        df.to_csv(os.path.join(args.outdir, "per_sample_binary.csv"), index=False)
    except Exception as e:
        print(f"[Warn] CSV save failed: {e}")

    return {"svm_predictions": preds, "svm_features": features, "labels_binary": labels_bin}


def build_argparser():
    ap = get_args_parser(description="DINOv2 K-Model + SVM (Multi-format images)")
    ap.add_argument("--svm-model-path", type=str, required=True, help="Path to joblib SVM pickle")
    ap.add_argument("--pretrained-linear-list", type=str, default=None, help="Comma-separated ckpt paths (order preserved)")
    ap.add_argument("--pretrained-linear-glob", type=str, default=None, help="Glob for ckpts")
    ap.add_argument("--require-exact-num-models", type=int, default=13, help="Require exact K heads (set -1 to disable)")
    ap.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--avg-pool", action="store_true", default=True)
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
    print("\n[Inference completed] Multi-format images → 13×3 probs → SVM → binary metrics printed.")


if __name__ == "__main__":
    main()
