import numpy as np
from functools import partial
import os
import glob
import joblib

import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.special import expit, softmax

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


def setup_external_loader(data_path, batch_size=1, num_workers=8):
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
        return torch.cat(probs_all, dim=-1)


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
        if class_to_idx is None or normal_idx is None:
            return 0 if int(lbl) == 0 else 1
        return 0 if int(lbl) == int(normal_idx) else 1

    return to_binary


def compute_binary_summary(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 0) & (y_pred == 0)).sum())
    tn = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 1) & (y_pred == 0)).sum())
    fn = int(((y_true == 0) & (y_pred == 1)).sum())
    N = tp + tn + fp + fn

    def div(a, b):
        return (a / b) if b != 0 else float("nan")

    accuracy    = div(tp + tn, N)
    ppv         = div(tp, tp + fp)
    sensitivity = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    npv         = div(tn, tn + fn)
    prevalence  = div(tp + fn, N)
    precision   = ppv

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

    # SVM 확률 추출
    if hasattr(svm, 'decision_function'):
        decision_scores = svm.decision_function(features)
        threshold = 0.29
        preds = (decision_scores >= threshold).astype(int)
        if len(decision_scores.shape) == 1:
            prob_class_1 = expit(decision_scores)
            prob_class_0 = 1 - prob_class_1
            svm_probabilities = np.stack([prob_class_0, prob_class_1], axis=1)
        else:
            svm_probabilities = softmax(decision_scores, axis=1)
    elif hasattr(svm, 'predict_proba'):
        svm_probabilities = svm.predict_proba(features)
    else:
        print("[Warning] SVM doesn't support probability. Using one-hot encoding.")
        svm_probabilities = np.zeros((len(preds), 2))
        svm_probabilities[np.arange(len(preds)), preds] = 1.0

    # 파일 경로 추출
    all_paths = []
    if hasattr(dataset, 'samples') and dataset.samples:
        all_paths = [os.path.abspath(sample[0]) for sample in dataset.samples]
        print(f"[Info] Found {len(all_paths)} paths from dataset.samples")
    elif hasattr(dataset, 'imgs') and dataset.imgs:
        all_paths = [os.path.abspath(img[0]) for img in dataset.imgs]
        print(f"[Info] Found {len(all_paths)} paths from dataset.imgs")
    elif hasattr(dataset, 'img_lst'):
        all_paths = [os.path.abspath(path) for path in dataset.img_lst]
        print(f"[Info] Found {len(all_paths)} paths from dataset.img_lst")
    elif hasattr(dataset, 'root'):
        print(f"[Info] Scanning files from dataset root: {dataset.root}")
        all_paths = []
        for root, dirs, files in os.walk(dataset.root):
            for file in sorted(files):
                if file.lower().endswith(('.dcm', '.png', '.jpg', '.jpeg')):
                    all_paths.append(os.path.abspath(os.path.join(root, file)))
        print(f"[Info] Found {len(all_paths)} files by scanning")

    if len(all_paths) == 0:
        all_paths = [f"unknown_{i}" for i in range(len(preds))]
        print("[Warning] Could not extract file paths from dataset")

    filenames = [os.path.basename(path) for path in all_paths]

    # Binary metrics
    if labels_bin.size > 0:
        import pandas as pd  # pandas를 여기서 import
        
        report = compute_binary_summary(labels_bin, preds)
        print('[품명]: 흉부 X-ray 기반 정상/비정상 분류 인공지능 소프트웨어')
        print('[형명]: CXR:NT-01')
        print('[제조번호]: NT_v1.1')
        print('[제조일자]: 25/10/18')

        print(f"Accuracy: {report['accuracy']*100:.2f}%")
        print(f"PPV: {report['ppv']*100:.2f}%")
        print(f"TP: {report['TP']}")
        print(f"TN: {report['TN']}")
        print(f"FP: {report['FP']}")
        print(f"FN: {report['FN']}")
        print(f"Precision: {report['precision']*100:.2f}%")
        print(f"Sensitivity: {report['sensitivity']*100:.2f}%")
        print(f"Specificity: {report['specificity']*100:.2f}%")
        print(f"NPV: {report['npv']*100:.2f}%")
        
        # AUC 계산
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            # prob_abnormal (클래스 1의 확률)을 사용하여 AUC 계산
            auc_score = roc_auc_score(labels_bin, svm_probabilities[:, 1])
            print(f"\n=== AUC ===")
            print(f"AUC (ROC): {auc_score:.4f}")
            
            # ROC curve 데이터 저장
            fpr, tpr, thresholds = roc_curve(labels_bin, svm_probabilities[:, 1])
            roc_data = pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr,
                'threshold': thresholds
            })
            roc_csv_path = os.path.join(args.outdir, "roc_curve_data.csv")
            roc_data.to_csv(roc_csv_path, index=False)
            print(f"[CSV] ROC curve data saved to: {roc_csv_path}")
        except Exception as e:
            print(f"[Warning] Could not calculate AUC: {e}")

        fp_indices = [i for i, (true_label, pred_label) in enumerate(zip(labels_bin, preds))
                      if true_label == 0 and pred_label == 1]
        fn_indices = [i for i, (true_label, pred_label) in enumerate(zip(labels_bin, preds))
                      if true_label == 1 and pred_label == 0]

        fp_paths = [all_paths[i] for i in fp_indices if i < len(all_paths)]
        with open("FP.txt", "w") as f:
            for path in fp_paths:
                f.write(path + "\n")
        print(f"Saved {len(fp_paths)} FP paths to FP.txt")

        fn_paths = [all_paths[i] for i in fn_indices if i < len(all_paths)]
        with open("FN.txt", "w") as f:
            for path in fn_paths:
                f.write(path + "\n")
        print(f"Saved {len(fn_paths)} FN paths to FN.txt")
    else:
        print("\n[Info] No labels found; only predictions available.")

    # 출력 디렉토리 생성
    os.makedirs(args.outdir, exist_ok=True)
    
    # 상세 CSV 생성
    try:
        import pandas as pd

        detailed_results = []
        for i in range(len(preds)):
            row = {
                'index': i,
                'filepath': all_paths[i] if i < len(all_paths) else f"unknown_{i}",
                'filename': filenames[i] if i < len(filenames) else f"unknown_{i}",
                'true_label': labels_bin[i] if i < len(labels_bin) else -1,
                'predicted_label': preds[i],
                'prob_normal': svm_probabilities[i, 0],
                'prob_abnormal': svm_probabilities[i, 1],
                'confidence': svm_probabilities[i, preds[i]],
            }

            if hasattr(svm, 'decision_function') and len(decision_scores.shape) == 1:
                row['decision_score'] = decision_scores[i]

            detailed_results.append(row)

        df_detailed = pd.DataFrame(detailed_results)

        detailed_csv_path = os.path.join(args.outdir, "detailed_results.csv")
        df_detailed.to_csv(detailed_csv_path, index=False)
        print(f"\n[CSV] Detailed results saved to: {detailed_csv_path}")
        print(f"[CSV] Columns: {list(df_detailed.columns)}")
        print(f"[CSV] Total rows: {len(df_detailed)}")

        print("\n=== Sample Results (first 5) ===")
        display_cols = ['filename', 'true_label', 'predicted_label', 'prob_normal', 'prob_abnormal', 'confidence']
        print(df_detailed[display_cols].head().to_string(index=False))

        print("\n=== Prediction Summary ===")
        print(f"Normal (0) predictions: {(preds == 0).sum()} ({(preds == 0).sum() / len(preds) * 100:.1f}%)")
        print(f"Abnormal (1) predictions: {(preds == 1).sum()} ({(preds == 1).sum() / len(preds) * 100:.1f}%)")
        print(f"Average confidence: {df_detailed['confidence'].mean():.3f}")
        print(f"Min confidence: {df_detailed['confidence'].min():.3f}")
        print(f"Max confidence: {df_detailed['confidence'].max():.3f}")

        # 간단 CSV도 저장
        df_simple = pd.DataFrame({"index": np.arange(len(preds)), "label_bin": labels_bin, "svm_pred": preds})
        df_simple.to_csv(os.path.join(args.outdir, "per_sample_binary_030.csv"), index=False)

    except Exception as e:
        print(f"[Error] Failed to create detailed CSV: {e}")
        import traceback
        traceback.print_exc()

    # Optional saves
    if args.save_features:
        np.save(os.path.join(args.outdir, "features.npy"), features)
        np.save(os.path.join(args.outdir, "labels_binary.npy"), labels_bin)
    if args.save_preds:
        np.save(os.path.join(args.outdir, "svm_preds.npy"), preds)

    return {"svm_predictions": preds, "svm_features": features, "labels_binary": labels_bin}


def build_argparser():
    ap = get_args_parser(description="DINOv2 K-Model + SVM (Binary via folder names)")
    ap.add_argument("--svm-model-path", type=str, required=True, help="Path to joblib SVM pickle")
    ap.add_argument("--pretrained-linear-list", type=str, default=None, help="Comma-separated ckpt paths (order preserved)")
    ap.add_argument("--pretrained-linear-glob", type=str, default=None, help="Glob for ckpts")
    ap.add_argument("--require-exact-num-models", type=int, default=13, help="Require exact K heads (set -1 to disable)")
    ap.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--avg-pool", action="store_true", default=True)
    ap.add_argument("--outdir", type=str, default="./svm_out_AUC_KTL")
    ap.add_argument("--save-features", action="store_true", default=True)
    ap.add_argument("--save-preds", action="store_true", default=True)
    return ap

def main():
    parser = build_argparser()
    args = parser.parse_args()

    # 출력 디렉토리 미리 생성 (run() 안에서도 만들지만 안전하게 한 번 더)
    os.makedirs(args.outdir, exist_ok=True)

    # 메인 실행
    run(args)


if __name__ == "__main__":
    main()

