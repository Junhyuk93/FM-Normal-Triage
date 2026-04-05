import os
import glob
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from functools import partial
import joblib
import argparse

from model_utils import ModelWithIntermediateLayers
from setup import setup_and_build_model, get_args_parser
from multi_preprocessing import ToTensor
from Weak_set import Chest_Single_Data_Generator_dinov2

from data import make_data_loader, make_dataset
from data.transforms import make_classification_eval_transform


# ------------------ 모델 순서 정의 ------------------
def get_model_order():
    """
    학습 시와 동일한 모델 순서를 정의
    SVM 학습 시 사용한 순서와 정확히 일치해야 함
    """
    return [
        "normal-triage-FM",      # 0번째 위치
        "normal-triage-FM-v2",   # 1번째 위치  
        "normal-triage-FM-v3",   # 2번째 위치
        "normal-triage-FM-v4",   # 3번째 위치
        "normal-triage-FM-v5",   # 4번째 위치
        "normal-triage-FM-v6",   # 5번째 위치
        "normal-triage-FM-v7",   # 6번째 위치
        "normal-triage-FM-v8",   # 7번째 위치
        "normal-triage-FM-v9",   # 8번째 위치
        "normal-triage-FM-v10",  # 9번째 위치
        "normal-triage-FM-v11",  # 10번째 위치
        "normal-triage-FM-v12",  # 11번째 위치
        "normal-triage-FM-v13"   # 12번째 위치
    ]

def sort_linear_checkpoints(linear_list):
    """
    checkpoint 경로를 올바른 순서로 정렬
    """
    expected_order = get_model_order()
    sorted_paths = [None] * len(expected_order)
    
    # 각 경로에서 모델명 추출하여 올바른 위치에 배치
    for path in linear_list:
        for i, expected_name in enumerate(expected_order):
            if expected_name in path:
                if sorted_paths[i] is not None:
                    print(f"Warning: Duplicate model found for {expected_name}")
                sorted_paths[i] = path
                break
        else:
            print(f"Warning: Unknown model path: {path}")
    
    # 누락된 모델 체크
    missing_models = []
    for i, (path, name) in enumerate(zip(sorted_paths, expected_order)):
        if path is None:
            missing_models.append(f"{i}: {name}")
    
    if missing_models:
        raise ValueError(f"Missing models:\n" + "\n".join(missing_models))
    
    return sorted_paths


# ------------------ Feature 생성 ------------------
def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    # intermediate_output element: (tokens, class_token)
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat((output, torch.mean(intermediate_output[-1][0], dim=1)), dim=-1)
        output = output.reshape(output.shape[0], -1)
    return output.float()

class LinearClassifier(nn.Module):
    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool

    def forward(self, x_tokens_list):
        x = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(x)

def _infer_linear_prefix(state_dict):
    """
    ckpt['model'] 안에서 *.linear.weight 키를 찾아 공통 prefix를 자동 추정.
    """
    for k in state_dict.keys():
        if k.endswith('linear.weight') and 'classifier' in k:
            return k[: -len('linear.')]
    return None

def load_linear_from_ckpt(classifier: LinearClassifier, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cuda")
    full_state_dict = ckpt["model"] if "model" in ckpt else ckpt
    prefix = _infer_linear_prefix(full_state_dict)
    if prefix is None:
        raise ValueError(f"Cannot infer classifier prefix automatically in: {ckpt_path}")

    state_dict = {k.replace(prefix, ''): v for k, v in full_state_dict.items() if k.startswith(prefix)}
    classifier.load_state_dict(state_dict, strict=False)
    return classifier

def setup_linear_classifier(sample_output, num_classes=3, n_blocks=4, avg_pool=True):
    out_dim = create_linear_input(sample_output, n_blocks, avg_pool).shape[1]
    classifier = LinearClassifier(out_dim, n_blocks, avg_pool, num_classes).cuda()
    return classifier

# ------------------ 데이터 로딩 ------------------
def setup_external_loader(data_path, batch_size=1, num_workers=8):
    dataset = make_dataset(
        dataset_str=data_path,
        transform=make_classification_eval_transform(),
    )
    loader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
    )
    return loader, dataset

# ------------------ FM Feature Extractor (수정됨) ------------------
class DINOv2_MultiModelExtractor(nn.Module):
    """
    13개의 개별 DINOv2 모델에서 각각 3-class 확률을 추출하여 SVM 입력으로 사용
    """
    def __init__(self, args, n_blocks=4, avg_pool=True, linear_ckpts=None):
        super().__init__()
        self.n_blocks = n_blocks
        self.avg_pool = avg_pool
        self.num_classes = getattr(args, "training_num_classes", 3)
        self.use_amp = True

        # backbone 모델 설정 (공통)
        base_model, autocast_dtype = setup_and_build_model(args)
        base_model = base_model.cuda().eval()
        self.autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)

        self.feature_model = ModelWithIntermediateLayers(
            base_model,
            self.n_blocks,
            self.autocast_ctx
        ).eval()

        # 더미 입력으로 크기 파악 후 linear 초기화
        with torch.no_grad():
            dummy = torch.randn(1, 3, 1024, 1024).cuda()
            sample_output = self.feature_model(dummy)

        # *** 중요: checkpoint를 올바른 순서로 정렬 ***
        if linear_ckpts:
            sorted_ckpts = sort_linear_checkpoints(linear_ckpts)
            self.model_order = get_model_order()
            
            # 순서 검증을 위한 정보 출력
            print(f"\n=== Model Loading Order ===")
            for i, (path, name) in enumerate(zip(sorted_ckpts, self.model_order)):
                print(f"Position {i:2d}: {name}")
                print(f"           -> {os.path.basename(path)}")
            print(f"===========================\n")
        else:
            sorted_ckpts = []
            self.model_order = []

        # 13개의 개별 linear classifiers (정확한 순서로)
        self.classifiers = nn.ModuleList()
        for i, path in enumerate(sorted_ckpts):
            clf = setup_linear_classifier(sample_output, num_classes=self.num_classes,
                                          n_blocks=self.n_blocks, avg_pool=self.avg_pool)
            load_linear_from_ckpt(clf, path)
            self.classifiers.append(clf)
            print(f"Loaded model {i:2d}: {self.model_order[i]} from {os.path.basename(path)}")

        self.num_models = len(self.classifiers)
        print(f"\nLoaded {self.num_models} individual models in correct order for SVM feature extraction")

    @torch.no_grad()
    def extract_svm_features(self, x):
        """
        각 모델에서 3-class 확률을 **정확한 순서로** 추출하여 연결
        """
        # backbone에서 intermediate features 추출
        x_tokens_list = self.feature_model(x)

        # 각 모델에서 3-class 확률 추출 (순서 보장됨)
        all_probs = []
        for i, clf in enumerate(self.classifiers):
            logits = clf(x_tokens_list)  # (batch_size, 3)
            # softmax를 적용하여 확률로 변환
            probs = torch.softmax(logits, dim=-1)  # (batch_size, 3)
            all_probs.append(probs)

        # 모든 확률을 연결: (batch_size, 13*3) = (batch_size, 39)
        if len(all_probs) > 0:
            combined_probs = torch.cat(all_probs, dim=-1)  # (batch_size, 39)
        else:
            raise ValueError("No classifiers loaded!")

        print(f"Debug: Combined probs shape: {combined_probs.shape}")  # 디버그용
        return combined_probs.cpu().numpy()

    @torch.no_grad()
    def extract_individual_logits(self, x):
        """
        개별 모델들의 logits을 리스트로 반환 (분석용)
        """
        x_tokens_list = self.feature_model(x)
        logits_list = []

        for clf in self.classifiers:
            logits = clf(x_tokens_list)
            probs = torch.softmax(logits, dim=-1)
            logits_list.append(probs.cpu().numpy())

        return logits_list

# ------------------ 유틸: 리스트/글롭 파싱 ------------------
def _parse_list_arg(comma_list: str):
    if not comma_list:
        return []
    return [s.strip() for s in comma_list.split(",") if s.strip()]

def _parse_glob_arg(glob_str: str):
    if not glob_str:
        return []
    paths = sorted(glob.glob(glob_str))
    return paths

# ------------------ SVM Inference Pipeline (수정됨) ------------------
def run_fm_svm_inference(args):
    """
    FM 모델로 feature 추출 후 SVM으로 최종 분류
    """
    print("Setting up FM Feature Extractor...")

    # 데이터 로더 설정
    dataloader, dataset = setup_external_loader(
        args.test_dataset,
        batch_size=args.batch_size
    )

    # linear checkpoint 리스트 파싱
    linear_list = []
    linear_list += _parse_list_arg(getattr(args, "pretrained_linear_list", None))
    linear_list += _parse_glob_arg(getattr(args, "pretrained_linear_glob", None))
    linear_list = [p for p in linear_list if os.path.exists(p)]

    if len(linear_list) == 0 and hasattr(args, 'pretrained_linear'):
        linear_list = [args.pretrained_linear]

    print(f"Found {len(linear_list)} linear checkpoints:")
    for i, path in enumerate(linear_list):
        print(f"  {i+1}. {path}")  # 전체 경로 표시

    # *** 중요: 13개 모델이 모두 있는지 확인 ***
    expected_count = 13
    if len(linear_list) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} models, got {len(linear_list)}")

    # FM Feature Extractor 초기화 (순서 보장됨)
    feature_extractor = DINOv2_MultiModelExtractor(
        args=args,
        n_blocks=4,
        avg_pool=True,
        linear_ckpts=linear_list  # 내부에서 올바른 순서로 정렬됨
    ).eval()

    # SVM 모델 로드
    if not os.path.exists(args.svm_model_path):
        raise FileNotFoundError(f"SVM model not found: {args.svm_model_path}")

    print(f"Loading SVM model from: {args.svm_model_path}")
    svm_model = joblib.load(args.svm_model_path)

    # Feature 추출 및 예측
    print("Extracting 13×3=39 probability features for SVM...")
    all_svm_features = []
    all_labels = []
    all_individual_logits = []

    with torch.no_grad():
        for image, label in tqdm(dataloader):
            image = image.cuda(non_blocking=True)

            # 13개 모델에서 각각 3-class 확률 추출 → SVM 입력용 39차원 feature
            svm_features = feature_extractor.extract_svm_features(image)
            all_svm_features.append(svm_features.squeeze(0))

            # 개별 모델 logits도 저장 (분석용)
            individual_logits = feature_extractor.extract_individual_logits(image)
            all_individual_logits.append(individual_logits)

            all_labels.append(label.item())

    # Feature 배열로 변환
    all_svm_features = np.array(all_svm_features)  # (N, 39) - 13 models × 3 classes
    all_labels = np.array(all_labels)

    print(f"SVM input features shape: {all_svm_features.shape}")
    print(f"Expected: (N, {len(linear_list) * 3}) where N is number of samples")

    # 차원 확인
    expected_dim = len(linear_list) * 3  # 13개 모델에서 각각 3개 확률
    if all_svm_features.shape[1] != expected_dim:
        print(f"ERROR: Feature dimension mismatch!")
        print(f"Expected: {expected_dim}, Got: {all_svm_features.shape[1]}")
        return None

    # SVM으로 예측 (normal=0, abnormal=1)
    print("Running SVM inference...")
    svm_predictions = svm_model.predict(all_svm_features)

    # 개별 모델들의 3-class 예측도 계산 (참고용)
    individual_predictions = []
    for i in range(len(all_individual_logits)):
        sample_preds = []
        for model_idx in range(len(linear_list)):
            model_logits = all_individual_logits[i][model_idx].squeeze(0)  # (3,)
            pred = np.argmax(model_logits)
            sample_preds.append(pred)
        individual_predictions.append(sample_preds)

    individual_predictions = np.array(individual_predictions)  # (N, 13)

    # 결과 분석
    if len(all_labels) > 0:  # 라벨이 있는 경우
        # SVM 결과 분석 (normal=0, abnormal=1)
        print(f"\n=== SVM Results (Normal=0, Abnormal=1) ===")
        svm_accuracy = np.mean(svm_predictions == all_labels)
        print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")

        # SVM 클래스별 정확도
        svm_class_names = {0: 'normal', 1: 'abnormal'}
        for cls in [0, 1]:
            cls_mask = (all_labels == cls)
            if np.sum(cls_mask) > 0:
                cls_acc = np.mean(svm_predictions[cls_mask] == all_labels[cls_mask])
                cls_name = svm_class_names.get(cls, str(cls))
                print(f"SVM {cls_name:>8}: {cls_acc * 100:.2f}%")

        # 개별 모델들의 3-class 성능도 계산 (참고용)
        print(f"\n=== Individual Model Performance (3-class) ===")
        model_order = get_model_order()

        for model_idx in range(len(linear_list)):
            model_preds = individual_predictions[:, model_idx]
            model_acc = np.mean(model_preds == all_labels)
            model_name = model_order[model_idx] if model_idx < len(model_order) else f"Model_{model_idx}"
            print(f"{model_name:>20}: {model_acc * 100:.2f}%")

        # 전체 개별 모델 평균 성능
        all_individual_acc = []
        for model_idx in range(len(linear_list)):
            model_preds = individual_predictions[:, model_idx]
            model_acc = np.mean(model_preds == all_labels)
            all_individual_acc.append(model_acc)

        avg_individual_acc = np.mean(all_individual_acc)
        print(f"Average individual model accuracy: {avg_individual_acc * 100:.2f}%")

    else:
        print(f"\n=== Results ===")
        print("No ground truth labels available for accuracy calculation.")
        for i, svm_pred in enumerate(svm_predictions):
            result = "normal" if svm_pred == 0 else "abnormal"
            print(f"Sample {i+1}: {result}")

    return {
        'svm_predictions': svm_predictions,
        'individual_predictions': individual_predictions,
        'svm_features': all_svm_features,
        'individual_logits': all_individual_logits,
        'labels': all_labels
    }

def main():
    # 기존 파서에 SVM 관련 인자 추가
    args_parser = get_args_parser(description="DINOv2 13-Model + SVM Binary Classification")

    # SVM 모델 경로 (고정)
    args_parser.add_argument("--svm-model-path", type=str, 
                             default="/workspace/inference/svm_weight.pickle",
                             help="Path to the trained SVM model (.pickle or .pkl file)")

    # 기존 linear 관련 인자들
    args_parser.add_argument("--pretrained-linear-list", type=str, default=None,
                             help="Comma-separated list of 13 linear ckpt paths.")
    args_parser.add_argument("--pretrained-linear-glob", type=str, default=None,
                             help="Glob pattern for 13 linear ckpts (e.g., '/path/to/*.pth').")

    args = args_parser.parse_args()

    # 필수 인자 체크
    if not hasattr(args, 'test_dataset') or not args.test_dataset:
        raise ValueError("--test-dataset is required")

    # SVM 모델 경로 고정 사용
    if not args.svm_model_path:
        args.svm_model_path = "/workspace/inference/svm_weight.pickle"
    
    print(f"Using SVM model: {args.svm_model_path}")

    # inference 실행
    results = run_fm_svm_inference(args)

    print(f"\nInference completed!")
    print(f"Final SVM output: Normal/Abnormal classification")
    print(f"Individual models: 13 × 3-class probabilities used as SVM input (순서: 입력된 명령어 순서)")
    return results

if __name__ == "__main__":
    results = main()
