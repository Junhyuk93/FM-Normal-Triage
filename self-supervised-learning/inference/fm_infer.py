import os
import glob
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from functools import partial

from model_utils import ModelWithIntermediateLayers
from setup import setup_and_build_model, get_args_parser
from multi_preprocessing import ToTensor
from Weak_set import Chest_Single_Data_Generator_dinov2

from data import make_data_loader, make_dataset
from data.transforms import make_classification_eval_transform



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
    예) 'classifiers_dict.classifier_4_blocks_avgpool_True_lr_0_00003.' 형태 등
    """
    for k in state_dict.keys():
        if k.endswith('linear.weight') and 'classifier' in k:
            # remove trailing 'linear.weight'
            return k[: -len('linear.') ]  # 남기는 건 '...linear.' 앞까지
    # fallback: 기존 로직이 필요한 케이스 대비
    return None

def load_linear_from_ckpt(classifier: LinearClassifier, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cuda")
    full_state_dict = ckpt["model"] if "model" in ckpt else ckpt
    prefix = _infer_linear_prefix(full_state_dict)
    if prefix is None:
        # 기존 방식(고정 prefix)을 쓰는 ckpt만 있는 경우를 대비한 매우 보수적인 폴백
        # 필요 시 여기 문자열을 수정하세요.
        raise ValueError(f"Cannot infer classifier prefix automatically in: {ckpt_path}")
    # prefix 예: 'classifiers_dict.classifier_4_blocks_avgpool_True_lr_0_00003.'
    state_dict = {k.replace(prefix, ''): v for k, v in full_state_dict.items() if k.startswith(prefix)}
    classifier.load_state_dict(state_dict, strict=False)
    return classifier


def setup_linear_classifier(sample_output, num_classes=3, n_blocks=4, avg_pool=True):
    out_dim = create_linear_input(sample_output, n_blocks, avg_pool).shape[1]
    classifier = LinearClassifier(out_dim, n_blocks, avg_pool, num_classes).cuda()
    return classifier

# ------------------ 데이터 로딩 ------------------
def setup_external_loader2(data_path, batch_size=1, num_workers=8):
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

# ------------------ 앙상블 모델 ------------------
class DINOv2_Ensembler(nn.Module):
    """
    1) feature backbone 1개 + 여러 linear heads (권장)
    2) 또는 backbone 자체도 여러 개 (느림)
    둘 다 지원. 주어진 리스트 길이에 따라 동작.
    """
    def __init__(self, args, n_blocks=4, avg_pool=True,
                 backbone_ckpts=None, linear_ckpts=None):
        super().__init__()
        self.n_blocks = n_blocks
        self.avg_pool = avg_pool
        self.num_classes = getattr(args, "training_num_classes", 3)
        self.use_amp = True

        # 공통 backbone을 하나 만든다 (선호)
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

        # linear heads
        self.classifiers = nn.ModuleList()
        if linear_ckpts:
            for p in linear_ckpts:
                clf = setup_linear_classifier(sample_output, num_classes=self.num_classes,
                                              n_blocks=self.n_blocks, avg_pool=self.avg_pool)
                load_linear_from_ckpt(clf, p)
                self.classifiers.append(clf)
        # backbone까지 여러 개인 경우(옵션)
        self.backbones = []
        if backbone_ckpts and len(backbone_ckpts) > 1:
            # 첫 번째는 이미 로드됨. 나머지를 저장(아주 큰 메모리/시간 소모 가능)
            for b in backbone_ckpts[1:]:
                # 개별 args를 복제해서 교체하기 애매하므로, 여기선 동일 args에 ckpt만 바꾸고 재빌드.
                # setup_and_build_model이 args.pretrained_weights를 참조한다면,
                # 외부에서 미리 args를 바꿔 호출하도록 하거나, 아래처럼 임시로 덮어씌운 뒤 빌드.
                prev = args.pretrained_weights
                args.pretrained_weights = b
                model_i, _dtype_i = setup_and_build_model(args)
                args.pretrained_weights = prev
                model_i = model_i.cuda().eval()
                fm_i = ModelWithIntermediateLayers(model_i, self.n_blocks, self.autocast_ctx).eval()
                self.backbones.append(fm_i)

        # 최소 1개 이상의 예측 경로가 있어야 함
        self.num_members = max(len(self.classifiers), 1)
        self.use_multi_backbone = (len(self.backbones) > 0)

    @torch.no_grad()
    def forward(self, x):
        logits_list = []

        # 1) 공통 backbone + 여러 linear
        feats = self.feature_model(x)
        for clf in self.classifiers:
            logits_list.append(clf(feats))

        # 2) 여러 backbone(있다면) + (linear가 있다면 그 linear / 없다면 linear 없이 불가)
        if self.use_multi_backbone:
            if len(self.classifiers) == 0:
                raise ValueError("Multiple backbones provided but no linear heads to map features to logits.")
            for fm in self.backbones:
                feats_i = fm(x)
                for clf in self.classifiers:
                    logits_list.append(clf(feats_i))

        # 최소 1개는 있어야 함
        if len(logits_list) == 0:
            raise ValueError("No ensemble members found. Provide linear_ckpts and/or backbone_ckpts.")

        # (N, C) 로짓 평균
        logits_stack = torch.stack(logits_list, dim=0)  # (M, N, C)
        mean_logits = logits_stack.mean(dim=0)          # (N, C)
        probs = torch.softmax(mean_logits, dim=1)       # (N, C)
        return probs

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

def main(args):
    # 확장 인자 추가 (기존 파서 기반으로)
    # 이미 get_args_parser를 썼다면, 아래처럼 덧붙여도 됨.
    # 단, 이 파일은 단독 실행을 가정.
    pass

if __name__ == "__main__":
    # 원래 파서 불러오기
    args_parser = get_args_parser(description="DINOv2 3-Class Ensemble Inference")
    # 추가 인자
    args_parser.add_argument("--pretrained-linear-list", type=str, default=None,
                             help="Comma-separated list of linear ckpt paths.")
    args_parser.add_argument("--pretrained-linear-glob", type=str, default=None,
                             help="Glob pattern for linear ckpts (e.g., '/path/to/*.pth').")
    args_parser.add_argument("--pretrained-weights-list", type=str, default=None,
                             help="Comma-separated list of backbone ckpt paths.")
    args_parser.add_argument("--pretrained-weights-glob", type=str, default=None,
                             help="Glob pattern for backbone ckpts.")

    args = args_parser.parse_args()

    # 데이터 로더
    externalloader, externalset = setup_external_loader2(
        args.test_dataset,
        batch_size=args.batch_size
    )

    # 리스트/글롭 해석
    linear_list = []
    linear_list += _parse_list_arg(getattr(args, "pretrained_linear_list", None))
    linear_list += _parse_glob_arg(getattr(args, "pretrained_linear_glob", None))
    linear_list = [p for p in linear_list if os.path.exists(p)]

    backbone_list = []
    backbone_list += _parse_list_arg(getattr(args, "pretrained_weights_list", None))
    backbone_list += _parse_glob_arg(getattr(args, "pretrained_weights_glob", None))
    backbone_list = [p for p in backbone_list if os.path.exists(p)]
    # 기본 backbone 1개는 기존 인자를 사용
    if len(backbone_list) == 0 and getattr(args, "pretrained_weights", None):
        backbone_list = [args.pretrained_weights]

    # 앙상블 모델 초기화
    #model = DINOv2_Ensembler(
    #    args=args,
    #    n_blocks=4,
    #    avg_pool=True,
    #    backbone_ckpts=backbone_list,
    #    linear_ckpts=linear_list if len(linear_list) > 0 else [args.pretrained_linear]
    #).eval()

    model = DINOv2_Ensembler(
    args,
    n_blocks=4,
    avg_pool=True,
    backbone_ckpts=backbone_list,
    linear_ckpts=linear_list
    ).eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for image, label in tqdm(externalloader):
            image = image.cuda(non_blocking=True)
            probs = model(image)                 # (B, C)
            all_probs.append(probs.squeeze(0).cpu().numpy())
            all_labels.append(label.item())

    all_probs = np.array(all_probs)             # (N, C)
    all_labels = np.array(all_labels)           # (N,)
    preds = np.argmax(all_probs, axis=1)
    accuracy = np.mean(preds == all_labels)

    print(f"\n[Ensemble] Members: {len(linear_list) if len(linear_list)>0 else 1} linear"
          f"{' x ' + str(len(backbone_list)) + ' backbones' if len(backbone_list)>1 else ''}")
    if len(linear_list) > 0:
        print("Linear ckpts used:")
        for p in linear_list:
            print(" -", p)
    if len(backbone_list) > 0:
        print("Backbone ckpts used:")
        for p in backbone_list:
            print(" -", p)

    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

    # per-class accuracy
    num_classes = getattr(args, "training_num_classes", 3)
    correct_per_class = np.zeros(num_classes, dtype=np.int64)
    total_per_class = np.zeros(num_classes, dtype=np.int64)

    for pred, label in zip(preds, all_labels):
        total_per_class[label] += 1
        if pred == label:
            correct_per_class[label] += 1

    class_names = {0: 'normal', 1: 'target', 2: 'others'}
    print("\nPer-class Accuracy:")
    for cls in range(num_classes):
        cls_name = class_names.get(cls, str(cls))
        acc = 100.0 * (correct_per_class[cls] / total_per_class[cls]) if total_per_class[cls] > 0 else 0.0
        print(f"{cls_name:>7}: {acc:.2f}%")
