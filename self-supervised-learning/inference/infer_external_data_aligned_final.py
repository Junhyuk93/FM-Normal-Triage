import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import glob

from multi_preprocessing import ToTensor
from Weak_set import Chest_Single_Data_Generator_dinov2

from setup import setup_and_build_model, get_args_parser
from model_utils import ModelWithIntermediateLayers


import os
import torch
import torch.nn as nn
import numpy as np
from model_utils import ModelWithIntermediateLayers
from setup import setup_and_build_model
from main import create_linear_input, setup_linear_classifiers


class All_DINOv2_Models(nn.Module):
    def __init__(self, model_ckpt_paths, classifier_ckpt_paths, args):
        super().__init__()
        self.n_blocks = 4
        self.avg_pool = True

        self.models = nn.ModuleList()
        self.classifiers = nn.ModuleList()

        for model_path, classifier_path in zip(model_ckpt_paths, classifier_ckpt_paths):
            # 1. setup DINOv2 backbone
            model, autocast_dtype = setup_and_build_model(args)
            model = model.cuda().eval()

            feature_model = ModelWithIntermediateLayers(
                model,
                self.n_blocks,
                lambda **kwargs: torch.cuda.amp.autocast(enabled=True, dtype=autocast_dtype)
            )
            feature_model.eval()

            # 2. dummy forward to determine output dim
            dummy_input = torch.randn(1, 3, 1024, 1024).cuda()
            sample_output = feature_model(dummy_input)
            classifier = setup_linear_classifiers(
                sample_output,
                num_classes=3,
                n_blocks=self.n_blocks,
                avg_pool=self.avg_pool
            )

            # 3. load classifier weights
            ckpt = torch.load(classifier_path, map_location="cuda")
            full_state_dict = ckpt["model"]
            prefix = f'classifiers_dict.classifier_{self.n_blocks}_blocks_avgpool_{self.avg_pool}_lr_0_00003.'
            state_dict = {
                k.replace(prefix, ''): v
                for k, v in full_state_dict.items()
                if k.startswith(prefix)
            }
            classifier.load_state_dict(state_dict)

            self.models.append(feature_model)
            self.classifiers.append(classifier)

    def forward(self, x):
        outputs = []
        for model, classifier in zip(self.models, self.classifiers):
            with torch.no_grad():
                feats = model(x)
                logits = classifier(feats)
                probs = torch.softmax(logits, dim=1)
                outputs.append(probs.squeeze(0).cpu().numpy())  # [3]
        return outputs  # list of 13 arrays, each of shape [3]


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=2):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)
    
def setup_linear_classifiers(sample_output, num_classes=2, n_blocks=None, avg_pool=None):
    out_dim = create_linear_input(sample_output, use_n_blocks=n_blocks, use_avgpool=avg_pool).shape[1]
    linear_classifier = LinearClassifier(
                    out_dim, use_n_blocks=n_blocks, use_avgpool=avg_pool, num_classes=num_classes
                )
    linear_classifier = linear_classifier.cuda()
    return linear_classifier


def load_data_from_directory(data_path, label_mapping=None):
    """
    디렉토리 구조에서 이미지 경로와 라벨을 직접 로드
    
    Args:
        data_path: 데이터 루트 경로
        label_mapping: 폴더명을 라벨로 매핑하는 딕셔너리 (예: {'normal': 0, 'abnormal': 1})
                      None이면 폴더명을 알파벳 순서로 정렬해서 자동 매핑
    
    Returns:
        image_paths: 상대 경로 리스트
        labels: 해당하는 라벨 리스트
    """
    image_paths = []
    labels = []
    
    # 지원하는 이미지 확장자 (대소문자 구분 없이)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.dcm', '*.dicom', 
                       '*.JPG', '*.JPEG', '*.PNG', '*.DCM', '*.DICOM']
    
    # 폴더별로 클래스 구조가 있는 경우
    class_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    if class_folders:
        # 클래스별 폴더가 있는 경우
        if label_mapping is None:
            # 자동 매핑: 폴더명을 알파벳 순서로 정렬해서 라벨 생성
            class_folders.sort()
            label_mapping = {folder: idx for idx, folder in enumerate(class_folders)}
        
        print(f"Found class folders: {class_folders}")
        print(f"Label mapping: {label_mapping}")
        
        for class_folder in class_folders:
            class_path = os.path.join(data_path, class_folder)
            
            # 라벨 매핑에서 해당 폴더의 라벨 찾기
            if class_folder in label_mapping:
                class_label = label_mapping[class_folder]
            else:
                # 매핑에 없는 폴더는 기본적으로 others(2)로 분류
                class_label = 2
                print(f"Warning: Folder '{class_folder}' not in label_mapping, assigning to 'others' (label 2)")
            
        for class_folder in class_folders:
            class_path = os.path.join(data_path, class_folder)
            
            # 라벨 매핑에서 해당 폴더의 라벨 찾기
            if class_folder in label_mapping:
                class_label = label_mapping[class_folder]
            else:
                # 매핑에 없는 폴더는 기본적으로 others(2)로 분류
                class_label = 2
                print(f"Warning: Folder '{class_folder}' not in label_mapping, assigning to 'others' (label 2)")
            
            print(f"Processing folder: {class_folder} -> label: {class_label}")
            
            # 폴더 내 모든 파일 확인 (디버깅용)
            all_files = os.listdir(class_path)
            print(f"  All files in {class_folder}: {all_files[:5]}...")  # 처음 5개만 출력
            
            folder_image_count = 0
            
            # 현재 폴더에서 직접 이미지 파일 찾기
            for ext in image_extensions:
                pattern = os.path.join(class_path, ext)
                files = glob.glob(pattern)
                if files:
                    print(f"  Found {len(files)} files with pattern {ext}")
                folder_image_count += len(files)
                
                for file_path in files:
                    rel_path = os.path.relpath(file_path, data_path)
                    image_paths.append(rel_path)
                    labels.append(class_label)
            
            # 하위 폴더들도 재귀적으로 검색
            for item in all_files:
                item_path = os.path.join(class_path, item)
                if os.path.isdir(item_path):
                    print(f"    Searching in subfolder: {item}")
                    subfolder_count = 0
                    for ext in image_extensions:
                        pattern = os.path.join(item_path, ext)
                        files = glob.glob(pattern)
                        if files:
                            print(f"      Found {len(files)} files with pattern {ext}")
                        subfolder_count += len(files)
                        
                        for file_path in files:
                            rel_path = os.path.relpath(file_path, data_path)
                            image_paths.append(rel_path)
                            labels.append(class_label)
                    
                    print(f"    Subfolder {item}: {subfolder_count} images")
                    folder_image_count += subfolder_count
            
            print(f"  Total: {folder_image_count} images in {class_folder}")
    
    else:
        # 단일 폴더에 모든 이미지가 있는 경우 (라벨을 파일명에서 추출해야 함)
        print("Single folder structure detected. You may need to modify label extraction logic.")
        
        for ext in image_extensions:
            pattern = os.path.join(data_path, ext)
            files = glob.glob(pattern)
            
            for file_path in files:
                rel_path = os.path.relpath(file_path, data_path)
                image_paths.append(rel_path)
                
                # 여기서 파일명 기반으로 라벨을 추출하는 로직을 추가
                # 3개 클래스: normal(0), target(1), others(2)
                filename = os.path.basename(file_path).lower()
                if 'normal' in filename:
                    labels.append(0)
                elif 'target' in filename:
                    labels.append(1)
                elif 'others' in filename:
                    labels.append(2)
                else:
                    labels.append(2)  # 기본값을 others로 설정
    
    print(f"Total images found: {len(image_paths)}")
    print(f"Label distribution: {dict(zip(*torch.unique(torch.tensor(labels), return_counts=True)))}")
    
    return image_paths, labels


def setup_external_loader(data_path, batch_size=1, num_workers=8, label_mapping=None):
    """
    pickle 파일 없이 직접 경로에서 데이터 로드
    
    Args:
        data_path: 이미지가 있는 루트 디렉토리
        batch_size: 배치 크기
        num_workers: 데이터 로더 워커 수
        label_mapping: 클래스 폴더명을 라벨로 매핑하는 딕셔너리
    """
    # 이미지 경로와 라벨을 직접 로드
    image_paths, labels = load_data_from_directory(data_path, label_mapping)
    
    transform_test = transforms.Compose([ToTensor()])
    dataset = Chest_Single_Data_Generator_dinov2(
        (1024, 1024), 
        data_path, 
        image_paths, 
        labels, 
        transform=transform_test
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, dataset


def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    externalloader, externalset = setup_external_loader(
        data_path='/workspace/dataset/v1/valid',
        batch_size=args.batch_size,
        label_mapping=label_mapping
    )

    model, autocast_dtype = setup_and_build_model(args)

    n_last_blocks = 4
    avg_pool = True
    lr_ = '0_00003'
    from functools import partial
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)

    sample_output = feature_model(externalset[0][0].unsqueeze(0).to(DEVICE))
    
    linear_classifier = setup_linear_classifiers(
        sample_output,
        args.training_num_classes,
        n_last_blocks,
        avg_pool
    )

    checkpoint = torch.load(args.pretrained_linear, map_location=DEVICE)
    full_state_dict = checkpoint["model"]
    prefix = f'classifiers_dict.classifier_{n_last_blocks}_blocks_avgpool_{avg_pool}_lr_{lr_}.'
    state_dict = {
    k.replace(prefix, ''): v
    for k, v in full_state_dict.items()
    if k.startswith(prefix)
    }

    linear_classifier.load_state_dict(state_dict)

    feature_model.eval()
    linear_classifier.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for image, _ ,label, _ in tqdm(externalloader):
            image = image.to(DEVICE)
            features = feature_model(image)
            logits = linear_classifier(features)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(label.cpu())
            probs = torch.softmax(logits, dim=1)  # 확률로 변환
            print(probs)

    print(f"all_preds : {all_preds}")
    print(f"all_labels : {all_labels}")
    print("Inference 완료. 결과는 all_preds, all_labels에 저장됨.")


if __name__ == "__main__":
    description = "DINOv2 External Chest Dataset Inference Script"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    main(args)