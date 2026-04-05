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
import cv2

from model_utils import ModelWithIntermediateLayers
from setup import setup_and_build_model, get_args_parser
from multi_preprocessing import ToTensorSingle
from Weak_set import DicomSingleImageDataset

# ------------------ DICOM to PNG 함수 (기존 코드와 동일) ------------------
import SimpleITK as sitk
from skimage import exposure

def resize_and_normalize(arr, out_size, out_channel, percentile=99):
    """
    Args:
        arr: numpy array
        out_size: (tuple) size of output array
        out_channel : channels of output array (duplicated stack)
        percentile : pixel intensity which will be preserved
    Returns:
        numpy ndarray
    """
    img = arr
    np_img = img.astype(np.float32)
    np_img -= np.min(np_img)
    np_img /= np.percentile(np_img, percentile)
    np_img[np_img>1] = 1
    np_img *= (2**8-1)
    np_img = np_img.astype(np.uint8)
    if out_channel > 1:
        np_img = np.stack((np_img,)*out_channel, axis=-1)
    return np_img

def dicom_to_png(path, out_size=1024, out_channel=3, keep_aspect_ratio=True):
    """
    Args:
        path : input dicom file path
        axis : target axis (x=0, y=1)
    Returns:
        None (save .png image)
    """
    if keep_aspect_ratio:
        ### Get pixel information
        spacings = []
        pixels = []
        psizes = []
        dcm = sitk.ReadImage(path)
        for i in range(2):
            spacings.append(dcm.GetSpacing()[i])
            pixels.append(dcm.GetSize()[i])
            psizes.append(spacings[-1] * pixels[-1])
        img = sitk.GetArrayFromImage(dcm)
        img = np.squeeze(img, axis=0)
        ### target (base axis = y axis)
        target_idx = np.where(psizes==np.min(psizes))[0][0]
        target_size = psizes[target_idx]
        ### crop
        target_x = int(target_size / spacings[0])
        target_y = int(target_size / spacings[1])
        offset_x = int((pixels[0] - target_x) * 0.5)
        if offset_x < 0:
            target_x = int((float(pixels[0])/float(target_x))*out_size)
            offset_x = int((out_size-target_x) * 0.5)
            tmp = resize_and_normalize(img[0, :target_y, :], (target_x,out_size), 3)
            result_img = np.zeros((out_size, out_size, 3), dtype='uint8')
            result_img[:, offset_x:offset_x+target_x, :] = tmp
        else:
            result_img = resize_and_normalize(img, (out_size,out_size), out_channel)
    else:
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))
        result_img = resize_and_normalize(img[0], (out_size, out_size), out_channel)
        
    result_img2 = exposure.equalize_hist(result_img)*255
    result_img2 = np.clip(result_img2, 0, 255).astype(np.uint8)
    return result_img2

# ------------------ 기존 DicomSingleImageDataset 재구현 ------------------
from torch.utils.data import Dataset

class DicomSingleImageDataset(Dataset):
    def __init__(self, img_size, input_img_path, mode=None, transform=None):
        self.img_size = img_size
        self.input_img_path = input_img_path  # 단일 파일 경로
        self.mode = mode
        self.transform = transform
    
    def __len__(self):
        return 1
        
    def __getitem__(self, idx):
        # 기존 코드와 동일한 방식으로 DICOM 처리
        temp_image = dicom_to_png(self.input_img_path)
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
        mask = np.zeros((1024, 1024))
        img = cv2.resize(temp_image, (self.img_size[0], self.img_size[1]))
        img = np.expand_dims(img, -1)
        mask = np.expand_dims(mask, -1)
        
        sample = {'image': img, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
            sample['image'] /= 255.
            sample['mask'] /= 255.
            
        return sample['image'], sample['mask']

# ------------------ Feature 생성 ------------------
def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
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

# ------------------ All Models (13개 모델을 담는 클래스) ------------------
class All_Models(nn.Module):
    def __init__(self, args, n_blocks=4, avg_pool=True, linear_ckpts=None):
        super().__init__()
        self.n_blocks = n_blocks
        self.avg_pool = avg_pool
        self.num_classes = getattr(args, "training_num_classes", 3)

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
            dummy = torch.randn(1, 3, 1024, 1024).cuda()  # 3채널 입력으로 변경
            sample_output = self.feature_model(dummy)

        # 13개의 개별 linear classifiers 초기화
        self.classifiers = nn.ModuleList()
        if linear_ckpts:
            for i, ckpt_path in enumerate(linear_ckpts):
                clf = setup_linear_classifier(sample_output, num_classes=self.num_classes,
                                              n_blocks=self.n_blocks, avg_pool=self.avg_pool)
                load_linear_from_ckpt(clf, ckpt_path)
                self.classifiers.append(clf)
                print(f"Loaded model {i+1}: {os.path.basename(ckpt_path)}")

        self.num_models = len(self.classifiers)
        print(f"Total {self.num_models} models loaded for feature extraction")

        # layer freeze (첫 번째 코드와 동일한 동작)
        self.layer_freeze()

    def layer_freeze(self):
        """첫 번째 코드의 layer_freeze와 동일한 동작"""
        for classifier in self.classifiers:
            for i, child in enumerate(classifier.children()):
                if i == 1 or i == 2:
                    for param in child.parameters():
                        param.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        """
        기존 코드의 forward와 유사한 구조
        13개 모델에서 각각 확률값을 추출하여 리스트로 반환
        """
        print(f"Input tensor shape: {x.shape}")
        
        # grayscale을 3채널로 변환 (DINOv2는 3채널 입력 필요)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            print(f"Converted to 3 channels: {x.shape}")
        
        # backbone에서 intermediate features 추출
        try:
            x_tokens_list = self.feature_model(x)
            print(f"Number of intermediate features: {len(x_tokens_list)}")
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return []

        # 각 모델에서 확률값 추출 (기존 코드처럼 .cpu().detach().numpy()[0] 형태)
        results = []
        for i, clf in enumerate(self.classifiers):
            try:
                logits = clf(x_tokens_list)  # (batch_size, 3)
                # softmax를 적용하여 확률로 변환
                probs = torch.softmax(logits, dim=-1)  # (batch_size, 3)
                # 기존 코드와 같은 형태로 변환: .cpu().detach().numpy()[0]
                prob_array = probs.cpu().detach().numpy()[0]  # (3,)
                results.append(prob_array)
                print(f"Model {i+1} probabilities: {prob_array}")
            except Exception as e:
                print(f"Error in model {i+1}: {e}")
                # 오류 발생시 더미 확률값 추가
                results.append(np.array([0.33, 0.33, 0.34]))

        print(f"Total results collected: {len(results)}")
        return results  # 13개 모델의 확률값 리스트 반환

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

# ------------------ Main Function ------------------
def main():
    # 기존 파서에 SVM 관련 인자 추가
    args_parser = get_args_parser(description="DINOv2 13-Model + SVM Binary Classification for Single Image")

    # 단일 이미지 파일 경로
    args_parser.add_argument("--image-path", type=str, required=True,
                             help="Path to the single DICOM file")

    # SVM 모델 경로
    args_parser.add_argument("--svm-model-path", type=str, required=True,
                             help="Path to the trained SVM model (.pickle or .pkl file)")

    # 기존 linear 관련 인자들
    args_parser.add_argument("--pretrained-linear-list", type=str, default=None,
                             help="Comma-separated list of 13 linear ckpt paths.")
    args_parser.add_argument("--pretrained-linear-glob", type=str, default=None,
                             help="Glob pattern for 13 linear ckpts.")

    args = args_parser.parse_args()

    # 필수 인자 체크
    if not args.image_path:
        raise ValueError("--image-path is required")

    if not args.svm_model_path:
        raise ValueError("--svm-model-path is required")

    print("Setting up All_Models for single image inference...")
    print(f"Loading image: {args.image_path}")

    # linear checkpoint 리스트 파싱
    linear_list = []
    if hasattr(args, 'pretrained_linear_list') and args.pretrained_linear_list:
        linear_list += _parse_list_arg(args.pretrained_linear_list)
    if hasattr(args, 'pretrained_linear_glob') and args.pretrained_linear_glob:
        linear_list += _parse_glob_arg(args.pretrained_linear_glob)
    
    # None 값들 제거하고 존재하는 파일만 필터링
    linear_list = [p for p in linear_list if p is not None and os.path.exists(p)]

    print(f"Found {len(linear_list)} linear checkpoints:")
    for i, path in enumerate(linear_list):
        if path is not None:
            print(f"  {i+1}. {os.path.basename(path)}")

    # 13개 모델이 모두 로드되었는지 확인
    if len(linear_list) == 0:
        print("ERROR: No valid linear checkpoint files found!")
        return None
    
    if len(linear_list) != 13:
        print(f"Warning: Expected 13 models but got {len(linear_list)} models")

    # 기존 코드와 동일한 방식으로 데이터셋과 데이터로더 설정
    transform_test = transforms.Compose([
        ToTensorSingle(),
    ])
    
    testset = DicomSingleImageDataset((1024, 1024), args.image_path, transform=transform_test)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    # All_Models 초기화
    model = All_Models(
        args=args,
        n_blocks=4,
        avg_pool=True,
        linear_ckpts=linear_list
    ).eval()
    model.cuda()

    # SVM 모델 로드
    if not os.path.exists(args.svm_model_path):
        raise FileNotFoundError(f"SVM model not found: {args.svm_model_path}")

    print(f"Loading SVM model from: {args.svm_model_path}")
    svm = joblib.load(args.svm_model_path)

    # 기존 코드와 동일한 inference 루프
    print("Running inference on single image...")
    pred_labels = []

    for sample in tqdm(testloader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        pred = model(sample[0].cuda())  # GPU로 이동
        pred_labels.append(pred)

    # 기존 코드와 동일한 후처리
    pred_labels2 = np.array(pred_labels)
    pred_labels2 = pred_labels2.reshape(pred_labels2.shape[0], -1)

    print("SVM input shape:", pred_labels2.shape)
    print("SVM input example:", pred_labels2[0])

    # SVM으로 예측 (기존 코드와 동일)
    new_n_pred = svm.predict(pred_labels2)

    # 결과 출력 (기존 코드와 동일)
    flag = 'normal' if new_n_pred[0] == 0 else 'abnormal'
    print(f"\n=== Result ===")
    print(f"Image: {os.path.basename(args.image_path)}")
    print(f"This image is {flag}")
    print(f"Ground Truth (assumed): normal")
    
    # 정확도 계산 (ground truth가 normal이라고 가정)
    is_correct = (new_n_pred[0] == 0)  # 0이면 정상 예측이므로 정답
    print(f"Correct prediction: {is_correct}")

    print(f"\nInference completed!")
    return {
        'svm_prediction': new_n_pred[0],
        'svm_features': pred_labels2[0],
        'individual_predictions': pred_labels[0],
        'image_path': args.image_path,
        'is_correct': is_correct
    }

if __name__ == "__main__":
    results = main()
