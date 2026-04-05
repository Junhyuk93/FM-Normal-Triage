"""
public_data.py 수정 버전 - 클래스 폴더 구조 없이 추론 가능
기존 파일을 백업하고 이 파일로 교체하세요.
"""
import os
from pathlib import Path
from typing import Callable, Optional
from PIL import Image
import pydicom
import numpy as np

class PublicDataInference:
    """추론 전용 데이터셋 - 클래스 폴더 구조 불필요"""
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        
        # 지원하는 확장자
        self.extensions = {'.dcm', '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        # 데이터 수집
        self.samples = []
        self._collect_samples()
        
        print(f"[PublicDataInference] Found {len(self.samples)} files")
    
    def _collect_samples(self):
        """파일 수집"""
        # 루트가 파일인 경우
        if self.root.is_file():
            if self.root.suffix.lower() in self.extensions:
                self.samples.append((self.root, 0))  # 더미 클래스 0
            return
        
        # 루트가 디렉토리인 경우
        # 1단계: 직접 파일 찾기
        for file_path in self.root.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.extensions:
                self.samples.append((file_path, 0))  # 더미 클래스 0
        
        # 2단계: 클래스 폴더가 있는 경우도 지원
        if not self.samples:
            for class_dir in self.root.iterdir():
                if class_dir.is_dir():
                    for file_path in class_dir.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() in self.extensions:
                            # 클래스 이름을 숫자로 변환 시도
                            try:
                                class_id = int(class_dir.name)
                            except ValueError:
                                class_id = 0  # 변환 실패시 0
                            self.samples.append((file_path, class_id))
    
    def __len__(self):
        return len(self.samples)
    
    def load_image(self, path: Path) -> Image.Image:
        """이미지 로드 (DICOM 포함)"""
        if path.suffix.lower() == '.dcm':
            # DICOM 파일
            dcm = pydicom.dcmread(str(path))
            img_array = dcm.pixel_array
            
            # 정규화
            img_array = img_array.astype(np.float32)
            if img_array.max() > img_array.min():
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
            img_array = (img_array * 255).astype(np.uint8)
            
            # Grayscale to RGB
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            return Image.fromarray(img_array)
        else:
            # 일반 이미지
            return Image.open(path).convert('RGB')
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        
        try:
            image = self.load_image(path)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            # 더미 이미지 반환
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target


# 기존 클래스와 호환성을 위한 별칭
class PublicData(PublicDataInference):
    """기존 PublicData 클래스와의 호환성"""
    
    def __init__(self, *args, **kwargs):
        # 'split' 인자 제거 (추론에서는 불필요)
        kwargs.pop('split', None)
        super().__init__(*args, **kwargs)


# 레지스트리 등록을 위한 더미 변수들
_DATASET_NAME = "normal-triage"
_DATASET_CLASS = PublicData
