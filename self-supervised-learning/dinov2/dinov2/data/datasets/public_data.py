import os
from glob import glob 
from PIL import Image
from collections import Counter

from typing import Optional, Callable, Any, Tuple
from torchvision.datasets import VisionDataset
import numpy as np

import SimpleITK as sitk

class NormalTraige(VisionDataset):
    def __init__(
            self,
            *,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
        ) -> None:
            super().__init__(root, transforms, transform, target_transform)

            self.root = root 
            print(f"Dataset root: {self.root}")

            # glob 수정: 재귀 탐색 + 파일만
            self.img_lst = sorted(glob(os.path.join(root, '**', '*'), recursive=True))
            self.img_lst = [f for f in self.img_lst if os.path.isfile(f) and f.split('.')[-1] in ['dcm']]
            
            # 3-class mapping
            self.class_ = {
                "normal": 0,
                "target": 1,
                "others": 2,
            }

            # class_idx 계산
            self.class_idx = []
            for f in self.img_lst:
                # 파일에서 top_dir 추출
                rel_path = os.path.relpath(f, self.root)  # 예: normal/xxx/file.dcm
                top_dir = rel_path.split(os.sep)[0]       # 'normal', 'target', 'others' 만 나오게

                try:
                    cls_id = self.class_[top_dir]
                except KeyError:
                    raise RuntimeError(f"Invalid class folder: {top_dir} in file: {f}")

                self.class_idx.append(cls_id)
            
            # 디버깅 출력
            print(f"총 클래스 수: {len(self.class_)}")
            print(f"총 이미지 수: {len(self.img_lst)}")
            print(f"클래스별 분포:")
            class_names = []
            for f in self.img_lst:
                rel_path = os.path.relpath(f, self.root)
                top_dir = rel_path.split(os.sep)[0]
                class_names.append(top_dir)
            for class_name, count in Counter(class_names).items():
                print(f"  {class_name}: {count}개")

    def __len__(self) -> int:
        return len(self.img_lst)  

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            if self.img_lst[index].split('.')[-1] == 'png':
                image_data = Image.open(self.img_lst[index]).convert(mode="RGB")
            elif self.img_lst[index].split('.')[-1] == 'dcm':
                image_data = sitk.GetArrayFromImage(sitk.ReadImage(self.img_lst[index])).squeeze()
                image_data = image_data - np.min(image_data)
                image_data = image_data / np.max(image_data)
                image_data = (image_data * 255).astype(np.uint8)
                image_data = Image.fromarray(image_data).convert(mode="RGB")
            else:
                raise RuntimeError(f"Unsupported file type: {self.img_lst[index]}")
        except Exception as e:
            raise RuntimeError(f"Cannot read image for sample {index} : {self.img_lst[index]}") from e
        
        target = self.class_idx[index]

        if self.transforms is not None:
            image, target = self.transforms(image_data, target)
        else:
            image = image_data

        return image, target
class osteo(VisionDataset):
    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.root = root 
        print(self.root)
        self.img_lst = glob(root + '/*/*')
        self.class_ = {
                        "normal" : 0,
                        "osteopenia" :0,
                        "osteoporosis":1,
                       }
        self.class_idx = [self.class_[self.img_lst[i].split('/')[-2]] for i in range(len(self.img_lst))]

    def __len__(self) -> int:
        return len(self.img_lst)  

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            if self.img_lst[index].split('.')[-1]=='png':
                image_data = Image.open(self.img_lst[index]).convert(mode="RGB")
                # image = ImageDataDecoder(image_data).decode()
            elif self.img_lst[index].split('.')[-1]=='dcm':
                image_data = sitk.GetArrayFromImage(sitk.ReadImage(self.img_lst[index])).squeeze()
                image_data = image_data - np.min(image_data)
                image_data = image_data / np.max(image_data)
                image_data = (image_data*255).astype(np.uint8)
                image_data = Image.fromarray(image_data).convert(mode="RGB")
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        
        target = self.class_idx[index]
        # target = TargetDecoder(target).decode()

        if self.transforms is not None:
            # image = self.transforms(image=image_data)['image'] # albumentations 에러뜸
            image, target = self.transforms(image_data, target) # torchvision transforms
            
        return image, target

class osteo_phase1(VisionDataset):
    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.root = root 
        print(self.root)
        normal = glob(root+'/normal/*')
        porosis = glob(root+'/osteoporosis/*')
        # self.img_lst = glob(root + '/*/*')
        normal.extend(porosis)
        self.img_lst = normal
        self.class_ = {
                        "normal" : 0,
                        "osteoporosis":1,
                       }
        self.class_idx = [self.class_[self.img_lst[i].split('/')[-2]] for i in range(len(self.img_lst))]

    def __len__(self) -> int:
        return len(self.img_lst)  

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = Image.open(self.img_lst[index]).convert(mode="RGB")
            # image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        
        target = self.class_idx[index]

        if self.transforms is not None:
            # image = self.transforms(image=image_data)['image'] # albumentations 에러뜸
            image, target = self.transforms(image_data, target) # torchvision transforms
            
        return image, target
    
class osteo_penia(VisionDataset):
    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.root = root 
        print(self.root)
        normal = glob(root+'/normal/*')
        penia = glob(root+'/osteopenia*/*')
        # self.img_lst = glob(root + '/*/*')
        normal.extend(penia)
        self.img_lst = normal
        self.class_ = {
                        "normal" : 0,
                        "osteopenia":1,
                        "osteopenia1":1,
                        "osteopenia2":1,
                        "osteopenia3":1,
                       }
        self.class_idx = [self.class_[self.img_lst[i].split('/')[-2]] for i in range(len(self.img_lst))]

    def __len__(self) -> int:
        return len(self.img_lst)  

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = Image.open(self.img_lst[index]).convert(mode="RGB")
            # image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        
        target = self.class_idx[index]

        if self.transforms is not None:
            # image = self.transforms(image=image_data)['image'] # albumentations 에러뜸
            image, target = self.transforms(image_data, target) # torchvision transforms
            
        return image, target

class Osteo_3cls(VisionDataset):
    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.root = root 
        print(self.root)
        self.img_lst = glob(root + '/*/*')
        self.class_ = {
                        "normal" : 0,
                        "osteopenia" :1,
                        "osteoporosis":2,
                       }

    def __len__(self) -> int:
        return len(self.img_lst)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = Image.open(self.img_lst[index]).convert(mode="RGB")
            # image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        
        target = self.class_[self.img_lst[index].split('/')[-2]]
        # target = TargetDecoder(target).decode()

        if self.transforms is not None:
            # image = self.transforms(image=image_data)['image'] # albumentations 에러뜸
            image, target = self.transforms(image_data, target) # torchvision transforms
            
        return image, target
    
class Genderosteo(VisionDataset):
    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.root = root 
        print(self.root)
        self.img_lst = glob(root + '/*/*')
        self.class_ = {
                        "M" : 0,
                        "F" :1,
                       }
    def __len__(self) -> int:
        return len(self.img_lst)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = Image.open(self.img_lst[index]).convert(mode="RGB")
            # image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        
        target = self.class_[self.img_lst[index].split('_')[-2]] # gender split
        # target = TargetDecoder(target).decode()

        if self.transforms is not None:
            # image = self.transforms(image=image_data)['image'] # albumentations 에러뜸
            image, target = self.transforms(image_data, target) # torchvision transforms
            
        return image, target

    def __len__(self) -> int:
        return len(self.img_lst)

class Ageosteo(VisionDataset):
    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.root = root 
        print(self.root)
        self.img_lst = glob(root + '/*/*')

    def __len__(self) -> int:
        return len(self.img_lst)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = Image.open(self.img_lst[index]).convert(mode="RGB")
            # image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        target = np.int8(self.img_lst[index].split('_')[-1].split('.')[0]) # age split
        # target = TargetDecoder(target).decode()

        if self.transforms is not None:
            # image = self.transforms(image=image_data)['image'] # albumentations 에러뜸
            image, target = self.transforms(image_data, target) # torchvision transforms
            
        return image, target

    def __len__(self) -> int:
        return len(self.img_lst)

class PD(VisionDataset):
    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.root = root 
        print(self.root)
        self.img_lst = glob(root + '/*/*')
        self.class_ = {
                        "chestdr" : 0,
                        "chexpert" :1,
                        "mimic":2,
                        "nih":3,
                        "padchest":4,
                        "vindr":5,
                        "paxray":6,
                        "objectcxr":7,
                        "brixia":8,
                        "jsrt":9,
                        "shenzhen":10,
                        "peru":11}


    def __len__(self) -> int:
        return len(self.img_lst)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = Image.open(self.img_lst[index]).convert(mode="RGB")
            # image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        
        target = self.class_[self.img_lst[index].split('/')[-3]]
        # target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image_data, target)

        return image, target

    def __len__(self) -> int:
        return len(self.img_lst)