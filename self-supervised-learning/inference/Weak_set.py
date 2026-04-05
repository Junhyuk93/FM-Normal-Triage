import cv2
import numpy as np
from torch.utils.data import Dataset

from chest_dcm_to_png import dicom_to_png

class Chest_Single_Data_Generator(Dataset):
    def __init__(self, img_size, data_root_path, input_img_paths, labels, mode=None, transform=None):
        self.img_size = img_size
        self.data_root_path = data_root_path
        self.input_img_paths = input_img_paths
        self.labels = labels
        self.mode = mode
        self.transform = transform
    
    def __len__(self):
        return len(self.input_img_paths)

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        input_img_path = self.data_root_path + '/' + self.input_img_paths[idx]
        print(input_img_path)
        temp_image = dicom_to_png(input_img_path)#, out_channel=1)
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
        mask =  np.zeros((1024,1024))

        img = cv2.resize(temp_image, (self.img_size[0], self.img_size[1]))

        img = np.expand_dims(img, -1)
        mask = np.expand_dims(mask, -1)
        
        sample = {'image': img, 'mask': mask, 'label': self.labels[idx]}   

        if self.transform:
            sample = self.transform(sample)
            sample['image'] /= 255.
            sample['mask'] /= 255.
            
        return sample['image'], sample['mask'], sample['label'], input_img_path
    

class Chest_Single_Data_Generator_dinov2(Dataset):
    def __init__(self, img_size, data_root_path, input_img_paths, labels, mode=None, transform=None):
        self.img_size = img_size
        self.data_root_path = data_root_path
        self.input_img_paths = input_img_paths
        self.labels = labels
        self.mode = mode
        self.transform = transform
    
    def __len__(self):
        return len(self.input_img_paths)

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        input_img_path = self.data_root_path + '/' + self.input_img_paths[idx]

        # DICOM → PNG 변환
        temp_image = dicom_to_png(input_img_path)

        # Grayscale로 변환
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)

        # 이미지 리사이즈
        img = cv2.resize(temp_image, (self.img_size[0], self.img_size[1]))

        # (H, W) → (H, W, 1)
        img = np.expand_dims(img, -1)

        # ✅ 3채널로 복제 (DINOv2 요구)
        img = np.repeat(img, 3, axis=-1)  # (H, W, 3)

        # 마스크는 사용 안 하더라도 형식 맞춰줌
        mask = np.zeros((self.img_size[0], self.img_size[1], 1))

        sample = {'image': img, 'mask': mask, 'label': self.labels[idx]}   

        if self.transform:
            sample = self.transform(sample)
            sample['image'] /= 255.
            sample['mask'] /= 255.
            
        return sample['image'], sample['mask'], sample['label'], input_img_path

    
class CustomDataset(Dataset):
    def __init__(self, img_size, input_img_paths, labels, mode=None, transform=None):
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.labels = labels
        self.mode = mode
        self.transform = transform
    
    def __len__(self):
        return len(self.input_img_paths)

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        input_img_path = self.input_img_paths[idx]
        temp_image = cv2.imread(input_img_path)
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
        mask =  np.zeros((1024,1024))

        img = cv2.resize(temp_image, (self.img_size[0], self.img_size[1]))

        img = np.expand_dims(img, -1)
        mask = np.expand_dims(mask, -1)
        
        sample = {'image': img, 'mask': mask, 'label': self.labels[idx]}   

        if self.transform:
            sample = self.transform(sample)
            sample['image'] /= 255.
            sample['mask'] /= 255.
            
        return sample['image'], sample['mask'], sample['label'], self.input_img_paths[idx]
    

class DicomSingleImageDataset(Dataset):
    def __init__(self, img_size, input_img_paths, mode=None, transform=None):
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.mode = mode
        self.transform = transform
    
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        temp_image = dicom_to_png(self.input_img_paths)
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
        mask =  np.zeros((1024,1024))

        img = cv2.resize(temp_image, (self.img_size[0], self.img_size[1]))

        img = np.expand_dims(img, -1)
        mask = np.expand_dims(mask, -1)
        
        sample = {'image': img, 'mask': mask}   

        if self.transform:
            sample = self.transform(sample)
            sample['image'] /= 255.
            sample['mask'] /= 255.
            
        return sample['image'], sample['mask']