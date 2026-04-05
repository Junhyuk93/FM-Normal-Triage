import torch
import numpy as np
import random
import cv2

class ToTensor(object):
    def __call__(self, data):
        if 'recon' in data.keys():
            input, recon, mask, label = data['image'], data['recon'], data['mask'], data['label']
            input = input.transpose((2, 0, 1)).astype(np.float32)
            recon = recon.transpose((2, 0, 1)).astype(np.float32)
            mask = mask.transpose((2, 0, 1)).astype(np.float32)
            
            data = {'image': torch.from_numpy(input), 'recon': torch.from_numpy(recon), 'mask': torch.from_numpy(mask), 'label': label}

            return data

        input, mask, label = data['image'], data['mask'], data['label']
        input = input.transpose((2, 0, 1)).astype(np.float32)
        mask = mask.transpose((2, 0, 1)).astype(np.float32)
        
        data = {'image': torch.from_numpy(input), 'mask': torch.from_numpy(mask), 'label': label}

        return data
    
class ToTensorSingle(object):
    def __call__(self, data):
        if 'recon' in data.keys():
            input, recon, mask, label = data['image'], data['recon'], data['mask']
            input = input.transpose((2, 0, 1)).astype(np.float32)
            recon = recon.transpose((2, 0, 1)).astype(np.float32)
            mask = mask.transpose((2, 0, 1)).astype(np.float32)
            
            data = {'image': torch.from_numpy(input), 'recon': torch.from_numpy(recon), 'mask': torch.from_numpy(mask)}

            return data

        input, mask = data['image'], data['mask']
        input = input.transpose((2, 0, 1)).astype(np.float32)
        mask = mask.transpose((2, 0, 1)).astype(np.float32)
        
        data = {'image': torch.from_numpy(input), 'mask': torch.from_numpy(mask)}

        return data

class Rotation_2D(object):
    def __call__(self, sample, degree = 10):
        p = 0.3
        if 'recon' in sample.keys():
            image = sample['image']
            recon = sample['recon']
            mask = sample['mask']
            label = sample['label']
            R_move = random.randint(-degree,degree)
            if random.random() < p:
                #print("_rotation_2D")
                M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), R_move, 1)
                image = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
                image = np.expand_dims(image, axis=-1)
                recon = cv2.warpAffine(recon,M,(recon.shape[1],recon.shape[0]))
                recon = np.expand_dims(recon, axis=-1)
                for i in range(mask.shape[2]):
                    mask[:,:,i] = cv2.warpAffine(mask[:,:,i],M,(image.shape[1],image.shape[0]))
                # mask = np.expand_dims(mask, axis=-1)
                #rotate_pimg = cv2.warpAffine(point_img,M,(img.shape[0],img.shape[1]))
            return {'image': image, 'recon': recon, 'mask': mask, 'label': label}

        image = sample['image']
        mask = sample['mask']
        label = sample['label']
        R_move = random.randint(-degree,degree)
        if random.random() < p:
            #print("_rotation_2D")
            M = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), R_move, 1)
            image = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
            image = np.expand_dims(image, axis=-1)
            for i in range(mask.shape[2]):
                mask[:,:,i] = cv2.warpAffine(mask[:,:,i],M,(image.shape[1],image.shape[0]))
            # mask = np.expand_dims(mask, axis=-1)
            #rotate_pimg = cv2.warpAffine(point_img,M,(img.shape[0],img.shape[1]))
        return {'image': image, 'mask': mask, 'label': label}


class Shift_2D(object):
    def __call__(self, sample, shift = 10):
        p = 0.3
        if 'recon' in sample.keys():
            image = sample['image']
            recon = sample['recon']
            mask = sample['mask']
            label = sample['label']
            
            x_move = random.randint(-shift,shift)
            y_move = random.randint(-shift,shift)
            if random.random() < p:
                shift_M = np.float32([[1,0,x_move], [0,1,y_move]])
                image = cv2.warpAffine(image, shift_M,(image.shape[1], image.shape[0]))
                recon = cv2.warpAffine(recon, shift_M,(recon.shape[1], recon.shape[0]))
                for i in range(mask.shape[2]):
                    mask[:,:,i] = cv2.warpAffine(mask[:,:,i], shift_M, (mask.shape[1], mask.shape[0]))
                image = np.expand_dims(image, axis=-1)
                recon = np.expand_dims(recon, axis=-1)
                
            return {'image': image, 'recon': recon, 'mask': mask, 'label': label}

        image = sample['image']
        mask = sample['mask']
        label = sample['label']
        
        x_move = random.randint(-shift,shift)
        y_move = random.randint(-shift,shift)
        if random.random() < p:
            shift_M = np.float32([[1,0,x_move], [0,1,y_move]])
            image = cv2.warpAffine(image, shift_M,(image.shape[1], image.shape[0]))
            for i in range(mask.shape[2]):
                mask[:,:,i] = cv2.warpAffine(mask[:,:,i], shift_M, (mask.shape[1], mask.shape[0]))
            image = np.expand_dims(image, axis=-1)
#             mask = np.expand_dims(mask, axis=-1)
            
        return {'image': image, 'mask': mask, 'label': label}

class RandomBlur(object):
    def __call__(self, sample):
        image = sample['image']
        p = 0.3
        if random.random() < p:
            image = sample['image']
            image = cv2.blur(image,(3,3))
            image = np.expand_dims(image, axis=-1)
            sample['image'] = image

        return sample