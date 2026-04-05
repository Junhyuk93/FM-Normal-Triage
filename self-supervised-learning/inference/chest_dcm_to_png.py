import numpy as np
import SimpleITK as sitk
from skimage import exposure

def _resize_and_normalize(arr, out_size, out_channel, percentile=99):
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
        # assert target_idx == 0, 'cant has larger width than height, filename:{}'.format(path)
        target_size = psizes[target_idx]
        ### crop
        target_x = int(target_size / spacings[0])
        target_y = int(target_size / spacings[1])
        offset_x = int((pixels[0] - target_x) * 0.5)
        if offset_x < 0:
            target_x = int((float(pixels[0])/float(target_x))*out_size)
            offset_x = int((out_size-target_x) * 0.5)
            tmp = _resize_and_normalize(img[0, :target_y, :], (target_x,out_size), 3)
            result_img = np.zeros((out_size, out_size, 3), dtype='uint8')
            result_img[:, offset_x:offset_x+target_x, :] = tmp
        else:
            # result_img = _resize_and_normalize(img[0, :target_y, offset_x:offset_x+target_x], (out_size,out_size), 3)
            result_img = _resize_and_normalize(img, (out_size,out_size), out_channel)
    else:
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))
        result_img = _resize_and_normalize(img[0], (out_size, out_size), out_channel)
        
    result_img2 = exposure.equalize_hist(result_img)*255
    result_img2 = np.clip(result_img2, 0, 255).astype(np.uint8)
    return result_img2