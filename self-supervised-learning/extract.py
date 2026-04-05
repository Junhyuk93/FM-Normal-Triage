from glob import glob 
import random
from tqdm import tqdm 
import shutil

nih_path = '/workspace/open_dataset/NIH_CXR/images_*/images/*.png'
nih_lst = glob(nih_path)
random.shuffle(nih_lst)


paxray_path = '/workspace/open_dataset/PaxRay++/images_patlas/*_frontal.png'
paxray_lst = glob(paxray_path)
random.shuffle(paxray_lst)


medfmc_path = '/workspace/open_dataset/MedFMC/MedFMC_*/chest/images/*.png'
medfmc_lst = glob(medfmc_path)
random.shuffle(medfmc_lst)


object_cxr_path = '/workspace/open_dataset/tmp/object-CXR/*/*.jpg'
object_cxr_lst = glob(object_cxr_path)

random.shuffle(object_cxr_lst)


def _copy(lst, target, num):
    for idx in tqdm(lst[:num]):
        shutil.copy(idx, target + idx.split('/')[-1])


_copy(nih_lst, '/workspace/test-cxr/nih/', 1000)
_copy(paxray_lst, '/workspace/test-cxr/paxray/', 1000)
_copy(meffmc_lst, '/workspace/test-cxr/medfmc/', 1000)
_copy(object_cxr_lst, '/workspace/test-cxr/object_cxr/', 1000)