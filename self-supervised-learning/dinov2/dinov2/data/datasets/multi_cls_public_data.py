
from glob import glob 
from PIL import Image

from typing import Optional, Callable, Any, Tuple
from torchvision.datasets import VisionDataset

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
        self.img_lst = glob(root + '/*/*/*')
        self.class_ = {
                        "medfmc" : 0,
                        "nih" :1,
                        "object_cxr":2,
                        "paxray":3        }


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
            image, target = self.transforms(image_data, target)

        return image, target

    def __len__(self) -> int:
        return len(self.img_lst)