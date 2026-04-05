import os
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

def setup_linear_classifier(sample_output, num_classes=3, n_blocks=4, avg_pool=True):
    out_dim = create_linear_input(sample_output, n_blocks, avg_pool).shape[1]
    classifier = LinearClassifier(out_dim, n_blocks, avg_pool, num_classes).cuda()
    return classifier

# ------------------ 모델 래퍼 ------------------
class DINOv2_Model(nn.Module):
    def __init__(self, model_ckpt_path, classifier_ckpt_path, args):
        super().__init__()
        self.n_blocks = 4
        self.avg_pool = True

        model, autocast_dtype = setup_and_build_model(args)
        model = model.cuda().eval()

        self.feature_model = ModelWithIntermediateLayers(
            model,
            self.n_blocks,
            partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
        ).eval()

        dummy_input = torch.randn(1, 3, 1024, 1024).cuda()
        sample_output = self.feature_model(dummy_input)

        self.classifier = setup_linear_classifier(sample_output, num_classes=3, n_blocks=self.n_blocks, avg_pool=self.avg_pool)
        ckpt = torch.load(classifier_ckpt_path, map_location="cuda")
        full_state_dict = ckpt["model"]
        prefix = f'classifiers_dict.classifier_{self.n_blocks}_blocks_avgpool_{self.avg_pool}_lr_0_00003.'
        state_dict = {k.replace(prefix, ''): v for k, v in full_state_dict.items() if k.startswith(prefix)}
        self.classifier.load_state_dict(state_dict)

    def forward(self, x):
        with torch.no_grad():
            feats = self.feature_model(x)
            logits = self.classifier(feats)
            probs = torch.softmax(logits, dim=1)
        return probs.squeeze(0).cpu().numpy()  # [3]

# ------------------ 데이터 로딩 ------------------
def extract_path_from_dataset_arg(dataset_arg):
    if ':' in dataset_arg:
        for part in dataset_arg.split(':'):
            if part.startswith("root="):
                return part.split('=')[1]
    return dataset_arg

def load_data_from_directory_fixed3(data_path):
    image_paths, labels = [], []
    image_extensions = ['.dcm', '.dicom']
    label_mapping = {'normal': 0, 'target': 1, 'others': 2}

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue

        label = label_mapping.get(folder, 2)

        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(folder, file))
                labels.append(label)

    return image_paths, labels

def setup_external_loader(data_path, batch_size=1, num_workers=8):
    image_paths, labels = load_data_from_directory_fixed3(data_path)
    transform = transforms.Compose([ToTensor()])
    dataset = Chest_Single_Data_Generator_dinov2((1024, 1024), data_path, image_paths, labels, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, dataset

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


def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    externalloader, externalset = setup_external_loader2(
        args.test_dataset,
        batch_size=args.batch_size
    )

    model = DINOv2_Model(
        model_ckpt_path=args.pretrained_weights,
        classifier_ckpt_path=args.pretrained_linear,
        args=args
    )
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for image, label in tqdm(externalloader):
            image = image.cuda()
            probs = model(image)
            all_probs.append(probs)
            all_labels.append(label.item())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = np.argmax(all_probs, axis=1)                  # shape: (N,)
    accuracy = np.mean(preds == all_labels)

    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # per-class accuracy 계산
    num_classes = 3
    correct_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)

    for pred, label in zip(preds, all_labels):
        total_per_class[label] += 1
        if pred == label:
            correct_per_class[label] += 1

    class_names = {0: 'normal', 1: 'target', 2: 'others'}
    print("\nPer-class Accuracy:")
    for cls in range(num_classes):
        cls_name = class_names[cls]
        acc = 100 * correct_per_class[cls] / total_per_class[cls] if total_per_class[cls] > 0 else 0.0
        print(f"{cls_name:>7}: {acc:.2f}%")


if __name__ == "__main__":
    args_parser = get_args_parser(description="Single Model DINOv2 3-Class Inference")
    args = args_parser.parse_args()
    main(args)
