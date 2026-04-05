import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from functools import partial
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from model_utils import ModelWithIntermediateLayers
from setup import setup_and_build_model, get_args_parser
from multi_preprocessing import ToTensor
from Weak_set import Chest_Single_Data_Generator_dinov2

# ----------------------------------------
# Feature 생성 함수
def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat((output, torch.mean(intermediate_output[-1][0], dim=1)), dim=-1)
        output = output.reshape(output.shape[0], -1)
    return output.float()

# Linear classifier
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

def setup_linear_classifiers(sample_output, num_classes=3, n_blocks=None, avg_pool=None):
    out_dim = create_linear_input(sample_output, use_n_blocks=n_blocks, use_avgpool=avg_pool).shape[1]
    classifier = LinearClassifier(out_dim, n_blocks, avg_pool, num_classes).cuda()
    return classifier

# ----------------------------------------
class All_DINOv2_Models(nn.Module):
    def __init__(self, model_ckpt_paths, classifier_ckpt_paths, args):
        super().__init__()
        self.n_blocks = 4
        self.avg_pool = True
        self.models = nn.ModuleList()
        self.classifiers = nn.ModuleList()

        for model_path, classifier_path in zip(model_ckpt_paths, classifier_ckpt_paths):
            model, autocast_dtype = setup_and_build_model(args)
            model = model.cuda().eval()

            feature_model = ModelWithIntermediateLayers(
                model,
                self.n_blocks,
                partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
            ).eval()

            dummy_input = torch.randn(1, 3, 1024, 1024).cuda()
            sample_output = feature_model(dummy_input)

            classifier = setup_linear_classifiers(sample_output, 3, self.n_blocks, self.avg_pool)
            ckpt = torch.load(classifier_path, map_location="cuda")
            full_state_dict = ckpt["model"]
            prefix = f'classifiers_dict.classifier_{self.n_blocks}_blocks_avgpool_{self.avg_pool}_lr_0_00003.'
            state_dict = {k.replace(prefix, ''): v for k, v in full_state_dict.items() if k.startswith(prefix)}
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
                outputs.append(probs.squeeze(0).cpu().numpy())  # shape: [3]
        return outputs  # list of 13 x [3] → concat to 39

# ----------------------------------------
def extract_path_from_dataset_arg(dataset_arg):
    if ':' in dataset_arg:
        for part in dataset_arg.split(':'):
            if part.startswith("root="):
                return part.split('=')[1]
    return dataset_arg

def load_data_from_directory_vin(data_path):
    image_paths, labels = [], []
    image_extensions = ['.dcm', '.dicom']

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue

        if folder == 'normal':
            label = 0
            for file in os.listdir(folder_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(folder, file))
                    labels.append(label)
        elif folder == 'abnormal':
            label = 1
            # abnormal/*/파일까지 모두 찾아야 함
            for disease_subfolder in os.listdir(folder_path):
                disease_path = os.path.join(folder_path, disease_subfolder)
                if not os.path.isdir(disease_path):
                    continue
                for file in os.listdir(disease_path):
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        rel_path = os.path.join(folder, disease_subfolder, file)
                        image_paths.append(rel_path)
                        labels.append(label)

    return image_paths, labels


def load_data_from_directory(data_path, label_mapping=None):
    image_paths, labels = [], []
    image_extensions = ['.dcm', '.dicom']
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue
        label = label_mapping.get(folder, 2) if label_mapping else 2
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(folder, file))
                labels.append(label)
    return image_paths, labels

def setup_external_loader(data_path, batch_size=1, num_workers=8, label_mapping=None):
    # image_paths, labels = load_data_from_directory(data_path, label_mapping)
    image_paths, labels = load_data_from_directory_vin(data_path)
    transform_test = transforms.Compose([ToTensor()])
    dataset = Chest_Single_Data_Generator_dinov2((1024, 1024), data_path, image_paths, labels, transform=transform_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, dataset

# ----------------------------------------
def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 14-class → 단일 target classification 위한 label mapping
    label_mapping = {
        'normal': 0,
        'cardiomegaly': 1,
        'advanced_tuberculosis': 2,
        'nodule': 3,
        'active_tuberculosis': 4,
        'consolidation': 5,
        'interstitial_opacity': 6,
        'pleural_effusion': 7,
        'pneumothorax': 8,
        'atelectasis': 9,
        'mediastinal_widening': 10,
        'support_device': 11,
        'pleural_calcification': 12,
        'pneumoperitoneum': 13
    }

    externalloader, externalset = setup_external_loader(
        extract_path_from_dataset_arg(args.test_dataset),
        batch_size=args.batch_size,
        label_mapping=label_mapping
    )

    model_ckpt_paths = [args.pretrained_weights] * 13
    classifier_ckpt_paths = ["/workspace/eval/v2_stable_e400/normal-triage-FM/running_checkpoint_linear_eval_36250.pth",
                             "/workspace/eval/v2_stable_e400/normal-triage-FM-v2/running_checkpoint_linear_eval_36250.pth",
                             "/workspace/eval/v2_stable_e400/normal-triage-FM-v3/running_checkpoint_linear_eval_36250.pth",
                             "/workspace/eval/v2_stable_e400/normal-triage-FM-v4/running_checkpoint_linear_eval_36250.pth",
                             "/workspace/eval/v2_stable_e400/normal-triage-FM-v5/running_checkpoint_linear_eval_36250.pth",
                             "/workspace/eval/v2_stable_e400/normal-triage-FM-v6/running_checkpoint_linear_eval_36250.pth",
                             "/workspace/eval/v2_stable_e400/normal-triage-FM-v7/running_checkpoint_linear_eval_23750.pth",
                             "/workspace/eval/v2_stable_e400/normal-triage-FM-v8/running_checkpoint_linear_eval_23750.pth",
                             "/workspace/eval/v2_stable_e400/normal-triage-FM-v9/running_checkpoint_linear_eval_23750.pth",
                             "/workspace/eval/v2_stable_e400/normal-triage-FM-v9/running_checkpoint_linear_eval_23750.pth",
                             "/workspace/eval/v2_stable_e400/normal-triage-FM-v9/running_checkpoint_linear_eval_23750.pth",
                             "/workspace/eval/v2_stable_e400/normal-triage-FM-v9/running_checkpoint_linear_eval_23750.pth",
                             "/workspace/eval/v2_stable_e400/normal-triage-FM-v9/running_checkpoint_linear_eval_23750.pth"
                             ]

    model = All_DINOv2_Models(model_ckpt_paths, classifier_ckpt_paths, args)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for image, _, label, _ in tqdm(externalloader):
            image = image.to(DEVICE)
            probs_list = model(image)
            probs_concat = np.concatenate(probs_list)  # shape: (39,)
            all_preds.append(probs_concat)
            all_labels.append(label.item())

    # --------------------
    # SVM 분류 및 평가
    print("\n[Step] SVM Prediction & Metric Evaluation")
    X = np.array(all_preds)
    y_true = [0 if y == 0 else 1 for y in all_labels]  # normal=0, others=1

    svm_model = joblib.load("/workspace/inference/svm_weight.pickle")
    y_pred = svm_model.predict(X)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    args_parser = get_args_parser(description="DINOv2 + SVM Final Inference")
    args = args_parser.parse_args()
    main(args)
