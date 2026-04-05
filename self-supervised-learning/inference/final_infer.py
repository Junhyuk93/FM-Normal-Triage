import torch
import torch.nn as nn
import pickle
import os
from tqdm import tqdm
import joblib  # For loading SVM model
import numpy as np

from model_utils import ModelWithIntermediateLayers
from multi_preprocessing import ToTensor
from setup import setup_and_build_model, get_args_parser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),
            ),
            dim=-1,
        )
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
        self.out_dim = out_dim

    def forward(self, x_tokens_list):
        x = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(x)


def setup_linear_classifier(sample_output, use_n_blocks, use_avgpool, ckpt_path):
    out_dim = create_linear_input(sample_output, use_n_blocks, use_avgpool).shape[1]
    classifier = LinearClassifier(out_dim, use_n_blocks, use_avgpool).to(DEVICE)
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    full_state_dict = ckpt['model']
    prefix = f'classifiers_dict.classifier_{use_n_blocks}_blocks_avgpool_{use_avgpool}_lr_0_00003.'
    state_dict = {k.replace(prefix, ''): v for k, v in full_state_dict.items() if k.startswith(prefix)}
    classifier.load_state_dict(state_dict)
    classifier.eval()
    return classifier


def load_pickle_data(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def infer_with_models(pickle_data, model_ckpt_list, target_diseases, svm_model_path):
    use_n_blocks = 4
    use_avgpool = True

    models = []
    classifiers = []

    args_parser = get_args_parser()
    args = args_parser.parse_args([])  # use default

    for ckpt_path in model_ckpt_list:
        model, autocast_dtype = setup_and_build_model(args)
        model.eval()
        model = model.to(DEVICE)
        feature_model = ModelWithIntermediateLayers(
            model,
            use_n_blocks,
            lambda **kwargs: torch.cuda.amp.autocast(enabled=True, dtype=autocast_dtype)
        )
        feature_model.eval()

        dummy_input = pickle_data[0][0].unsqueeze(0).to(DEVICE)
        sample_output = feature_model(dummy_input)
        classifier = setup_linear_classifier(sample_output, use_n_blocks, use_avgpool, ckpt_path)

        models.append(feature_model)
        classifiers.append(classifier)

    X_features = []
    y_labels = []

    for image, _, label, path in tqdm(pickle_data):
        image = image.unsqueeze(0).to(DEVICE)
        image_probs = []

        for model, classifier, target_disease in zip(models, classifiers, target_diseases):
            with torch.no_grad():
                feats = model(image)
                logits = classifier(feats)
                probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
                image_probs.append(probs.astype(np.float32))

        X_features.append(np.concatenate(image_probs))
        y_labels.append(label)

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)

    return preds, y_labels, X_features



if __name__ == '__main__':
    pickle_path = '/path/to/All_Test.pickle'
    svm_model_path = '/path/to/svm_model.pkl'
    model_ckpt_list = [
        '/path/to/model1.pth',
        '/path/to/model2.pth',
        '/path/to/model3.pth',
        '/path/to/model4.pth',
        '/path/to/model5.pth',
        '/path/to/model6.pth',
        '/path/to/model7.pth',
        '/path/to/model8.pth',
        '/path/to/model9.pth',
        '/path/to/model10.pth',
        '/path/to/model11.pth',
        '/path/to/model12.pth',
        '/path/to/model13.pth',
    ]

    data = load_pickle_data(pickle_path)
    preds, labels, features = infer_with_models(data, model_ckpt_list, svm_model_path)