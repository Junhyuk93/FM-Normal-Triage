"""
linear.py 파일 참고하여 추론 코드 작성
"""
    # autocast_dtype 유무 차이 비교 필요!
from setup_2stage import setup_and_build_model, get_args_parser
from model_utils import ModelWithIntermediateLayers, evaluate_2stage
from functools import partial
from data import make_data_loader, make_dataset
from data.transforms import make_classification_eval_transform
import torch
import torch.nn as nn


@torch.no_grad()
def evaluate_linear_classifiers(
    feature_model,
    linear_classifiers,
    data_loader,
):

    postprocessors = {k: LinearPostprocessor(v, class_mapping) for k, v in linear_classifiers.classifiers_dict.items()}
    metrics = {k: metric.clone() for k in linear_classifiers.classifiers_dict}

    _, results_dict_temp = evaluate_2stage(
        feature_model,
        data_loader,
        postprocessors,
        metrics,
        torch.cuda.current_device(),
    )

    return results_dict


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=2):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes

        # default 1 layer
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
            
    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)

def setup_linear_classifiers(sample_output, num_classes=2, n_blocks=None, avg_pool=None):
    
    out_dim = create_linear_input(sample_output, use_n_blocks=n_blocks, use_avgpool=avg_pool).shape[1]
    linear_classifier = LinearClassifier(
                    out_dim, use_n_blocks=n_blocks, use_avgpool=avg_pool, num_classes=num_classes
                )
    linear_classifier = linear_classifier.cuda()
    return linear_classifier

def make_eval_data_loader(test_dataset_str, batch_size, num_workers):
    test_dataset = make_dataset(
        dataset_str=test_dataset_str,
        transform=make_classification_eval_transform(),
    )
    test_data_loader = make_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
    )
    return test_data_loader, test_dataset


def main(args):
    
    test_loader, test_dataset = make_eval_data_loader(test_dataset_str=args.test_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=16)

    model, autocast_dtype = setup_and_build_model(args)

    ## Non-porosis vs porosis
    n_last_blocks1 = 1 # or 4
    avg_pool1 = True
    lr_1 = '0_00125'
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model1 = ModelWithIntermediateLayers(model, n_last_blocks1, autocast_ctx)
    # print(feature_model)
    sample_output1 = feature_model1(test_dataset[0][0].unsqueeze(0).cuda())

    linear_classifier1 = setup_linear_classifiers(
        sample_output1,
        args.training_num_classes,
        n_last_blocks1,
        avg_pool1
    )

    state_dict1 = torch.load(args.pretrained_linear1)
    state_dict1 = {k.replace(f'classifiers_dict.classifier_{n_last_blocks1}_blocks_avgpool_{avg_pool1}_lr_{lr_1}.',''):v for k,v in state_dict1.items()}
    
    linear_classifier1.load_state_dict(state_dict1)

    ## Normal vs Osteopenia
    n_last_blocks2 = 4 # or 4
    avg_pool2 = True
    lr_2 = '0_00003'
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model2 = ModelWithIntermediateLayers(model, n_last_blocks2, autocast_ctx)
    sample_output2 = feature_model2 (test_dataset[0][0].unsqueeze(0).cuda())

    linear_classifier2 = setup_linear_classifiers(
        sample_output2,
        args.training_num_classes,
        n_last_blocks2 ,
        avg_pool2
    )

    state_dict2 = torch.load(args.pretrained_linear2)
    state_dict2 = {k.replace(f'classifiers_dict.classifier_{n_last_blocks2}_blocks_avgpool_{avg_pool2}_lr_{lr_2}.',''):v for k,v in state_dict2.items()}
    
    linear_classifier2.load_state_dict(state_dict2)
    
    evaluate_2stage(
        feature_model1,
        feature_model2,
        test_loader,
        linear_classifier1,
        linear_classifier2)
    
if __name__ == "__main__":
    description = "Osteo Binary Classification Inference"
    
    # get_args_parser 함수 사용 --> 모델 config 읽기
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    main(args)
