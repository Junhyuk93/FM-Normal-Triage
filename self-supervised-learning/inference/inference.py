"""
linear.py 파일 참고하여 추론 코드 작성
"""
    # autocast_dtype 유무 차이 비교 필요!
from setup import setup_and_build_model, get_args_parser
from model_utils import ModelWithIntermediateLayers, evaluate
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
    # metric_type,
    # metrics_file_path,
    # training_num_classes,
    # iteration,
    # prefixstring="",
    # class_mapping=None,
    # best_classifier_on_val=None,
):
    # logger.info("running validation !")

    # num_classes = training_num_classes
    # metric = build_metric(metric_type, num_classes=num_classes)
    postprocessors = {k: LinearPostprocessor(v, class_mapping) for k, v in linear_classifiers.classifiers_dict.items()}
    metrics = {k: metric.clone() for k in linear_classifiers.classifiers_dict}

    _, results_dict_temp = evaluate(
        feature_model,
        data_loader,
        postprocessors,
        metrics,
        torch.cuda.current_device(),
    )

    # logger.info("")
    # results_dict = {}
    # max_accuracy = 0
    # best_classifier = ""
    # for i, (classifier_string, metric) in enumerate(results_dict_temp.items()):
    #     logger.info(f"{prefixstring} -- Classifier: {classifier_string} * {metric}")
    #     if (
    #         best_classifier_on_val is None and metric["top-1"].item() > max_accuracy
    #     ) or classifier_string == best_classifier_on_val:
    #         max_accuracy = metric["top-1"].item()
    #         best_classifier = classifier_string

    # results_dict["best_classifier"] = {"name": best_classifier, "accuracy": max_accuracy}

    # logger.info(f"best classifier: {results_dict['best_classifier']}")

    # if distributed.is_main_process():
    #     with open(metrics_file_path, "a") as f:
    #         f.write(f"iter: {iteration}\n")
    #         for k, v in results_dict.items():
    #             f.write(json.dumps({k: v}) + "\n")
    #         f.write("\n")

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

        # gelee layer
        # self.linear = nn.Sequential(
        #     nn.Linear(self.out_dim, int(self.out_dim/2)),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(int(self.out_dim/2), int(self.out_dim/4)),
        #     nn.ReLU(),
        #     nn.Linear(int(self.out_dim/4), int(self.num_classes)))

        # 2 layer
        # self.linear = nn.Sequential(
        #     nn.Linear(self.out_dim, int(self.out_dim/2)),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(int(self.out_dim/2), int(self.num_classes)))
            
    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)

def setup_linear_classifiers(sample_output, num_classes=2, n_blocks=None, avg_pool=None):
    # linear_classifiers_dict = nn.ModuleDict()
    # optim_param_groups = []
    # for n in n_last_blocks_list:
    #     for avgpool in [False, True]:
    #         for _lr in learning_rates:
    #             lr = scale_lr(_lr, batch_size)
    #             out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
    #             linear_classifier = LinearClassifier(
    #                 out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
    #             )
    #             linear_classifier = linear_classifier.cuda()
    #             linear_classifiers_dict[
    #                 f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")
    #             ] = linear_classifier
    #             optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    # linear_classifiers = AllClassifiers(linear_classifiers_dict)
    # if distributed.is_enabled():
    #     linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)

    # return linear_classifiers, optim_param_groups
    
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
        # sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
        # collate_fn=_pad_and_collate if metric_type == MetricType.IMAGENET_REAL_ACCURACY else None,
    )
    return test_data_loader, test_dataset


def main(args):
    
    test_loader, test_dataset = make_eval_data_loader(test_dataset_str=args.test_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=16)

    model, autocast_dtype = setup_and_build_model(args)

    # n_last_blocks_list = [1, 4]
    # n_last_blocks = max(n_last_blocks_list)
    
    # n_last_blocks_list = [4]
    n_last_blocks = 4 # or 4
    avg_pool = True
    lr_ = '0_00003'
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)
    # print(feature_model)
    sample_output = feature_model(test_dataset[0][0].unsqueeze(0).cuda())

    linear_classifier = setup_linear_classifiers(
        sample_output,
        args.training_num_classes,
        n_last_blocks,
        avg_pool
    )

    state_dict = torch.load(args.pretrained_linear)
    state_dict = {k.replace(f'classifiers_dict.classifier_{n_last_blocks}_blocks_avgpool_{avg_pool}_lr_{lr_}.',''):v for k,v in state_dict.items()}
    '''
    classifier_4_blocks_avgpool_False_lr_0_05.
    classifier_4_blocks_avgpool_False_lr_0_05000.

    classifier_1_blocks_avgpool_True_lr_0_05000
    '''
    # print(state_dict)
    
    linear_classifier.load_state_dict(state_dict)
    # print(linear_classifiers)
    
    evaluate(
        feature_model,
        test_loader,
        linear_classifier)
    
    # test_on_datasets 함수 참고해서 inference 코드 작성
    
    # metric별 출력물 저장 코드 
    # AUC, ACC(macro, micro 둘 다), confusion matrix, 
    # sensitivity, specificity, F1 Score 
    # (앞에 것들 잘 나오면 npv, ppv)
    
    # 위 실험 웨이트 3개에서 실험해야함

if __name__ == "__main__":
    description = "Osteo Binary Classification Inference"
    
    # get_args_parser 함수 사용 --> 모델 config 읽기
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    main(args)
