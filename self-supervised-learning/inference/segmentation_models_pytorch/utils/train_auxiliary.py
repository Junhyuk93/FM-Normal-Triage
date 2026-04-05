import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter


class Epoch:

    def __init__(self, model, c_loss, s_loss, c_metrics, s_metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.c_loss = c_loss
        self.s_loss = s_loss
        self.c_metrics = c_metrics
        self.s_metrics = s_metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.c_loss.to(self.device)
        self.s_loss.to(self.device)
        for metric in self.c_metrics:
            metric.to(self.device)
        for metric in self.s_metrics:
            metric.to(self.device)

    def _format_logs(self, c_logs, s_logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in c_logs.items()]
        str_logs += ['{} - {:.4}'.format(k, v) for k, v in s_logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, mask, label):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        c_logs = {}
        s_logs = {}
        c_loss_meter = AverageValueMeter()
        s_loss_meter = AverageValueMeter()
        c_metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.c_metrics}
        s_metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.s_metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, mask, label in iterator:
                x, mask, label = x.to(self.device), mask.to(self.device), label.to(self.device)
                c_loss, s_loss, y_pred = self.batch_update(x, mask, label)

                # update loss classification logs
                c_loss_value = c_loss.cpu().detach().numpy()
                c_loss_meter.add(c_loss_value)
                c_loss_logs = {self.c_loss.__name__: c_loss_meter.mean}
                c_logs.update(c_loss_logs)

                # update classification metrics logs
                for metric_fn in self.c_metrics:
                    metric_value = metric_fn(y_pred[1], label).cpu().detach().numpy()
                    c_metrics_meters[metric_fn.__name__].add(metric_value)
                c_metrics_logs = {k: v.mean for k, v in c_metrics_meters.items()}
                c_logs.update(c_metrics_logs)

                # update loss segmentation logs
                s_loss_value = s_loss.cpu().detach().numpy()
                s_loss_meter.add(s_loss_value)
                s_loss_logs = {self.s_loss.__name__: s_loss_meter.mean}
                s_logs.update(s_loss_logs)

                # update segmentation metrics logs
                for metric_fn in self.s_metrics:
                    metric_value = metric_fn(y_pred[0], mask).cpu().detach().numpy()
                    s_metrics_meters[metric_fn.__name__].add(metric_value)
                s_metrics_logs = {k: v.mean for k, v in s_metrics_meters.items()}
                s_logs.update(s_metrics_logs)

                if self.verbose:
                    s = self._format_logs(c_logs, s_logs)
                    iterator.set_postfix_str(s)

        return c_logs, s_logs


class TrainEpoch(Epoch):

    def __init__(self, model, c_loss, s_loss, c_metrics, s_metrics, optimizer, scheduler=None, device='cpu', verbose=True):
        super().__init__(
            model=model,
            c_loss=c_loss,
            s_loss=s_loss,
            c_metrics=c_metrics,
            s_metrics=s_metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, mask, label):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        s_loss = self.s_loss(prediction[0], mask)
        c_loss = self.c_loss(prediction[1], label)
        total_loss = 0.8*s_loss + c_loss
        total_loss.backward()
#         s_loss.backward(retain_graph=True)
#         c_loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return c_loss, s_loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, c_loss, s_loss, c_metrics, s_metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            c_loss=c_loss,
            s_loss=s_loss,
            c_metrics=c_metrics,
            s_metrics=s_metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, mask, label):
        with torch.no_grad():
            prediction = self.model.forward(x)
            s_loss = self.s_loss(prediction[0], mask)
            c_loss = self.c_loss(prediction[1], label)
        return c_loss, s_loss, prediction
