import numpy as np
import torch
import math
import logging
import properscoring as ps
# from joblib import Parallel, delayed
from einops import rearrange
# import CRPS.CRPS as pscore

logger = logging.getLogger('TADnet.metrics')


# ND：相对误差
def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor):
    """
    用于计算整个测试集的ND的中间量记录
    """
    zero_index = (labels != 0)
    diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
    summation = torch.sum(torch.abs(labels[zero_index])).item()

    return [diff, summation]


# q-risk: 分位数损失
def accuracy_ROU(q: float, rou: torch.Tensor, labels: torch.Tensor):
    """
    input_shape = (batch, time_step)
    """
    zero_index = (labels != 0)
    error = labels[zero_index] - rou[zero_index]
    ql = torch.max((q - 1) * error, q * error)
    numerator = 2 * torch.sum(ql).item()
    denominator = torch.abs(labels[zero_index]).sum().item()

    return [numerator, denominator]


# RMSE：均方根误差
def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor):
    diff = torch.sum(torch.mul((mu - labels), (mu - labels))).item()
    return diff
    
# MAE: 平均绝对误差
def accuracy_MAE(mu: torch.Tensor, labels: torch.Tensor):
    diff = torch.sum(torch.abs(mu - labels)).item()
    return diff



def accuracy_CRPS(samples: torch.Tensor, labels: torch.Tensor):
    
    labels = labels.data.cpu().numpy()
    samples = samples.data.cpu().numpy()
    crps = ps.crps_ensemble(labels, samples).sum(axis=0)

    return crps  

# 评估可靠性
def PICP(low, up, label):
    sample_num = label.shape[0] * label.shape[1]
    inclu = (label >= low) & (label <= up)
    inclu_num = inclu.sum().item()
    return inclu_num, sample_num



# 预测区间平均带宽(PI normalized average width，PINAW)
def accuracy_NAPS(samples: torch.Tensor):
    out=torch.zeros((samples.shape[1]), device=samples.device)
    for i in range(10,100,10):
        q_up=samples.quantile(1-i/200, dim=-1)
        q_low=samples.quantile(i/200, dim=-1)
        out = out + torch.sum((q_up - q_low), dim=0)
    out= torch.sum(out/9).item()
    
    return out

def accuracy_MRAE(samples: torch.Tensor, labels: torch.Tensor):
    '''
    采样数为100， 求的是MRAE中的 sum(eta)
    '''
    samples_num = samples.shape[-1]
    stride = round(5*samples_num/100) 
    labels = labels.clone().unsqueeze(-1)
    df1 = torch.sum(samples > labels, dim=[0,1])
    eta = df1[[i-1 for i in range(stride, samples_num, stride)]]
    return eta.data.cpu().numpy()


def quantile_loss(target, quantile_forecast, q):
        return 2.0 * np.sum(
            np.abs(
                (quantile_forecast - target)
                * ((target <= quantile_forecast) - q)
            )
        )
    

class ProbMetrics():
    '''
    用于完整数据集（需要分批次进行推理）的指标计算类
    '''
    def __init__(self) -> None:
        self.metrics = {'num': np.zeros(1),
                        'ND': np.zeros(2),  # numerator, denominator
                        'RMSE': np.zeros(1),  # numerator, denominator, time step count
                        'MAE': np.zeros(1),
                        'rou10': np.zeros(1),
                        'rou90': np.zeros(1),  # numerator, denominator
                        'NAPS':np.zeros(1),
                        'MRAE':np.zeros(1),
                        'CRPS':np.zeros(1)}
        
    def update(self, predict50, predict_sample, labels):
        predict50 = predict50[:,:,0]
        labels = labels[:,:,0]
        sample_num = predict_sample.shape[-1]
        predict10 = predict_sample[:,:, int(0.1 * sample_num)]
        predict90 = predict_sample[:,:, int(0.9 * sample_num)]
        
        self.metrics['num'] = self.metrics['num'] + labels.shape[0] * labels.shape[1]
        self.metrics['ND'] = self.metrics['ND'] + accuracy_ND(predict50, labels)
        self.metrics['rou10'] = self.metrics['rou10'] + accuracy_ROU(0.1, predict10, labels)
        self.metrics['rou90'] = self.metrics['rou90'] + accuracy_ROU(0.9, predict90, labels)
        self.metrics['RMSE'] = self.metrics['RMSE'] + accuracy_RMSE(predict50, labels)
        self.metrics['MAE'] = self.metrics['MAE'] + accuracy_MAE(predict50, labels)
        
        self.metrics['NAPS'] = self.metrics['NAPS'] + accuracy_NAPS(predict_sample)
        self.metrics['MRAE'] = self.metrics['MRAE'] + accuracy_MRAE(predict_sample, labels)
        self.metrics['CRPS'] = self.metrics['CRPS'] + accuracy_CRPS(predict_sample, labels)
    
    def result(self):
        summary_metric = {}
        summary_metric['ND'] = self.metrics['ND'][0] / self.metrics['ND'][1]
        summary_metric['rou10'] = self.metrics['rou10'][0] / self.metrics['rou10'][1]
        summary_metric['rou90'] = self.metrics['rou90'][0] / self.metrics['rou90'][1]
        summary_metric['RMSE'] = np.sqrt(self.metrics['RMSE'][0] / self.metrics['num'][0])
        summary_metric['MAE'] = self.metrics['MAE'][0] / self.metrics['num'][0]
        summary_metric['NAPS'] = (self.metrics['NAPS'][0] / self.metrics['num'][0])
        summary_metric['CRPS'] = (self.metrics['CRPS'].sum() / self.metrics['num'][0])
        summary_metric['MRAE'] = np.abs((self.metrics['MRAE'] / self.metrics['num'][0]) - np.arange(0.05,1,0.05)).mean()
        # summary_metric['CRPS_LIST'] = (self.metrics['CRPS'] / self.metrics['num']).tolist()
        # print(summary_metric)

        
        return summary_metric



# def init_metrics():
#     metrics = {
#         'ND': np.zeros(2),  # numerator, denominator
#         'RMSE': np.zeros(2),  # numerator, denominator, time step count
#         'MAE': np.zeros(2),
#         'rou10': np.zeros(2),
#         'rou90': np.zeros(2),  # numerator, denominator
#         'CRPS':[]
#     }
#     return metrics




# def update_metrics(raw_metrics, predict50, predict_sample, labels):
#     predict50 = predict50[:,:,0]
#     labels = labels[:,:,0]
#     sample_num = predict_sample.shape[-1]
#     predict10 = predict_sample[:,:, int(0.1 * sample_num)]
#     predict90 = predict_sample[:,:, int(0.9 * sample_num)]
    
#     raw_metrics['ND'] = raw_metrics['ND'] + accuracy_ND(predict50, labels)
#     raw_metrics['rou10'] = raw_metrics['rou10'] + accuracy_ROU(0.1, predict10, labels)
#     raw_metrics['rou90'] = raw_metrics['rou90'] + accuracy_ROU(0.9, predict90, labels)
#     raw_metrics['RMSE'] = raw_metrics['RMSE'] + accuracy_RMSE(predict50, labels)
#     raw_metrics['MAE'] = raw_metrics['MAE'] + accuracy_MAE(predict50, labels)
#     raw_metrics['CRPS'] = raw_metrics['CRPS'] + accuracy_CRPS(predict_sample, labels)
    

#     return raw_metrics


# def final_metrics(raw_metrics):
#     summary_metric = {}
#     summary_metric['ND'] = raw_metrics['ND'][0] / raw_metrics['ND'][1]
#     summary_metric['rou10'] = raw_metrics['rou10'][0] / raw_metrics['rou10'][1]
#     summary_metric['rou90'] = raw_metrics['rou90'][0] / raw_metrics['rou90'][1]
#     summary_metric['RMSE'] = np.sqrt(raw_metrics['RMSE'][0] / raw_metrics['RMSE'][1]) 
#     summary_metric['MAE'] = np.sqrt(raw_metrics['MAE'][0] / raw_metrics['MAE'][1])
#     summary_metric['CRPS'] = np.mean(raw_metrics['CRPS'])
    
#     return summary_metric
