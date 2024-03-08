from utils import *
from metrics import ProbMetrics

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import os
import logging
import time

from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# from my_datasets import PrepareData
import pandas as pd


# logger = logging.getLogger('Train')


def train(model: nn.Module,
          optimizer: optim,
          train_loader,
          epoch: int):
    model.train()
    loss_epoch = np.zeros(len(train_loader))

    for i, x in enumerate(tqdm(train_loader, ncols=100, position=0)):
        # dataloader出来的数据格式均为(batch,feature, seq_len)
        # 前向
        optimizer.zero_grad()
        loss = model.loss(x['source'], x['target'], x['time_cov'])
        loss.backward()
        optimizer.step()
        # 记录训练过程数据
        loss_epoch[i] = loss.item()
    return loss_epoch


def valuation(model: nn.Module,
              val_loader,
              epoch: int,
              device,
              target_len,
              logger
              ):
    # loss_epoch = np.zeros(len(test_loader))
    model.eval()
    loss_epoch = np.zeros(len(val_loader))
    with torch.no_grad():
        metrics = ProbMetrics()
        for i, x in enumerate(tqdm(val_loader, ncols=100, position=0)):  
            labels = x['target'].to(device)
            y_hat, y_sample_hat = model.predict(x['source'], x['time_cov'])
            loss = model.loss(x['source'], x['target'], x['time_cov'])
            loss_epoch[i] = loss.item()
           
            metrics.update(y_hat, y_sample_hat, labels)

        summary_metric = metrics.result()
        metrics_string = '; '.join('{}: {:.5f}'.format(k, v) for k, v in summary_metric.items())
        logger.info('Full test metrics: ' + metrics_string)

    return summary_metric, loss_epoch


def train_and_val(model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   test_loader: DataLoader,
                   optimizer: optim,
                   num_epochs,
                   device,
                   exp_result_dir,
                   params_dict,
                   logger,
                   key_metrics = 'CRPS',
                   scheduler = None):
    # 生成训练记录文件夹
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    result_dir = os.path.join(exp_result_dir, time_now)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    #  设置记录日志
    logger_name = model._get_name()
    set_logger(logger, os.path.join(result_dir, logger_name + '.log'))
    params_dict_len = params_dict.__len__()
    model_params = '{'
    for i, (k, v) in enumerate(params_dict.items()):
        if k != 'device':
            model_params += f''' "{k}":{v} '''     
            model_params += ', \n'
    model_params += ' "end":"end" }'

    json_path = os.path.join(result_dir, 'params.json')
    f_json = open(json_path, 'w', newline='\n')
    f_json.write(model_params)
    f_json.close()

    logger.info(f'model_params: \n {model_params}')
    logger.info('Begin train and val')

    # 使用TensorBoard
    # writer = SummaryWriter(result_dir)

    # 训练
    best_test_metrics = float('inf')
    best_epoch = 0
    loss_train_list = []
    loss_test_list = []
    best_metric = {}
    for epoch in range(num_epochs):

        logger.info(f'epoch:{epoch} training...')
        loss_train = train(model, optimizer, train_loader, epoch)
        logger.info(f'epoch:{epoch} loss_train={loss_train.mean()}')
        loss_train_list.append(loss_train.mean())
        # scheduler.step(loss_train.mean())

        logger.info(f'epoch:{epoch} val...')
        metrics, loss_test = valuation(model, val_loader, epoch, device, params_dict['target_len'],logger=logger)
        loss_test_list.append(loss_test.mean())
        # _, _ = valuation(model, test_loader, epoch, device, params_dict['target_len'],logger=logger)

        if metrics[key_metrics] <= best_test_metrics:
            best_epoch = epoch
            best_metric = metrics
            best_test_metrics = metrics[key_metrics]
            model_path = os.path.join(result_dir, 'best_model')
            torch.save(model.state_dict(), model_path)
            best_metrics_string = ';   '.join(' \n {}: {:.5f}'.format(k, v) for k, v in best_metric.items())
            logger.info('Current best val metrics: {}, produced in epoch: {}'.format(best_metrics_string, best_epoch))
            
        logger.info(F'Current best CRPS: {best_test_metrics}, produced in epoch: {best_epoch}')

    logger.info('\n ********** End ************')
    df = pd.DataFrame({'train':loss_train_list,
                       'test':loss_test_list})
    df.to_csv(os.path.join(result_dir, 'loss.csv'))
    metrics_string = ';  '.join('\n {}: {:.5f}'.format(k, v) for k, v in best_metric.items())
    logger.info('\n The best val metrics: {}, produced in epoch: {}'.format(metrics_string, best_epoch))
    
    return result_dir



    

