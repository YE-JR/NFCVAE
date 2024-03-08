import logging
import numpy as np
import os
from tqdm import tqdm
import logging
import json
import torch
import random
from models.SingleSeq.DeepAR import DeepAR
from models.SingleSeq.FCVAE import FCVAE
from models.SingleSeq.CVAE import CVAE
from models.SingleSeq.QSQF import QSQF
from models.SingleSeq.QRTrans import QRTrans
from models.SingleSeq.QRNN import QRNN
from models.SingleSeq.DeepTCN import DeepTCN
from models.SingleSeq.FCVAE_C import FCVAE_C
from models.SingleSeq.FCVAEv2 import FCVAEv2
from models.SingleSeq.FCVAE_T import FCVAE_T
from models.SingleSeq.CVAE_C import CVAE_C

logger = logging.getLogger('TADNet.Utils')


################# 参数设置 ###################
class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    """

    def __init__(self, json_path=None):
        if json_path is not None:
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by params.dict['learning_rate']"""
        return self.__dict__


############# 设置训练记录logger ################
def set_logger(_logger, log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    """

    # _logger = logging.getLogger(logger_name)
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%D  %H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)
    _logger.addHandler(TqdmHandler(fmt))
    
    
# 随机种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True
    
def get_model(model_name, params:Params):
    if model_name == 'FCVAE':
        return FCVAE(enc_dim=params.enc_dim, 
                    cond_channels_dim=params.cond_channels_dim, 
                    post_channels_dim=params.post_channels_dim, 
                    z_dim=params.z_dim, 
                    kernel_size=params.kernel_size, 
                    dropout=params.dropout, 
                    hidden_size=params.hidden_size, 
                    num_heads=params.num_heads, 
                    dec_length=params.target_len,
                    device=params.device).to(params.device)
    elif model_name == 'FCVAEv2':
        return FCVAEv2(enc_dim=params.enc_dim, 
                    cond_channels_dim=params.cond_channels_dim, 
                    post_channels_dim=params.post_channels_dim, 
                    z_dim=params.z_dim, 
                    kernel_size=params.kernel_size, 
                    dropout=params.dropout, 
                    hidden_size=params.hidden_size, 
                    num_heads=params.num_heads, 
                    dec_length=params.target_len,
                    device=params.device).to(params.device)
    

    elif model_name == 'FCVAE_C':
        return FCVAE_C(enc_dim=params.enc_dim, 
                    cond_channels_dim=params.cond_channels_dim, 
                    post_channels_dim=params.post_channels_dim, 
                    z_dim=params.z_dim, 
                    kernel_size=params.kernel_size, 
                    dropout=params.dropout, 
                    hidden_size=params.hidden_size, 
                    dec_length=params.target_len,
                    device=params.device).to(params.device)
    elif model_name == 'CVAE_C':
        return CVAE_C(enc_dim=params.enc_dim, 
                    cond_channels_dim=params.cond_channels_dim, 
                    post_channels_dim=params.post_channels_dim, 
                    z_dim=params.z_dim, 
                    kernel_size=params.kernel_size, 
                    dropout=params.dropout, 
                    hidden_size=params.hidden_size, 
                    dec_length=params.target_len,
                    device=params.device).to(params.device)
    
    elif model_name == 'FCVAE_T':
        return FCVAE_T(enc_dim=params.enc_dim, 
                    num_layers=params.num_layers, 
                    post_channels_dim=params.post_channels_dim, 
                    z_dim=params.z_dim, 
                    kernel_size=params.kernel_size, 
                    dropout=params.dropout, 
                    hidden_size=params.hidden_size, 
                    num_heads = params.num_heads,
                    dec_length=params.target_len,
                    device=params.device).to(params.device)
        
    elif model_name == 'CVAE':
        return CVAE(enc_dim=params.enc_dim, 
                    cond_channels_dim=params.cond_channels_dim, 
                    post_channels_dim=params.post_channels_dim, 
                    z_dim=params.z_dim, 
                    kernel_size=params.kernel_size, 
                    dropout=params.dropout, 
                    hidden_size=params.hidden_size, 
                    num_heads=params.num_heads, 
                    dec_length=params.target_len,
                    device=params.device).to(params.device)
        
        
    elif model_name == 'QSQF':
        return QSQF(params=params, device=params.device).to(params.device)
    
    
    elif model_name == 'QRTrans':
        return QRTrans(device=params.device,
                       enc_input_size=params.enc_dim,
                       num_heads=params.num_heads,
                       num_layers=params.num_layers,
                       hidden_size=params.hidden_size).to(params.device)
        
        
    elif model_name == 'QRNN':
        return QRNN(device=params.device,
                       enc_dim=params.enc_dim,
                       source_len=params.source_len,
                       target_len=params.target_len,
                       hidden_size=params.hidden_size).to(params.device)
        
        
    elif model_name == 'DeepAR':
        return DeepAR(params=params).to(params.device)
    


    elif model_name == 'DeepTCN':
        return DeepTCN(enc_dim=params.enc_dim,
                       hidden_dim=params.hidden_size,
                       channels_dim=params.channels_dim,
                       kernel_size=params.kernel_size,
                       dec_length=params.target_len,
                       dropout=params.dropout,
                       device=params.device).to(params.device)