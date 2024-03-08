from utils import Params, setup_seed, get_model
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import os
import logging
from dataprocess.gefdatasets import ProcessGEF
from train import train_and_val, valuation
import sys
import json

def main(data_name = 'Zone1', model_name='FCVAE', cuda_id=0):
    model_name = model_name 
    data_dir = 'data/GEFCom2014'
    data_name = data_name

    json_dir = os.path.join('experiments/config', f'{model_name}_params.json') 
    
    params = Params(json_dir)
    # 读取数据
    prepare_data = ProcessGEF(data_dir=data_dir, 
                              Zone=data_name,
                              source_len=params.source_len,
                              target_len=params.target_len)
    params.enc_dim=prepare_data.get_cov_dim() + 1

                

    # 创建文件夹，保存训练信息
    exp_dir = 'experiments/log/' + data_name
    exp_result_dir = os.path.join(exp_dir, str(params.source_len)+'_'+str(params.target_len), model_name)
    if not os.path.exists(exp_result_dir):
        os.makedirs(exp_result_dir)

    # 设置训练设备
    cuda_exist = torch.cuda.is_available()
    if cuda_exist:
        params.device = torch.device(f'cuda:{cuda_id}')
    else:
        params.device = torch.device('cpu')


    model = get_model(model_name, params)
    
    # 数据加载器
    train_dataset = prepare_data.get_dataset(flag='train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True)
    
    val_dataset = prepare_data.get_dataset(flag='val')
    val_loader = DataLoader(dataset=val_dataset, batch_size=params.batch_size, shuffle=False)

    test_dataset = prepare_data.get_dataset(flag='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=params.batch_size, shuffle=False)

    # 优化器、损失函数、学习率调整
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    
    logger_name = model._get_name()
    logger = logging.getLogger(logger_name)
    # 训练
    setup_seed(20)
    result_dir = train_and_val(model=model,
                                    train_loader=train_loader,
                                    val_loader=test_loader,
                                    test_loader=test_loader,
                                    optimizer=optimizer,
                                    num_epochs=params.num_epochs,
                                    device=params.device,
                                    logger=logger,
                                    exp_result_dir=exp_result_dir,
                                    params_dict=params.__dict__)
    
    model.load_state_dict(torch.load(os.path.join(result_dir, 'best_model')))
    summary_metric,_ = valuation(model=model,
                                val_loader=test_loader,
                                epoch=0,
                                device=params.device,
                                logger=logger,
                                target_len=params.target_len,)
    
    metric_json = json.dumps(summary_metric, sort_keys=False, indent=4, separators=(',', ': '))
    # 显示数据类型
    f = open(f'{result_dir}/test_metric.json', 'w')
    f.write(metric_json)
    f.close()
    # df = pd.DataFrame(summary_metric)
    # df.to_csv(os.path.join(result_dir, 'summary_metric.csv'))
    
    
    
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
    # main('Zone6', 'FCVAE', '0')
    
        
        