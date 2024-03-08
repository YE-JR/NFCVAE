import pandas as pd
import numpy as np
from scipy import stats
import os
from torch.utils.data import Dataset
import json

class ProcessGEF:
    def __init__(self, data_dir='data/GEFCom2014', 
                       Zone='Zone1',
                    #    split_percentage = [0.6,0.2,0.2],
                       source_len=96,
                       target_len=8,
                       stride=1) -> None:
        
        
        # self.split_percentage = split_percentage
        self.source_len = source_len
        self.target_len = target_len
        self.stride = stride
        
        
        
        data_path = os.path.join(data_dir, Zone + '.csv')
        data = pd.read_csv(data_path)
        
        
            
        data = data.fillna(0)
        data.loc[data.TARGETVAR>=1,'TARGETVAR']=0.99
        data.loc[data.TARGETVAR<=0,'TARGETVAR']=0.01


        data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
        data = data.sort_values('TIMESTAMP', ignore_index=True)
        self.data = data.reset_index()

        self.date = data['TIMESTAMP']
        self.data = self.data.set_index('TIMESTAMP',drop=True)
        
        self.cov_column = ['U10','V10','U100','V100']
        self.time_cov_name = ["month", "day", "weekday", "hour"]
        self._process_cov()

        # if target_len == 1:
        #     lag_cov = []
        #     for i in range(2,4):
        #         self.data[f'lag{i}'] =  self.data['TARGETVAR'].shift(i)
        #         lag_cov.append(f'lag{i}')
        #     self.data = self.data.fillna(0.01)
        #     self.cov_column = lag_cov + self.cov_column

        # self._get_split_point()
        self.split_point = {'train_start':'2012-01-01 01:00:00',
                            'train_end':'2013-05-01 00:00:00',
                            'val_start':'2013-04-27 01:00:00',
                            'val_end':'2013-08-01 00:00:00',
                            'test_start':'2013-07-28 01:00:00',
                            'test_end':'2014-01-01 00:00:00'}

    def get_cov_dim(self):
        return len(self.cov_column)
    
    def _process_cov(self):
        '''
        处理时变协变量
        '''
        
        for col in self.cov_column:
            self.data[col] = stats.zscore(self.data[col].values)

        # for i in self.time_cov_name:
        #     self.cov_column.append(i)
        #     if i == 'month':
        #         self.data.insert(0, 'month', stats.zscore(self.date.dt.month.values))
        #     elif i == 'weekday':
        #         self.data.insert(0, 'weekday', stats.zscore(self.date.dt.weekday.values))
        #     elif i == 'day':
        #         self.data.insert(0, 'day', stats.zscore(self.date.dt.day.values))
        #     elif i == 'hour':
        #         self.data.insert(0, 'hour', stats.zscore(self.date.dt.hour.values))
        #     elif i == 'minute':
        #         self.data.insert(0, 'minute', stats.zscore(self.date.dt.minute.values))
        #     else:
        #         raise Exception(f'{i} is not a timestamp feature!')

    def get_dataset(self, flag='train'):

        date_start = self.split_point[flag + '_start']
        date_end = self.split_point[flag + '_end']
        data = self.data[date_start:date_end].copy()
        # data = data.drop('TIMESTAMP')
        
        dataset =  SingleTimeSerDataset(data=data, 
                                        instance_colum = 'TARGETVAR',
                                        cov_colums = self.cov_column,
                                        source_len=self.source_len, 
                                        target_len=self.target_len,
                                        stride=self.stride)
        
        return dataset
    
    def get_dataframe(self, flag='train'):
        '''
        给传统机器学习用的
        '''
        X_column = self.cov_column.copy()
        X_column.append('TARGETVAR')
        Y_column = []
        date_start = self.split_point[flag + '_start']
        date_end = self.split_point[flag + '_end']
        data = self.data[date_start:date_end].copy()
        for i in range(1, self.target_len + 1):
            data[f'lag{i}'] =  data['TARGETVAR'].shift(-i)
            Y_column.append(f'lag{i}')
            
        
        
        
        return data.iloc[:-self.target_len], X_column, Y_column
        
        
        
        
    
    def get_plot_data(self):
        '''
        返回最后测试集最后一个窗口的样本
        '''
        date_start = self.split_point['test_start']
        date_end = self.split_point['test_end']
        data = self.data[date_start:date_end].copy()
        date_index = data.index.values
        dataset = SingleTimeSerDataset(data=data, 
                                        instance_colum = 'TARGETVAR',
                                        cov_colums = self.cov_column,
                                        source_len=self.source_len, 
                                        target_len=self.target_len,
                                        stride=self.stride)
        
        return dataset, pd.to_datetime(date_index), data
    


    

    # def _get_split_point(self):
    #     data_len = self.data.shape[0]
    #     self.split_point = {}
    #     self.split_point['train_start'] = 0
    #     self.split_point['train_end'] = int(self.split_percentage[0] * data_len)
    #     self.split_point['val_start'] = self.split_point['train_end'] - self.source_len
    #     self.split_point['val_end'] = self.split_point['train_end'] + int(self.split_percentage[1] * data_len)
    #     self.split_point['test_start'] = self.split_point['val_end'] - self.source_len
    #     self.split_point['test_end'] = data_len - 1


class SingleTimeSerDataset(Dataset):
    '''
    风电数据集处理：
        所有风力发电数据按照最大出力归一化到 [0,1] 之间，表征出力的比例
        输出结果不再进行转换还原
        不考虑静态变量，不同的序列分开训练
    '''
        
    def __init__(self, data, instance_colum, cov_colums, 
                 source_len, target_len, stride) -> None:
        
        self.data = data
        self.instance_colum = instance_colum
        self.time_cov = data[cov_colums].values
        self.source_len = source_len
        self.target_len = target_len
        self.stride = stride
        
        
        self.window_len = source_len + target_len
        
        self.sample_num = (data.shape[0] - self.window_len) // stride + 1
        
        
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, index):
        
        start_index = index * self.stride
        end_index = start_index + self.window_len
        
        data_arr = self.data[self.instance_colum].values
        
        source = data_arr[start_index : start_index+self.source_len]
        target = data_arr[start_index+self.source_len : end_index]
        
       
        time_cov = self.time_cov[start_index:end_index]
        
        
        
        x = {'source': np.expand_dims(source, -1).astype(np.float32),
             'target': np.expand_dims(target, -1).astype(np.float32),
             'time_cov': time_cov.astype(np.float32)}
        
        return x
    
    
