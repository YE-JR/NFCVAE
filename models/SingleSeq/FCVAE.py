import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math
import torch.nn.functional as F
from einops import rearrange
from math import log

############# 先验编码器 / 条件编码器 #############
class ConditionEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 channels_dim: list, 
                 z_dim:int,
                 kernel_size:int, 
                 dropout:float, 
                 hidden_size:int, 
                 num_heads:int, 
                 dec_length:int) -> None:
        
        super().__init__()
        
        self.cnn_block = TemporalConvNet(input_dim=input_dim, 
                                         channels_dim=channels_dim, 
                                         kernel_size=kernel_size,
                                         dropout=dropout)
        self.grn = GRN(channels_dim[-1], hidden_size)
        self.grn_dec = GRN(input_dim-1, hidden_size)

        self.attention = AttentionNet(embed_dim=hidden_size,
                                      num_heads=num_heads,
                                      dec_length=dec_length)

        self.fc_mean = nn.Sequential(nn.Linear(hidden_size, z_dim),
                                     nn.LayerNorm(z_dim),
                                     nn.ELU()) 
        self.fc_std = nn.Sequential(nn.Linear(hidden_size, z_dim),
                                    nn.LayerNorm(z_dim),
                                    nn.ELU())
        

    def forward(self, source, cov):
        """
        输出的长度为需要预测的长度
        """
        enc_len = source.shape[1]
        enc_data = torch.cat([source, cov[:, :enc_len, :]], dim=-1)
        feature_future = self.grn_dec(cov[:, enc_len:, :])
        cnn_output = self.cnn_block(enc_data.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, seq_len, feature)
        feature_past = self.grn(cnn_output)
        attn_input = torch.cat((feature_past, feature_future), dim=1)
        attn_output = self.attention(attn_input)


        mean_p = self.fc_mean(attn_output)
        logs_p = self.fc_std(attn_output)
        

        return mean_p, logs_p
       
############ 后验编码器 ############
class PosteriorEncoder(nn.Module):
    '''
    p(z|x)
    '''
    def __init__(self, 
                 dec_length, 
                 z_dim,
                 input_dim, 
                 channels_dim:list=[16,32,64], 
                 kernel_size=7):
        super(PosteriorEncoder, self).__init__()
        self.dec_length = dec_length
        self.z_dim = z_dim
        channels_dim = channels_dim.copy() # 避免改变外部输入的参数
        # channels_dim.append(z_dim)
        
        
        self.net = TemporalConvNet(input_dim=input_dim, 
                                   channels_dim=channels_dim, 
                                   kernel_size=kernel_size, 
                                   dropout=0.0)
        self.dowmsample = nn.Linear(input_dim, channels_dim[-1])
        self.lastlayer = nn.Sequential(nn.Linear(channels_dim[-1], z_dim),
                                       nn.LayerNorm(z_dim),
                                       nn.ReLU(),
                                       nn.Linear(z_dim,z_dim))
        
        
    def forward(self, source, target, cov):
        '''
        x.shape = [batch, seq_len, feature_dim]
        '''
        x = torch.cat([torch.cat([source, target],dim=1),
                       cov],dim=-1)

        h1 = self.net(x.permute(0,2,1)).permute(0,2,1)
        h2 = F.relu(self.dowmsample(x) + h1)

        z_q = self.lastlayer(h2)
        
        return z_q[:, -self.dec_length:, :]


############ 解码器 ############
class PredictDecoder(nn.Module):
    def __init__(self, dec_length, input_dim:int, hidden_dim:int, output_dim:int=1):
        super(PredictDecoder, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                 nn.LayerNorm(hidden_dim),
                                 nn.ELU(),
                                 nn.Linear(hidden_dim,output_dim),
                                 nn.ELU())
        
    def forward(self, z):
        return self.net(z)


########### 主模型 ################
class FCVAE(nn.Module):
    def __init__(self, enc_dim: int, 
                       cond_channels_dim: list, 
                       post_channels_dim: list,
                       z_dim:int,
                       kernel_size:int, 
                       dropout:float, 
                       hidden_size:int, 
                       num_heads:int, 
                       dec_length:int,
                       device):
        super().__init__()
        self.device = device
        self.dec_length = dec_length
        self.z_dim =z_dim
        # p(f(z)|c)
        self.condition_encoder = ConditionEncoder(input_dim=enc_dim, 
                                                 channels_dim=cond_channels_dim, 
                                                 z_dim=z_dim,
                                                 kernel_size=kernel_size, 
                                                 dropout=dropout, 
                                                 hidden_size=hidden_size, 
                                                 num_heads=num_heads, 
                                                 dec_length=dec_length) 
        # q(z|x)
        self.posterior_encoder = PosteriorEncoder(dec_length=dec_length,
                                                  z_dim=z_dim,
                                                  channels_dim=post_channels_dim,
                                                  input_dim=enc_dim,
                                                  kernel_size=kernel_size) 
        # p(x|c)
        self.predict_decoder = PredictDecoder(dec_length=dec_length,
                                              input_dim=z_dim,
                                              hidden_dim=hidden_size,
                                              output_dim=1)     
        # f(z)
        self.flow = Flow(z_dim=z_dim)                          
        
        
    def forward(self, source, target, cov):
        
        mean_p, logs_p = self.condition_encoder(source, cov)
        z_q = self.posterior_encoder(source, target, cov)
        z_p, logdet_sum = self.flow(z_q) # 由先验采样出来的z_q经过flow变换后得到的z_p
        y_hat = self.predict_decoder(z_q)

        
        return y_hat, z_p, mean_p, logs_p, logdet_sum
        
        
    def loss(self, source, target, cov):
        source = source.to(self.device)
        target = target.to(self.device)
        cov = cov.to(self.device)
        y_hat, z_p, mean_p, logs_p, logdet_sum = self(source, target, cov)
        gaussian = torch.distributions.normal.Normal(mean_p, torch.exp(logs_p))
        log_p = gaussian.log_prob(z_p)
        l1 = F.l1_loss(target, y_hat)
        lh = calc_loss(log_p, logdet_sum)
        
        return l1 + lh
    
    def parallel_sample(self, mean_p, std_p, sample_num):
        mean_sample = mean_p.clone().unsqueeze_(-1)
        mean_sample = mean_sample.repeat([1,1,1,sample_num])
        mean_sample = rearrange(mean_sample,'b s f n -> (n b) s f')
        
        std_sample = std_p.clone().unsqueeze_(-1)
        std_sample = std_sample.repeat([1,1,1,sample_num])
        std_sample = rearrange(std_sample,'b s f n -> (n b) s f')
        
        # 采样：z_p.shape = [sample_num * batch, seq_len, feature_dim]
        z_p = mean_sample + torch.randn_like(mean_sample) * std_sample
        
        return z_p
    
    def predict(self, source, cov, is_get_z = False, sample_num=100):
        source = source.to(self.device)
        cov = cov.to(self.device)
        batch_size = source.shape[0]
        mean_p, logs_p = self.condition_encoder(source, cov)
        std_p = torch.exp(logs_p)
            

        z_p = self.parallel_sample(mean_p, std_p, sample_num)
        z = self.flow.reverse(z_p)
        
        y_hat_sample = self.predict_decoder(z)
        
        y_hat_sample = torch.clamp(y_hat_sample, 0, 1)
        y_hat_list = torch.split(y_hat_sample, [batch_size]*sample_num, 0)
        y_hat_sample = torch.cat(y_hat_list, 2)
        y_hat_sample = torch.sort(y_hat_sample, dim=-1)[0]
        
        y_hat = torch.median(y_hat_sample, dim=-1, keepdim=True)[0]

        if is_get_z:
            return y_hat, y_hat_sample, z_p, z
        else:
            return y_hat, y_hat_sample
    
    def recurrent(self, source, cov, sample_num=100):
        '''
        通过逐步递归实现多步预测 (效果很烂)
        调用此方法：需要使用 1 步预测 训练出的模型
        '''
        source = source.to(self.device)
        cov = cov.to(self.device)
        source_len = cov.shape[1]
        target_len = cov.shape[1] - source.shape[1]
        y_hat_list = []
        y_hat_sample_list = []
        for i in range(target_len):
            source_step = source[:, i:, :]
            cov_step = cov[:, i: source_len+1+i, :]
            y_hat_step, y_hat_sample_step = self.predict(source_step, cov_step, sample_num)
            y_hat_list.append(y_hat_step)
            y_hat_sample_list.append(y_hat_sample_step)
            source = torch.cat([source, y_hat_step], dim=1)
            
        return torch.cat(y_hat_list, dim=1), torch.cat(y_hat_sample_list, dim=1)
        

def calc_loss(log_p, logdet):
    _, seq_length, sample_num = log_p.shape
    log_p = torch.sum(log_p, [-1,-2])
    n_pixel = seq_length * sample_num
    loss = logdet + log_p
    return (-loss / (log(2) * n_pixel)).mean()
############################################ 内部模块 ##############################################
############################ 门控模块 ##########################
class GatedLinearUnit(nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 dropout_rate=None):

        super(GatedLinearUnit, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        self.W4 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.W5 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if 'bias' not in n:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in n:
                torch.nn.init.zeros_(p)

    def forward(self, x):
        """
        输入的维度：(batch, time_step, feature)
        """
        if self.dropout_rate:
            x = self.dropout(x)
        output = self.sigmoid(self.W4(x)) * self.W5(x)
        return output


class GRN(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        """
        符号参考tft这篇论文
        input_size: 输入x的特征维度
        static_cov的特征维度embedding_size = hidden_size
        """
        super().__init__()
        self.dowmsample = None

        if input_size != hidden_size:
            self.dowmsample = nn.Linear(input_size, hidden_size)

        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(input_size, hidden_size)  # 用以处理x
        self.W3 = nn.Linear(hidden_size, hidden_size)  # 用以处理static_embedding
        self.elu = nn.ELU()
        self.glu = GatedLinearUnit(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x, cond=None):
        """
        x = (batch, seq_len, feature_size)
        """
        if self.dowmsample is not None:
            res = self.dowmsample(x)
        else:
            res = x
        if cond is not None:
            eta2 = self.elu(self.W2(x) + self.W3(cond))
        eta2 = self.elu(self.W2(x))
        eta1 = self.W1(eta2)
        output = self.layernorm(res + self.glu(eta1))

        return output
    
#################### 卷积 模块 ######################
class Chomp1d(nn.Module):
    """
    用来做padding的
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0):
        """
        n_inputs:输入变量（batch,channel,seq_len）的channel
        n_outputs:输出变量的通道数，可以理解为输出的特征维度
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    输出序列的长度与输入序列相等
    n_inputs：输入序列的通道数（维数）
    num_channels：各TCN块的通道数；（最终的TCN网络是由len(num_channels)层TCN块堆叠而成的）
    """

    def __init__(self, input_dim, channels_dim, kernel_size=7, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(channels_dim)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 层数越深越稀疏
            in_channels = input_dim if i == 0 else channels_dim[i - 1]
            out_channels = channels_dim[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    
####################### 注意力网络 #######################
class PositionalEmbedding(nn.Module):
    """
    位置信息嵌入
    attention的输入维度(seq_len, batch, model_dim)
    位置信息要与attention的维度匹配
    """

    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)  # (seq_len, model_dim)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (batch, seq_len, model_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.shape[1], :]


class AttentionNet(nn.Module):
    """
    attention 的本质是获取各个位置之间的关系,不涉及特征维度的变化

    """
    def __init__(self, embed_dim, num_heads, dec_length) -> None:
        super().__init__()
        # attention的输入维度(batch, sqe, embed_dim)
        # model_dim就是feature的维度
        self.dec_length = dec_length
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.pe_embedding = PositionalEmbedding(embed_dim=embed_dim)
        self.fc_key = nn.Linear(embed_dim, embed_dim)
        self.fc_value = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        pos = self.pe_embedding(x)
        x = x + pos
        query = x[:, -self.dec_length:, :]
        key = self.fc_key(x[:, :-self.dec_length, :])
        value = self.fc_value(x[:, :-self.dec_length, :])
        attn_output, _ = self.attention(query, key, value)
        output = self.layer_norm(attn_output)
        return self.relu(output)


    

############## 流模型 #################
class ComplexNet(nn.Module):
    def __init__(self,
                 input_dim, 
                 channels_dim,
                 dropout=0) -> None:
        
        super().__init__()
        
        
        self.grn = GRN(input_dim, input_dim)
        self.tcn = TemporalConvNet(input_dim, 
                                    channels_dim, 
                                    kernel_size=7, 
                                    dropout=dropout)
        self.elu = nn.ELU()
    def forward(self, x):
        '''
        x.shape = (batch, seq_len, channels)
        '''
        h1 = self.grn(x).permute(0, 2, 1)
        output = self.tcn(h1).permute(0, 2, 1)
        return self.elu(output) 

# TODO: 
class Flip(nn.Module):
    def forward(self, x):
        x = torch.flip(x, [-1])
        logdet = torch.tensor(0).to(dtype=x.dtype, device=x.device)
            # logdet = torch.zeros(x.shape[0]).to(dtype=x.dtype, device=x.device) 
        return x, logdet
        
    def reverse(self, x):
        x = torch.flip(x, [-1])
        return x


class ElementwiseAffine(nn.Module):
    '''
    对应glow中的Actnorm
    (batch_size, channels, dec_len)
    '''
    def __init__(self, z_dim):
        super().__init__()
        self.m = nn.Parameter(torch.zeros(z_dim))
        self.logs = nn.Parameter(torch.ones(z_dim))

    def forward(self, x):
        _, dec_len, _ = x.shape
        y = self.m + torch.exp(self.logs) * x
        logdet = dec_len * torch.sum(self.logs)
        return y, logdet
            
        
    def reverse(self, x):
        x = (x - self.m) * torch.exp(-self.logs)
        return x
        


class InvLinear(nn.Module):
    '''
    如何在神经网络优化的过程中使得 W 满足可逆？
    '''
    def __init__(self, in_features) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, in_features, bias=False)

    def forward(self, x):
        _, dec_len, _ = x.shape
        
        out = self.fc(x)
        logdet = (
            dec_len * torch.slogdet(self.fc.weight.double())[1].float()
        )
        return out, logdet
            
            
    def reverse(self, x):
        return F.linear(
                x, self.fc.weight.inverse()
            )



class ResidualCouplingLayer(nn.Module):
    '''
    对应glow中的Affine coupling layer
    '''
    def __init__(self,
                z_dim,
                mean_only=False):
        assert z_dim % 2 == 0, "隐变量 z 的维度 z_dim 必须是2的倍数"
        
        super().__init__()
        self.z_dim = z_dim
        self.half_zdim = z_dim // 2
        self.mean_only = mean_only

        # 如果只输出均值，那么维度就不用乘2；乘2时为了如果需要标准差时，将一半的通道作为标准差
        last_channel = self.half_zdim * (2 - mean_only)
        self.enc = ComplexNet(input_dim = self.half_zdim, 
                              channels_dim = [self.half_zdim, z_dim, 
                                              z_dim, last_channel])
        

    def forward(self, x):
        '''
        x.shape = (batch, seq_len, feature)
        '''
        x0, x1 = torch.split(x, [self.half_zdim]*2, -1)
        stats = self.enc(x0)
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_zdim]*2, -1)
        else:
            m = stats
            logs = torch.zeros_like(m)
            
        x1 = m + x1 * torch.exp(logs)
        x = torch.cat([x0, x1], -1)
        logdet = torch.sum(logs)
        return x, logdet
        
    def reverse(self, x):
        x0, x1 = torch.split(x, [self.half_zdim]*2, -1)
        stats = self.enc(x0)
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_zdim]*2, -1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        x1 = (x1 - m) * torch.exp(-logs)
        x = torch.cat([x0, x1], -1)
        return x
        

class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                z_dim,
                n_flows=2):
        super().__init__()
        self.n_flows = n_flows
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(z_dim=z_dim,
                                                    mean_only=True))
            self.flows.append(Flip())

    def forward(self, x):
        '''
        x.shape = (batch, feature, seq_len)
        '''
        logdet_sum = torch.zeros(x.shape[0]).to(x.device)
        # logdet_sum = 0
        for flow in self.flows:
            x, logdet = flow(x)
            logdet_sum += logdet
        return x, logdet_sum
        
    def reverse(self, x):
        for flow in reversed(self.flows):
            x = flow.reverse(x)
        return x
        


class Flow(nn.Module):
    def __init__(self, 
                 z_dim) -> None:
        '''
        params: 
            hidden_size: 静态协变量经过编码器之后的维度
            channels: 目标时间序列经过squeezing之后的维度,在实际操作中将目标序列长度压缩为1,将通道大小扩张为序列长度的大小
                      需要注意的是, channels必须为偶数
            kernel_size: 为1维卷积的卷积核大小
        '''
        super().__init__()
        self.flows = nn.ModuleList()
        # self.flows.append(InvReLU())
        self.flows.append(ElementwiseAffine(z_dim=z_dim))
        self.flows.append(InvLinear(in_features=z_dim))
        # self.flows.append(ResidualCouplingBlock(z_dim))
        
    def forward(self, x):
        '''
        params:
            x:在flow前向过程时, x为由需要预测的目标进过线性encoder升维过后得到的隐变量,
              在flow逆向过程时, x为由历史序列经过encoder和decoder后的隐变量
              x.shape = (batch_size, feature_size, seq_len), x在输入时seq_len会被压缩为1, 以增大feature的维度 
            g:为静态协变量经过encoder所得到的隐变量
            reverse: 为True则为flow的逆向过程, 为False则为Flow的前向过程
        output:
            在前向时需要logdet_sum, 用来计算KL散度损失
        '''
        logdet_sum = torch.zeros(x.shape[0]).to(x.device)
        for flow in self.flows:
            x, logdet = flow(x)
            logdet_sum += logdet
        return x, logdet_sum
        
    def reverse(self, x):
        # 在flow逆向过程时，x为由历史序列经过encoder和decoder后的隐变量
        for flow in reversed(self.flows):
            x = flow.reverse(x)
        return x


class ELUPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()
    def forward(self, x):
        return self.elu(x) + 1.
