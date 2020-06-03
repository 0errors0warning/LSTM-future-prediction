# # # -*- coding: utf-8 -*-
# # """
# # Created on Tue Apr  7 12:50:45 2020

# # @author: MSI-PC
# # """
import pywt
import numpy as np
import torch
from torch import nn
import xlrd
import matplotlib.pyplot as plt
def ReLu(x):
    x = (np.abs(x) + x) / 2.0
    return x
def wavlet(data):
    print(data.shape)
    w = pywt.Wavelet('db8')  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    print("maximum level is " + str(maxlev))
    threshold = 0.01  # Threshold for filtering
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))  # 将噪声滤波
    datarec = pywt.waverec(coeffs, 'db8')[:len(data):]  # 将信号进行小波重构
    print(datarec.shape)
    return datarec
def wavletseq(data):
    gkd=wavlet(data.T[0])
    data=data.T
    data[0]=gkd
    data=data.T
    return data
def run_train_lstm():
    plt.figure(figsize=(50, 10), dpi=100)
    inp_dim = 2
    out_dim = 1
    mid_dim = 3
    mid_layers = 1
    batch_size = 64
    mod_dir = '.'

    '''load data'''
    data,std,mean = load_data_hour()
    inp_dim = data.shape[1]
    #=====================如果希望预测大概趋势，取消注释下行代码=======================
    # data=wavletseq(data)
    #=================================================================
    data_x = data[:-1, :]
    data_y = data[+1:, 0]
    assert data_x.shape[1] == inp_dim

    train_size = int(len(data_x)-70)
    test_size=len(data_x)-train_size
    train_x = data_x[:train_size]
    train_y = data_y[:train_size]
    train_x = train_x.reshape((train_size, inp_dim))
    train_y = train_y.reshape((train_size, out_dim))
    print("train x=",train_x.shape)
    print("train y=",train_y.shape)
    '''build model'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegLSTM(inp_dim, out_dim, mid_dim, mid_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    #==========以下是训练过程，模型生成以后可以把下面注释掉===============
    '''train'''
    var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
    var_y = torch.tensor(train_y, dtype=torch.float32, device=device)

    batch_var_x = list()
    batch_var_y = list()

    for i in range(batch_size):
        j = train_size - i
        batch_var_x.append(var_x[j:])
        batch_var_y.append(var_y[j:])

    from torch.nn.utils.rnn import pad_sequence
    batch_var_x = pad_sequence(batch_var_x)
    batch_var_y = pad_sequence(batch_var_y)

    with torch.no_grad():
        weights =  np.tanh(np.arange(len(train_y)) * (np.e / len(train_y)))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print("Training Start")
    for e in range(len(data_y)):
        out = net(batch_var_x)
        loss = criterion(out, batch_var_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if e % 64 == 0:
            print('Epoch: {:4}, Loss: {:.5f}'.format(e, loss.item()))
    torch.save(net.state_dict(), '{}/net1.pth'.format(mod_dir))
    print("Save in:", '{}/net1.pth'.format(mod_dir))
    #=======================================================
    net.load_state_dict(torch.load('{}/net1.pth'.format(mod_dir), map_location=lambda storage, loc: storage))
    net = net.eval()

    test_x = data_x.copy()
    test_x[train_size:, 0] = 0
    #++++++++++++++++++++++希望预测的未来小时数
    pl=6
    #++++++++++++++++++++++
    cap=np.zeros([pl,inp_dim])
    test_x=np.vstack((test_x,cap))

    test_x = test_x[:, np.newaxis, :]
    test_x = torch.tensor(test_x, dtype=torch.float32, device=device)
    eval_size = 1
    zero_ten = torch.zeros((mid_layers, eval_size, mid_dim), dtype=torch.float32, device=device)

    test_y, hc = net.output_y_hc(test_x[:train_size], (zero_ten, zero_ten))

    test_x[train_size + 1, 0, 0] = test_y[-1]
    for i in range(train_size + 1, len(data) +pl- 2):
        test_y, hc = net.output_y_hc(test_x[i:i + 1], hc)
        test_x[i + 1, 0, 0] = test_y[-1]
    pred_y = test_x[1:, 0, 0]

    pred_y = pred_y.cpu().data.numpy()
    print("real y=",((data_y*std+mean)) )
    pred_y=(pred_y*std+mean)
    data_y=(data_y*std+mean)

    k=test_size
    corection=sum((pred_y[len(data_y)-k:len(data_y):]-data_y[len(data_y)-k::]))/(k)
    pred_y[len(data_y)-k::]-=corection
    #++++++++++++++++++++调节显示长度，数字越大，时间段越长============================
    length=60
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++==================


    corection=sum((pred_y[len(data_y)-length:len(data_y):]-data_y[len(data_y)-length::]))/(length)
    pred_y[len(data_y)-length::]-=corection
    print("pred_y y=",(pred_y)[len(data_y)::])
    h=max(pred_y[len(data_y)-length::].max(),data_y[len(data_y)-length::].max())
    l=min(pred_y[len(data_y)-length::].min(),data_y[len(data_y)-length::].min())
    plt.ylim(l, h)
    plt.xlim(0, length+pl)
    plt.plot(pred_y[len(data_y)-length::], 'r', label='pred', alpha=0.3)
    plt.plot(data_y[len(data_y)-length::], 'b', label='real', alpha=0.3)
    errorfactor=sum(abs(pred_y[len(data_y)-length:len(data_y):]-data_y[len(data_y)-length::]))/length
    evl_data=sum(data_y[len(data_y)-length::])/length
    evl_pred=sum(data_y[len(pred_y)-length:len(data_y):])/length


    pred_fluctatefactor=(max((pred_y[len(data_y)-length:len(data_y):]))-min((pred_y[len(data_y)-length:len(data_y):])))/evl_pred
    data_fluctatefactor=(max(abs(data_y[len(data_y)-length:len(data_y):]))-min(abs(data_y[len(data_y)-length:len(data_y):])))/evl_data
    print("原波动率为：",data_fluctatefactor*100,"%")
    print("预测波动率为：",pred_fluctatefactor*100,"%")
    print("预波动幅度的平均差值为：",abs(data_fluctatefactor*100-pred_fluctatefactor*100),"%")
    print("真实值和预测值两者的平均差值：",errorfactor)
    plt.plot([train_size, train_size], [-1, 2], color='k', label='train | pred')
    plt.legend(loc='best')
    plt.savefig('lstm_reg.png')



class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()

        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y


    def output_y_hc(self, x, hc):
        y, hc = self.rnn(x, hc)  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, hc


class RegGRU(nn.Module):
    def __init__(self, inp_dim, out_dim, mod_dim, mid_layers):
        super(RegGRU, self).__init__()

        self.rnn = nn.GRU(inp_dim, mod_dim, mid_layers)
        self.reg = nn.Linear(mod_dim, out_dim)

    def forward(self, x):
        x, h = self.rnn(x)  # (seq, batch, hidden)

        seq_len, batch_size, hid_dim = x.shape
        x = x.view(-1, hid_dim)
        x = self.reg(x)
        x = x.view(seq_len, batch_size, -1)
        return x

    def output_y_h(self, x, h):
        y, h = self.rnn(x, h)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, h


def load_data_hour():
    data = xlrd.open_workbook('LSTM1.xlsx')
    table = data.sheet_by_name('Sheet1')
    i1=0
    i2=1
    a=np.array(table.col_values(i1))[1::]
    b=np.array(table.col_values(i2))[1::]
    seq_number=np.vstack((a,b))
    # +++++++++++++添加关联数据代码+++++++++++++++++++++++++
#+++==============k是想要添加数据列的序号，计数从零开始=======================
    # k=2        #比如k=2是第三列
#+++==============k是想要添加数据列的序号，计数从零开始=======================
    # a=np.array(table.col_values(k))[1::]
    # seq_number=np.vstack((seq_number,a))
    # +++++++++++++添加关联数据代码+++++++++++++++++++++++++
    
    
    # 添加多列只要如下复制即可
    
    
    
    #+++==============k是想要添加数据列的序号，计数从零开始=======================
    # k=3       #比如k=2是第三列
#+++==============k是想要添加数据列的序号，计数从零开始=======================
    # a=np.array(table.col_values(k))[1::]
    # seq_number=np.vstack((seq_number,a))
    # +++++++++++++添加关联数据代码+++++++++++++++++++++++++
    
    
    seq=(seq_number.T).astype(np.float64)
    a=seq.mean(axis=0)[0]
    b=seq.std(axis=0)[0]    
    seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
    return seq,b,a

def load_data_week():
    data = xlrd.open_workbook('LSTM1.xlsx')
    table = data.sheet_by_name('Sheet1')
    a=np.array(table.col_values(1)[1::])
    b=np.array(table.col_values(7)[1::])
    seq_number=np.vstack((a,b))
    for i in range(7,10):
        b=np.array(table.col_values(i+1)[1::])
        seq_number=np.vstack((seq_number,b))
    seq=seq_number.T
    a=seq.mean(axis=0)[0]
    b=seq.std(axis=0)[0]
    seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
    print(seq.std(axis=0)[0])
    return seq,b,a

if __name__ == '__main__':
    run_train_lstm()