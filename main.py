import os
import config
if config.USE_CUDA:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_ID
    
# import 其他库/模块
import time
import numpy as np
import scipy.io
import torch
from torch.utils.data import TensorDataset, DataLoader
import data_gen
import data_pre
from model import detector_model_mmse
import det_train
import anonymizer
import det_eval
import functions

#设置工作环境
#os.chdir(r'/home/uceemko/p3-env8/mmse_mlp')
#print("Current work path is ", os.getcwd())

# 设置参数
noise_var = 0.1
precoder_type = "mmse"

#模型以及优化器、损失函数定义
if precoder_type == "mmse":
    detector_net = detector_model_mmse(2*config.Nr, config.N_user, config.dropout_rate)
#detector optimizer
detector_optimizer = torch.optim.AdamW(detector_net.parameters(), weight_decay=config.kernel_regular)
#损失函数
detector_loss_fn = torch.nn.NLLLoss()

#将模型和损失函数放到GPU上
if config.USE_CUDA:
    detector_net = torch.nn.DataParallel(detector_net) # 这个可能会引起一些问题，后续可以考虑改成单GPU训练
    detector_net = detector_net.cuda()
    detector_loss_fn = detector_loss_fn.cuda()

#将模型的参数设置为可训练
for p in detector_net.parameters():
    p.requires_grad = True
    
ser_clean_list, der_clean_list = [], []
ser_anon_list,  der_anon_list  = [], []

for ite1 in range(config.N_block):
    # 决定发送用户
    transmit_user = np.random.randint(1, config.N_user + 1)  # NumPy 的范围是左闭右开
    
    # 生成数据
    if precoder_type == "mmse":
        block_data = data_gen.data_gen_mmse(transmit_user=transmit_user, noise_var=noise_var)
    
    # 数据预处理
    det_inputs = data_pre.data_pre(block_data)
    
    # detector网络
    detector_net.train()
    predict_output = det_train.det_train(det_inputs, detector_net, detector_optimizer, detector_loss_fn)

    # clean 指标（评估模式，不更新参数）
    detector_net.eval()
    with torch.no_grad():
        metrics_clean = det_eval.det_eval(block_data, detector_net, detector_loss_fn)
        ser_clean = functions.ser(block_data)

    # 匿名器
    for p in detector_net.parameters():
        p.requires_grad_(False)
    block_data_anon, pert = anonymizer.anonymize(block_data, det_inputs, detector_net)
    for p in detector_net.parameters():
        p.requires_grad_(True)   # 如果后面还要继续训练

    # 匿名输入过detector得到匿名的检测结果
    detector_net.eval()
    with torch.no_grad():
        metrics_anon = det_eval.det_eval(block_data_anon, detector_net, detector_loss_fn)
        ser_anon = functions.ser(block_data_anon)
    
    # 记录
    ser_clean_list.append(ser_clean)
    ser_anon_list.append(ser_anon)
    der_clean_list.append(metrics_clean["der"])
    der_anon_list.append(metrics_anon["der"])

ser_clean_mean = np.mean(ser_clean_list)
ser_anon_mean  = np.mean(ser_anon_list)
der_clean_mean = np.mean(der_clean_list)
der_anon_mean  = np.mean(der_anon_list)

print(f"SER clean mean: {ser_clean_mean:.6f}")
print(f"SER anon  mean: {ser_anon_mean:.6f}")
print(f"DER clean mean: {der_clean_mean:.6f}")
print(f"DER anon  mean: {der_anon_mean:.6f}")