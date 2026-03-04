import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import config

def det_train(det_inputs, detector_net, detector_optimizer, detector_loss_fn):
    # 解构det_inputs
    x_train = det_inputs["x_train"]
    y_train = det_inputs["y_train"]
    x_test = det_inputs["x_test"]
    y_test = det_inputs["y_test"]
    
    # 训练dataset
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    # 测试dataset
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    # 训练dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    # 测试dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    #记录每个epoch的预测结果
    pred_historys = np.empty((0, config.N_user))

    #第二个循环：epoch的循环
    for epoch in range(config.n_epoch):    
        #epoch开始时间
        epoch_start_time = time.time()
        
        #训练模式
        detector_net.train()
        
        #batch的数量
        len_train_dataloader = len(train_dataloader)
        # 训练和测试的数据迭代器
        data_train_iter = iter(train_dataloader)
        data_test_iter = iter(test_dataloader)
        # 测试集数据
        data_test = next(data_test_iter)
        
        #第三个循环：batch的循环
        for ite3 in range(len_train_dataloader):            
            #加载这个batch的训练用数据
            data_batch_train = next(data_train_iter)
            #这个batch的数据和标签
            batch_inputs, batch_labels = data_batch_train
            #梯度清零
            detector_optimizer.zero_grad()
            
            #源域的数据和标签放到GPU上
            if config.USE_CUDA:
                batch_inputs = batch_inputs.cuda()
                batch_labels = batch_labels.cuda()
                
            #将数据输入网络得到输出
            class_output = detector_net(batch_inputs)
            #计算损失
            err = detector_loss_fn(class_output, batch_labels.long())
            #反向传播
            err.backward()
            #更新参数
            detector_optimizer.step()
            
        # 在每个epoch的最后，评估所有指标
        detector_net.eval()
        with torch.no_grad():
            # 评估训练集指标
            epoch_train_class_loss = 0.0 # 每个epoch的总训练集分类损失
            epoch_train_probability = np.empty((0, config.N_user)) # 每个epoch的总训练集分类概率
            epoch_train_correct_prob = [] # 每个epoch的总训练集正确分类概率
            
            # 遍历整个源域数据集
            for data_batch_train in train_dataloader:
                batch_train_inputs, batch_train_labels = data_batch_train
                if config.USE_CUDA:
                    batch_train_inputs = batch_train_inputs.cuda()
                    batch_train_labels = batch_train_labels.cuda()
                
                # 前向传播
                train_class_output = detector_net(batch_train_inputs)
                
                # 分类损失
                train_class_loss = detector_loss_fn(train_class_output, batch_train_labels)
                epoch_train_class_loss += train_class_loss.item() * len(batch_train_inputs)
                
                # 记录正确分类概率
                batch_train_probability = torch.exp(train_class_output).cpu().numpy() # 每个batch的分类概率
                epoch_train_probability = np.vstack((epoch_train_probability, batch_train_probability)) # 每个epoch的分类概率
                batch_train_correct_probability = batch_train_probability[np.arange(len(batch_train_labels)), batch_train_labels.cpu().numpy()]
                epoch_train_correct_prob.extend(batch_train_correct_probability)

                
            # 计算平均值
            train_class_loss = epoch_train_class_loss / len(train_dataset)
            epoch_train_correct_prob = np.mean(epoch_train_correct_prob)

            # 评估测试集指标
            test_inputs, test_labels = data_test
            if config.USE_CUDA:
                test_inputs = test_inputs.cuda()
                test_labels = test_labels.cuda()
            
            # 前向传播
            test_class_output = detector_net(test_inputs)
            
            # 分类损失
            test_class_loss = detector_loss_fn(test_class_output, test_labels)
            # 记录正确分类概率
            test_probability = torch.exp(test_class_output).cpu().numpy()
            test_correct_probability = test_probability[np.arange(len(test_labels)), test_labels.cpu().numpy()]
            test_correct_probability = np.mean(test_correct_probability, axis=0)
            
            # 记录每个epoch的分类概率
            pred_historys = np.vstack((pred_historys, np.mean(test_probability, axis=0)))

        # epoch结束时间
        epoch_end_time = time.time()

        # 修改输出语句
        print(
            f'epoch: {epoch+1}, '
            f'epoch_time: {epoch_end_time - epoch_start_time:.2f}, '
            f'train_class_loss: {train_class_loss:.2f}, '
            f'train_correct_prob: {epoch_train_correct_prob:.2f}, '
            f'test_class_loss: {test_class_loss.item():.2f}, '
            f'test_correct_prob: {test_correct_probability:.2f}'
        )
            
