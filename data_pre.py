import numpy as np
import config
import itertools

def data_pre(block_data):
    k = block_data["k"]
    noise_var = block_data["noise_var"]
    H = block_data["H"]
    s = block_data["s"]
    s_bits = block_data["s_bits"]
    n = block_data["n"]
    x = block_data["x"]
    y = block_data["y_clean"]
    
    # 数据增强：使用itertools.product生成所有可能的调制符号组合
    combinations = itertools.product(range(config.Mod), repeat=config.Nt)
    Aug_matrix = np.array(list(combinations)).T  # 转置使每列为一个组合
    Aug_matrix = np.exp(1j * (2 * np.pi * Aug_matrix / config.Mod + np.pi/4))  # 调制
    
    # 限制数据增强组合数量上限
    N_combinations = config.Mod ** config.Nt # 数据增强组合数量
    N_combinations_max = config.Mod ** 5  # 数据增强组合数量上限
    if N_combinations > N_combinations_max:
        Aug_filter = np.random.choice(N_combinations, N_combinations_max, replace=False)
        Aug_matrix_filtered = Aug_matrix[:, Aug_filter]
        N_combinations_filtered = N_combinations_max
    else:
        N_combinations_filtered = N_combinations
        Aug_matrix_filtered = Aug_matrix
        
    # 针对每个用户进行数据增强，并拼接得到训练输入数据
    H_reshaped = np.empty((config.Nr, 0))
    # 随机生成噪声大小
    noise_var_aug = np.exp(np.random.uniform(np.log(config.noise_var_aug_min), np.log(config.noise_var_aug_max))) #对数均匀分布随机数
    
    for i in range(config.N_user):
        n_aug = np.random.normal(0, np.sqrt(noise_var_aug/2), (config.Nr, N_combinations_filtered)) \
                + 1j * np.random.normal(0, np.sqrt(noise_var_aug/2), (config.Nr, N_combinations_filtered))
        H_aug = H[:, :, i] @ Aug_matrix_filtered + n_aug
        H_reshaped = np.hstack((H_reshaped, H_aug))
    # 分离实部和虚部
    H_reshaped = np.vstack((np.real(H_reshaped), np.imag(H_reshaped)))
    pilot_train = H_reshaped.T
    
    # 生成训练集标签：每个用户对应一个数字标签
    pilot_label = np.zeros((config.N_user * N_combinations_filtered))
    for i in range(config.N_user):
        pilot_label[i * N_combinations_filtered : (i + 1) * N_combinations_filtered] = i
    pilot_label = pilot_label.T
    # 生成训练集标签：每个用户对应一个独热编码标签
    # pilot_label = np.zeros((N_user, N_combinations_filtered * N_user))
    # for i in range(N_user):
    #     pilot_label[i, i * N_combinations_filtered : (i + 1) * N_combinations_filtered] = 1
    # pilot_label = pilot_label.T
    
    # 生成测试集数据
    signal_train = np.vstack([np.real(y), np.imag(y)]).T
    # 生成测试集标签：每个用户对应一个数字标签
    signal_label = np.zeros((y.shape[1]))
    signal_label[:] = k - 1 # 用户编号从1开始，标签从0开始
    signal_label = signal_label.T
    # 生成测试集标签：每个用户对应一个独热编码标签
    # signal_label = np.zeros((N_user, y.shape[1]))
    # signal_label[N_user_index-1, :] = 1
    # signal_label = signal_label.T

    # 数据归一化（使用训练数据的均值和标准差）
    mean_train = np.mean(pilot_train, axis=0)
    std_train = np.std(pilot_train, axis=0) + config.eps
    pilot_train_normalized = (pilot_train - mean_train) / std_train
    signal_train_normalized = (signal_train - mean_train) / std_train
    
    return {
    "x_train": pilot_train_normalized,
    "y_train": pilot_label,
    "x_test": signal_train_normalized,
    "y_test": signal_label,
    "norm_mean": mean_train,
    "norm_std": std_train,
    }

