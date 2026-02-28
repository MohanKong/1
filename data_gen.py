import numpy as np
import config

def data_gen_mmse(transmit_user, noise_var): # 生成数据的函数，输入参数为配置字典 cfg # 根据 cfg 中的参数生成训练数据和测试数据 # 返回训练数据和测试数据 pass
    # 信道矩阵
    H = (np.random.randn(config.Nr, config.Nt, config.N_user) + 1j * np.random.randn(config.Nr, config.Nt, config.N_user)) / np.sqrt(2)
    # 生成随机符号
    s_bits = np.random.randint(0, config.Mod, (config.Nr, config.N_symbol))
    # PSK调制
    s = np.exp(1j * (2 * np.pi * s_bits / config.Mod + np.pi / 4))
    # 产生加性高斯噪声
    n = np.random.normal(0, np.sqrt(noise_var / 2), (config.Nr, config.N_symbol)) + 1j * np.random.normal(0, np.sqrt(noise_var / 2),(config.Nr, config.N_symbol))
    # MMSE编码
    W_mmse = np.zeros((config.Nt, config.Nr, config.N_user), dtype=complex)
    for ite2 in range(config.N_user):       
        W_mmse_t = H[:, :, ite2].conj().T @ H[:, :, ite2] + noise_var * np.eye(config.Nt)
        W_mmse[:,:,ite2] = np.linalg.inv(W_mmse_t) @ H[:, :, ite2].conj().T
    # 生成x
    x = W_mmse[:, :, transmit_user - 1] @ s
    # 生成信号
    y = H[:, :, transmit_user - 1] @ x + n
    
    block_data = {
        "k": transmit_user,
        "noise_var": noise_var,
        "H": H,
        "s_bits": s_bits,
        "s": s,
        "n": n,
        "x": x,
        "y_clean": y,
    }
    return block_data