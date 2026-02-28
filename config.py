USE_CUDA = False      # 想用GPU就True；强制CPU就False
GPU_ID = "0"

#变量定义
batch_size = 64  # 批大小
dropout_rate = 0.05 # Dropout概率
kernel_regular = 0.0005 #正则化参数
cuda = True # 是否使用GPU
n_epoch = 200  # 训练轮数
noise_var = 0.01 # 定义 noise_var 的值

#数据导入
N_block = 1000
Nr = 5
Nt = 5
N_user = 5

# （建议新增，但也可先不加）
N_symbol = 50
Mod = "QPSK"
precoder_type = "identity"   # 预留：identity/mmse/isa
complex_repr = "ri"          # "ri" 或 "complex"
#标准差偏移量
eps = 1e-6
#随机生成噪声大小
noise_var_aug_max = 1 #噪声方差最大值
noise_var_aug_min = 0.001 #噪声方差最小值

dropout_rate = 0.05
kernel_regular = 0.0005

# cfg = {
#     "USE_CUDA": USE_CUDA,
#     "GPU_ID": GPU_ID,
#     "cuda": cuda,

#     "batch_size": batch_size,
#     "dropout_rate": dropout_rate,
#     "kernel_regular": kernel_regular,
#     "n_epoch": n_epoch,

#     # --- data_gen 必要/推荐/预留 ---
#     "noise_var": noise_var,
#     "N_block": N_block,
#     "Nr": Nr,
#     "Nt": Nt,
#     "N_user": N_user,

#     "N_symbol": N_symbol,
#     "mod": mod,

#     "precoder_type": precoder_type,
#     "complex_repr": complex_repr,
# }