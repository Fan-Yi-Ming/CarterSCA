import random
import cupy as cp
import numpy as np
from datetime import datetime


def generate_random_hex_string(length=16) -> str:
    """
    生成指定长度的随机十六进制字符串，并以空格分隔每两个字符。

    :param length: 十六进制字符串的长度（默认为16）
    :return: 格式化的十六进制字符串
    """
    hex_chars = '0123456789ABCDEF'
    hex_string = ''.join(random.choice(hex_chars) for _ in range(length * 2))
    formatted_hex_string = ' '.join(hex_string[i:i + 2] for i in range(0, len(hex_string), 2))
    return formatted_hex_string


def index_str_to_range(index_str: str):
    if '-' in index_str:
        start, end = map(int, index_str.split('-'))
        # 为了包含结束值，需要 end + 1
        return list(range(start, end + 1))
    else:
        # 单个数字返回包含该数字的列表
        return [int(index_str)]


def hw(byte):
    """计算单个字节的汉明重量"""
    weight = 0
    while byte:
        weight += byte & 1
        byte >>= 1
    return weight


def analyze_process_cpa_cpu(data_arr_2d, sample_arr_2d):
    epsilon = 1e-10

    data_mean = np.mean(data_arr_2d, axis=0, keepdims=True)
    data_std = np.std(data_arr_2d, axis=0, keepdims=True)
    data_std = np.maximum(data_std, epsilon)
    data_norm = (data_arr_2d - data_mean) / data_std

    sample_mean = np.mean(sample_arr_2d, axis=0, keepdims=True)
    sample_std = np.std(sample_arr_2d, axis=0, keepdims=True)
    sample_std = np.maximum(sample_std, epsilon)
    sample_norm = (sample_arr_2d - sample_mean) / sample_std

    trace_number = data_arr_2d.shape[0]
    correlation_arr_2d = np.dot(data_norm.T, sample_norm) / (trace_number - 1)
    correlation_arr_2d = np.clip(correlation_arr_2d, -1.0, 1.0)

    return correlation_arr_2d.astype(np.float32)


def analyze_process_cpa_gpu(data_arr_2d, sample_arr_2d, batch_size=10000):
    epsilon = 1e-10

    data_arr_2d_gpu = cp.ascontiguousarray(cp.asarray(data_arr_2d))
    sample_arr_2d_gpu = cp.ascontiguousarray(cp.asarray(sample_arr_2d))

    trace_number, sbox_size = data_arr_2d_gpu.shape
    sample_number = sample_arr_2d_gpu.shape[1]

    data_mean = cp.mean(data_arr_2d_gpu, axis=0, keepdims=True)
    data_std = cp.std(data_arr_2d_gpu, axis=0, keepdims=True)
    data_std = cp.maximum(data_std, epsilon)

    sample_mean = cp.mean(sample_arr_2d_gpu, axis=0, keepdims=True)
    sample_std = cp.std(sample_arr_2d_gpu, axis=0, keepdims=True)
    sample_std = cp.maximum(sample_std, epsilon)

    data_arr_2d_gpu -= data_mean
    data_arr_2d_gpu /= data_std

    sample_arr_2d_gpu -= sample_mean
    sample_arr_2d_gpu /= sample_std

    correlation_arr_2d_gpu = cp.zeros((sbox_size, sample_number), dtype=data_arr_2d_gpu.dtype)

    num_batches = (sample_number + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, sample_number)

        sample_batch_norm = sample_arr_2d_gpu[:, start:end]

        batch_correlation = cp.dot(data_arr_2d_gpu.T, sample_batch_norm) / (trace_number - 1)
        batch_correlation = cp.clip(batch_correlation, -1.0, 1.0)

        correlation_arr_2d_gpu[:, start:end] = batch_correlation

    result = correlation_arr_2d_gpu.get()
    return result.astype(np.float32)


def rank_sbox_key_guesses(correlation_arr_2d, candidates):
    sbox_size = correlation_arr_2d.shape[0]

    abs_corr = np.abs(correlation_arr_2d)
    max_abs_val = np.max(abs_corr, axis=1)  # 每个密钥的最大|r|
    max_abs_pos = np.argmax(abs_corr, axis=1)  # 最大|r|出现的位置

    # 获取带符号的实际最大相关系数
    actual_max_val = correlation_arr_2d[np.arange(sbox_size), max_abs_pos]

    # 按|r|降序排序
    sorted_indices = np.argsort(-max_abs_val)

    # 返回前candidates个结果
    sbox_key_arr = sorted_indices[:candidates].astype(np.uint8)
    sbox_keycorr_arr = actual_max_val[sorted_indices[:candidates]]
    sbox_keypos_arr = max_abs_pos[sorted_indices[:candidates]]

    return sbox_key_arr, sbox_keycorr_arr, sbox_keypos_arr


def report_sbox_key_guesses(sbox_index, sbox_key_arr, sbox_keycorr_arr, sbox_keypos_arr, sample_first_pos):
    best_key = sbox_key_arr[0]
    print(f"The Sbox{sbox_index} best correlation Key byte {best_key:02X}:")

    candidates = sbox_key_arr.shape[0]
    for i in range(candidates):
        key_candidate = sbox_key_arr[i]
        correlation_value = sbox_keycorr_arr[i]
        relative_pos = sbox_keypos_arr[i]
        absolute_pos = sample_first_pos + relative_pos
        print(f"Key byte candidate: {key_candidate:02X}, "
              f"value: {correlation_value:.3f}, "
              f"at relative position: {relative_pos}, "
              f"absolute position: {absolute_pos}")


def analyze_process_cpa_cp(sbox_index, data_arr_2d, sample_arr_2d,
                           sbox_key_arr_2d, sbox_keycorr_arr_2d, sbox_keypos_arr_2d,
                           candidates, batch_size=10000):
    """
    使用CuPy对单个S盒执行高性能、分批处理、内存优化的相关性功耗分析(CPA)

    核心特性：
    · 同时考虑正负相关性，采用最大绝对值|r|作为关键指标
    · 内存效率优化，支持大规模能量迹分析
    · 精简相关系数矩阵结构，消除冗余零列
    · 使用标准Pearson相关系数(分母N-1)，兼容ChipWhisperer需调整分母为N
    """
    start_time = datetime.now()
    print(f"启动Sbox{sbox_index}分析，时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ===============================================================
    # 1. 数据传输至GPU：中间值矩阵与功耗轨迹
    # ===============================================================
    data_arr_2d_gpu = cp.asarray(data_arr_2d)  # 维度: (trace_number, sbox_size)
    sample_arr_2d_gpu = cp.asarray(sample_arr_2d)  # 维度: (trace_number, sample_number)

    trace_number, sbox_size = data_arr_2d_gpu.shape
    sample_number = sample_arr_2d_gpu.shape[1]

    print(f"密钥假设数: {sbox_size} | 候选密钥数: {candidates}"
          f" | 能量迹数: {trace_number} | 样本点数: {sample_number}")

    # ===============================================================
    # 2. 中间值矩阵全局标准化预处理
    # ===============================================================
    epsilon = 1e-12
    data_mean = cp.mean(data_arr_2d_gpu, axis=0, keepdims=True)
    data_std_val = cp.std(data_arr_2d_gpu, axis=0, keepdims=True,
                          ddof=1)  # (1, sbox_size) 兼容ChipWhisperer需设ddof=0
    data_std_val = cp.maximum(data_std_val, epsilon)
    data_std = (data_arr_2d_gpu - data_mean) / data_std_val
    data_std = cp.ascontiguousarray(data_std)

    # ===============================================================
    # 3. 初始化结果存储矩阵
    # ===============================================================
    correlation_arr_2d_gpu = cp.zeros((sbox_size, sample_number), dtype=cp.float32)  # 维度: (sbox_size, sample_number)

    # ===============================================================
    # 4. 分批相关系数计算（内存优化核心）
    #    按样本点维度分批次处理，控制GPU内存峰值使用
    # ===============================================================
    num_batches = (sample_number + batch_size - 1) // batch_size
    print(f"开始分批处理，总计 {num_batches} 批")

    for batch_idx in range(num_batches):
        batch_start_time = datetime.now()
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, sample_number)

        # 提取当前批次功耗样本
        sample_batch = sample_arr_2d_gpu[:, start_idx:end_idx]

        # 当前批次功耗数据标准化
        sample_batch_mean = cp.mean(sample_batch, axis=0, keepdims=True)
        sample_batch_std_val = cp.std(sample_batch, axis=0, keepdims=True,
                                      ddof=1)  # (1, sbox_size) 兼容ChipWhisperer需设ddof=0
        sample_batch_std_val = cp.maximum(sample_batch_std_val, epsilon)
        sample_batch_std = (sample_batch - sample_batch_mean) / sample_batch_std_val
        sample_batch_std = cp.ascontiguousarray(sample_batch_std)

        # Pearson相关系数计算（标准分母N-1）
        batch_correlation = cp.dot(data_std.T, sample_batch_std) / (trace_number - 1)

        # 数值边界处理与内存优化
        batch_correlation = cp.clip(batch_correlation, -1.0, 1.0)
        batch_correlation = cp.ascontiguousarray(batch_correlation)

        # 批次结果写入总矩阵
        correlation_arr_2d_gpu[:, start_idx:end_idx] = batch_correlation

        # 即时释放批次资源，维持低内存占用
        del sample_batch, sample_batch_mean, sample_batch_std_val, sample_batch_std, batch_correlation
        cp.get_default_memory_pool().free_all_blocks()

        batch_time = (datetime.now() - batch_start_time).total_seconds()
        print(f"批次 {batch_idx + 1:>3}/{num_batches} 完成，耗时 {batch_time:.3f} 秒")

    # ===============================================================
    # 5. 关键候选密钥提取（基于绝对相关系数）
    # ===============================================================
    abs_corr = cp.abs(correlation_arr_2d_gpu)  # (sbox_size, sample_number)
    max_abs_val = cp.max(abs_corr, axis=1)  # (sbox_size,) 各密钥假设最大|r|
    max_abs_pos = cp.argmax(abs_corr, axis=1)  # (sbox_size,) 对应样本位置

    # 获取带符号的实际最大相关系数
    actual_max_val = correlation_arr_2d_gpu[cp.arange(sbox_size), max_abs_pos]

    # 按|r|降序选取前candidates个最佳假设
    top_idx = cp.argsort(-max_abs_val)[:candidates]

    # 结果回传至CPU存储
    sbox_key_arr_2d[sbox_index] = np.uint8(top_idx.get())  # 最优密钥字节索引
    sbox_keycorr_arr_2d[sbox_index] = actual_max_val[top_idx].get()  # 对应相关系数（带符号）
    sbox_keypos_arr_2d[sbox_index] = max_abs_pos[top_idx].get()  # 泄漏点样本位置

    # ===============================================================
    # 6. 相关系数矩阵准备（后续可视化与分析）
    # ===============================================================
    correlation_arr_2d = correlation_arr_2d_gpu.get()  # 传输至主机内存

    # ===============================================================
    # 7. GPU内存彻底清理，准备后续S盒处理
    # ===============================================================
    del data_arr_2d_gpu, sample_arr_2d_gpu, data_std, correlation_arr_2d_gpu, abs_corr
    del max_abs_val, max_abs_pos, actual_max_val, top_idx
    cp.get_default_memory_pool().free_all_blocks()

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"Sbox{sbox_index}分析完成！总用时 {total_time:.3f} 秒")

    return correlation_arr_2d
