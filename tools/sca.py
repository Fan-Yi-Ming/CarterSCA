import random
import cupy as cp
import numpy as np


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

    data_arr_2d_gpu = cp.ascontiguousarray(cp.asarray(data_arr_2d, dtype=cp.float32))

    trace_number, sbox_size = data_arr_2d_gpu.shape
    sample_number = sample_arr_2d.shape[1]

    data_mean = cp.mean(data_arr_2d_gpu, axis=0, keepdims=True)
    data_std = cp.std(data_arr_2d_gpu, axis=0, keepdims=True)
    data_std = cp.maximum(data_std, epsilon)

    data_arr_2d_gpu -= data_mean
    data_arr_2d_gpu /= data_std
    data_transposed = data_arr_2d_gpu.T

    correlation_arr_2d = np.zeros((sbox_size, sample_number), dtype=np.float32)

    num_batches = (sample_number + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, sample_number)

        sample_batch = cp.ascontiguousarray(cp.asarray(sample_arr_2d[:, start:end], dtype=cp.float32))

        sample_mean = cp.mean(sample_batch, axis=0, keepdims=True)
        sample_std = cp.std(sample_batch, axis=0, keepdims=True)
        sample_std = cp.maximum(sample_std, epsilon)

        sample_batch -= sample_mean
        sample_batch /= sample_std

        correlation_batch = cp.dot(data_transposed, sample_batch) / (trace_number - 1)
        correlation_batch = cp.clip(correlation_batch, -1.0, 1.0)
        correlation_arr_2d[:, start:end] = correlation_batch.get()

    cp.get_default_memory_pool().free_all_blocks()

    return correlation_arr_2d


def analyze_process_cpa_gpu2(data_arr_2d, sample_arr_2d, batch_size=10000):
    epsilon = 1e-10

    data_arr_2d_gpu = cp.ascontiguousarray(cp.asarray(data_arr_2d))
    sample_arr_2d_gpu = cp.ascontiguousarray(cp.asarray(sample_arr_2d))

    trace_number, sbox_size = data_arr_2d_gpu.shape
    sample_number = sample_arr_2d.shape[1]

    data_mean = cp.mean(data_arr_2d_gpu, axis=0, keepdims=True)
    data_std = cp.std(data_arr_2d_gpu, axis=0, keepdims=True)
    data_std = cp.maximum(data_std, epsilon)
    data_arr_2d_gpu -= data_mean
    data_arr_2d_gpu /= data_std
    data_transposed = data_arr_2d_gpu.T

    sample_mean = cp.mean(sample_arr_2d_gpu, axis=0, keepdims=True)
    sample_std = cp.std(sample_arr_2d_gpu, axis=0, keepdims=True)
    sample_std = cp.maximum(sample_std, epsilon)
    sample_arr_2d_gpu -= sample_mean
    sample_arr_2d_gpu /= sample_std

    correlation_arr_2d_gpu = cp.zeros((sbox_size, sample_number), dtype=data_arr_2d_gpu.dtype)

    num_batches = (sample_number + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, sample_number)

        sample_batch_norm = sample_arr_2d_gpu[:, start:end]

        correlation_batch = cp.dot(data_transposed, sample_batch_norm) / (trace_number - 1)
        correlation_batch = cp.clip(correlation_batch, -1.0, 1.0)
        correlation_arr_2d_gpu[:, start:end] = correlation_batch

    correlation_arr_2d = correlation_arr_2d_gpu.get()
    cp.get_default_memory_pool().free_all_blocks()

    return correlation_arr_2d.astype(np.float32)


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
