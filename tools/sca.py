import random
import cupy as cp
import numpy as np


def generate_random_hex_string(length) -> str:
    """生成指定长度的随机十六进制字符串，并以空格分隔每两个字符。"""
    hex_chars = '0123456789ABCDEF'
    hex_string = ''.join(random.choice(hex_chars) for _ in range(length * 2))
    formatted_hex_string = ' '.join(hex_string[i:i + 2] for i in range(0, len(hex_string), 2))
    return formatted_hex_string


def index_str_to_range(index_str: str):
    """"
    将索引字符串转换为索引范围列表

    该函数处理两种格式的输入：
    1. 单个数字：如 "5" -> 返回 [5]
    2. 范围表示：如 "3-7" -> 返回 [3, 4, 5, 6, 7]
    """
    if '-' in index_str:
        start, end = map(int, index_str.split('-'))
        # 为了包含结束值，需要 end + 1
        return list(range(start, end + 1))
    else:
        # 单个数字返回包含该数字的列表
        return [int(index_str)]


def hex_dump(data: bytes, bytes_per_line=16):
    """以16进制格式打印所有数据"""
    print(f"总字节数: {len(data)}")
    print("-" * 80)

    for i in range(0, len(data), bytes_per_line):
        # 当前行的字节切片
        chunk = data[i:i + bytes_per_line]

        # 十六进制部分
        hex_str = ' '.join(f"{b:02x}" for b in chunk)
        # 对齐
        hex_str = hex_str.ljust(bytes_per_line * 3 - 1)

        # ASCII部分
        ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)

        # 打印
        print(f"{i:04x}: {hex_str} | {ascii_str}")

    print("-" * 80)


def hw(byte):
    """计算单个字节的汉明重量"""
    weight = 0
    while byte:
        weight += byte & 1
        byte >>= 1
    return weight


def get_bit(byte, index):
    """获取字节指定索引位置的值"""
    if not 0 <= index < 8:
        raise ValueError(f"Bit index must be 0-7, got {index}")
    return (byte >> index) & 1


def d_func(byte, mode):
    """选择函数 - 用于DPA攻击中的轨迹分类"""
    if 0 <= mode < 8:
        # 基于特定位的分类
        return get_bit(byte, mode) == 1
    if 8 <= mode <= 15:
        # 基于汉明重量的分类，阈值为(mode-7)
        threshold = mode - 7
        return hw(byte) >= threshold
    raise ValueError(f"d_func mode must be 0-15, got {mode}")


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


def rank_sbox_key_guesses(value_arr_2d, candidates):
    sbox_size = value_arr_2d.shape[0]

    abs_value = np.abs(value_arr_2d)
    max_abs_val = np.max(abs_value, axis=1)  # 每个密钥的最大|r|
    max_abs_pos = np.argmax(abs_value, axis=1)  # 最大|r|出现的位置

    # 获取带符号的实际最大值
    actual_max_val = value_arr_2d[np.arange(sbox_size), max_abs_pos]

    # 按|r|降序排序
    sorted_indices = np.argsort(-max_abs_val)

    # 返回前candidates个结果
    sbox_key_arr = sorted_indices[:candidates].astype(np.uint8)
    sbox_keyvalue_arr = actual_max_val[sorted_indices[:candidates]]
    sbox_keypos_arr = max_abs_pos[sorted_indices[:candidates]]

    return sbox_key_arr, sbox_keyvalue_arr, sbox_keypos_arr


def report_sbox_key_guesses(sbox_index, sbox_key_arr, sbox_keyvalue_arr, sbox_keypos_arr, sample_first_pos):
    best_key = sbox_key_arr[0]
    print(f"The Sbox{sbox_index} best candidate: {best_key:02X}")

    candidates = sbox_key_arr.shape[0]
    for i in range(candidates):
        key_candidate = sbox_key_arr[i]
        correlation_value = sbox_keyvalue_arr[i]
        relative_pos = sbox_keypos_arr[i]
        absolute_pos = sample_first_pos + relative_pos
        print(f"Key byte candidate: {key_candidate:02X}, "
              f"value: {correlation_value:.3f}, "
              f"at relative position: {relative_pos}, "
              f"absolute position: {absolute_pos}")
