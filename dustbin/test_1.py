import cupy as cp


def calculate_gpu_memory_cpa(n_traces, n_time_points, n_key_guesses=256, dtype=cp.float32):
    """
    计算CPA攻击所需的GPU显存
    """
    bytes_per_element = 4  # float32占用4字节

    # 1. 能量轨迹矩阵 (n_traces × n_time_points)
    traces_memory = n_traces * n_time_points * bytes_per_element

    # 2. 假设能量矩阵 (n_traces × n_key_guesses)
    hypothetical_memory = n_traces * n_key_guesses * bytes_per_element

    # 3. 相关系数矩阵 (n_time_points × n_key_guesses)
    correlation_memory = n_time_points * n_key_guesses * bytes_per_element

    # 4. 中间计算结果
    # 标准化后的轨迹矩阵
    traces_std_memory = n_traces * n_time_points * bytes_per_element
    # 标准化后的假设能量矩阵
    power_std_memory = n_traces * n_key_guesses * bytes_per_element

    # 总显存需求 (峰值使用)
    total_memory = (traces_memory + hypothetical_memory + correlation_memory +
                    traces_std_memory + power_std_memory)

    return {
        'traces_memory_MB': traces_memory / (1024 ** 2),
        'hypothetical_memory_MB': hypothetical_memory / (1024 ** 2),
        'correlation_memory_MB': correlation_memory / (1024 ** 2),
        'total_peak_memory_MB': total_memory / (1024 ** 2),
        'minimum_required_MB': (traces_memory + hypothetical_memory + correlation_memory) / (1024 ** 2)
    }


# 你的参数
n_traces = 10000
n_time_points = 20000
n_key_guesses = 256

memory_requirements = calculate_gpu_memory_cpa(n_traces, n_time_points, n_key_guesses)

print("=== CPA攻击GPU显存需求分析 ===")
print(f"能量轨迹数量: {n_traces}")
print(f"每个轨迹点数: {n_time_points}")
print(f"密钥猜测数: {n_key_guesses}")
print(f"能量轨迹矩阵: {memory_requirements['traces_memory_MB']:.2f} MB")
print(f"假设能量矩阵: {memory_requirements['hypothetical_memory_MB']:.2f} MB")
print(f"相关系数矩阵: {memory_requirements['correlation_memory_MB']:.2f} MB")
print(f"最低显存需求: {memory_requirements['minimum_required_MB']:.2f} MB")
print(f"峰值显存需求: {memory_requirements['total_peak_memory_MB']:.2f} MB")