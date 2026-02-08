import os
from datetime import datetime
import cupy as cp
import numpy as np
import trsfile
from trsfile import SampleCoding, Trace, Header, TracePadding
from trsfile.parametermap import TraceParameterMap, TraceSetParameterMap, TraceParameterDefinitionMap

from crypto.aes import Aes, aes_inv_keyexpansion
from tools.sca import hw, index_str_to_range

from typing import Tuple
from multiprocessing import Pool, cpu_count


def analyze_process_cpa_cp(sbox_index, data_arr_2d, sample_arr_2d,
                           sbox_key_arr_2d, sbox_keycorr_arr_2d, sbox_keypos_arr_2d,
                           candidates, batch_size=10000):
    """使用CuPy对单个S盒执行CPA分析"""
    start_time = datetime.now()
    print(f"启动Sbox{sbox_index}分析，时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 数据传输至GPU
    data_arr_2d_gpu = cp.asarray(data_arr_2d)  # (trace_number, sbox_size)
    sample_arr_2d_gpu = cp.asarray(sample_arr_2d)  # (trace_number, sample_number)

    trace_number, sbox_size = data_arr_2d_gpu.shape
    sample_number = sample_arr_2d_gpu.shape[1]

    print(f"密钥假设数: {sbox_size} | 候选密钥数: {candidates}"
          f" | 能量迹数: {trace_number} | 样本点数: {sample_number}")

    # 中间值矩阵标准化
    epsilon = 1e-12
    data_mean = cp.mean(data_arr_2d_gpu, axis=0, keepdims=True)
    data_std_val = cp.std(data_arr_2d_gpu, axis=0, keepdims=True, ddof=1)
    data_std_val = cp.maximum(data_std_val, epsilon)
    data_std = (data_arr_2d_gpu - data_mean) / data_std_val
    data_std = cp.ascontiguousarray(data_std)

    # 初始化结果存储矩阵
    correlation_arr_2d_gpu = cp.zeros((sbox_size, sample_number), dtype=cp.float32)

    # 分批相关系数计算
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
        sample_batch_std_val = cp.std(sample_batch, axis=0, keepdims=True, ddof=1)
        sample_batch_std_val = cp.maximum(sample_batch_std_val, epsilon)
        sample_batch_std = (sample_batch - sample_batch_mean) / sample_batch_std_val
        sample_batch_std = cp.ascontiguousarray(sample_batch_std)

        # Pearson相关系数计算
        batch_correlation = cp.dot(data_std.T, sample_batch_std) / (trace_number - 1)

        # 数值边界处理
        batch_correlation = cp.clip(batch_correlation, -1.0, 1.0)
        batch_correlation = cp.ascontiguousarray(batch_correlation)

        # 批次结果写入总矩阵
        correlation_arr_2d_gpu[:, start_idx:end_idx] = batch_correlation

        # 释放批次资源
        del sample_batch, sample_batch_mean, sample_batch_std_val, sample_batch_std, batch_correlation
        cp.get_default_memory_pool().free_all_blocks()

        batch_time = (datetime.now() - batch_start_time).total_seconds()
        print(f"批次 {batch_idx + 1:>3}/{num_batches} 完成，耗时 {batch_time:.3f} 秒")

    # 关键候选密钥提取（基于绝对相关系数）
    abs_corr = cp.abs(correlation_arr_2d_gpu)
    max_abs_val = cp.max(abs_corr, axis=1)
    max_abs_pos = cp.argmax(abs_corr, axis=1)
    actual_max_val = correlation_arr_2d_gpu[cp.arange(sbox_size), max_abs_pos]

    # 按|r|降序选取前candidates个最佳假设
    top_idx = cp.argsort(-max_abs_val)[:candidates]

    # 结果回传至CPU存储
    sbox_key_arr_2d[sbox_index] = np.uint8(top_idx.get())
    sbox_keycorr_arr_2d[sbox_index] = actual_max_val[top_idx].get()
    sbox_keypos_arr_2d[sbox_index] = max_abs_pos[top_idx].get()

    correlation_arr_2d = correlation_arr_2d_gpu.get()

    # GPU内存清理
    del data_arr_2d_gpu, sample_arr_2d_gpu, data_std, correlation_arr_2d_gpu, abs_corr
    del max_abs_val, max_abs_pos, actual_max_val, top_idx
    cp.get_default_memory_pool().free_all_blocks()

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"Sbox{sbox_index}分析完成！总用时 {total_time:.3f} 秒")

    return correlation_arr_2d


def process_single_trace(args: Tuple) -> Tuple[int, np.array, np.array]:
    """处理单条能量迹线，提取样本数据并计算中间值"""
    # 解包输入参数
    (traceset_path, trace_index, sample_first_pos, sample_number,
     sbox_num, sbox_size, crypto_direction, sbox_key_arr_2d) = args

    # 打开迹线文件
    traceset = trsfile.open(traceset_path, 'r')

    # 提取样本点数据
    trace = traceset[trace_index]
    sample_arr = np.float32(trace[sample_first_pos:sample_first_pos + sample_number])
    data_arr_2d = np.zeros((sbox_num, sbox_size), dtype=cp.float32)

    # 初始化AES算法对象
    aes = Aes()

    Nb = 4  # AES状态矩阵列数
    wi = np.zeros(Nb, dtype=np.uint32)  # 轮密钥数组

    if crypto_direction == 0:
        # 获取随机数输入参数
        input_arr = np.frombuffer(bytes(trace.parameters["INPUT"].value), dtype=np.uint8)

        for i in range(sbox_size):
            aes.set_state(input_arr)

            # 构造猜测轮密钥（所有字节相同）
            wi[0] = (i << 24) | (i << 16) | (i << 8) | i
            wi[1] = wi[0]
            wi[2] = wi[0]
            wi[3] = wi[0]

            # 执行第一轮加密操作
            aes.add_roundkey(wi)
            aes.sub_state()

            # 获取中间值并计算汉明重量
            intermediate_arr = aes.get_state()
            hw_intermediate_arr = np.array([hw(byte) for byte in intermediate_arr], dtype=np.float32)
            data_arr_2d[:, i] = hw_intermediate_arr

    if crypto_direction == 1:
        # 获取随机数输入参数
        output_arr = np.frombuffer(bytes(trace.parameters["OUTPUT"].value), dtype=np.uint8)
        for i in range(sbox_size):
            aes.set_state(output_arr)

            # 构造猜测轮密钥（所有字节相同）
            wi[0] = (i << 24) | (i << 16) | (i << 8) | i
            wi[1] = wi[0]
            wi[2] = wi[0]
            wi[3] = wi[0]

            # 执行第一轮解密操作
            aes.add_roundkey(wi)
            aes.sub_state()

            # 获取中间值并计算汉明重量
            intermediate_arr = aes.get_state()
            hw_intermediate_arr = np.array([hw(byte) for byte in intermediate_arr], dtype=np.float32)
            data_arr_2d[:, i] = hw_intermediate_arr

    # 关闭迹线文件
    if traceset:
        traceset.close()

    return trace_index, sample_arr, data_arr_2d


class Aes128CPA:
    """AES-128 CPA攻击处理器"""

    def __init__(self):
        """初始化CPA攻击参数"""
        # 分析参数
        self.candidates = 4  # 每个S盒保留的密钥候选数量
        self.num_processes = cpu_count()  # 进程数
        self.num_batch_size = 10000  # GPU批次处理大小

        # 能量迹参数
        self.trace_number = 0  # 能量迹总数
        self.sample_first_pos = 0  # 样本点起始位置
        self.sample_number = 0  # 样本点数量

        # 攻击方向
        self.crypto_direction = 0  # 0加密 1解密

        # S盒参数
        self.sbox_num = 16  # AES S盒数量
        self.sbox_size = 256  # 8bit S盒大小
        self.sbox_index_str = "0-15"  # 分析的S盒索引范围

        # 攻击结果存储
        self.sbox_key_result_path = "../crypto/usim_cpa_sbox_key_result.npz"
        self.sbox_key_arr_2d = np.zeros((self.sbox_num, self.candidates), dtype=np.uint8)
        self.sbox_keycorr_arr_2d = np.zeros((self.sbox_num, self.candidates), dtype=np.float32)
        self.sbox_keypos_arr_2d = np.zeros((self.sbox_num, self.candidates), dtype=np.int32)

        # 能量迹文件相关
        self.traceset = None  # 原始能量迹数据集
        self.traceset_path: str = ""  # 原始能量迹文件路径
        self.traceset2_switch: bool = True  # 是否启用相关系数曲线存储
        self.traceset2 = None  # 相关系数曲线数据集

        # 分析数据数组
        self.data_arr_3d = None  # 中间值数组: [sbox_num, trace_number, sbox_size]
        self.sample_arr_2d = None  # 样本点数据: [trace_number, sample_number]

    def init_process(self):
        """初始化分析过程"""
        self.open_traceset()
        self.open_traceset2()

        # 初始化数据数组
        self.data_arr_3d = np.zeros((self.sbox_num, self.trace_number, self.sbox_size), dtype=np.float32)
        self.sample_arr_2d = np.zeros((self.trace_number, self.sample_number), dtype=np.float32)

    def analyze(self):
        """执行完整的CPA分析流程"""
        self.init_process()
        self.load_samples_and_creat_intermediates()

        # 分析每个S盒
        for sbox_index in index_str_to_range(self.sbox_index_str):
            correlation_arr_2d = analyze_process_cpa_cp(sbox_index, self.data_arr_3d[sbox_index], self.sample_arr_2d,
                                                        self.sbox_key_arr_2d, self.sbox_keycorr_arr_2d,
                                                        self.sbox_keypos_arr_2d, self.candidates, self.num_batch_size)

            # 存储相关系数曲线
            if self.traceset2_switch:
                start_time = datetime.now()
                print(f"正在将Sbox{sbox_index}的相关性数据写入TraceSet")

                for i in range(self.sbox_size):
                    trace_parameter_map = TraceParameterMap()
                    trace = Trace(
                        sample_coding=SampleCoding.FLOAT,
                        samples=correlation_arr_2d[i],
                        parameters=trace_parameter_map,
                        title=f"Sbox{sbox_index}-KeyGuess_0x{i:02X}")
                    self.traceset2.append(trace)

                total_time = (datetime.now() - start_time).total_seconds()
                print(f"Sbox{sbox_index}的相关性数据写入TraceSet完毕！总用时 {total_time:.3f} 秒")

        self.finish_process()

    def finish_process(self):
        """完成分析过程"""
        self.report()
        self.recovery_key()
        self.close_traceset()
        self.close_traceset2()

    def load_samples_and_creat_intermediates(self):
        """加载样本点数据并创建中间值"""
        start_time = datetime.now()
        print(f"开始加载样本数据并创建AES加密Sbox输出中间值")

        try:
            # 分批处理参数
            num_batches = (self.trace_number + self.num_processes - 1) // self.num_processes
            print(f"开始加载 {self.trace_number} 条迹线，使用 {self.num_processes} 个进程，总共 {num_batches} 批")

            # 多进程并行处理
            with Pool(processes=self.num_processes) as pool:
                for batch_idx in range(num_batches):
                    print(f"开始加载第 {batch_idx + 1:>3}/{num_batches} 批...")

                    # 计算当前批次范围
                    start_idx = batch_idx * self.num_processes
                    end_idx = min((batch_idx + 1) * self.num_processes, self.trace_number)
                    current_batch_size = end_idx - start_idx

                    # 准备任务参数
                    batch_tasks = []
                    for i in range(current_batch_size):
                        trace_index = start_idx + i
                        batch_tasks.append((
                            self.traceset_path, trace_index, self.sample_first_pos, self.sample_number,
                            self.sbox_num, self.sbox_size, self.crypto_direction, self.sbox_key_arr_2d[:, 0]
                        ))

                    # 并行执行
                    batch_results = pool.map(process_single_trace, batch_tasks)

                    # 排序并存储结果
                    batch_results.sort(key=lambda x: x[0])
                    for trace_index, sample_arr, data_arr_2d in batch_results:
                        self.sample_arr_2d[trace_index] = sample_arr
                        self.data_arr_3d[:, trace_index, :] = data_arr_2d

            print(f"所有批加载完成！")

        except Exception as e:
            print(f"加载过程中发生错误: {e}")
            raise

        total_time = (datetime.now() - start_time).total_seconds()
        print(f"完成样本数据加载和中间值创建！总用时 {total_time:.3f} 秒")

    def report(self):
        """生成攻击结果报告"""
        for i in index_str_to_range(self.sbox_index_str):
            best_key = self.sbox_key_arr_2d[i][0]
            print(f"The Sbox{i} best correlation Key byte {best_key:02X}:")

            for j in range(self.candidates):
                key_candidate = self.sbox_key_arr_2d[i][j]
                correlation_value = self.sbox_keycorr_arr_2d[i][j]
                relative_pos = self.sbox_keypos_arr_2d[i][j]
                absolute_pos = self.sample_first_pos + relative_pos
                print(f"Key byte candidate: {key_candidate:02X}, "
                      f"value: {correlation_value:.3f}, "
                      f"at relative position: {relative_pos}, "
                      f"absolute position: {absolute_pos}")

    def open_traceset(self):
        """打开原始能量迹文件"""
        self.traceset = trsfile.open(self.traceset_path, 'r')
        self.trace_number = self.traceset.get_headers()[Header.NUMBER_TRACES]
        number_samples_per_trace = self.traceset.get_headers()[Header.NUMBER_SAMPLES]

        # 自动调整样本点范围
        if self.sample_first_pos < 0:
            self.sample_first_pos = 0
            print(f"自动调整样本点起始位置为: 0")

        if self.sample_first_pos + self.sample_number > number_samples_per_trace:
            self.sample_number = number_samples_per_trace - self.sample_first_pos
            print(f"自动调整样本点数量为: {self.sample_number}")

    def open_traceset2(self):
        """创建相关系数曲线文件"""
        if self.traceset2_switch:
            # 初始化参数映射
            traceset_parameter_map = TraceSetParameterMap()
            trace_parameter_definition_map = TraceParameterDefinitionMap()

            # 定义文件头信息
            headers = {
                Header.TRS_VERSION: 2,
                Header.TITLE_SPACE: 255,
                Header.SAMPLE_CODING: SampleCoding.FLOAT,
                Header.LABEL_X: " ",
                Header.SCALE_X: 1.0,
                Header.LABEL_Y: " ",
                Header.SCALE_Y: 1.0,
                Header.TRACE_SET_PARAMETERS: traceset_parameter_map,
                Header.TRACE_PARAMETER_DEFINITIONS: trace_parameter_definition_map
            }

            # 构建新文件名
            base_name = os.path.splitext(os.path.basename(self.traceset_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            new_base_name = f"{base_name}+UsimCPA({timestamp})"
            traceset2_path = os.path.join(
                os.path.dirname(self.traceset_path),
                new_base_name + os.path.splitext(self.traceset_path)[1]
            )

            # 创建新TRS文件
            self.traceset2 = trsfile.trs_open(
                path=traceset2_path,
                mode='w',
                engine='TrsEngine',
                headers=headers,
                padding_mode=TracePadding.AUTO,
                live_update=True
            )

    def close_traceset(self):
        """关闭原始能量迹文件"""
        if self.traceset:
            self.traceset.close()

    def close_traceset2(self):
        """关闭相关系数曲线文件"""
        if self.traceset2_switch and self.traceset2:
            self.traceset2.close()

    def recovery_key(self):
        """执行密钥恢复"""
        key = self.sbox_key_arr_2d[:, 0]

        if self.crypto_direction == 0:
            print("KEY:", ' '.join(f'{x:02X}' for x in key))

        if self.crypto_direction == 1:
            temp = self.sbox_key_arr_2d[:, 0]
            aes = Aes()
            aes.set_state(temp)
            aes.shift_rows()
            temp = aes.get_state()

            Nb = 4
            wi = np.zeros(Nb, dtype=np.uint32)
            for i in range(4):
                wi[i] = ((np.uint32(temp[i * Nb + 0]) << 24) |
                         (np.uint32(temp[i * Nb + 1]) << 16) |
                         (np.uint32(temp[i * Nb + 2]) << 8) |
                         np.uint32(temp[i * Nb + 3]))

            Nk = 4
            key = aes_inv_keyexpansion(wi, 10, None, Nk)
            print("KEY:", ' '.join(f'{x:02X}' for x in key))


if __name__ == '__main__':
    # 创建CPA攻击实例
    aes128_cpa = Aes128CPA()

    # # 第一轮攻击配置（加密）
    # aes128_cpa.traceset_path = "..\\traceset\\aes128_en.trs"
    # aes128_cpa.traceset2_switch = False
    # aes128_cpa.sample_first_pos = 20000
    # aes128_cpa.sample_number = 50000
    # aes128_cpa.crypto_direction = 0
    # aes128_cpa.sbox_index_str = "0-15"
    # aes128_cpa.analyze()

    # 第二轮攻击配置（解密）
    aes128_cpa.traceset_path = "..\\traceset\\aes128_de2+LowPass(20251123194356)+StaticAlign(20251123194809).trs"
    aes128_cpa.traceset2_switch = False
    aes128_cpa.sample_first_pos = 200000
    aes128_cpa.sample_number = 250000
    aes128_cpa.crypto_direction = 1
    aes128_cpa.sbox_index_str = "0-15"
    aes128_cpa.analyze()
