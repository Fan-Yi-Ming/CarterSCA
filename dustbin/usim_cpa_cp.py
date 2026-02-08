import os
from datetime import datetime
import cupy as cp
import numpy as np
import trsfile
import trsfile.traceparameter as tp
from trsfile import SampleCoding, Trace, Header, TracePadding
from trsfile.parametermap import TraceParameterMap, TraceSetParameterMap, TraceParameterDefinitionMap

from crypto.aes import Aes, aes_inv_keyexpansion
from tools.sca import hw, index_str_to_range


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
    data_mean = cp.mean(data_arr_2d_gpu, axis=0, keepdims=True)  # (1, sbox_size)
    data_std_val = cp.std(data_arr_2d_gpu, axis=0, keepdims=True,
                          ddof=1)  # (1, sbox_size) 兼容ChipWhisperer需设ddof=0
    data_std_val = cp.maximum(data_std_val, epsilon)  # 数值稳定性保障
    data_std = (data_arr_2d_gpu - data_mean) / data_std_val  # 标准化中间值
    data_std = cp.ascontiguousarray(data_std)  # 内存连续布局优化

    # ===============================================================
    # 3. 初始化结果存储矩阵
    # ===============================================================
    correlation_arr_2d_gpu = cp.zeros((sbox_size, sample_number), dtype=cp.float32)  # 维度: (sbox_size, sample_number)

    # ===============================================================
    # 4. 分批相关系数计算（内存优化核心）
    #    按样本点维度分批次处理，控制GPU内存峰值使用
    # ===============================================================
    num_batches = (sample_number + batch_size - 1) // batch_size
    print(f"开始分批处理，总计{num_batches}批")

    for batch_idx in range(num_batches):
        batch_start_time = datetime.now()
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, sample_number)
        this_batch_size = end_idx - start_idx

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
        print(f"批次 {batch_idx + 1:>3}/{num_batches} 完成，"
              f"处理样本 {this_batch_size:>5} 点，耗时 {batch_time:.3f} 秒")

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
    sbox_key_arr_2d[sbox_index] = top_idx.get()  # 最优密钥字节索引
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


class USIMCPACP:
    """
    USIM CPA攻击处理器

    实现两轮攻击策略：
    - 第一轮攻击: 恢复 K ⊕ OPC (密钥与OPC的异或值)
    - 第二轮攻击: 使用第一轮结果恢复完整密钥K和OPC
    """

    def __init__(self):
        """初始化USIMCPA攻击参数和数据结构"""

        # 分析参数
        self.candidates = 4  # 每个S盒保留的密钥候选数量

        # 能量迹参数
        self.trace_number = 0  # 能量迹总数
        self.sample_first_pos = 0  # 样本点起始位置
        self.sample_number = 0  # 样本点数量

        # 攻击轮次参数
        self.attack_round_index = 0  # 当前攻击轮次: 0-第一轮, 1-第二轮
        self.attack_round_times = 2  # 总攻击轮次数

        # S盒参数
        self.sbox_num = 16  # AES S盒数量
        self.sbox_size = 256  # 8bit S盒大小
        self.sbox_index_str = "0-15"  # 分析的S盒索引范围

        # 攻击结果存储
        self.sbox_key_result_path = ".\\usim_sbox_key_result.npz"  # 结果文件路径
        # 攻击结果数组: [attack_round_times, sbox_num, candidates]
        self.sbox_key_arr_3d = np.zeros((self.attack_round_times, self.sbox_num, self.candidates), dtype=np.uint8)
        self.sbox_keycorr_arr_3d = np.zeros((self.attack_round_times, self.sbox_num, self.candidates), dtype=np.float32)
        self.sbox_keypos_arr_3d = np.zeros((self.attack_round_times, self.sbox_num, self.candidates), dtype=np.int32)

        # 能量迹文件相关
        self.traceset = None  # 原始能量迹数据集
        self.traceset_path = ""  # 原始能量迹文件路径
        self.traceset2_switch: bool = True  # 是否启用相关系数曲线存储
        self.traceset2 = None  # 相关系数曲线数据集

        # 分析数据数组
        self.data_arr_3d = None  # 中间值数组: [sbox_num, trace_number, sbox_size]
        self.sample_arr_2d = None  # 样本点数据: [trace_number, sample_number]

    def init_process(self):
        """
        初始化分析过程

        执行步骤:
        1. 打开原始能量迹文件
        2. 创建相关系数曲线文件
        3. 加载历史攻击结果(如果存在)
        4. 初始化数据数组
        """
        self.open_traceset()  # 打开原始能量迹文件
        self.open_traceset2()  # 创建相关系数曲线文件
        self.load_sbox_key_result()  # 加载历史攻击结果

        # 初始化中间值数组: [16个S盒, trace_number条能量迹, 256个密钥假设]
        self.data_arr_3d = np.zeros((self.sbox_num, self.trace_number, self.sbox_size), dtype=np.float32)

        # 初始化样本点数组: [trace_number条能量迹, sample_number个样本点]
        self.sample_arr_2d = np.zeros((self.trace_number, self.sample_number), dtype=np.float32)

    def analyze(self):
        """
        执行完整的CPA分析流程

        流程:
        1. 初始化分析环境
        2. 创建AES加密中间值
        3. 多线程并行分析各个S盒
        4. 完成分析并保存结果
        """
        self.init_process()  # 初始化分析环境

        # 加载样本点数据到内存并创建AES加密过程中S盒输出的中间值（基于汉明重量能量消耗模型）
        self.load_samples_and_creat_intermediates()
        # 为每个指定的S盒提交分析任务
        for sbox_index in index_str_to_range(self.sbox_index_str):
            correlation_arr_2d = analyze_process_cpa_cp(sbox_index, self.data_arr_3d[sbox_index], self.sample_arr_2d,
                                                        self.sbox_key_arr_3d[self.attack_round_index],
                                                        self.sbox_keycorr_arr_3d[self.attack_round_index],
                                                        self.sbox_keypos_arr_3d[self.attack_round_index],
                                                        self.candidates, 10000)
            if self.traceset2_switch:
                start_time = datetime.now()
                print(f"正在将Sbox{sbox_index}的相关性数据写入TraceSet，"
                      f"时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                for i in range(self.sbox_size):
                    trace_parameter_map = TraceParameterMap()
                    trace = Trace(
                        sample_coding=SampleCoding.FLOAT,
                        samples=correlation_arr_2d[i],
                        parameters=trace_parameter_map,
                        title=f"Sbox{sbox_index}-KeyGuess_0x{i:02X}")
                    self.traceset2.append(trace)
                # 记录完成时间并计算耗时
                total_time = (datetime.now() - start_time).total_seconds()
                print(f"Sbox{sbox_index}的相关性数据写入TraceSet完毕！总用时 {total_time:.3f} 秒")

            print("=" * 80)

        self.finish_process()  # 完成分析流程

    def finish_process(self):
        """
        完成分析过程

        执行步骤:
        1. 生成分析报告
        2. 尝试恢复完整密钥
        3. 保存攻击结果
        4. 关闭能量迹文件
        """
        self.report()  # 生成分析报告
        self.recovery_key()  # 尝试恢复完整密钥
        self.save_sbox_key_result()  # 保存攻击结果
        self.close_traceset()  # 关闭能量迹文件
        self.close_traceset2()  # 关闭能量迹文件

    def load_samples_and_creat_intermediates(self):
        """
        加载样本点数据到内存并创建AES加密过程中S盒输出的中间值（基于汉明重量能量消耗模型）

        主要完成两个任务：
        1. 将能量迹样本点数据加载到内存中的二维数组
        2. 根据攻击轮次计算对应的中间值：
           - 攻击轮次0：恢复第一轮轮密钥 K⊕OPC，计算第一轮S盒输出（AddRoundKey + SubBytes）
           - 攻击轮次1：恢复完整密钥K和OPC，计算第二轮S盒输出，使用第一轮攻击结果

        中间值基于汉明重量模型计算，用于后续的相关性能量分析攻击
        """
        aes = Aes()  # 初始化AES算法对象

        # 记录开始时间
        start_time = datetime.now()
        print(f"开始加载样本数据并创建AES加密S盒输出中间值，时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 遍历所有能量迹
        for i in range(self.trace_number):
            # 任务1：加载样本点数据到内存
            # 从能量迹中提取指定范围的样本点数据到二维数组
            trace = self.traceset[i]
            trace_data = trace[self.sample_first_pos:self.sample_first_pos + self.sample_number]
            self.sample_arr_2d[i] = np.copy(trace_data)

            # 从能量迹参数中获取随机数输入（RND）
            input_arr = np.frombuffer(bytes(trace.parameters["RND"].value), dtype=np.uint8)

            Nb = 4  # AES状态矩阵列数
            wi = np.zeros(Nb, dtype=np.uint32)  # 轮密钥数组

            # 任务2：创建中间值数据
            # 第一轮攻击：恢复 K ⊕ OPC
            if self.attack_round_index == 0:
                # 遍历所有可能的密钥字节值（0-255）
                for j in range(self.sbox_size):
                    aes.set_state(input_arr)  # 设置AES状态为随机数输入

                    # 构造测试轮密钥 [kk kk kk kk] 形式（每个字节相同）
                    wi[0] = (j << 24) | (j << 16) | (j << 8) | j
                    wi[1] = wi[0]
                    wi[2] = wi[0]
                    wi[3] = wi[0]

                    aes.add_roundkey(wi)  # 执行轮密钥加操作
                    aes.sub_state()  # 执行S盒替换操作
                    intermediate_arr = aes.get_state()  # 获取S盒输出中间值

                    # 计算中间值的汉明重量作为假设能量消耗
                    hw_intermediate_arr = np.array([hw(byte) for byte in intermediate_arr], dtype=np.float32)

                    # 为所有S盒存储假设能量消耗值到三维数据数组
                    self.data_arr_3d[:, i, j] = hw_intermediate_arr  # 使用切片批量赋值

            # 第二轮攻击：恢复完整密钥K和OPC
            elif self.attack_round_index == 1:
                # 设置输入状态为随机数输入
                aes.set_state(input_arr)

                # 使用第一轮攻击结果构造第一轮轮密钥
                for j in range(Nb):
                    wi[j] = ((np.uint32(self.sbox_key_arr_3d[0][j * Nb + 0][0]) << 24) |
                             (np.uint32(self.sbox_key_arr_3d[0][j * Nb + 1][0]) << 16) |
                             (np.uint32(self.sbox_key_arr_3d[0][j * Nb + 2][0]) << 8) |
                             np.uint32(self.sbox_key_arr_3d[0][j * Nb + 3][0]))

                # 执行完整的第一轮加密操作
                aes.add_roundkey(wi)  # 第一轮轮密钥加
                aes.sub_state()  # 第一轮S盒替换
                aes.shift_rows()  # 第一轮行移位
                aes.mix_columns()  # 第一轮列混合
                input_arr = aes.get_state()  # 获取第二轮输入状态

                # 遍历所有可能的第二轮轮密钥字节值
                for j in range(self.sbox_size):
                    aes.set_state(input_arr)  # 设置第二轮输入状态

                    # 构造第二轮测试轮密钥 [kk kk kk kk] 形式
                    wi[0] = (j << 24) | (j << 16) | (j << 8) | j
                    wi[1] = wi[0]
                    wi[2] = wi[0]
                    wi[3] = wi[0]

                    aes.add_roundkey(wi)  # 第二轮轮密钥加
                    aes.sub_state()  # 第二轮S盒替换
                    intermediate_arr = aes.get_state()  # 获取第二轮S盒输出中间值

                    # 计算汉明重量作为假设能量消耗
                    hw_intermediate_arr = np.array([hw(byte) for byte in intermediate_arr], dtype=np.float32)

                    # 为所有S盒存储假设能量消耗值到三维数据数组
                    self.data_arr_3d[:, i, j] = hw_intermediate_arr  # 使用切片批量赋值

        # 记录完成时间并计算耗时
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"完成样本数据加载和中间值创建！总用时 {total_time:.3f} 秒")

    def report(self):
        """
        生成并打印攻击结果报告

        显示每个S盒的最佳密钥候选及其相关系数和位置信息
        """
        # 遍历指定的S盒索引
        for i in index_str_to_range(self.sbox_index_str):
            # 获取当前S盒的最佳密钥字节
            best_key = self.sbox_key_arr_3d[self.attack_round_index][i][0]
            print(f"The Sbox{i} best correlation Key byte {best_key:02X}:")

            # 显示所有候选密钥信息
            for j in range(self.candidates):
                key_candidate = self.sbox_key_arr_3d[self.attack_round_index][i][j]
                correlation_value = self.sbox_keycorr_arr_3d[self.attack_round_index][i][j]
                relative_pos = self.sbox_keypos_arr_3d[self.attack_round_index][i][j]
                absolute_pos = self.sample_first_pos + relative_pos
                print(f"Key byte candidate: {key_candidate:02X}, "
                      f"value: {correlation_value:.3f}, "
                      f"at relative position: {relative_pos}, "
                      f"absolute position: {absolute_pos}")

    def open_traceset(self):
        """
        打开原始能量迹TRS文件并初始化参数

        自动调整样本点范围以防止越界访问
        """
        # 打开TRS格式的能量迹文件
        self.traceset = trsfile.open(self.traceset_path, 'r')

        # 获取能量迹数量
        self.trace_number = self.traceset.get_headers()[Header.NUMBER_TRACES]

        # 获取每条能量迹的样本点数量
        sample_number_per_trace = self.traceset.get_headers()[Header.NUMBER_SAMPLES]

        # 自动调整样本点起始位置
        if self.sample_first_pos < 0:
            self.sample_first_pos = 0
            print(f"自动调整样本点起始位置为: 0")

        # 自动调整样本点数量防止越界
        if self.sample_first_pos + self.sample_number > sample_number_per_trace:
            self.sample_number = sample_number_per_trace - self.sample_first_pos
            print(f"自动调整样本点数量为: {self.sample_number}")

    def open_traceset2(self):
        """
        创建并打开用于存储相关系数曲线的TRS文件

        新文件基于原始文件名添加时间戳和攻击类型标识
        """
        if self.traceset2_switch:
            # ============================ TraceSet参数初始化 ============================
            # 初始化TraceSet参数映射
            traceset_parameter_map = TraceSetParameterMap()
            traceset_parameter_map['ATTACK_ROUND_INDEX'] = tp.IntegerArrayParameter([self.attack_round_index])

            # 定义TraceSet中每条Trace的参数定义
            trace_parameter_definition_map = TraceParameterDefinitionMap()

            # 定义TRS文件头信息
            headers = {
                Header.TRS_VERSION: 2,
                Header.TITLE_SPACE: 255,
                Header.SAMPLE_CODING: SampleCoding.FLOAT,  # 数据编码格式
                Header.LABEL_X: " ",  # X轴标签
                Header.SCALE_X: 1.0,  # X轴缩放比例
                Header.LABEL_Y: " ",  # Y轴标签
                Header.SCALE_Y: 1.0,  # Y轴缩放比例
                Header.TRACE_SET_PARAMETERS: traceset_parameter_map,  # 跟踪集参数
                Header.TRACE_PARAMETER_DEFINITIONS: trace_parameter_definition_map  # Trace的参数定义
            }

            # 构建新文件名
            base_name = os.path.splitext(os.path.basename(self.traceset_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            new_base_name = f"{base_name}+USIMCPA({timestamp})"
            traceset2_path = os.path.join(
                os.path.dirname(self.traceset_path),
                new_base_name + os.path.splitext(self.traceset_path)[1]
            )

            # 创建新的TRS文件用于存储相关系数曲线
            self.traceset2 = trsfile.trs_open(
                path=traceset2_path,  # 文件路径
                mode='w',  # 写入模式
                engine='TrsEngine',  # 存储引擎
                headers=headers,  # 文件头信息
                padding_mode=TracePadding.AUTO,  # 自动填充
                live_update=True  # 实时更新(性能略有损失)
            )

    def close_traceset(self):
        """
        关闭原始能量迹文件曲线文件
        """
        if self.traceset:
            self.traceset.close()

    def close_traceset2(self):
        """
        关闭相关系数曲线文件
        """
        if self.traceset2_switch:
            if self.traceset2:
                self.traceset2.close()

    def save_sbox_key_result(self):
        """
        保存S盒密钥攻击结果到NPZ文件

        保存内容包括:
        - sbox_key_arr_3d: 密钥候选数组
        - sbox_keycorr_arr_3d: 相关系数数组
        - sbox_keypos_arr_3d: 位置信息数组
        """
        try:
            np.savez(
                self.sbox_key_result_path,
                sbox_key_arr_3d=self.sbox_key_arr_3d,
                sbox_keycorr_arr_3d=self.sbox_keycorr_arr_3d,
                sbox_keypos_arr_3d=self.sbox_keypos_arr_3d
            )
            print(f"Sbox密钥攻击结果已保存到: {self.sbox_key_result_path}")
        except Exception as e:
            print(f"保存Sbox密钥攻击结果失败: {e}")

    def load_sbox_key_result(self):
        """
        从NPZ文件加载历史S盒密钥攻击结果

        如果文件不存在则跳过加载
        """
        if not os.path.exists(self.sbox_key_result_path):
            return

        try:
            data = np.load(self.sbox_key_result_path)
            self.sbox_key_arr_3d = data['sbox_key_arr_3d']
            self.sbox_keycorr_arr_3d = data['sbox_keycorr_arr_3d']
            self.sbox_keypos_arr_3d = data['sbox_keypos_arr_3d']
            print(f"已从文件加载Sbox密钥攻击结果: {self.sbox_key_result_path}")
        except Exception as e:
            print(f"加载Sbox密钥攻击结果失败: {e}")

    def recovery_key(self):
        """
        执行密钥恢复操作

        根据攻击轮次恢复不同的密钥信息:
        - 攻击轮次0: 显示 KEY ⊕ OPC
        - 攻击轮次1: 计算并显示完整密钥KEY和OPC
        """
        # 获取第一轮攻击结果(KEY ⊕ OPC)
        key_xor_opc = self.sbox_key_arr_3d[0, :, 0]

        # 第一轮攻击: 显示 KEY ⊕ OPC
        if self.attack_round_index == 0:
            print("KEY ⊕ OPC:", ' '.join(f'{x:02X}' for x in key_xor_opc))

        # 第二轮攻击: 恢复完整密钥K和OPC
        elif self.attack_round_index == 1:
            Nb = 4  # AES状态矩阵列数
            wi = np.zeros(Nb, dtype=np.uint32)  # 轮密钥数组

            # 从第二轮攻击结果构造轮密钥
            for i in range(4):
                wi[i] = ((np.uint32(self.sbox_key_arr_3d[1][i * Nb + 0][0]) << 24) |
                         (np.uint32(self.sbox_key_arr_3d[1][i * Nb + 1][0]) << 16) |
                         (np.uint32(self.sbox_key_arr_3d[1][i * Nb + 2][0]) << 8) |
                         np.uint32(self.sbox_key_arr_3d[1][i * Nb + 3][0]))

            Nk = 4  # AES-128密钥字数
            # 执行逆密钥扩展获取第一轮轮密钥
            key = aes_inv_keyexpansion(wi, 1, None, Nk)
            # 计算OPC: OPC = (KEY ⊕ OPC) ⊕ KEY
            opc = np.bitwise_xor(key_xor_opc, key)

            print("KEY:", ' '.join(f'{x:02X}' for x in key))
            print("OPC:", ' '.join(f'{x:02X}' for x in opc))


if __name__ == '__main__':
    """
    主函数: USIM CPA攻击示例

    演示完整的两轮攻击流程:
    1. 第一轮攻击恢复 K ⊕ OPC
    2. 第二轮攻击恢复完整密钥K和OPC
    """
    # # 创建 USIMCPACP 攻击实例
    # usim_cpa_cp = USIMCPACP()
    # # 设置能量迹文件路径
    # usim_cpa_cp.traceset_path = "..\\traceset\\usim_simulation.trs"
    # usim_cpa_cp.traceset2_switch = False
    #
    # # 第一轮攻击配置
    # usim_cpa_cp.sample_first_pos = 0  # 样本点起始位置
    # usim_cpa_cp.sample_number = 160  # 样本点数量
    # usim_cpa_cp.attack_round_index = 0  # 第一轮攻击
    # usim_cpa_cp.sbox_index_str = "0-15"  # 分析所有16个S盒
    # usim_cpa_cp.analyze()  # 执行第一轮攻击
    #
    # # 第二轮攻击配置
    # usim_cpa_cp.sample_first_pos = 0  # 样本点起始位置
    # usim_cpa_cp.sample_number = 160  # 样本点数量
    # usim_cpa_cp.attack_round_index = 1  # 第二轮攻击
    # usim_cpa_cp.sbox_index_str = "0-15"  # 分析所有16个S盒
    # usim_cpa_cp.analyze()  # 执行第二轮攻击

    # 创建 USIMCPACP 攻击实例
    usim_cpa_cp = USIMCPACP()

    # 设置能量迹文件路径
    usim_cpa_cp.traceset_path = "../traceset/old/test.trs"
    usim_cpa_cp.traceset2_switch = False

    # 第一轮攻击配置
    usim_cpa_cp.sample_first_pos = 1000000  # 样本点起始位置
    usim_cpa_cp.sample_number = 430000  # 样本点数量
    usim_cpa_cp.attack_round_index = 0  # 第一轮攻击
    usim_cpa_cp.sbox_index_str = "0-15"  # 分析所有16个S盒
    usim_cpa_cp.analyze()  # 执行第一轮攻击
