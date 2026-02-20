import os
import numpy as np
import trsfile
import trsfile.traceparameter as tp
from datetime import datetime
from trsfile import SampleCoding, Trace, Header, TracePadding
from trsfile.parametermap import TraceParameterMap, TraceSetParameterMap, TraceParameterDefinitionMap
from typing import Tuple
from multiprocessing import Pool, cpu_count
from tools.aes import Aes, aes_inv_keyexpansion
from tools.sca import hw, index_str_to_range, analyze_process_cpa_cp


def process_single_trace(args: Tuple) -> Tuple[int, np.array, np.array]:
    """
    处理单条能量迹线，提取样本数据并计算中间值

    Args:
        args: 包含处理参数的元组

    Returns:
        Tuple[int, np.array, np.array]: 迹线索引、样本数据数组、中间值数据数组
    """
    # 解包输入参数
    (traceset_path, trace_index, sample_first_pos, sample_number,
     sbox_num, sbox_size, attack_round_index, sbox_key_arr_2d) = args

    # 打开迹线文件
    traceset = trsfile.open(traceset_path, 'r')

    # 提取样本点数据
    trace = traceset[trace_index]
    sample_arr = np.array(trace[sample_first_pos:sample_first_pos + sample_number], dtype=np.float32)
    data_arr_2d = np.zeros((sbox_num, sbox_size), dtype=np.float32)

    # 初始化AES算法对象
    aes = Aes()

    # 获取随机数输入参数
    input_arr = np.frombuffer(bytes(trace.parameters["INPUT"].value), dtype=np.uint8)[0:16]

    Nb = 4  # AES状态矩阵列数
    wi = np.zeros(Nb, dtype=np.uint32)  # 轮密钥数组

    # 第一轮攻击：恢复 K ⊕ OPC
    if attack_round_index == 0:
        for i in range(sbox_size):
            aes.set_state(input_arr)  # 设置AES状态

            # 构造猜测轮密钥（所有字节相同）
            wi[0] = (i << 24) | (i << 16) | (i << 8) | i
            wi[1] = wi[0]
            wi[2] = wi[0]
            wi[3] = wi[0]

            # 执行第一轮加密操作
            aes.add_roundkey(wi)  # 轮密钥加
            aes.sub_state()  # S盒替换

            # 获取中间值并计算汉明重量
            intermediate_arr = aes.get_state()
            hw_intermediate_arr = np.array([hw(byte) for byte in intermediate_arr], dtype=np.float32)
            data_arr_2d[:, i] = hw_intermediate_arr

    # 第二轮攻击：恢复完整密钥K和OPC
    elif attack_round_index == 1:
        # 使用第一轮攻击结果执行完整的第一轮加密
        aes.set_state(input_arr)

        # 构造第一轮轮密钥
        for i in range(Nb):
            wi[i] = ((np.uint32(sbox_key_arr_2d[0][i * Nb + 0]) << 24) |
                     (np.uint32(sbox_key_arr_2d[0][i * Nb + 1]) << 16) |
                     (np.uint32(sbox_key_arr_2d[0][i * Nb + 2]) << 8) |
                     np.uint32(sbox_key_arr_2d[0][i * Nb + 3]))

        # 执行完整的第一轮加密
        aes.add_roundkey(wi)  # 轮密钥加
        aes.sub_state()  # S盒替换
        aes.shift_rows()  # 行移位
        aes.mix_columns()  # 列混合
        input_arr = aes.get_state()  # 获取第二轮输入

        # 处理第二轮加密
        for i in range(sbox_size):
            aes.set_state(input_arr)  # 设置第二轮输入状态

            # 构造猜测轮密钥
            wi[0] = (i << 24) | (i << 16) | (i << 8) | i
            wi[1] = wi[0]
            wi[2] = wi[0]
            wi[3] = wi[0]

            # 执行第二轮加密操作
            aes.add_roundkey(wi)  # 轮密钥加
            aes.sub_state()  # S盒替换

            # 获取中间值并计算汉明重量
            intermediate_arr = aes.get_state()
            hw_intermediate_arr = np.array([hw(byte) for byte in intermediate_arr], dtype=np.float32)
            data_arr_2d[:, i] = hw_intermediate_arr

    # 关闭迹线文件
    if traceset:
        traceset.close()

    return trace_index, sample_arr, data_arr_2d


class UsimCPA:
    """
    两轮攻击：
    - 第一轮攻击: 恢复 K ⊕ OPC (密钥与OPC的异或值)
    - 第二轮攻击: 使用第一轮结果恢复完整密钥K和OPC
    """

    def __init__(self):
        """初始化USIMCPA攻击参数和数据结构"""

        # 分析参数
        self.candidates = 4  # 每个S盒保留的密钥候选数量
        self.num_processes = cpu_count()  # 进程数，默认为系统CPU核心数量
        self.num_batch_size = 10000  # GPU计算相关系数时一次最多处理的样本点数

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
        self.sbox_key_result_path = "../usim_cpa_sbox_key_result.npz"  # 结果文件路径
        # 攻击结果数组: [attack_round_times, sbox_num, candidates]
        self.sbox_key_arr_3d = np.zeros((self.attack_round_times, self.sbox_num, self.candidates), dtype=np.uint8)
        self.sbox_keycorr_arr_3d = np.zeros((self.attack_round_times, self.sbox_num, self.candidates), dtype=np.float32)
        self.sbox_keypos_arr_3d = np.zeros((self.attack_round_times, self.sbox_num, self.candidates), dtype=np.int32)

        # 能量迹文件相关
        self.traceset = None  # 原始能量迹数据集
        self.traceset_path: str = ""  # 原始能量迹文件路径
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

        # 初始化数据数组
        self.data_arr_3d = np.zeros((self.sbox_num, self.trace_number, self.sbox_size), dtype=np.float32)
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

        # 初始化分析环境
        self.init_process()

        # 加载样本点数据到内存并创建AES加密过程中S盒输出的中间值（基于汉明重量能量消耗模型）
        self.load_samples_and_creat_intermediates()

        # 为每个指定的S盒提交分析任务
        for sbox_index in index_str_to_range(self.sbox_index_str):
            # 执行CPA分析过程，计算相关性矩阵
            correlation_arr_2d = analyze_process_cpa_cp(sbox_index, self.data_arr_3d[sbox_index], self.sample_arr_2d,
                                                        self.sbox_key_arr_3d[self.attack_round_index],
                                                        self.sbox_keycorr_arr_3d[self.attack_round_index],
                                                        self.sbox_keypos_arr_3d[self.attack_round_index],
                                                        self.candidates, self.num_batch_size)

            # 如果启用了第二个迹线集，将相关性数据写入
            if self.traceset2_switch:
                start_time = datetime.now()
                print(f"正在将Sbox{sbox_index}的相关性数据写入TraceSet，"
                      f"时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

                # 遍历所有可能的密钥猜测值，创建迹线并添加到迹线集
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

        # 完成分析流程
        self.finish_process()

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
        number_samples_per_trace = self.traceset.get_headers()[Header.NUMBER_SAMPLES]

        # 自动调整样本点起始位置
        if self.sample_first_pos < 0:
            self.sample_first_pos = 0
            print(f"自动调整样本点起始位置为: 0")

        # 自动调整样本点数量防止越界
        if self.sample_first_pos + self.sample_number > number_samples_per_trace:
            self.sample_number = number_samples_per_trace - self.sample_first_pos
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
            new_base_name = f"{base_name}+UsimCPA({timestamp})"
            traceset2_path = os.path.join(
                os.path.dirname(self.traceset_path),
                new_base_name + os.path.splitext(self.traceset_path)[1]
            )

            # 创建新的TRS文件用于存储相关系数曲线
            self.traceset2 = trsfile.trs_open(
                path=traceset2_path,
                mode='w',
                engine='TrsEngine',
                headers=headers,
                padding_mode=TracePadding.AUTO,
                live_update=True
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

    def load_samples_and_creat_intermediates(self):
        """
        加载样本点数据到内存并创建AES加密过程中S盒输出的中间值

        基于汉明重量能量消耗模型，主要完成两个任务：
        1. 将能量迹样本点数据加载到内存中的二维数组
        2. 根据攻击轮次计算对应的中间值
        """
        start_time = datetime.now()
        print(f"开始加载样本数据并创建AES加密Sbox输出中间值，时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # 计算分批处理参数
            num_batches = (self.trace_number + self.num_processes - 1) // self.num_processes
            print(f"开始加载 {self.trace_number} 条迹线，使用 {self.num_processes} 个进程并行处理，"
                  f"总共分为 {num_batches} 批进行加载")

            # 多进程并行加载迹线数据
            with Pool(processes=self.num_processes) as pool:
                for batch_idx in range(num_batches):
                    print(f"开始加载第 {batch_idx + 1:>3}/{num_batches} 批...")

                    # 计算当前批次处理的迹线范围
                    start_idx = batch_idx * self.num_processes
                    end_idx = min((batch_idx + 1) * self.num_processes, self.trace_number)
                    current_batch_size = end_idx - start_idx

                    # 准备当前批次的任务参数列表
                    batch_tasks = []
                    for i in range(current_batch_size):
                        trace_index = start_idx + i
                        batch_tasks.append((
                            self.traceset_path,
                            trace_index,
                            self.sample_first_pos,
                            self.sample_number,
                            self.sbox_num,
                            self.sbox_size,
                            self.attack_round_index,
                            self.sbox_key_arr_3d[:, :, 0]
                        ))

                    # 并行执行当前批次的处理任务
                    batch_results = pool.map(process_single_trace, batch_tasks)

                    # 按原始索引排序，确保输出文件中的迹线顺序正确
                    batch_results.sort(key=lambda x: x[0])

                    # 将处理结果顺序写入输出数组
                    for trace_index, sample_arr, data_arr_2d in batch_results:
                        self.sample_arr_2d[trace_index] = sample_arr
                        self.data_arr_3d[:, trace_index, :] = data_arr_2d

            print(f"所有批加载完成！")

        except Exception as e:
            print(f"加载过程中发生错误: {e}")
            raise

        # 计算并输出总耗时
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
            if self.sbox_key_arr_3d.shape[2] != self.candidates:
                self.candidates = self.sbox_key_arr_3d.shape[2]
                print(f"候选密钥数量（candidates）已调整为 {self.candidates} 以匹配历史数据。"
                      f"如需重置，请删除: {self.sbox_key_result_path}")
        except Exception as e:
            print(f"加载Sbox密钥攻击结果失败: {e}")
            try:
                os.remove(self.sbox_key_result_path)
                print(f"已删除损坏的文件: {self.sbox_key_result_path}")
            except Exception as delete_error:
                print(f"删除损坏文件失败: {delete_error}")

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
    usim_cpa = UsimCPA()

    # 第一轮攻击
    usim_cpa.num_processes = 16
    usim_cpa.num_batch_size = 10000
    usim_cpa.traceset_path = "D:\\traceset\\c51_Milenage\\milenage.trs"
    usim_cpa.traceset2_switch = False
    usim_cpa.sample_first_pos = 150000
    usim_cpa.sample_number = 200000
    usim_cpa.attack_round_index = 0
    usim_cpa.sbox_index_str = "0-15"
    usim_cpa.analyze()

    # 第二轮攻击
    usim_cpa.num_processes = 16
    usim_cpa.num_batch_size = 10000
    usim_cpa.traceset_path = "D:\\traceset\\c51_Milenage\\milenage.trs"
    usim_cpa.traceset2_switch = False
    usim_cpa.sample_first_pos = 150000
    usim_cpa.sample_number = 200000
    usim_cpa.attack_round_index = 1
    usim_cpa.sbox_index_str = "0-15"
    usim_cpa.analyze()
