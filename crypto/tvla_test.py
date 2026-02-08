import os
from datetime import datetime
from typing import Tuple

import numpy as np
import trsfile
from trsfile import SampleCoding, Trace, Header, TracePadding
from trsfile.parametermap import TraceParameterMap, TraceSetParameterMap, TraceParameterDefinitionMap
from multiprocessing import Pool, cpu_count
import scipy.stats as stats


def process_single_trace(args: Tuple) -> Tuple[int, int]:
    """处理单条迹线，返回迹线索引和分组值"""
    (traceset_path, trace_index) = args
    traceset = trsfile.open(traceset_path, 'r')
    trace = traceset[trace_index]
    group_value = trace.parameters["TVLA_GROUP"].value[0]
    if group_value not in [0x00, 0x01]:
        raise ValueError("TVLA_GROUP value must be 0 or 1.")
    return trace_index, group_value


def process_single_trace2(args: Tuple) -> Tuple[int, int, np.array]:
    """处理单条迹线，返回迹线索引、分组值和样本数据"""
    (traceset_path, trace_index, sample_first_pos, sample_number) = args
    traceset = trsfile.open(traceset_path, 'r')
    trace = traceset[trace_index]
    group_value = trace.parameters["TVLA_GROUP"].value[0]
    if group_value not in [0x00, 0x01]:
        raise ValueError("TVLA_GROUP value must be 0 or 1.")
    sample_arr = np.float32(trace[sample_first_pos:sample_first_pos + sample_number])
    return trace_index, group_value, sample_arr


def perform_ttest_vectorized(fix_sample_arr_2d, rnd_sample_arr_2d):
    """执行向量化的t检验，返回t值和p值"""
    fix_mean = np.mean(fix_sample_arr_2d, axis=0)
    rnd_mean = np.mean(rnd_sample_arr_2d, axis=0)

    fix_var = np.var(fix_sample_arr_2d, axis=0, ddof=1)
    rnd_var = np.var(rnd_sample_arr_2d, axis=0, ddof=1)

    fix_var = np.maximum(fix_var, 1e-12)
    rnd_var = np.maximum(rnd_var, 1e-12)

    n_fix = fix_sample_arr_2d.shape[0]
    n_rnd = rnd_sample_arr_2d.shape[0]

    # Welch's t-statistic
    se = np.sqrt(fix_var / n_fix + rnd_var / n_rnd)
    se = np.maximum(se, 1e-10)
    t_values = (fix_mean - rnd_mean) / se

    varexpr_fix = fix_var / n_fix
    varexpr_rnd = rnd_var / n_rnd

    df = (varexpr_fix + varexpr_rnd) ** 2 / (
            varexpr_fix ** 2 / (n_fix - 1) +
            varexpr_rnd ** 2 / (n_rnd - 1)
    )

    df = np.where(df > 1e12, np.inf, df)
    df = np.maximum(df, 1.0)

    p_values = 2 * stats.t.cdf(-np.abs(t_values), df)

    return t_values, p_values


class TVLATest:
    """TVLA测试类，用于执行基于t检验的泄露检测"""

    def __init__(self):
        # 进程配置
        self.num_processes = cpu_count()

        # 能量迹参数
        self.trace_number = 0
        self.sample_first_pos = 0
        self.sample_number = 0

        # TVLA参数
        self.t_value_threshold = 4.5

        # 能量迹文件
        self.traceset = None
        self.traceset_path = ""
        self.traceset2 = None

        # 分析数据
        self.fix_trace_number = 0
        self.rnd_trace_number = 0
        self.fix_sample_arr_2d = None
        self.rnd_sample_arr_2d = None
        self.t_values = None
        self.p_values = None

    def init_process(self):
        """初始化分析过程"""
        self.open_traceset()
        self.open_traceset2()

        # 初始化样本点数组
        self.fix_sample_arr_2d = np.zeros((self.fix_trace_number, self.sample_number), dtype=np.float32)
        self.rnd_sample_arr_2d = np.zeros((self.rnd_trace_number, self.sample_number), dtype=np.float32)

        self.t_values = np.zeros(self.sample_number, dtype=np.float32)
        self.p_values = np.zeros(self.sample_number, dtype=np.float32)

    def open_traceset(self):
        """打开原始能量迹TRS文件并初始化参数"""
        self.traceset = trsfile.open(self.traceset_path, 'r')
        self.trace_number = self.traceset.get_headers()[Header.NUMBER_TRACES]
        sample_number_per_trace = self.traceset.get_headers()[Header.NUMBER_SAMPLES]

        # 自动调整样本点范围
        if self.sample_first_pos < 0:
            self.sample_first_pos = 0
            print(f"自动调整样本点起始位置为: 0")

        if self.sample_first_pos + self.sample_number > sample_number_per_trace:
            self.sample_number = sample_number_per_trace - self.sample_first_pos
            print(f"自动调整样本点数量为: {self.sample_number}")

        start_time = datetime.now()
        print(f"开始分别统计固定组组数和随机组组数，时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        self.fix_trace_number = 0
        self.rnd_trace_number = 0

        try:
            # 分批统计分组数量
            num_batches = (self.trace_number + self.num_processes - 1) // self.num_processes
            print(f"开始加载 {self.trace_number} 条迹线，使用 {self.num_processes} 个进程并行处理，"
                  f"总共分为 {num_batches} 批进行加载")

            with Pool(processes=self.num_processes) as pool:
                for batch_idx in range(num_batches):
                    print(f"开始加载第 {batch_idx + 1:>3}/{num_batches} 批...")

                    start_idx = batch_idx * self.num_processes
                    end_idx = min((batch_idx + 1) * self.num_processes, self.trace_number)
                    current_batch_size = end_idx - start_idx

                    batch_tasks = []
                    for i in range(current_batch_size):
                        trace_index = start_idx + i
                        batch_tasks.append((self.traceset_path, trace_index))

                    batch_results = pool.map(process_single_trace, batch_tasks)
                    batch_results.sort(key=lambda x: x[0])

                    for trace_index, group_value in batch_results:
                        if group_value == 0x00:
                            self.fix_trace_number += 1
                        if group_value == 0x01:
                            self.rnd_trace_number += 1

            print(f"所有批加载完成！")

        except Exception as e:
            print(f"加载过程中发生错误: {e}")
            raise

        total_time = (datetime.now() - start_time).total_seconds()
        print(f"完成固定组组数和随机组组数统计！总用时 {total_time:.3f} 秒")

    def open_traceset2(self):
        """创建并打开用于存储TVLA结果的TRS文件"""
        traceset_parameter_map = TraceSetParameterMap()
        trace_parameter_definition_map = TraceParameterDefinitionMap()

        headers = {
            Header.TRS_VERSION: 2,
            Header.TITLE_SPACE: 255,
            Header.SAMPLE_CODING: SampleCoding.FLOAT,
            Header.LABEL_X: "Sample Point",
            Header.SCALE_X: 1.0,
            Header.LABEL_Y: "Value",
            Header.SCALE_Y: 1.0,
            Header.TRACE_SET_PARAMETERS: traceset_parameter_map,
            Header.TRACE_PARAMETER_DEFINITIONS: trace_parameter_definition_map
        }

        # 构建新文件名
        base_name = os.path.splitext(os.path.basename(self.traceset_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_base_name = f"{base_name}+TVLA({timestamp})"
        traceset2_path = os.path.join(
            os.path.dirname(self.traceset_path),
            new_base_name + os.path.splitext(self.traceset_path)[1]
        )

        # 创建新的TRS文件
        self.traceset2 = trsfile.trs_open(
            path=traceset2_path,
            mode='w',
            engine='TrsEngine',
            headers=headers,
            padding_mode=TracePadding.AUTO,
            live_update=True
        )

    def close_traceset(self):
        """关闭所有打开的能量迹文件"""
        if self.traceset:
            self.traceset.close()
        if self.traceset2:
            self.traceset2.close()

    def load_sample_data(self):
        """加载样本数据"""
        fix_indices = 0
        rnd_indices = 0

        start_time = datetime.now()
        print(f"开始分别加载固定组数据和随机组数据，时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # 分批处理参数
            num_batches = (self.trace_number + self.num_processes - 1) // self.num_processes
            print(f"开始加载 {self.trace_number} 条迹线，使用 {self.num_processes} 个进程并行处理，"
                  f"总共分为 {num_batches} 批进行加载")

            # 多进程并行加载
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
                            self.traceset_path,
                            trace_index,
                            self.sample_first_pos,
                            self.sample_number,
                        ))

                    # 并行执行
                    batch_results = pool.map(process_single_trace2, batch_tasks)
                    batch_results.sort(key=lambda x: x[0])

                    # 处理结果
                    for trace_index, group_value, sample_arr, in batch_results:
                        if group_value == 0x00:
                            self.fix_sample_arr_2d[fix_indices] = sample_arr
                            fix_indices += 1
                        if group_value == 0x01:
                            self.rnd_sample_arr_2d[rnd_indices] = sample_arr
                            rnd_indices += 1

            print(f"所有批加载完成！")

        except Exception as e:
            print(f"加载过程中发生错误: {e}")
            raise

        total_time = (datetime.now() - start_time).total_seconds()
        print(f"完成加载固定组数据和随机组数据加载！总用时 {total_time:.3f} 秒")

    def calculate_t_test(self):
        """计算t值和p值"""
        self.init_process()
        self.load_sample_data()

        print("开始计算T_Values P_Values...")
        start_time = datetime.now()

        use_trace_number = min(self.fix_trace_number, self.rnd_trace_number)
        print(f"使用 {use_trace_number} 条迹线进行T-test")
        self.t_values, self.p_values = perform_ttest_vectorized(
            self.fix_sample_arr_2d[:use_trace_number],
            self.rnd_sample_arr_2d[:use_trace_number]
        )

        # 保存结果到TRS文件
        trace = Trace(
            sample_coding=SampleCoding.FLOAT,
            samples=self.t_values,
            parameters=TraceParameterMap(),
            title="T_Values"
        )
        self.traceset2.append(trace)

        # 保存正阈值
        trace_positive = Trace(
            sample_coding=SampleCoding.FLOAT,
            samples=np.full(self.sample_number, self.t_value_threshold),
            parameters=TraceParameterMap(),
            title="Positive_Threshold"
        )
        self.traceset2.append(trace_positive)

        # 保存负阈值
        trace_negative = Trace(
            sample_coding=SampleCoding.FLOAT,
            samples=np.full(self.sample_number, -self.t_value_threshold),
            parameters=TraceParameterMap(),
            title="Negative_Threshold"
        )
        self.traceset2.append(trace_negative)

        trace = Trace(
            sample_coding=SampleCoding.FLOAT,
            samples=self.p_values,
            parameters=TraceParameterMap(),
            title="P_Values"
        )
        self.traceset2.append(trace)

        end_time = datetime.now()
        elapsed_time = end_time - start_time
        print(f"T_Values P_Values计算完成，耗时: {elapsed_time.total_seconds():.2f} 秒")

        self.close_traceset()


if __name__ == '__main__':
    # ==================== 使用示例 ====================
    tvla_test = TVLATest()

    tvla_test.num_processes = 16

    tvla_test.sample_first_pos = 0
    tvla_test.sample_number = 200000
    tvla_test.t_value_threshold = 4.5

    tvla_test.traceset_path = "..\\traceset\\tvla_test_usim158+LowPass(20251125120754)+StaticAlign(20251125121346)+StaticAlign(20251125123124).trs"

    # 执行TVLA测试
    tvla_test.calculate_t_test()
