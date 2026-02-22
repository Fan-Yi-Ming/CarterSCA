import os
import scipy.stats as stats
import numpy as np
import trsfile
import time
from datetime import datetime
from typing import Tuple
from trsfile import SampleCoding, Trace, Header, TracePadding
from trsfile.parametermap import TraceParameterMap, TraceSetParameterMap, TraceParameterDefinitionMap
from multiprocessing import Pool, cpu_count


def process_single_trace(traceset_path, trace_index) -> Tuple[int]:
    traceset = trsfile.open(traceset_path, 'r')
    trace = traceset[trace_index]
    traceset.close()

    group_value = trace.parameters["TVLA_GROUP"].value[0]
    if group_value not in [0x00, 0x01]:
        raise ValueError("TVLA_GROUP value must be 0 or 1.")

    return group_value


def process_single_trace2(traceset_path, trace_index, sample_first_pos, sample_number) -> Tuple[int, np.array]:
    traceset = trsfile.open(traceset_path, 'r')
    trace = traceset[trace_index]
    traceset.close()

    group_value = trace.parameters["TVLA_GROUP"].value[0]
    if group_value not in [0x00, 0x01]:
        raise ValueError("TVLA_GROUP value must be 0 or 1.")
    sample_arr = np.array(trace[sample_first_pos:sample_first_pos + sample_number], dtype=np.float32)

    return group_value, sample_arr


def t_test_core(fix_sample_arr_2d, rnd_sample_arr_2d):
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

    def __init__(self):
        self.process_number = cpu_count()

        self.trace_number: int = 0
        self.sample_first_pos: int = 0
        self.sample_number: int = 0

        self.t_value_threshold: float = 4.5

        self.fix_trace_number: int = 0
        self.rnd_trace_number: int = 0
        self.fix_sample_arr_2d = None
        self.rnd_sample_arr_2d = None
        self.t_values = None
        self.p_values = None

        self.traceset = None
        self.traceset_path: str = ""
        self.traceset2 = None

    def init_process(self):
        self.open_traceset()
        self.open_traceset2()

        self.fix_sample_arr_2d = np.zeros((self.fix_trace_number, self.sample_number), dtype=np.float32)
        self.rnd_sample_arr_2d = np.zeros((self.rnd_trace_number, self.sample_number), dtype=np.float32)
        self.t_values = np.zeros(self.sample_number, dtype=np.float32)
        self.p_values = np.zeros(self.sample_number, dtype=np.float32)

    def open_traceset(self):
        self.traceset = trsfile.open(self.traceset_path, 'r')
        self.trace_number = self.traceset.get_headers()[Header.NUMBER_TRACES]
        sample_number_per_trace = self.traceset.get_headers()[Header.NUMBER_SAMPLES]

        if self.sample_first_pos < 0:
            self.sample_first_pos = 0
            print(f"自动调整样本点起始位置为: 0")

        if self.sample_first_pos + self.sample_number > sample_number_per_trace:
            self.sample_number = sample_number_per_trace - self.sample_first_pos
            print(f"自动调整样本点数量为: {self.sample_number}")

        self.fix_trace_number = 0
        self.rnd_trace_number = 0

        start_time = time.monotonic()
        print(f"开始分别统计固定组组数和随机组组数")

        batch_size = 100
        batch_number = (self.trace_number + batch_size - 1) // batch_size

        # 多进程并行加载
        with Pool(processes=self.process_number) as pool:
            for batch_idx in range(batch_number):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, self.trace_number)
                current_batch_size = end_idx - start_idx
                print(f"批{batch_idx + 1}/{batch_number} ({current_batch_size}条)")

                # 准备任务参数
                batch_tasks = [(self.traceset_path, start_idx + i)
                               for i in range(current_batch_size)]

                # 并行处理
                batch_results = pool.starmap(process_single_trace, batch_tasks)

                # 写入结果
                for group_value in batch_results:
                    if group_value == 0x00:
                        self.fix_trace_number += 1
                    if group_value == 0x01:
                        self.rnd_trace_number += 1

        elapsed_time = time.monotonic() - start_time
        print(f"完成统计 固定组组数 {self.fix_trace_number} 随机组组数 {self.rnd_trace_number} "
              f"用时 {elapsed_time:.3f} 秒")

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

        base_name = os.path.splitext(os.path.basename(self.traceset_path))[0]
        timestamp = datetime.now().strftime("%H%M%S")
        new_base_name = f"{base_name}+TVLA({timestamp})"
        traceset2_path = os.path.join(os.path.dirname(self.traceset_path),
                                      new_base_name + os.path.splitext(self.traceset_path)[1])

        self.traceset2 = trsfile.trs_open(
            path=traceset2_path,
            mode='w',
            engine='TrsEngine',
            headers=headers,
            padding_mode=TracePadding.AUTO,
            live_update=True
        )

    def close_traceset(self):
        if self.traceset:
            self.traceset.close()

    def close_traceset2(self):
        if self.traceset2:
            self.traceset2.close()

    def load_samples(self):
        fix_indices = 0
        rnd_indices = 0

        start_time = time.monotonic()
        print(f"开始加载 {self.trace_number} 条迹线")

        batch_size = 100
        batch_number = (self.trace_number + batch_size - 1) // batch_size

        # 多进程并行加载
        with Pool(processes=self.process_number) as pool:
            for batch_idx in range(batch_number):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, self.trace_number)
                current_batch_size = end_idx - start_idx
                print(f"批{batch_idx + 1}/{batch_number} ({current_batch_size}条)")

                # 准备任务参数
                batch_tasks = [(self.traceset_path, start_idx + i,
                                self.sample_first_pos, self.sample_number)
                               for i in range(current_batch_size)]

                # 并行处理
                batch_results = pool.starmap(process_single_trace2, batch_tasks)

                # 写入结果
                for group_value, sample_arr in batch_results:
                    if group_value == 0x00:
                        self.fix_sample_arr_2d[fix_indices] = sample_arr
                        fix_indices += 1
                    if group_value == 0x01:
                        self.rnd_sample_arr_2d[rnd_indices] = sample_arr
                        rnd_indices += 1

        elapsed_time = time.monotonic() - start_time
        print(f"所有迹线加载完成 用时 {elapsed_time:.3f}秒")

    def t_test(self):
        self.init_process()
        self.load_samples()

        start_time = time.monotonic()
        print("开始计算T_Values P_Values")

        useful_trace_number = min(self.fix_trace_number, self.rnd_trace_number)
        print(f"使用 {useful_trace_number} 条迹线进行T-test")
        self.t_values, self.p_values = t_test_core(
            self.fix_sample_arr_2d[:useful_trace_number],
            self.rnd_sample_arr_2d[:useful_trace_number]
        )

        # T_Values
        trace = Trace(
            sample_coding=SampleCoding.FLOAT,
            samples=self.t_values,
            parameters=TraceParameterMap(),
            title="T_Values"
        )
        self.traceset2.append(trace)

        # 正阈值
        trace = Trace(
            sample_coding=SampleCoding.FLOAT,
            samples=np.full(self.sample_number, self.t_value_threshold),
            parameters=TraceParameterMap(),
            title="Positive_Threshold"
        )
        self.traceset2.append(trace)

        # 负阈值
        trace = Trace(
            sample_coding=SampleCoding.FLOAT,
            samples=np.full(self.sample_number, -self.t_value_threshold),
            parameters=TraceParameterMap(),
            title="Negative_Threshold"
        )
        self.traceset2.append(trace)

        # P_Values
        trace = Trace(
            sample_coding=SampleCoding.FLOAT,
            samples=self.p_values,
            parameters=TraceParameterMap(),
            title="P_Values"
        )
        self.traceset2.append(trace)

        elapsed_time = time.monotonic() - start_time
        print(f"T_Values P_Values计算完成 用时 {elapsed_time:.3f} 秒")

        self.close_traceset()
        self.close_traceset2()


if __name__ == '__main__':
    tvla_test = TVLATest()
    tvla_test.process_number = 16

    tvla_test.sample_first_pos = 380825
    tvla_test.sample_number = 120000
    tvla_test.t_value_threshold = 4.5
    tvla_test.traceset_path = "D:\\traceset\\aes128_en_tvla+LowPass(155914)+StaticAlign(160423).trs"

    tvla_test.t_test()
