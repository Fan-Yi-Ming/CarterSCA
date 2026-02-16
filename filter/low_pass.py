import os
import copy
from multiprocessing import Pool, cpu_count
import numpy as np
import trsfile
from datetime import datetime
from typing import Tuple
import trsfile.traceparameter as tp
from trsfile import Header, TracePadding, SampleCoding, Trace


def fast_lowpass_filter(signal: np.ndarray, weight: float) -> np.ndarray:
    """
    零相位延迟双向一阶低通滤波器（前向+反向滤波消除相位偏移）
    """
    signal = np.array(signal, dtype=np.float32, copy=True)
    alpha = 1.0 / (weight + 1.0)  # 滤波系数

    # 前向滤波
    forward_pass = np.zeros_like(signal)
    forward_pass[0] = signal[0]
    for i in range(1, len(signal)):
        forward_pass[i] = alpha * signal[i] + (1.0 - alpha) * forward_pass[i - 1]

    # 反向滤波（补偿相位延迟）
    backward_pass = np.zeros_like(signal)
    backward_pass[-1] = forward_pass[-1]
    for i in range(len(signal) - 2, -1, -1):
        backward_pass[i] = alpha * forward_pass[i] + (1.0 - alpha) * backward_pass[i + 1]

    return backward_pass


def process_single_trace(args: Tuple) -> Tuple[int, Trace]:
    """
    多进程工作函数：对单条迹线执行低通滤波
    返回：(原始索引, 滤波后迹线对象)
    """
    trace_index, trace, weight = args

    # 执行零相位低通滤波
    low_pass_samples = fast_lowpass_filter(trace.samples, weight)

    # 构造滤波后迹线（强制转为浮点编码）
    low_pass_trace = Trace(
        sample_coding=SampleCoding.FLOAT,
        samples=low_pass_samples,
        parameters=trace.parameters,
        title=trace.title)

    return trace_index, low_pass_trace


class LowPass:
    """
    多进程零相位低通滤波器（适用于大规模 .trs 能量迹集）
    高效抑制高频噪声，同时完全保持信号时序对齐
    """

    def __init__(self):
        # 滤波参数
        self.weight = 1.0  # 滤波权重（越大越平滑）
        self.num_processes = cpu_count()  # 并行进程数

        # 文件句柄
        self.traceset = None  # 输入迹线集
        self.traceset_path: str = ""  # 输入文件路径
        self.traceset2 = None  # 输出迹线集

    def open_traceset(self):
        """打开原始文件并创建带时间戳的新输出文件"""
        self.traceset = trsfile.open(self.traceset_path, 'r')

        # 生成新文件名：原名 + LowPass + 时间戳
        base_name = os.path.splitext(os.path.basename(self.traceset_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_base_name = f"{base_name}+LowPass({timestamp})"
        traceset2_path = os.path.join(os.path.dirname(self.traceset_path),
                                      new_base_name + os.path.splitext(self.traceset_path)[1])

        # 复制头信息并更新关键字段
        new_headers = copy.deepcopy(self.traceset.get_headers())
        new_headers[Header.NUMBER_TRACES] = 0
        new_headers[Header.SAMPLE_CODING] = SampleCoding.FLOAT
        new_headers[Header.TRACE_SET_PARAMETERS]["LOW_PASS_WEIGHT"] = tp.FloatArrayParameter([self.weight])

        # 创建输出文件（实时更新模式）
        self.traceset2 = trsfile.trs_open(
            path=traceset2_path,
            mode='w',
            engine='TrsEngine',
            headers=new_headers,
            padding_mode=TracePadding.AUTO,
            live_update=True
        )

    def close_traceset(self):
        """安全关闭输入输出文件"""
        if self.traceset:
            self.traceset.close()
            self.traceset = None
        if self.traceset2:
            self.traceset2.close()
            self.traceset2 = None

    def low_pass(self):
        """主滤波流程：分批多进程并行处理 + 顺序写入 + 进度显示"""
        self.open_traceset()

        try:
            number_traces = self.traceset.get_headers()[Header.NUMBER_TRACES]
            print(f"开始处理 {number_traces} 条迹线...")
            print(f"使用 {self.num_processes} 个进程并行处理")

            num_batches = (number_traces + self.num_processes - 1) // self.num_processes
            print(f"总共分为 {num_batches} 批进行处理")

            with Pool(processes=self.num_processes) as pool:
                for batch_idx in range(num_batches):
                    print(f"\n开始处理第 {batch_idx + 1}/{num_batches} 批...")

                    start_idx = batch_idx * self.num_processes
                    end_idx = min((batch_idx + 1) * self.num_processes, number_traces)
                    batch_size = end_idx - start_idx

                    # 构造当前批次任务
                    tasks = [(start_idx + i,
                              self.traceset[start_idx + i],
                              self.weight) for i in range(batch_size)]

                    # 并行滤波
                    results = pool.map(process_single_trace, tasks)
                    results.sort(key=lambda x: x[0])  # 保持原始顺序

                    # 顺序写入输出文件
                    for _, trace in results:
                        self.traceset2.append(trace)

                    print(f"第 {batch_idx + 1} 批处理完成，写入 {batch_size} 条迹线")
                    print(f"进度: {end_idx}/{number_traces} ({end_idx / number_traces * 100:.1f}%)")

            print(f"\n所有批处理完成！总共处理了 {number_traces} 条迹线")

        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            raise
        finally:
            self.close_traceset()


if __name__ == '__main__':
    # ==================== 使用示例 ====================
    low_pass = LowPass()

    low_pass.num_processes = 16

    low_pass.weight = 20.0

    low_pass.traceset_path = "D:\\traceset\\c51_aes128\\aes128_de.trs"

    low_pass.low_pass()
