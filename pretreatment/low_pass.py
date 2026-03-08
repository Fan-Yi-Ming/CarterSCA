import os
import copy
import numpy as np
import trsfile.traceparameter as tp
import trsfile
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
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


def process_single_trace(trace, weight) -> Trace:
    # 执行零相位低通滤波
    low_pass_samples = fast_lowpass_filter(trace.samples, weight)

    # 构造滤波后迹线
    low_pass_trace = Trace(
        sample_coding=SampleCoding.FLOAT,
        samples=low_pass_samples,
        parameters=trace.parameters,
        title=trace.title,
        headers=trace.headers)

    return low_pass_trace


class LowPass:
    def __init__(self):
        self.num_processes: int = cpu_count()

        # 滤波参数
        self.weight: float = 1.0

        self.traceset = None
        self.traceset_path: str = ""
        self.traceset2 = None

    def open_traceset(self):
        self.traceset = trsfile.open(self.traceset_path, 'r')

        headers = copy.deepcopy(self.traceset.get_headers())
        headers[Header.NUMBER_TRACES] = 0
        headers[Header.SAMPLE_CODING] = SampleCoding.FLOAT
        headers[Header.TRACE_SET_PARAMETERS]["LOW_PASS_WEIGHT"] = tp.FloatArrayParameter([self.weight])

        base_name = os.path.splitext(os.path.basename(self.traceset_path))[0]
        timestamp = datetime.now().strftime("%H%M%S")
        new_base_name = f"{base_name}+LowPass({timestamp})"
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
        if self.traceset2:
            self.traceset2.close()

    def low_pass(self):
        self.open_traceset()
        trace_number = self.traceset.get_headers()[Header.NUMBER_TRACES]
        start_time = time.perf_counter()
        print(f"开始低通滤波 {trace_number} 条迹线")

        batch_size = self.num_processes
        batch_number = (trace_number + batch_size - 1) // batch_size

        # 多进程并行加载
        with Pool(processes=self.num_processes) as pool:
            for batch_idx in range(batch_number):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, trace_number)
                current_batch_size = end_idx - start_idx
                print(f"批{batch_idx + 1}/{batch_number} ({current_batch_size}条)")

                # 准备任务参数
                batch_tasks = [(self.traceset[start_idx + i], self.weight) for i in range(current_batch_size)]

                # 并行处理
                batch_results = pool.starmap(process_single_trace, batch_tasks)

                # 写入结果
                for low_pass_trace in batch_results:
                    self.traceset2.append(low_pass_trace)

        elapsed_time = time.perf_counter() - start_time
        print(f"低通滤波完成 用时:{elapsed_time:.3f}秒")
        self.close_traceset()


if __name__ == '__main__':
    low_pass = LowPass()
    low_pass.num_processes = 16

    low_pass.weight = 20.0
    low_pass.traceset_path = "D:\\traceset\\milenage.trs"

    low_pass.low_pass()
