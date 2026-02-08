import os
import copy
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import trsfile
from trsfile import Header, TracePadding, SampleCoding, Trace
import trsfile.traceparameter as tp


def fast_lowpass_filter(signal: np.ndarray, weight: float) -> np.ndarray:
    """
    使用双向滤波的快速低通滤波器

    参数:
        signal: 输入信号，一维numpy数组
        weight: 滤波权重，值越大平滑效果越强

    返回:
        滤波后的信号，零相位延迟
    """
    # 将输入信号转换为float32类型，确保数值精度和计算效率
    signal = np.array(signal, dtype=np.float32, copy=True)

    # 计算滤波系数：权重越大，alpha越小，平滑效果越强
    alpha = 1.0 / (weight + 1.0)

    # 前向滤波：从信号起始位置开始滤波，消除高频噪声
    forward_pass = np.zeros_like(signal)
    forward_pass[0] = signal[0]  # 初始化第一个样本

    # 前向递归滤波：当前值 = α×当前输入 + (1-α)×前一个滤波值
    for i in range(1, len(signal)):
        forward_pass[i] = alpha * signal[i] + (1.0 - alpha) * forward_pass[i - 1]

    # 反向滤波：从信号末尾开始反向滤波，消除相位失真
    backward_pass = np.zeros_like(signal)
    backward_pass[-1] = forward_pass[-1]  # 初始化最后一个样本

    # 反向递归滤波：确保零相位延迟
    for i in range(len(signal) - 2, -1, -1):
        backward_pass[i] = alpha * forward_pass[i] + (1.0 - alpha) * backward_pass[i + 1]

    return backward_pass


def process_trace(trace_index: int, trace: Trace, weight: float) -> tuple[int, Trace]:
    """
    处理单条Trace的函数，用于多线程执行

    参数:
        trace_index: Trace索引号
        trace: 原始Trace对象
        weight: 滤波权重参数

    返回:
        tuple[int, Trace]: 包含处理后的Trace索引和滤波后的Trace对象
    """
    # 对当前Trace的样本数据应用低通滤波
    low_pass = fast_lowpass_filter(trace.samples, weight)

    # 创建滤波后的Trace对象，保留原始参数和标题信息
    low_pass_trace = Trace(
        sample_coding=SampleCoding.FLOAT,  # 使用浮点数编码存储滤波结果
        samples=low_pass,  # 滤波后的样本数据
        parameters=trace.parameters,  # 保留原始Trace的参数信息
        title=trace.title  # 保留原始Trace的标题
    )

    return trace_index, low_pass_trace


class LowPassFilter:
    """
    低通滤波器类，用于对TraceSet进行批量滤波处理

    该类实现了多线程并行处理，每批处理的Trace数量等于线程数，
    确保高效利用计算资源
    """

    def __init__(self):
        """初始化低通滤波器参数"""
        # 低通滤波参数：权重值越大，滤波效果越强，信号越平滑
        self.weight = 1.0

        # 多线程参数：同时处理Trace的线程数量，也作为每批处理的Trace数量
        self.max_workers = 4

        # TraceSet相关变量
        self.traceset = None  # 待滤波的原始TraceSet对象
        self.traceset_path: str = ""  # 原始TraceSet文件路径
        self.traceset2 = None  # 低通滤波后的TraceSet对象
        self.traceset2_dir: str = ""  # 滤波后TraceSet保存目录

    def open_traceset(self):
        """
        打开原始TraceSet并创建滤波后的TraceSet文件

        该方法执行以下操作：
        1. 打开原始TraceSet文件（只读模式）
        2. 创建新的输出文件路径
        3. 复制并修改文件头信息
        4. 创建新的TraceSet文件（写入模式）
        """
        # 以只读模式打开原始TraceSet文件
        self.traceset = trsfile.open(self.traceset_path, 'r')

        # 构建新的输出文件路径：在原文件名基础上添加时间戳和滤波标识
        base_name = os.path.splitext(os.path.basename(self.traceset_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_base_name = f"{base_name}+LowPass({timestamp})"
        traceset2_path = os.path.join(os.path.dirname(self.traceset_path),
                                      new_base_name + os.path.splitext(self.traceset_path)[1])

        # 深度复制原始文件头信息，为创建新文件做准备
        new_headers = copy.deepcopy(self.traceset.get_headers())
        new_headers[Header.NUMBER_TRACES] = 0  # 重置Trace数量为0
        new_headers[Header.SAMPLE_CODING] = SampleCoding.FLOAT  # 设置样本编码为浮点数

        # 在TraceSet参数中添加滤波权重信息，便于后续追溯处理参数
        traceset_parameter_map = new_headers[Header.TRACE_SET_PARAMETERS]
        traceset_parameter_map['LOW_PASS_WEIGHT'] = tp.FloatArrayParameter([self.weight])

        # 创建新的TraceSet文件，用于存储滤波后的Trace数据
        self.traceset2 = trsfile.trs_open(
            path=traceset2_path,  # 新文件路径
            mode='w',  # 写入模式
            engine='TrsEngine',  # 使用TRS引擎
            headers=new_headers,  # 更新后的头部信息
            padding_mode=TracePadding.AUTO,  # 自动填充模式
            live_update=True  # 启用实时更新，支持进度预览
        )

    def close_traceset(self):
        """安全关闭当前打开的所有TraceSet文件"""
        if self.traceset:
            self.traceset.close()
        if self.traceset2:
            self.traceset2.close()

    def low_pass(self):
        """
        批量多线程低通滤波处理主函数

        处理流程：
        1. 打开TraceSet文件
        2. 计算总批次数
        3. 按批次处理：
           - 准备当前批次任务
           - 使用线程池并行处理
           - 等待批次完成
           - 按顺序写入结果
        4. 关闭文件

        每批处理的Trace数量等于线程数，确保充分利用多核CPU资源
        """
        # 步骤1：打开输入和输出TraceSet文件
        self.open_traceset()

        # 获取Trace总数，用于进度计算和批次划分
        number_traces = self.traceset.get_headers()[Header.NUMBER_TRACES]

        # 打印处理开始信息和参数配置
        print(f"开始批量多线程低通滤波处理")
        print(f"总Trace数: {number_traces}, 线程数: {self.max_workers}")

        # 步骤2：准备批次处理数据
        # 将所有Trace及其索引打包成列表，便于按索引访问
        all_traces = list(enumerate(self.traceset))

        # 计算总批次数：使用向上取整确保所有Trace都被处理
        total_batches = (number_traces + self.max_workers - 1) // self.max_workers

        # 步骤3：按批次循环处理所有Trace
        for batch_num in range(total_batches):
            # 计算当前批次的起始和结束索引
            start_idx = batch_num * self.max_workers
            end_idx = min((batch_num + 1) * self.max_workers, number_traces)
            current_batch_size = end_idx - start_idx

            # 打印当前批次信息
            print(f"\n处理第 {batch_num + 1}/{total_batches} 批Trace ({start_idx}-{end_idx - 1})...")

            # 使用线程池并行处理当前批次
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交当前批次的所有任务，直接传入具体参数
                future_to_index = {}
                for i in range(start_idx, end_idx):
                    trace_index, trace = all_traces[i]
                    # 直接传入具体参数而不是打包成元组
                    future = executor.submit(process_trace, trace_index, trace, self.weight)
                    future_to_index[future] = trace_index

                # 等待当前批次所有任务完成
                batch_results = [None] * current_batch_size
                completed_in_batch = 0

                for future in as_completed(future_to_index):
                    try:
                        # 获取任务结果：Trace索引和滤波后的Trace对象
                        trace_index, low_pass_trace = future.result()

                        # 计算当前Trace在批次中的位置
                        batch_position = trace_index - start_idx

                        # 按原始顺序存储结果
                        batch_results[batch_position] = low_pass_trace
                        completed_in_batch += 1

                        # 打印批次内进度
                        print(f"批次 {batch_num + 1} 完成: {completed_in_batch}/{current_batch_size}")

                    except Exception as e:
                        # 错误处理：记录失败Trace信息，但不中断整体处理
                        trace_index = future_to_index[future]
                        print(f"处理Trace {trace_index} 时发生错误: {e}")

            # 当前批次处理完成，按顺序写入文件
            print(f"正在写入第 {batch_num + 1} 批Trace...")
            for low_pass_trace in batch_results:
                if low_pass_trace is not None:
                    self.traceset2.append(low_pass_trace)

            # 打印批次完成统计信息
            successful_traces = len([r for r in batch_results if r is not None])
            print(f"第 {batch_num + 1} 批处理完成，已写入 {successful_traces} 条Trace")

        # 步骤4：处理完成，关闭所有文件
        self.close_traceset()
        print("\n低通滤波处理完成!")


if __name__ == '__main__':
    """
    主程序入口：配置滤波器参数并执行滤波处理

    使用示例：
    1. 创建滤波器实例
    2. 配置处理参数（线程数、权重、文件路径）
    3. 执行批量多线程滤波处理
    """
    # 创建低通滤波器实例
    low_pass_filter = LowPassFilter()

    # 配置滤波处理参数
    low_pass_filter.max_workers = 4  # 设置16个线程并行处理
    low_pass_filter.weight = 5.0
    low_pass_filter.traceset_path = "../traceset/aes128_en_1000.trs"
    low_pass_filter.traceset2_dir = "../traceset"

    # 执行批量多线程低通滤波处理
    low_pass_filter.low_pass()
