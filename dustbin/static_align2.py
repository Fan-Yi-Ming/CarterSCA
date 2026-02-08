import copy
import os
import numpy as np
import trsfile
from trsfile import Header, TracePadding, Trace
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from dataclasses import dataclass


@dataclass
class AlignmentResult:
    """对齐结果数据类"""
    trace_index: int
    aligned_trace: Optional[Trace]
    best_shift: int
    best_corr: float


def compute_correlation(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
    """计算两个数组之间的相关系数

    Args:
        x_arr: 第一个输入数组
        y_arr: 第二个输入数组

    Returns:
        float: 两个数组之间的皮尔逊相关系数
    """
    # 使用 np.corrcoef 计算相关性矩阵
    corr_matrix = np.corrcoef(x_arr, y_arr)
    # 提取 x_arr 和 y_arr 之间的相关系数（相关矩阵的左上角元素）
    return float(corr_matrix[0, 1])


def align_single_trace(trace_index: int, trace: Trace, pattern_arr: np.ndarray,
                       pattern_first_sample_pos: int, pattern_sample_number: int,
                       shift_range: list, threshold: float, ref_trace_index: int) -> AlignmentResult:
    """处理单条Trace的对齐函数，用于多线程执行

    Args:
        trace_index: Trace索引号
        trace: 原始Trace对象
        pattern_arr: 模板数据数组
        pattern_first_sample_pos: 模板起始位置
        pattern_sample_number: 模板样本点数量
        shift_range: 偏移范围列表
        threshold: 相关性阈值
        ref_trace_index: 参考Trace索引

    Returns:
        AlignmentResult: 对齐结果对象
    """
    # 如果是参考迹线，直接返回原迹线
    if trace_index == ref_trace_index:
        return AlignmentResult(
            trace_index=trace_index,
            aligned_trace=trace,
            best_shift=0,
            best_corr=1.0
        )

    best_shift = 0  # 最佳偏移量
    best_corr = -1.0  # 最佳相关系数，初始化为可能的最小值

    # 在允许的偏移范围内搜索最佳移位
    for shift in shift_range:
        # 计算当前移位后的模板位置
        start_pos = pattern_first_sample_pos + shift
        end_pos = start_pos + pattern_sample_number

        # 计算当前移位下的相关系数
        corr_temp = compute_correlation(pattern_arr, trace.samples[start_pos:end_pos])
        # 更新最佳偏移量和相关系数
        if corr_temp > best_corr:
            best_shift = shift
            best_corr = corr_temp

    # 根据阈值判断是否保留该迹线
    if best_corr >= threshold:
        # 如果存在偏移，对迹线进行循环移位
        if best_shift != 0:
            aligned_samples = np.roll(trace.samples, -best_shift)
            # 创建对齐后的Trace对象
            aligned_trace = Trace(
                sample_coding=trace.sample_coding,
                samples=aligned_samples,
                parameters=trace.parameters,
                title=trace.title
            )
            return AlignmentResult(
                trace_index=trace_index,
                aligned_trace=aligned_trace,
                best_shift=best_shift,
                best_corr=best_corr
            )
        else:
            return AlignmentResult(
                trace_index=trace_index,
                aligned_trace=trace,
                best_shift=best_shift,
                best_corr=best_corr
            )
    else:
        # 相关系数低于阈值，返回None表示排除该迹线
        return AlignmentResult(
            trace_index=trace_index,
            aligned_trace=None,
            best_shift=best_shift,
            best_corr=best_corr
        )


class StaticAlign:
    """静态对齐类，用于对能量迹进行基于模板匹配的对齐处理"""

    def __init__(self):
        # 初始化对齐参数
        self.ref_trace_index: int = 0  # 模板匹配参考能量迹的索引
        self.pattern_first_sample_pos: int = 0  # 模板匹配的样本点起始位置
        self.pattern_sample_number: int = 0  # 模板匹配的样本点数量
        self.shift_max: int = 0  # 最大允许偏移量
        self.step_size: int = 1  # 步进长度，默认为1（逐个移动）
        self.threshold: float = 0.0  # 相关性阈值，大于该值的迹线将被保留
        self.max_workers: int = 4  # 多线程参数，同时处理的线程数量

        # 迹线集相关变量
        self.traceset = None  # 待对齐的原始能量迹集
        self.traceset_path: str = ""  # 待对齐的能量迹集文件路径
        self.traceset2 = None  # 对齐后的能量迹集
        self.traceset2_dir: str = ""  # 对齐后的能量迹集保存目录

    def open_traceset(self):
        """打开原始迹线集并创建对齐后的迹线集文件"""
        # 打开原始的 TraceSet 文件（只读模式）
        self.traceset = trsfile.open(self.traceset_path, 'r')

        # 构建新的文件路径，用于保存对齐后的 TraceSet
        base_name = os.path.splitext(os.path.basename(self.traceset_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_base_name = f"{base_name}+StaticAlign({timestamp})"
        new_traceset_path = os.path.join(os.path.dirname(self.traceset_path),
                                         new_base_name + os.path.splitext(self.traceset_path)[1])

        # 复制原始文件头信息，并将迹线数量重置为0
        new_headers = copy.deepcopy(self.traceset.get_headers())
        new_headers[Header.NUMBER_TRACES] = 0

        # 创建新的 TraceSet 文件（写入模式）
        self.traceset2 = trsfile.trs_open(
            path=new_traceset_path,  # 新文件路径
            mode='w',  # 写入模式
            engine='TrsEngine',  # 存储引擎
            headers=new_headers,  # 新的头部信息
            padding_mode=TracePadding.AUTO,  # 填充模式
            live_update=True  # 启用实时更新，便于预览
        )

    def close_traceset(self):
        """关闭当前打开的 TraceSet 文件"""
        if self.traceset:
            self.traceset.close()
        if self.traceset2:
            self.traceset2.close()

    def align(self):
        """执行能量迹对齐操作

        基于模板匹配方法，通过计算相关系数找到最佳偏移量，
        并对符合条件的迹线进行循环移位对齐
        """
        # 参数验证
        if self.threshold <= 0:
            raise ValueError("相关性阈值必须大于0")
        if self.step_size <= 0:
            raise ValueError("步进长度必须大于0")
        if self.step_size > self.shift_max:
            print(f"警告：步进长度({self.step_size})大于最大偏移量({self.shift_max})，将步进长度设置为最大偏移量")
            self.step_size = self.shift_max

        # 打开迹线集文件
        self.open_traceset()

        # 获取迹线的总采样点数和总Trace数
        number_of_samples = self.traceset.get_headers()[Header.NUMBER_SAMPLES]
        number_traces = self.traceset.get_headers()[Header.NUMBER_TRACES]

        # 自动调整模板的采样数量，防止超出迹线范围
        if self.pattern_first_sample_pos + self.pattern_sample_number > number_of_samples:
            self.pattern_sample_number = number_of_samples - self.pattern_first_sample_pos
            print(f"自动调整后模版最多允许：{self.pattern_sample_number} 个点")

        # 从参考迹线中提取模板数据
        pattern_arr = self.traceset[self.ref_trace_index].samples[
                      self.pattern_first_sample_pos:self.pattern_first_sample_pos + self.pattern_sample_number]

        # 计算最大左移量，考虑边界限制
        shift_left = min(self.pattern_first_sample_pos, self.shift_max)
        print(f"最大允许左移：{shift_left} 个点")

        # 计算最大右移量，考虑边界限制
        shift_right = min(number_of_samples - (self.pattern_first_sample_pos + self.pattern_sample_number),
                          self.shift_max)
        print(f"最大允许右移：{shift_right} 个点")

        # 使用步进长度生成偏移范围
        shift_range = list(range(-shift_left, shift_right + 1, self.step_size))
        # 确保包含边界点
        if shift_range[-1] != shift_right:
            shift_range.append(shift_right)
        print(f"搜索偏移范围：{len(shift_range)} 个位置 (步长: {self.step_size})")
        print(f"使用多线程处理，线程数: {self.max_workers}")

        # 设置批量大小等于线程数
        batch_size = self.max_workers

        # 将所有Trace分成多个批次
        all_traces = list(enumerate(self.traceset))
        total_batches = (number_traces + batch_size - 1) // batch_size

        # 按批次处理所有Trace
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, number_traces)
            current_batch_size = end_idx - start_idx

            print(f"\n处理第 {batch_num + 1}/{total_batches} 批Trace ({start_idx}-{end_idx - 1})...")

            # 使用线程池并行处理当前批次
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交当前批次的所有任务
                future_to_index = {}
                for i in range(start_idx, end_idx):
                    trace_index, trace = all_traces[i]
                    future = executor.submit(
                        align_single_trace,
                        trace_index, trace, pattern_arr,
                        self.pattern_first_sample_pos, self.pattern_sample_number,
                        shift_range, self.threshold, self.ref_trace_index
                    )
                    future_to_index[future] = trace_index

                # 等待当前批次所有任务完成
                batch_results: list[Optional[AlignmentResult]] = [None] * current_batch_size
                completed_in_batch = 0
                included_in_batch = 0

                for future in as_completed(future_to_index):
                    try:
                        result: AlignmentResult = future.result()

                        # 计算在当前批次中的位置
                        batch_position = result.trace_index - start_idx
                        batch_results[batch_position] = result
                        completed_in_batch += 1

                        # 打印进度
                        if result.aligned_trace is not None:
                            included_in_batch += 1
                            print(f"批次 {batch_num + 1} 完成: {completed_in_batch}/{current_batch_size}, "
                                  f"Trace {result.trace_index} 偏移: {result.best_shift}, 相关性: {result.best_corr:.3f}")
                        else:
                            print(f"批次 {batch_num + 1} 完成: {completed_in_batch}/{current_batch_size}, "
                                  f"Trace {result.trace_index} 排除, 相关性: {result.best_corr:.3f} (阈值: {self.threshold})")

                    except Exception as e:
                        trace_index = future_to_index[future]
                        print(f"处理Trace {trace_index} 时发生错误: {e}")

            # 当前批次处理完成，按顺序写入文件
            print(f"正在写入第 {batch_num + 1} 批Trace...")
            written_in_batch = 0
            for result in batch_results:
                if result is not None and result.aligned_trace is not None:
                    self.traceset2.append(result.aligned_trace)
                    written_in_batch += 1

            print(f"第 {batch_num + 1} 批处理完成，已写入 {written_in_batch} 条Trace")

        # 关闭迹线集文件
        self.close_traceset()
        print("\n静态对齐处理完成!")


if __name__ == '__main__':
    # 创建对齐实例并设置参数
    align = StaticAlign()

    # 设置对齐参数
    align.ref_trace_index = 0  # 模板参考能量迹的索引
    align.pattern_first_sample_pos = 312495  # 模板的样本点起始位置
    align.pattern_sample_number = 2800  # 模板的样本点数量
    align.shift_max = 1000  # 最大偏移量
    align.step_size = 10  # 步进长度
    align.threshold = 0.8  # 相关性阈值
    align.max_workers = 8  # 设置线程数

    # 设置文件路径
    align.traceset_path = "..\\traceset\\tvla_test.trs"
    align.traceset2_dir = "../traceset"

    # 执行对齐操作
    align.align()
