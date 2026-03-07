import copy
import os
import numpy as np
import trsfile
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Tuple, Optional
from trsfile import Header, TracePadding, Trace


def compute_correlation(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
    corr_matrix = np.corrcoef(x_arr, y_arr)
    return float(corr_matrix[0, 1])


def process_single_trace(trace_index, trace,
                         ref_trace_index, pattern_arr,
                         shift_positions, threshold) -> Tuple[Optional[Trace], float, int]:
    # 参考迹线直接保留
    if trace_index == ref_trace_index:
        return trace, 1.0, 0

    best_corr = -1.0
    best_shift = 0

    # 遍历所有候选偏移量，寻找相关性最高的匹配位置
    for shift, start_pos, end_pos in shift_positions:
        trace_section = trace.samples[start_pos:end_pos]
        corr_temp = compute_correlation(pattern_arr, trace_section)
        if corr_temp > best_corr:
            best_shift = shift
            best_corr = corr_temp

    # 相关性达标则保留并对齐后的迹线，否则丢弃
    if best_corr >= threshold:
        if best_shift != 0:
            aligned_samples = np.roll(trace.samples, -best_shift)
            aligned_trace = Trace(
                sample_coding=trace.sample_coding,
                samples=aligned_samples,
                parameters=trace.parameters,
                title=trace.title,
                headers=trace.headers)
        else:
            aligned_trace = trace
        return aligned_trace, best_corr, best_shift
    else:
        return None, best_corr, best_shift


class StaticAlign:
    def __init__(self):
        self.num_processes: int = cpu_count()

        # 对齐参数
        self.ref_trace_index: int = 0  # 参考迹线索引
        self.pattern_first_sample_pos: int = 0  # 模板起始点
        self.pattern_sample_number: int = 0  # 模板长度
        self.shift_max: int = 0  # 最大允许偏移量
        self.step_size: int = 1  # 搜索步长
        self.threshold: float = 0.0  # 相关性保留阈值

        self.traceset = None
        self.traceset_path: str = ""
        self.traceset2 = None

    def open_traceset(self):
        self.traceset = trsfile.open(self.traceset_path, 'r')

        headers = copy.deepcopy(self.traceset.get_headers())
        headers[Header.NUMBER_TRACES] = 0

        base_name = os.path.splitext(os.path.basename(self.traceset_path))[0]
        timestamp = datetime.now().strftime("%H%M%S")
        new_base_name = f"{base_name}+StaticAlign({timestamp})"
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

    def align(self):
        # 参数合法性检查
        if self.threshold <= 0:
            raise ValueError("相关性阈值必须大于0")
        if self.step_size <= 0:
            raise ValueError("步进长度必须大于0")
        if self.pattern_first_sample_pos < 0:
            raise ValueError("模板起始位置不能为负数")
        if self.shift_max < 0:
            raise ValueError("最大偏移量不能为负数")
        if self.step_size > self.shift_max:
            raise ValueError("步进长度不能大于最大偏移量")

        self.open_traceset()
        headers = self.traceset.get_headers()
        sample_number = headers[Header.NUMBER_SAMPLES]
        trace_number = headers[Header.NUMBER_TRACES]
        if not (0 <= self.ref_trace_index < trace_number):
            raise ValueError(f"参考迹线索引 {self.ref_trace_index} 超出范围 [0, {trace_number - 1}]")

        start_time = time.perf_counter()
        print(f"开始静态对齐 {trace_number} 条迹线")

        # 防止模板越界
        if self.pattern_first_sample_pos + self.pattern_sample_number > sample_number:
            self.pattern_sample_number = sample_number - self.pattern_first_sample_pos
            print(f"调整模板长度: {self.pattern_sample_number}")

        # 提取参考模板
        pattern_arr = self.traceset[self.ref_trace_index].samples[
                      self.pattern_first_sample_pos:
                      self.pattern_first_sample_pos + self.pattern_sample_number]

        # 计算实际可搜索的左右偏移范围
        shift_left = min(self.pattern_first_sample_pos, self.shift_max)
        shift_right = min(sample_number - (self.pattern_first_sample_pos + self.pattern_sample_number),
                          self.shift_max)

        # 生成所有候选偏移量
        shift_range = list(range(-shift_left, shift_right + 1, self.step_size))
        if shift_range and shift_range[-1] != shift_right:
            shift_range.append(shift_right)
        print(f"对齐范围: 左移{shift_left}, 右移{shift_right}, 步长{self.step_size}, 次数: {len(shift_range)}")

        # 预计算每个偏移对应的切片索引
        shift_positions = [(shift,
                            self.pattern_first_sample_pos + shift,
                            self.pattern_first_sample_pos + shift + self.pattern_sample_number)
                           for shift in shift_range]

        batch_size = self.num_processes
        batch_number = (trace_number + batch_size - 1) // batch_size

        # 多进程并行加载
        with Pool(processes=self.num_processes) as pool:
            total_included = total_excluded = 0
            for batch_idx in range(batch_number):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, trace_number)
                current_batch_size = end_idx - start_idx
                print(f"批{batch_idx + 1}/{batch_number} ({current_batch_size}条)")

                # 准备任务参数
                batch_tasks = [(start_idx + i, self.traceset[start_idx + i],
                                self.ref_trace_index, pattern_arr,
                                shift_positions, self.threshold) for i in range(current_batch_size)]

                # 并行处理
                batch_results = pool.starmap(process_single_trace, batch_tasks)
                batch_included = batch_excluded = 0
                for i, (trace, corr, shift) in enumerate(batch_results):
                    if trace is not None:
                        self.traceset2.append(trace)
                        batch_included += 1
                        print(f"包含迹线 {start_idx + i}, 偏移: {shift}, 相关系数: {corr:.3f}")
                    else:
                        batch_excluded += 1
                        print(f"排除迹线 {start_idx + i}, 相关系数: {corr:.3f}")
                total_included += batch_included
                total_excluded += batch_excluded

        print(f"最终包含: {total_included} 条迹线")
        print(f"最终排除: {total_excluded} 条迹线")
        elapsed_time = time.perf_counter() - start_time
        print(f"静态对齐完成 用时:{elapsed_time:.3f}秒")
        self.close_traceset()


if __name__ == '__main__':
    static_align = StaticAlign()
    static_align.num_processes = 16

    static_align.ref_trace_index = 0
    static_align.pattern_first_sample_pos = 520000
    static_align.pattern_sample_number = 876165
    static_align.shift_max = 400000
    static_align.step_size = 100
    static_align.threshold = 0.4
    static_align.traceset_path = "D:\\traceset\\usim+LowPass(190339).trs"

    static_align.align()
