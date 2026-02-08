import copy
import os
from multiprocessing import Pool, cpu_count
from typing import Tuple, Optional
import numpy as np
import trsfile
from trsfile import Header, TracePadding, Trace
from datetime import datetime


def compute_correlation(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
    """
    计算两个一维数组的皮尔逊相关系数
    """
    corr_matrix = np.corrcoef(x_arr, y_arr)
    return float(corr_matrix[0, 1])


def process_single_trace_alignment(args: Tuple) -> Tuple[int, Optional[Trace], float, int]:
    """
    多进程工作函数：对单条能量迹进行模板匹配与对齐
    返回：(原始索引, 对齐后迹线或None, 最佳相关系数, 最佳偏移量)
    """
    trace_index, trace, pattern_arr, shift_positions, threshold, ref_trace_index = args

    # 参考迹线直接保留（相关系数记为1.0，偏移0）
    if trace_index == ref_trace_index:
        return trace_index, trace, 1.0, 0

    best_shift = 0
    best_corr = -1.0

    # 遍历所有候选偏移量，寻找相关性最高的匹配位置
    for shift, start_pos, end_pos in shift_positions:
        trace_segment = trace.samples[start_pos:end_pos]
        corr_temp = compute_correlation(pattern_arr, trace_segment)
        if corr_temp > best_corr:
            best_shift = shift
            best_corr = corr_temp

    # 相关性达标则保留并对齐后的迹线，否则丢弃
    if best_corr >= threshold:
        if best_shift != 0:
            aligned_samples = np.roll(trace.samples, -best_shift)  # 循环移位实现对齐
            aligned_trace = Trace(
                sample_coding=trace.sample_coding,
                samples=aligned_samples,
                parameters=trace.parameters,
                title=trace.title)
        else:
            aligned_trace = trace
        return trace_index, aligned_trace, best_corr, best_shift
    else:
        return trace_index, None, best_corr, best_shift


class StaticAlign:
    """
    基于模板匹配的静态能量迹多进程对齐工具
    支持大批量 .trs 文件的高效对齐与低质量迹线自动过滤
    """

    def __init__(self):
        # 对齐参数
        self.ref_trace_index: int = 0  # 参考迹线索引
        self.pattern_first_sample_pos: int = 0  # 模板起始采样点
        self.pattern_sample_number: int = 0  # 模板长度
        self.shift_max: int = 0  # 最大允许偏移量
        self.step_size: int = 1  # 搜索步长
        self.threshold: float = 0.0  # 相关性保留阈值

        # 并行参数
        self.num_processes: int = cpu_count()  # 默认使用全部CPU核心

        # 文件句柄
        self.traceset = None  # 输入迹线集（只读）
        self.traceset_path: str = ""  # 输入文件路径
        self.traceset2 = None  # 输出迹线集（写入）

    def open_traceset(self):
        """打开原始文件并创建带时间戳的新输出文件"""
        self.traceset = trsfile.open(self.traceset_path, 'r')

        # 生成新文件名：原名 + StaticAlign + 时间戳
        base_name = os.path.splitext(os.path.basename(self.traceset_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_base_name = f"{base_name}+StaticAlign({timestamp})"
        new_path = os.path.join(os.path.dirname(self.traceset_path),
                                new_base_name + os.path.splitext(self.traceset_path)[1])

        # 复制头信息并重置迹线计数为0
        new_headers = copy.deepcopy(self.traceset.get_headers())
        new_headers[Header.NUMBER_TRACES] = 0

        # 创建输出文件（实时更新模式，便于观察进度）
        self.traceset2 = trsfile.trs_open(
            path=new_path,
            mode='w',
            engine='TrsEngine',
            headers=new_headers,
            padding_mode=TracePadding.AUTO,
            live_update=True
        )

    def close_traceset(self):
        """安全关闭输入输出文件，防止句柄泄漏"""
        if self.traceset:
            self.traceset.close()
            self.traceset = None
        if self.traceset2:
            self.traceset2.close()
            self.traceset2 = None

    def align(self):
        """主对齐流程：参数检查 → 模板提取 → 多进程并行对齐 → 结果统计"""
        # ==================== 参数合法性检查 ====================
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
        number_samples = headers[Header.NUMBER_SAMPLES]
        number_traces = headers[Header.NUMBER_TRACES]

        if not (0 <= self.ref_trace_index < number_traces):
            raise ValueError(f"参考迹线索引 {self.ref_trace_index} 超出范围 [0, {number_traces - 1}]")

        # 防止模板越界
        if self.pattern_first_sample_pos + self.pattern_sample_number > number_samples:
            self.pattern_sample_number = number_samples - self.pattern_first_sample_pos
            print(f"调整模板长度: {self.pattern_sample_number}")

        # 提取参考模板
        pattern_arr = self.traceset[self.ref_trace_index].samples[
                      self.pattern_first_sample_pos:
                      self.pattern_first_sample_pos + self.pattern_sample_number]

        # 计算实际可搜索的左右偏移范围
        shift_left = min(self.pattern_first_sample_pos, self.shift_max)
        shift_right = min(number_samples - (self.pattern_first_sample_pos + self.pattern_sample_number),
                          self.shift_max)
        print(f"搜索范围: 左移{shift_left}, 右移{shift_right}")

        # 生成所有候选偏移量
        shift_range = list(range(-shift_left, shift_right + 1, self.step_size))
        if shift_range and shift_range[-1] != shift_right:
            shift_range.append(shift_right)
        print(f"搜索位置数: {len(shift_range)}")

        # 预计算每个偏移对应的切片索引（提升后续多进程效率）
        shift_positions = [(shift,
                            self.pattern_first_sample_pos + shift,
                            self.pattern_first_sample_pos + shift + self.pattern_sample_number)
                           for shift in shift_range]

        # ==================== 多进程并行对齐 ====================
        try:
            print(f"开始处理 {number_traces} 条迹线...")
            print(f"使用 {self.num_processes} 个进程并行处理")

            # 按进程数分批，避免一次性加载过多迹线到内存
            batch_count = (number_traces + self.num_processes - 1) // self.num_processes
            print(f"总共分为 {batch_count} 批进行处理")

            with Pool(processes=self.num_processes) as pool:
                total_included = total_excluded = 0

                for batch_idx in range(batch_count):
                    print(f"\n开始处理第 {batch_idx + 1}/{batch_count} 批...")

                    start_idx = batch_idx * self.num_processes
                    end_idx = min(start_idx + self.num_processes, number_traces)

                    tasks = [(start_idx + i,
                              self.traceset[start_idx + i],
                              pattern_arr,
                              shift_positions,
                              self.threshold,
                              self.ref_trace_index) for i in range(end_idx - start_idx)]

                    results = pool.map(process_single_trace_alignment, tasks)
                    results.sort(key=lambda x: x[0])  # 按原始顺序排序

                    batch_included = batch_excluded = 0
                    for idx, aligned_trace, corr, shift in results:
                        if aligned_trace is not None:
                            self.traceset2.append(aligned_trace)
                            batch_included += 1
                            print(f"包含迹线 {idx}, 偏移: {shift}, 相关系数: {corr:.3f}")
                        else:
                            batch_excluded += 1
                            print(f"排除迹线 {idx}, 相关系数: {corr:.3f}")

                    total_included += batch_included
                    total_excluded += batch_excluded
                    print(f"第 {batch_idx + 1} 批处理完成")
                    print(f"本批包含: {batch_included} 条, 排除: {batch_excluded} 条")
                    print(f"进度: {end_idx}/{number_traces} ({end_idx / number_traces * 100:.1f}%)")

                print(f"\n所有批处理完成！")
                print(f"总共处理: {number_traces} 条迹线")
                print(f"最终包含: {total_included} 条迹线")
                print(f"最终排除: {total_excluded} 条迹线")

        except Exception as e:
            print(f"对齐过程中发生错误: {e}")
            raise
        finally:
            self.close_traceset()


if __name__ == '__main__':
    # ==================== 使用示例 ====================
    static_align = StaticAlign()

    static_align.num_processes = 16

    static_align.ref_trace_index = 0
    static_align.pattern_first_sample_pos = 444237
    static_align.pattern_sample_number = 40894
    static_align.shift_max = 100
    static_align.step_size = 1
    static_align.threshold = 0.8

    static_align.traceset_path = "..\\traceset\\test5001+LowPass(20251130233305)+StaticAlign(20251130235145).trs"

    static_align.align()
