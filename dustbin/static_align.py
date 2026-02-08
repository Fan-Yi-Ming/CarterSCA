import copy
import os
import numpy as np
import trsfile
from trsfile import Header, TracePadding
from datetime import datetime


def compute_correlation(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
    """计算两个数组的皮尔逊相关系数"""
    corr_matrix = np.corrcoef(x_arr, y_arr)
    return float(corr_matrix[0, 1])


class StaticAlign:
    """基于模板匹配的能量迹对齐类"""

    def __init__(self):
        # 对齐参数
        self.ref_trace_index: int = 0  # 参考迹线索引
        self.pattern_first_sample_pos: int = 0  # 模板起始位置
        self.pattern_sample_number: int = 0  # 模板长度
        self.shift_max: int = 0  # 最大偏移量
        self.step_size: int = 1  # 搜索步长
        self.threshold: float = 0.0  # 相关性阈值

        # 文件相关
        self.traceset = None  # 原始迹线集
        self.traceset_path: str = ""  # 输入文件路径
        self.traceset2 = None  # 对齐后迹线集
        self.traceset2_dir: str = ""  # 输出目录

    def open_traceset(self):
        """打开迹线集文件"""
        self.traceset = trsfile.open(self.traceset_path, 'r')

        # 生成输出文件名（添加时间戳）
        base_name = os.path.splitext(os.path.basename(self.traceset_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_base_name = f"{base_name}+StaticAlign({timestamp})"
        new_traceset_path = os.path.join(os.path.dirname(self.traceset_path),
                                         new_base_name + os.path.splitext(self.traceset_path)[1])

        # 复制文件头并重置迹线数量
        new_headers = copy.deepcopy(self.traceset.get_headers())
        new_headers[Header.NUMBER_TRACES] = 0

        # 创建输出文件
        self.traceset2 = trsfile.trs_open(
            path=new_traceset_path,
            mode='w',
            engine='TrsEngine',
            headers=new_headers,
            padding_mode=TracePadding.AUTO,
            live_update=True
        )

    def close_traceset(self):
        """关闭迹线集文件"""
        self.traceset.close()
        self.traceset2.close()

    def align(self):
        """执行对齐操作"""
        # 参数验证
        if self.threshold <= 0:
            raise ValueError("相关性阈值必须大于0")
        if self.step_size <= 0:
            raise ValueError("步进长度必须大于0")
        if self.pattern_first_sample_pos < 0:
            raise ValueError("模板起始位置不能为负数")
        if self.step_size > self.shift_max:
            print(f"警告：步进长度大于最大偏移量，自动调整")
            self.step_size = self.shift_max

        # 打开文件
        self.open_traceset()
        number_of_samples = self.traceset.get_headers()[Header.NUMBER_SAMPLES]
        number_of_traces = self.traceset.get_headers()[Header.NUMBER_TRACES]

        # 调整模板长度防止越界
        if self.pattern_first_sample_pos + self.pattern_sample_number > number_of_samples:
            self.pattern_sample_number = number_of_samples - self.pattern_first_sample_pos
            print(f"调整模板长度: {self.pattern_sample_number}")

        # 提取模板数据
        pattern_arr = self.traceset[self.ref_trace_index].samples[
                      self.pattern_first_sample_pos:self.pattern_first_sample_pos + self.pattern_sample_number]

        # 计算搜索范围
        shift_left = min(self.pattern_first_sample_pos, self.shift_max)
        shift_right = min(number_of_samples - (self.pattern_first_sample_pos + self.pattern_sample_number),
                          self.shift_max)
        print(f"搜索范围: 左移{shift_left}, 右移{shift_right}")

        # 生成偏移位置列表
        shift_range = list(range(-shift_left, shift_right + 1, self.step_size))
        if shift_range[-1] != shift_right:
            shift_range.append(shift_right)
        print(f"搜索位置数: {len(shift_range)}")

        # 预计算所有移位位置
        shift_positions = []
        for shift in shift_range:
            start_pos = self.pattern_first_sample_pos + shift
            end_pos = start_pos + self.pattern_sample_number
            shift_positions.append((shift, start_pos, end_pos))

        # 处理每条迹线
        for index, trace in enumerate(self.traceset):
            # 跳过参考迹线
            if index == self.ref_trace_index:
                self.traceset2.append(trace)
                print(f"包含参考迹线: {index}")
                continue

            # 搜索最佳偏移位置
            best_shift = 0
            best_corr = -1.0
            for shift, start_pos, end_pos in shift_positions:
                corr_temp = compute_correlation(pattern_arr, trace.samples[start_pos:end_pos])
                if corr_temp > best_corr:
                    best_shift = shift
                    best_corr = corr_temp

            # 根据阈值决定是否保留
            if best_corr >= self.threshold:
                # 应用偏移
                if best_shift != 0:
                    trace.samples = np.roll(trace.samples, -best_shift)
                self.traceset2.append(trace)
                print(f"包含迹线 {index}, 偏移: {best_shift}, 相关系数: {best_corr:.3f}")
            else:
                print(f"排除迹线 {index}, 相关系数: {best_corr:.3f}")

        print(f"完成! 处理 {number_of_traces} 条迹线")
        self.close_traceset()


if __name__ == '__main__':
    # 使用示例
    align = StaticAlign()

    # 设置参数
    align.ref_trace_index = 0
    align.pattern_first_sample_pos = 20000
    align.pattern_sample_number = 10000
    align.shift_max = 1000
    align.step_size = 1
    align.threshold = 0.8

    # 设置文件路径
    align.traceset_path = "../traceset/aes128_en_1000.trs"
    align.traceset2_dir = "../traceset"

    # 执行对齐
    align.align()
