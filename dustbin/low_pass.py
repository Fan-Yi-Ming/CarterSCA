import os
import copy

import numpy as np
import trsfile
from datetime import datetime

from trsfile import Header, TracePadding, SampleCoding, Trace


def fast_lowpass_filter(signal: np.ndarray, weight: float) -> np.ndarray:
    """
    使用双向滤波的快速低通滤波器

    参数:
        signal: 输入信号，一维numpy数组
        weight: 滤波权重，值越大平滑效果越强

    返回:
        滤波后的信号，零相位延迟
    """
    signal = np.array(signal, dtype=float, copy=True)

    # 前向滤波
    alpha = 1.0 / (weight + 1.0)
    forward_pass = np.zeros_like(signal)
    forward_pass[0] = signal[0]

    for i in range(1, len(signal)):
        forward_pass[i] = alpha * signal[i] + (1.0 - alpha) * forward_pass[i - 1]

    # 反向滤波（消除相位失真）
    backward_pass = np.zeros_like(signal)
    backward_pass[-1] = forward_pass[-1]

    for i in range(len(signal) - 2, -1, -1):
        backward_pass[i] = alpha * forward_pass[i] + (1.0 - alpha) * backward_pass[i + 1]

    return backward_pass


class LowPassFilter:
    """低通滤波器类，用于对能量迹线集进行滤波处理"""

    def __init__(self):
        # 初始化低通滤波参数
        self.weight = 1.0
        # 迹线集相关变量
        self.traceset = None  # 待滤波的原始能量迹集
        self.traceset_path: str = ""  # 待滤波的能量迹集文件路径
        self.traceset2 = None  # 低通滤波后的能量迹集
        self.traceset2_dir: str = ""  # 低通滤波后的能量迹集保存目录

    def open_traceset(self):
        """打开原始迹线集并创建滤波后的迹线集文件"""
        # 打开原始的 TraceSet 文件（只读模式）
        self.traceset = trsfile.open(self.traceset_path, 'r')

        # 构建新的文件路径，用于保存滤波后的 TraceSet
        base_name = os.path.splitext(os.path.basename(self.traceset_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_base_name = f"{base_name}+LowPass({timestamp})"
        traceset2_path = os.path.join(os.path.dirname(self.traceset_path),
                                      new_base_name + os.path.splitext(self.traceset_path)[1])

        # 复制原始文件头信息，并将迹线数量重置为0
        new_headers = copy.deepcopy(self.traceset.get_headers())
        new_headers[Header.NUMBER_TRACES] = 0
        new_headers[Header.SAMPLE_CODING] = SampleCoding.FLOAT

        # 创建新的 TraceSet 文件（写入模式）
        self.traceset2 = trsfile.trs_open(
            path=traceset2_path,  # 新文件路径
            mode='w',  # 写入模式
            engine='TrsEngine',  # 存储引擎
            headers=new_headers,  # 新的头部信息
            padding_mode=TracePadding.AUTO,  # 填充模式
            live_update=True  # 启用实时更新，便于预览
        )

    def close_traceset(self):
        """关闭当前打开的 TraceSet 文件"""
        self.traceset.close()
        self.traceset2.close()

    def low_pass(self):
        """执行低通滤波处理"""
        # 打开迹线集文件
        self.open_traceset()

        # 获取迹线总数
        number_traces = self.traceset.get_headers()[Header.NUMBER_TRACES]

        # 遍历所有迹线，进行滤波处理
        for index, trace in enumerate(self.traceset):
            # 对当前迹线应用低通滤波
            low_pass = fast_lowpass_filter(trace.samples, self.weight)

            # 创建滤波后的迹线对象
            low_pass_trace = Trace(
                sample_coding=SampleCoding.FLOAT,  # 使用浮点数编码存储滤波后的样本
                samples=low_pass,  # 滤波后的样本数据
                parameters=trace.parameters,  # 保留原始迹线的参数信息
                title=trace.title  # 保留原始迹线的标题
            )

            # 将滤波后的迹线添加到新的迹线集中
            self.traceset2.append(low_pass_trace)

            # 打印进度
            print(f"LowPass finished {index + 1}，total {number_traces}")

        # 关闭迹线集文件
        self.close_traceset()


if __name__ == '__main__':
    # 创建低通滤波器实例
    low_pass_filter = LowPassFilter()

    # 设置滤波参数
    low_pass_filter.weight = 5.0
    low_pass_filter.traceset_path = "../traceset/aes128_en_1000.trs"
    low_pass_filter.traceset2_dir = "../traceset"

    # 执行低通滤波处理
    low_pass_filter.low_pass()
