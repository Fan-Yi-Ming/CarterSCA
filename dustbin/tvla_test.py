import os
from datetime import datetime

import numpy as np
import trsfile
from trsfile import SampleCoding, Trace, Header, TracePadding
from trsfile.parametermap import TraceParameterMap, TraceSetParameterMap, TraceParameterDefinitionMap


class TVLATest:
    """
    TVLA (Test Vector Leakage Assessment) 测试类

    该类实现了针对加密设备的TVLA测试，用于检测侧信道泄漏
    基于固定组和随机组的t-test统计分析方法

    属性:
        trace_number: 能量迹总数量
        sample_first_pos: 分析使用的样本点起始位置
        sample_number: 分析使用的样本点数量
        traceset: 原始能量迹数据集
        traceset_path: 原始能量迹文件路径
        traceset2: TVLA结果数据集
    """

    def __init__(self):
        """初始化TVLA测试参数和数据结构"""

        # 能量迹参数
        self.trace_number = 0  # 能量迹总数
        self.sample_first_pos = 0  # 样本点起始位置
        self.sample_number = 0  # 样本点数量

        # TVLA参数
        self.t_value_threshold = 4.5  # t-value判定阈值
        self.fixed_indices = None  # 固定组能量迹索引数组
        self.random_indices = None  # 随机组能量迹索引数组

        # 能量迹文件相关
        self.traceset = None  # 原始能量迹数据集
        self.traceset_path = ""  # 原始能量迹文件路径
        self.traceset2 = None  # TVLA结果数据集

        # 分析数据数组
        self.sample_arr_2d = None  # 样本点数据: [trace_number, sample_number]
        self.t_values = None  # t-value结果数组: [sample_number]

    def init_process(self):
        """
        初始化TVLA测试过程

        执行步骤:
        1. 打开原始能量迹文件
        2. 创建TVLA结果文件
        3. 加载能量迹数据并分组
        4. 初始化数据数组
        """
        self.open_traceset()  # 打开原始能量迹文件
        self.open_traceset2()  # 创建TVLA结果文件

        # 初始化样本点数组: [trace_number条能量迹, sample_number个样本点]
        self.sample_arr_2d = np.zeros((self.trace_number, self.sample_number), dtype=np.float32)

        # 初始化t-value数组: [sample_number个样本点]
        self.t_values = np.zeros(self.sample_number, dtype=np.float32)

        # 重置分组数组
        self.fixed_indices = np.array([], dtype=int)
        self.random_indices = np.array([], dtype=int)

    def analyze(self):
        """
        执行完整的TVLA分析流程

        流程:
        1. 初始化分析环境
        2. 加载样本数据
        3. 计算t-value
        4. 检测泄漏点
        5. 完成分析并保存结果
        """
        self.init_process()  # 初始化分析环境

        # 加载样本数据到内存
        self.load_sample_data()

        # 计算t-value
        self.calculate_t_values()

        self.finish_process()  # 完成分析流程

    def finish_process(self):
        """
        完成分析过程

        执行步骤:
        1. 生成测试报告
        2. 保存TVLA结果
        3. 关闭能量迹文件
        """
        self.report()  # 生成测试报告
        self.close_traceset()  # 关闭能量迹文件

    def load_sample_data(self):
        """
        加载样本点数据到内存

        从原始能量迹中提取指定范围的样本点数据
        """
        print("开始加载样本点数据...")
        start_time = datetime.now()

        # 使用列表临时存储索引，最后一次性转换为numpy数组（性能更好）
        fixed_indices_list = []
        random_indices_list = []

        # 遍历所有能量迹，提取样本点数据
        for i in range(self.trace_number):
            # 提取指定范围的样本点数据
            trace = self.traceset[i]
            trace_data = trace[self.sample_first_pos:self.sample_first_pos + self.sample_number]
            self.sample_arr_2d[i] = np.copy(trace_data)

            group_value = trace.parameters["TVLA_GROUP"].value[0]
            if group_value == 0:
                fixed_indices_list.append(i)
            elif group_value == 1:
                random_indices_list.append(i)
            else:
                # 处理意外的分组值
                print(f"警告: 能量迹 {i} 有未知的分组值: {group_value}")

        # 一次性转换为numpy数组
        self.fixed_indices = np.array(fixed_indices_list, dtype=int)
        self.random_indices = np.array(random_indices_list, dtype=int)

        print(f"分组统计 - 固定组: {len(self.fixed_indices)}, 随机组: {len(self.random_indices)}")

        end_time = datetime.now()
        elapsed_time = end_time - start_time
        print(f"样本点数据加载完成，耗时: {elapsed_time.total_seconds():.2f} 秒")

    def calculate_t_values(self):
        """
        计算t-value统计量

        使用Welch's t-test公式计算固定组和随机组在每个样本点上的t-value:
        t = (μ1 - μ2) / √(s1²/n1 + s2²/n2)

        其中:
        - μ1, μ2: 两组样本的均值
        - s1², s2²: 两组样本的方差
        - n1, n2: 两组样本的数量
        """
        print("开始计算t-value...")
        start_time = datetime.now()

        fixed_samples = self.sample_arr_2d[self.fixed_indices]  # 形状: [n_fixed, sample_number]
        random_samples = self.sample_arr_2d[self.random_indices]  # 形状: [n_random, sample_number]

        n_fixed = len(self.fixed_indices)
        n_random = len(self.random_indices)

        print(f"计算统计量 - 固定组: {n_fixed}条, 随机组: {n_random}条")

        # 计算两组的均值
        mean_fixed = np.mean(fixed_samples, axis=0)  # 形状: [sample_number]
        mean_random = np.mean(random_samples, axis=0)  # 形状: [sample_number]

        # 计算两组的方差
        var_fixed = np.var(fixed_samples, axis=0, ddof=1)  # 无偏估计 (除以n-1)
        var_random = np.var(random_samples, axis=0, ddof=1)  # 无偏估计 (除以n-1)

        # 计算均值差
        mean_diff = mean_fixed - mean_random

        # 计算标准误
        std_error = np.sqrt(var_fixed / n_fixed + var_random / n_random)

        # 避免除以零，将零值替换为一个小值
        std_error[std_error == 0] = 1e-10

        # 计算t-value
        self.t_values = mean_diff / std_error

        t_trace = Trace(
            sample_coding=SampleCoding.FLOAT,  # 使用浮点数编码存储滤波后的样本
            samples=self.t_values,  # 滤波后的样本数据
            parameters=TraceParameterMap(),  # 保留原始迹线的参数信息
            title="T_Test"  # 保留原始迹线的标题
        )
        # 将滤波后的迹线添加到新的迹线集中
        self.traceset2.append(t_trace)

        end_time = datetime.now()
        elapsed_time = end_time - start_time
        print(f"t-value计算完成，耗时: {elapsed_time.total_seconds():.2f} 秒")

    def report(self):
        pass

    def open_traceset(self):
        """
        打开原始能量迹TRS文件并初始化参数

        自动调整样本点范围以防止越界访问
        """
        # 打开TRS格式的能量迹文件
        self.traceset = trsfile.open(self.traceset_path, 'r')

        # 获取能量迹数量
        self.trace_number = self.traceset.get_headers()[Header.NUMBER_TRACES]

        # 获取每条能量迹的样本点数量
        sample_number_per_trace = self.traceset.get_headers()[Header.NUMBER_SAMPLES]

        # 自动调整样本点起始位置
        if self.sample_first_pos < 0:
            self.sample_first_pos = 0
            print(f"自动调整样本点起始位置为: 0")

        # 自动调整样本点数量防止越界
        if self.sample_first_pos + self.sample_number > sample_number_per_trace:
            self.sample_number = sample_number_per_trace - self.sample_first_pos
            print(f"自动调整样本点数量为: {self.sample_number}")

    def open_traceset2(self):
        """
        创建并打开用于存储TVLA结果的TRS文件

        新文件基于原始文件名添加时间戳和测试类型标识
        """
        # ============================ TraceSet参数初始化 ============================
        # 初始化TraceSet参数映射
        traceset_parameter_map = TraceSetParameterMap()

        # 定义TraceSet中每条Trace的参数定义
        trace_parameter_definition_map = TraceParameterDefinitionMap()

        # 定义TRS文件头信息
        headers = {
            Header.TRS_VERSION: 2,
            Header.TITLE_SPACE: 255,
            Header.SAMPLE_CODING: SampleCoding.FLOAT,  # 数据编码格式
            Header.LABEL_X: "Sample Point",  # X轴标签
            Header.SCALE_X: 1.0,  # X轴缩放比例
            Header.LABEL_Y: "t-value",  # Y轴标签
            Header.SCALE_Y: 1.0,  # Y轴缩放比例
            Header.TRACE_SET_PARAMETERS: traceset_parameter_map,  # 跟踪集参数
            Header.TRACE_PARAMETER_DEFINITIONS: trace_parameter_definition_map  # Trace的参数定义
        }

        # 构建新文件名
        base_name = os.path.splitext(os.path.basename(self.traceset_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        new_base_name = f"{base_name}+TVLA({timestamp})"
        traceset2_path = os.path.join(
            os.path.dirname(self.traceset_path),
            new_base_name + os.path.splitext(self.traceset_path)[1]
        )

        # 创建新的TRS文件用于存储TVLA结果
        self.traceset2 = trsfile.trs_open(
            path=traceset2_path,  # 文件路径
            mode='w',  # 写入模式
            engine='TrsEngine',  # 存储引擎
            headers=headers,  # 文件头信息
            padding_mode=TracePadding.AUTO,  # 自动填充
            live_update=True  # 实时更新
        )

    def close_traceset(self):
        """
        关闭所有打开的能量迹文件

        包括原始能量迹文件和TVLA结果文件
        """
        if self.traceset:
            self.traceset.close()
        if self.traceset2:
            self.traceset2.close()


if __name__ == '__main__':
    """
    主函数: TVLA测试示例

    执行完整的TVLA测试流程
    """
    # 创建TVLA测试实例
    tvla_test = TVLATest()

    # 设置能量迹文件路径
    tvla_test.traceset_path = "../traceset/aes128_tvla+LowPass(20251123202603)+StaticAlign(20251123202803).trs"

    # TVLA测试配置
    tvla_test.sample_first_pos = 0  # 样本点起始位置
    tvla_test.sample_number = 100000  # 样本点数量
    tvla_test.t_value_threshold = 4.5  # t-value阈值

    print("开始TVLA测试...")

    # 执行TVLA测试
    tvla_test.analyze()

    print(f"TVLA测试完成")
