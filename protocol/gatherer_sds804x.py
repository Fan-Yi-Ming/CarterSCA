import gc
import math
import struct
import time
import pyvisa
from trsfile import trs_open
from trsfile import Trace, SampleCoding, TracePadding
from trsfile.parametermap import TraceParameterMap


class GathererError(Exception):
    """
    自定义异常类，用于处理采集过程中出现的错误。
    """
    pass


class GathererTimeout(Exception):
    """
    自定义异常类，用于处理采集过程中的超时错误。
    """
    pass


class GathererSDS804X:
    def __init__(self):
        """
        初始化 SDS804X 示波器采集器。

        创建示波器连接对象、通道参数存储和轨迹数据集。
        """
        self.instrument = None  # PyVISA仪器连接对象
        self.channels_parameters = None  # 存储各通道波形参数
        self.traceset = None  # 轨迹数据集对象

    def open_instrument(self, resource_name: str = ""):
        """
        建立与示波器的连接。

        Args:
            resource_name: VISA资源地址字符串，例如 'TCPIP::192.168.1.100::INSTR'
        """
        self.instrument = pyvisa.ResourceManager().open_resource(resource_name)
        self.instrument.chunk_size = 20 * 1024 * 1024  # 设置数据传输块大小为20MB
        self.de_arm()  # 初始化为停止触发状态
        print(f"成功打开示波器: {resource_name}")

    def close_instrument(self):
        """
        安全关闭示波器连接。

        停止采集并释放VISA资源。
        """
        if self.instrument:
            resource_name = self.instrument.resource_name
            self.de_arm()  # 停止触发
            self.instrument.close()  # 关闭连接
            print(f"成功关闭示波器: {resource_name}")

    def wait_acquisition_ready(self, timeout: float = 10.0):
        """
        等待单次触发采集完成且数据就绪。

        在单次触发模式下，持续查询触发状态直到采集完成（进入Stop状态）。

        Args:
            timeout: 最大等待时间（秒）

        Raises:
            GathererTimeout: 超过指定时间仍未完成采集
        """
        start_time = time.perf_counter()
        while True:
            trigger_status = self.instrument.query(":TRIGger:STATus?").strip()
            elapsed_time = time.perf_counter() - start_time

            # 单次触发完成标志：示波器进入停止状态
            if trigger_status == "Stop":
                break
            elif elapsed_time > timeout:
                raise GathererTimeout(
                    f"等待单次采集完成超时：已等待 {elapsed_time:.3f} 秒，超时时间为 {timeout} 秒")
            time.sleep(0.1)  # 查询间隔100ms

    def update_channels_parameters(self, timeout: float = 10.0):
        """
        更新所有启用且可见通道的波形参数。

        通过强制触发方式获取最新的通道配置参数，包括垂直/水平设置、采样参数等。

        Args:
            timeout: 采集完成等待超时时间（秒）

        Raises:
            GathererTimeout: 更新过程中超时
        """
        # 执行强制触发
        self.instrument.write(":TRIGger:MODE FTRIG")  # 设置为强制触发模式

        # 等待采集完成
        self.wait_acquisition_ready(timeout=timeout)

        # 初始化通道参数存储
        self.channels_parameters = {}

        # 波形前导码参数地址映射表
        # 格式：参数名: [地址偏移, 数据类型]
        param_addr_type = {
            "data_bytes": [0x3c, "i"],  # 数据字节数
            "point_num": [0x74, 'i'],  # 波形点数（模拟通道专用）
            "fp": [0x84, 'i'],  # 起始点位置
            "sp": [0x88, 'i'],  # 数据点间隔
            "vdiv": [0x9c, 'f'],  # 垂直档位（原始值）
            "offset": [0xa0, 'f'],  # 垂直偏移（原始值）
            "code": [0xa4, 'f'],  # 码字值/div
            "adc_bit": [0xac, 'h'],  # ADC位数
            "interval": [0xb0, 'f'],  # 采样间隔
            "delay": [0xb4, 'd'],  # 水平延迟（秒）
            "tdiv": [0x144, 'h'],  # 水平时基（枚举索引）
            "probe": [0x148, 'f']  # 探头系数
        }

        # 数据类型对应的字节长度
        data_byte = {"i": 4, "f": 4, "h": 2, "d": 8}

        # 水平时基枚举值表（秒/格）
        tdiv_enum = [
            200e-12, 500e-12, 1e-9, 2e-9, 5e-9, 10e-9, 20e-9, 50e-9, 100e-9, 200e-9, 500e-9,
            1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6, 500e-6,
            1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3,
            1, 2, 5, 10, 20, 50, 100, 200, 500, 1000
        ]

        # 遍历所有通道（1-4）
        for channel_num in range(1, 5):
            # 检查通道是否启用且可见
            channel_enabled = self.instrument.query(f":CHANnel{channel_num}:SWITch?").strip() == "ON"
            channel_visible = self.instrument.query(f":CHANnel{channel_num}:VISible?").strip() == "ON"

            if channel_enabled and channel_visible:
                channel_name = f"C{channel_num}"
                self.channels_parameters[channel_name] = {}

                # 设置波形源并获取前导码
                self.instrument.write(f":WAVeform:SOURce {channel_name}")
                self.instrument.write(":WAVeform:PREamble?")

                # 读取原始数据并跳过头部
                recv_all = self.instrument.read_raw()
                recv = recv_all[recv_all.find(b'#') + 11:]

                # 解析各个参数
                for param_name, (addr_offset, data_type) in param_addr_type.items():
                    byte_count = data_byte[data_type]
                    param_bytes = recv[addr_offset:addr_offset + byte_count]
                    param_value = struct.unpack(data_type, param_bytes)[0]
                    self.channels_parameters[channel_name][param_name] = param_value

                # 后处理：转换枚举值和应用探头系数
                tdiv_index = self.channels_parameters[channel_name]["tdiv"]
                self.channels_parameters[channel_name]["tdiv"] = tdiv_enum[tdiv_index]

                probe_factor = self.channels_parameters[channel_name]["probe"]
                self.channels_parameters[channel_name]["vdiv"] *= probe_factor
                self.channels_parameters[channel_name]["offset"] *= probe_factor

        # 输出更新结果
        updated_channels = ', '.join(self.channels_parameters.keys())
        print(f"成功更新通道参数，打开且显示的通道: {updated_channels}")

    def get_channel_parameters(self, channel_name: str):
        """
        获取指定通道的参数配置。

        Args:
            channel_name: 通道标识符，如 "C1", "C2"

        Returns:
            通道参数字典

        Raises:
            GathererError: 指定通道不存在或参数未初始化
        """
        if channel_name in self.channels_parameters:
            return self.channels_parameters[channel_name]
        else:
            raise GathererError(f"{self.instrument.resource_name} 通道未找到: {channel_name}")

    def arm(self, delay: float = 0.1):
        """
        配置示波器为单次触发就绪状态。

        Args:
            delay: 准备状态稳定延迟时间（秒）
        """
        self.instrument.write(":TRIGger:MODE SINGle")  # 设置为单次触发模式
        self.instrument.write(":TRIGger:RUN")  # 启动触发准备
        time.sleep(delay)  # 等待仪器稳定

    def de_arm(self):
        """
        停止示波器触发采集。

        将示波器置于安全停止状态。
        """
        self.instrument.write(":TRIGger:STOP")  # 停止触发

    def open_traceset(self, traceset_path: str = "", headers: dict = None):
        """
        打开一个 TraceSet 文件，准备存储采集的数据。

        Args:
            traceset_path: 文件路径
            headers: TraceSet 的头部信息
        """
        self.traceset = trs_open(path=traceset_path,  # 路径
                                 mode='w',  # 模式: r, w, x, a (默认为 x)
                                 engine='TrsEngine',  # 可选: 如何存储 TraceSet (默认为 TrsEngine)
                                 headers=headers,  # 头部信息
                                 padding_mode=TracePadding.AUTO,  # 填充模式
                                 live_update=True)  # 可选: 更新 TRS 文件以进行实时预览 (小性能损失)
        print(f"成功创建 TraceSet 文件: {traceset_path}")

    def close_traceset(self):
        """
        关闭当前打开的 TraceSet 文件。
        """
        self.traceset.close()

    def acquisition(self, trace_parameter_map: TraceParameterMap = None, timeout: float = 10.0):
        """
        配置示波器读取波形数据。

        Args:
            trace_parameter_map: 包含每个通道波形参数的映射
            timeout: 超时等待时间（秒）

        Raises:
            GathererTimeout: 如果超时
            GathererError: 如果读取数据时发生错误
        """
        # 等待触发完成
        self.wait_acquisition_ready(timeout=timeout)

        # 遍历通道并获取波形数据
        for channel_name, channel_parameters in self.channels_parameters.items():
            self.instrument.write(":WAVeform:STARt 0")  # 设置波形起始位置为 0
            self.instrument.write(f":WAVeform:SOURce {channel_name}")  # 设置波形数据来源为当前通道
            # 获取实际采样点数，确保与初始采样点数一致
            points = channel_parameters["point_num"]
            if int(float(self.instrument.query(f":ACQuire:POINts?").strip())) != points:
                raise GathererError(f"{self.instrument.resource_name} 实际采样点数与初始采样点数不同")
            # 获取一次可获取波形数据的最大点数
            one_piece_num = float(self.instrument.query(":WAVeform:MAXPoint?").strip())
            read_times = math.ceil(points / one_piece_num)
            # 如果点数超过最大值，设置每次读取的点数
            if points > one_piece_num:
                self.instrument.write(f":WAVeform:POINt {one_piece_num}")
            # 根据ADC位数设置波形数据宽度
            if channel_parameters["adc_bit"] > 8:
                self.instrument.write(":WAVeform:WIDTh WORD")
            else:
                self.instrument.write(":WAVeform:WIDTh BYTE")
            # 初始化接收字节数据
            recv_byte = b''
            # 分段读取波形数据
            for i in range(0, read_times):
                start = i * one_piece_num
                # 设置每个切片的起始点
                self.instrument.write(f":WAVeform:STARt {start}")  # 设置每个切片的起始位置
                # 获取每个切片的波形数据
                self.instrument.write(":WAVeform:DATA?")  # 读取波形数据
                recv_rtn = self.instrument.read_raw()  # 读取原始数据
                # 解析返回的原始数据块
                block_start = recv_rtn.find(b'#')
                data_digit = int(recv_rtn[block_start + 1:block_start + 2])
                data_start = block_start + 2 + data_digit
                data_len = int(recv_rtn[block_start + 2:data_start])
                recv_byte += recv_rtn[data_start:data_start + data_len]
            # 根据ADC位数转换接收到的字节数据
            if channel_parameters["adc_bit"] > 8:
                convert_data = struct.unpack("%dh" % points, recv_byte)  # 转换为有符号短整型
            else:
                convert_data = struct.unpack("%db" % points, recv_byte)  # 转换为有符号字节型
            # 清理临时数据并进行垃圾回收
            del recv_byte  # 清除临时数据，释放内存
            gc.collect()  # 强制垃圾回收
            # 创建 Trace 对象并添加到 TraceSet
            trace = Trace(sample_coding=SampleCoding.SHORT,  # 12bit示波器用有符号短整型
                          samples=convert_data,
                          parameters=trace_parameter_map,
                          title=channel_name)
            self.traceset.append(trace)  # 将新采集的 Trace 添加到 TraceSet 中
