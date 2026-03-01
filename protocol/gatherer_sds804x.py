import math
import struct
import time
import pyvisa
from pyvisa import constants
from trsfile import trs_open
from trsfile import Trace, SampleCoding, TracePadding
from trsfile.parametermap import TraceParameterMap


class GathererError(Exception):
    """采集过程错误"""
    pass


class GathererSDS804X:
    """SDS804X示波器控制器"""

    def __init__(self):
        self.instrument = None  # VISA仪器对象
        self.channels_parameters = None  # 通道参数快照
        self.traceset = None  # TRS迹线文件对象

    def open_instrument(self, resource_name: str = ""):
        """打开示波器连接"""
        rm = pyvisa.ResourceManager()
        self.instrument = rm.open_resource(
            resource_name=resource_name,
            access_mode=constants.AccessModes.exclusive_lock,
            open_timeout=5000,
            timeout=10000,
            chunk_size=1 * 1024 * 1024,
        )
        self.de_arm()
        print(f"示波器已连接: {resource_name}")

    def close_instrument(self):
        """关闭示波器连接"""
        if self.instrument:
            self.de_arm()
            self.instrument.close()
            print(f"示波器已断开")

    def wait_for_trigger_stop(self, timeout: float = 10.0):
        """等待触发停止"""
        start_time = time.perf_counter()
        while True:
            trigger_status = self.instrument.query(":TRIGger:STATus?").strip()
            elapsed_time = time.perf_counter() - start_time

            if trigger_status == "Stop":
                break
            if elapsed_time > timeout:
                raise GathererError(f"触发超时")
            time.sleep(0.01)

    def get_enabled_channels(self):
        """获取当前启用的通道列表"""
        enabled_channels = []
        for channel_index in range(1, 5):
            channel_enabled = self.instrument.query(f":CHANnel{channel_index}:SWITch?").strip() == "ON"
            channel_visible = self.instrument.query(f":CHANnel{channel_index}:VISible?").strip() == "ON"
            if channel_enabled and channel_visible:
                channel = f"C{channel_index}"
                enabled_channels.append(channel)
        return enabled_channels

    def read_channel_preamble(self, channel: str):
        """读取指定通道的前导码"""
        self.instrument.write(f":WAVeform:SOURce {channel}")
        self.instrument.write(":WAVeform:PREamble?")
        received_bytes = self.instrument.read_raw()
        preamble_bytes = received_bytes[received_bytes.find(b'#') + 11:]
        return preamble_bytes

    def parse_channel_preamble(self, preamble_bytes):
        """解析通道前导数据，提取通道参数并进行必要的转换"""
        # 参数格式定义：参数名 -> [偏移地址, 数据类型]
        param_format = {
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

        # 数据类型字节数映射
        type_size = {"i": 4, "f": 4, "h": 2, "d": 8}

        # 水平时基枚举值（秒/格）
        tdiv_enum = [
            200e-12, 500e-12, 1e-9, 2e-9, 5e-9, 10e-9, 20e-9, 50e-9, 100e-9, 200e-9, 500e-9,
            1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6, 500e-6,
            1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3,
            1, 2, 5, 10, 20, 50, 100, 200, 500, 1000
        ]

        # 解析原始参数
        channel_parameters = {}
        for param_name, (addr_offset, data_type) in param_format.items():
            byte_count = type_size[data_type]
            param_bytes = preamble_bytes[addr_offset:addr_offset + byte_count]
            param_value = struct.unpack(data_type, param_bytes)[0]
            channel_parameters[param_name] = param_value

        # 转换时基值
        tdiv_index = channel_parameters["tdiv"]
        channel_parameters["tdiv"] = tdiv_enum[tdiv_index]

        # 应用探头系数
        probe_factor = channel_parameters["probe"]
        channel_parameters["vdiv"] *= probe_factor
        channel_parameters["offset"] *= probe_factor

        return channel_parameters

    def snapshot_channels_parameters(self, timeout: float = 10.0):
        """快照当前通道参数"""
        self.instrument.write(":WAVeform:STARt 0")
        self.instrument.write(":TRIGger:MODE FTRIG")
        self.wait_for_trigger_stop(timeout=timeout)

        self.channels_parameters = {}
        enabled_channels = self.get_enabled_channels()
        for channel in enabled_channels:
            preamble_bytes = self.read_channel_preamble(channel)
            channel_parameters = self.parse_channel_preamble(preamble_bytes)
            self.channels_parameters[channel] = channel_parameters

        temp = ', '.join(self.channels_parameters.keys())
        print(f"通道参数快照已保存: {temp}")

    def get_channel_parameters(self, channel: str):
        """获取指定通道的参数"""
        if channel in self.channels_parameters:
            return self.channels_parameters[channel]
        else:
            raise GathererError(f"通道未找到 {channel}")

    def verify_channel_parameters(self, channel: str):
        """验证指定通道的参数是否与快照一致"""
        if channel not in self.channels_parameters:
            raise GathererError(f"通道 {channel} 不在快照中")

        preamble_bytes = self.read_channel_preamble(channel)
        current_params = self.parse_channel_preamble(preamble_bytes)
        snapshot_params = self.channels_parameters[channel]

        if current_params != snapshot_params:
            for param in snapshot_params.keys():
                if param in current_params and snapshot_params[param] != current_params[param]:
                    old_val = snapshot_params[param]
                    new_val = current_params[param]
                    if isinstance(old_val, (int, float)):
                        change = ((new_val - old_val) / old_val * 100) if old_val != 0 else float('inf')
                        print(f"{param}: {old_val} → {new_val} ({change:+.2f}%)")
                    else:
                        print(f"{param}: {old_val} → {new_val}")
            raise GathererError(f"通道 {channel} 参数已修改")

    def arm(self, timeout: float = 1.0):
        """激活示波器（单次触发模式）"""
        self.instrument.write(":TRIGger:RUN")
        trigger_mode = self.instrument.query(":TRIGger:MODE?").strip()
        if trigger_mode != "SINGle":
            self.instrument.write(":TRIGger:MODE SINGle")

        start_time = time.perf_counter()
        while True:
            trigger_status = self.instrument.query(":TRIGger:STATus?").strip()
            elapsed_time = time.perf_counter() - start_time

            if trigger_status == "Ready":
                break
            if elapsed_time > timeout:
                raise GathererError(f"激活示波器超时")
            time.sleep(0.01)

    def de_arm(self):
        """取消激活示波器"""
        self.instrument.write(":TRIGger:STOP")

    def open_traceset(self, traceset_path: str = "", headers: dict = None):
        """打开TRS迹线文件"""
        self.traceset = trs_open(
            path=traceset_path,
            mode='w',
            engine='TrsEngine',
            headers=headers,
            padding_mode=TracePadding.AUTO,
            live_update=True
        )

    def close_traceset(self):
        """关闭TRS迹线文件"""
        if self.traceset:
            self.traceset.close()

    def acquisition(self, trace_parameter_map: TraceParameterMap = None, timeout: float = 10.0):
        """采集波形数据并保存为迹线"""
        self.wait_for_trigger_stop(timeout=timeout)

        enabled_channels = self.get_enabled_channels()
        for channel in enabled_channels:
            self.instrument.write(":WAVeform:STARt 0")
            self.verify_channel_parameters(channel)

            # 分批读取配置
            point_number = self.channels_parameters[channel]["point_num"]
            batch_size = int(float(self.instrument.query(":WAVeform:MAXPoint?").strip()))
            batch_number = math.ceil(point_number / batch_size)
            if point_number > batch_size:
                self.instrument.write(f":WAVeform:POINt {batch_size}")

            # 设置数据宽度
            if self.channels_parameters[channel]["adc_bit"] > 8:
                self.instrument.write(":WAVeform:BYTeorder LSB")
                self.instrument.write(":WAVeform:WIDTh WORD")
            else:
                self.instrument.write(":WAVeform:WIDTh BYTE")

            # 分批读取数据
            received_bytes_all = b''
            for batch_idx in range(0, batch_number):
                start_idx = batch_idx * batch_size
                current_batch_size = min(batch_size, point_number - batch_idx * batch_size)
                self.instrument.write(f":WAVeform:STARt {start_idx}")
                self.instrument.write(f":WAVeform:POINt {current_batch_size}")
                self.instrument.write(":WAVeform:DATA?")
                received_bytes = self.instrument.read_raw()

                # 解析数据块
                block_start = received_bytes.find(b'#')
                data_digit = int(received_bytes[block_start + 1:block_start + 2])
                data_start = block_start + 2 + data_digit
                data_len = int(received_bytes[block_start + 2:data_start])
                received_bytes_all += received_bytes[data_start:data_start + data_len]

            # 转换数据格式
            if self.channels_parameters[channel]["adc_bit"] > 8:
                convert_data = struct.unpack("<%dh" % point_number, received_bytes_all)
                sample_coding = SampleCoding.SHORT
            else:
                convert_data = struct.unpack("%db" % point_number, received_bytes_all)
                sample_coding = SampleCoding.BYTE

            # 创建并保存迹线
            trace = Trace(
                sample_coding=sample_coding,
                samples=convert_data,
                parameters=trace_parameter_map,
                title=channel
            )
            self.traceset.append(trace)
