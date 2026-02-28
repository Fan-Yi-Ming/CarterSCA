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


class GathererSDS804X2:
    """SDS804X示波器控制器"""

    def __init__(self):
        self.instrument = None  # VISA仪器对象
        self.channels_parameters = None  # 通道参数快照
        self.traceset = None  # TRS迹线集

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
        print(f"成功打开示波器 {resource_name}")

    def close_instrument(self):
        """关闭示波器连接"""
        if self.instrument:
            self.de_arm()
            self.instrument.close()
            print("成功关闭示波器")

    def wait_for_trigger_stop(self, timeout: float = 10.0):
        """等待触发停止"""
        start_time = time.perf_counter()
        while True:
            trigger_status = self.instrument.query(":TRIGger:STATus?").strip()
            elapsed_time = time.perf_counter() - start_time

            if trigger_status == "Stop":
                return
            if elapsed_time > timeout:
                raise GathererError(f"采集超时 已等待 {elapsed_time:.3f} 秒")
            time.sleep(0.01)

    def get_channels_parameters(self):
        """获取所有启用通道的波形参数"""
        channels_parameters = {}

        # 参数格式定义：参数名 -> [偏移地址, 数据类型]
        parameter_format = {
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

        type_size = {"i": 4, "f": 4, "h": 2, "d": 8}

        # 水平时基枚举值（秒/格）
        tdiv_enum = [
            200e-12, 500e-12, 1e-9, 2e-9, 5e-9, 10e-9, 20e-9, 50e-9, 100e-9, 200e-9, 500e-9,
            1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6, 500e-6,
            1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3,
            1, 2, 5, 10, 20, 50, 100, 200, 500, 1000
        ]

        # 遍历通道1-4
        for channel_index in range(1, 5):
            channel_enabled = self.instrument.query(f":CHANnel{channel_index}:SWITch?").strip() == "ON"
            channel_visible = self.instrument.query(f":CHANnel{channel_index}:VISible?").strip() == "ON"

            if channel_enabled and channel_visible:
                channel_name = f"C{channel_index}"
                channels_parameters[channel_name] = {}

                # 获取前导码
                self.instrument.write(f":WAVeform:SOURce {channel_name}")
                self.instrument.write(":WAVeform:PREamble?")
                received_bytes = self.instrument.read_raw()
                received_bytes = received_bytes[received_bytes.find(b'#') + 11:]

                # 解析参数
                for param_name, (addr_offset, data_type) in parameter_format.items():
                    byte_count = type_size[data_type]
                    param_bytes = received_bytes[addr_offset:addr_offset + byte_count]
                    param_value = struct.unpack(data_type, param_bytes)[0]
                    channels_parameters[channel_name][param_name] = param_value

                # 转换时基值并应用探头系数
                tdiv_index = channels_parameters[channel_name]["tdiv"]
                channels_parameters[channel_name]["tdiv"] = tdiv_enum[tdiv_index]

                probe_factor = channels_parameters[channel_name]["probe"]
                channels_parameters[channel_name]["vdiv"] *= probe_factor
                channels_parameters[channel_name]["offset"] *= probe_factor

        return channels_parameters

    def snapshot_channels_parameters(self, timeout: float = 10.0):
        """快照当前通道参数"""
        self.instrument.write(":TRIGger:MODE FTRIG")
        self.wait_for_trigger_stop(timeout=timeout)

        self.channels_parameters = self.get_channels_parameters()

        temp = ', '.join(self.channels_parameters.keys())
        print(f"完成通道参数快照 打开且显示的通道 {temp}")

    def get_channel_parameters(self, channel_name: str):
        """获取指定通道的参数"""
        if channel_name in self.channels_parameters:
            return self.channels_parameters[channel_name]
        else:
            raise GathererError(f"通道未找到: {channel_name}")

    def arm(self, delay: float = 0.1):
        """激活示波器（单次触发模式）"""
        trigger_mode = self.instrument.query(":TRIGger:MODE?")
        if trigger_mode != "SINGle":
            self.instrument.write(":TRIGger:MODE SINGle")
        self.instrument.write(":TRIGger:RUN")
        while True:
            trigger_status = self.instrument.query(":TRIGger:STATus?").strip()
            if trigger_status == "Ready":
                break
            time.sleep(0.01)
        time.sleep(delay)

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

        # 验证参数一致性
        current_channels_parameters = self.get_channels_parameters()
        if current_channels_parameters != self.channels_parameters:
            for channel in set(self.channels_parameters.keys()) | set(current_channels_parameters.keys()):
                if channel not in self.channels_parameters:
                    print(f"  {channel}: 仅存在于当前配置")
                elif channel not in current_channels_parameters:
                    print(f"  {channel}: 仅存在于快照配置")
                elif self.channels_parameters[channel] != current_channels_parameters[channel]:
                    print(f"  {channel}: 参数已修改")
                    for param in self.channels_parameters[channel].keys():
                        if param in current_channels_parameters[channel] and self.channels_parameters[channel][param] != \
                                current_channels_parameters[channel][param]:
                            old_val = self.channels_parameters[channel][param]
                            new_val = current_channels_parameters[channel][param]
                            if isinstance(old_val, (int, float)):
                                change = ((new_val - old_val) / old_val * 100) if old_val != 0 else float('inf')
                                print(f"    {param}: {old_val} → {new_val} ({change:+.2f}%)")
                            else:
                                print(f"    {param}: {old_val} → {new_val}")
            raise GathererError("当前通道参数与快照不一致")

        # 逐个通道采集
        for channel_name, channel_parameters in current_channels_parameters.items():
            self.instrument.write(":WAVeform:STARt 0")
            self.instrument.write(f":WAVeform:SOURce {channel_name}")

            # 分批读取配置
            point_number = channel_parameters["point_num"]
            batch_size = int(float(self.instrument.query(":WAVeform:MAXPoint?").strip()))
            batch_number = math.ceil(point_number / batch_size)
            if point_number > batch_size:
                self.instrument.write(f":WAVeform:POINt {batch_size}")

            # 设置数据宽度
            if channel_parameters["adc_bit"] > 8:
                self.instrument.write(":WAVeform:WIDTh WORD")
            else:
                self.instrument.write(":WAVeform:WIDTh BYTE")

            # 分批读取数据
            received_bytes_all = b''
            for batch_idx in range(0, batch_number):
                start_idx = batch_idx * batch_size
                self.instrument.write(f":WAVeform:STARt {start_idx}")
                self.instrument.write(":WAVeform:DATA?")
                received_bytes = self.instrument.read_raw()

                # 解析数据块
                block_start = received_bytes.find(b'#')
                data_digit = int(received_bytes[block_start + 1:block_start + 2])
                data_start = block_start + 2 + data_digit
                data_len = int(received_bytes[block_start + 2:data_start])
                received_bytes_all += received_bytes[data_start:data_start + data_len]

            # 转换数据格式
            if channel_parameters["adc_bit"] > 8:
                convert_data = struct.unpack("%dh" % point_number, received_bytes_all)
                sample_coding = SampleCoding.SHORT
            else:
                convert_data = struct.unpack("%db" % point_number, received_bytes_all)
                sample_coding = SampleCoding.BYTE

            # 创建并保存迹线
            trace = Trace(
                sample_coding=sample_coding,
                samples=convert_data,
                parameters=trace_parameter_map,
                title=channel_name
            )
            self.traceset.append(trace)
