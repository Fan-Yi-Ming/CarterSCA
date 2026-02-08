import time
import trsfile.traceparameter as tp
from trsfile import Header, SampleCoding
from trsfile.parametermap import TraceSetParameterMap, TraceParameterDefinitionMap, TraceParameterMap
from trsfile.traceparameter import ParameterType, TraceParameterDefinition

from protocol.c51_aes128.target_c51_aes128 import TargetC51Aes128
from protocol.gatherer_sds804x import GathererSDS804X
from tools.sca import generate_random_hex_string

if __name__ == '__main__':
    # ============================ 目标设备初始化 ============================
    target_c51_aes128_direction = 0  # 0:加密 1:解密
    target_c51_aes128_key_hex = "2B 7E 15 16 28 AE D2 A6 AB F7 15 88 09 CF 4F 3C"  # AES密钥
    target_c51_aes128 = TargetC51Aes128(port="COM3", baudrate=115200, timeout=1.0)
    target_c51_aes128.init(target_c51_aes128_key_hex)  # 初始化AES设备

    # ============================ 示波器配置参数 ============================
    gatherer_sds804x_resource_name = "TCPIP0::169.254.114.206::inst0::INSTR"  # 示波器地址
    gatherer_sds804x_ref_channel_name = "C1"  # 参考通道
    gatherer_sds804x_arm_delay = 0.1  # 触发准备延时(秒)
    gatherer_sds804x_acquisition_timeout = 5.0  # 采集超时(秒)
    gatherer_sds804x_tvla_acquisition_fix_times = 1000  # 固定组采集次数
    gatherer_sds804x_tvla_acquisition_rnd_times = 1000  # 随机组采集次数
    gatherer_sds804x_traceset_path = "..\\..\\traceset\\tvla_test.trs"  # 数据保存路径

    # ============================ 示波器初始化 ============================
    gatherer_sds804x = GathererSDS804X()
    gatherer_sds804x.open_instrument(gatherer_sds804x_resource_name)  # 连接示波器
    gatherer_sds804x.arm(gatherer_sds804x_arm_delay)  # 进入触发准备
    gatherer_sds804x.update_channels_parameters(timeout=gatherer_sds804x_acquisition_timeout)  # 更新通道参数

    # ============================ TraceSet元数据定义 ============================
    traceset_parameter_map = TraceSetParameterMap()  # 参数映射容器
    key_arr = bytes.fromhex(target_c51_aes128_key_hex)
    traceset_parameter_map["KEY"] = tp.ByteArrayParameter(key_arr)  # 保存密钥到文件头

    # 定义Trace参数结构
    trace_parameter_definition_map = TraceParameterDefinitionMap()
    trace_parameter_definition_map["INPUT"] = TraceParameterDefinition(ParameterType.BYTE, 16, 0)  # 16字节输入
    trace_parameter_definition_map["OUTPUT"] = TraceParameterDefinition(ParameterType.BYTE, 16, 16)  # 16字节输出
    trace_parameter_definition_map["TVLA_GROUP"] = TraceParameterDefinition(ParameterType.BYTE, 1, 32)  # 组别标识
    # 0x00=固定组 0x01=随机组

    # 构建TRS文件头
    ref_channel_parameters = gatherer_sds804x.get_channel_parameters(gatherer_sds804x_ref_channel_name)
    headers = {
        Header.TRS_VERSION: 2,  # TRS版本
        Header.TITLE_SPACE: 255,  # 标题空间
        Header.SAMPLE_CODING: SampleCoding.SHORT,  # 采样编码
        Header.LABEL_X: "s",  # X轴标签(时间)
        Header.SCALE_X: ref_channel_parameters["interval"],  # 时间间隔
        Header.LABEL_Y: "v",  # Y轴标签(电压)
        Header.SCALE_Y: ref_channel_parameters["vdiv"] / ref_channel_parameters["code"],  # 电压比例
        Header.TRACE_SET_PARAMETERS: traceset_parameter_map,  # 参数映射(包含密钥)
        Header.TRACE_PARAMETER_DEFINITIONS: trace_parameter_definition_map  # 参数定义
    }
    print(f"TRS文件头设置完成，参考通道: {gatherer_sds804x_ref_channel_name}")

    # 创建TRS文件
    gatherer_sds804x.open_traceset(traceset_path=gatherer_sds804x_traceset_path, headers=headers)

    # ============================ 固定组采集 ============================
    for i in range(gatherer_sds804x_tvla_acquisition_fix_times):
        start_time = time.monotonic()  # 开始计时

        gatherer_sds804x.arm(delay=gatherer_sds804x_arm_delay)  # 示波器准备

        # 固定输入数据
        input_hex = "FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF"  # 16字节固定值
        input_arr = bytes.fromhex(input_hex)

        # 设置Trace参数
        trace_parameter_map = TraceParameterMap()
        trace_parameter_map["INPUT"] = tp.ByteArrayParameter(input_arr)  # 输入参数

        # AES加密处理
        output_arr = target_c51_aes128.process(target_c51_aes128_direction, input_hex)
        trace_parameter_map["OUTPUT"] = tp.ByteArrayParameter(output_arr)  # 输出参数
        trace_parameter_map["TVLA_GROUP"] = tp.ByteArrayParameter(bytes([0x00]))  # 固定组标识

        # 执行采集
        gatherer_sds804x.acquisition(trace_parameter_map=trace_parameter_map,
                                     timeout=gatherer_sds804x_acquisition_timeout)

        # 输出采集信息
        elapsed_time = time.monotonic() - start_time
        print(f"固定组第 {i + 1} 次采集完成，耗时: {elapsed_time:.2f} 秒")

    # ============================ 随机组采集 ============================
    for i in range(gatherer_sds804x_tvla_acquisition_rnd_times):
        start_time = time.monotonic()  # 开始计时

        gatherer_sds804x.arm(delay=gatherer_sds804x_arm_delay)  # 示波器准备

        # 随机输入数据
        input_hex = generate_random_hex_string(16)  # 16字节随机数
        input_arr = bytes.fromhex(input_hex)

        # 设置Trace参数
        trace_parameter_map = TraceParameterMap()
        trace_parameter_map["INPUT"] = tp.ByteArrayParameter(input_arr)  # 输入参数

        # AES加密处理
        output_arr = target_c51_aes128.process(target_c51_aes128_direction, input_hex)
        trace_parameter_map["OUTPUT"] = tp.ByteArrayParameter(output_arr)  # 输出参数
        trace_parameter_map["TVLA_GROUP"] = tp.ByteArrayParameter(bytes([0x01]))  # 随机组标识

        # 执行采集
        gatherer_sds804x.acquisition(trace_parameter_map=trace_parameter_map,
                                     timeout=gatherer_sds804x_acquisition_timeout)

        # 输出采集信息
        elapsed_time = time.monotonic() - start_time
        print(f"随机组第 {i + 1} 次采集完成，耗时: {elapsed_time:.2f} 秒")

    # ============================ 资源清理 ============================
    target_c51_aes128.close()  # 关闭目标设备
    gatherer_sds804x.de_arm()  # 示波器停止采集
    gatherer_sds804x.close_traceset()  # 关闭TRS文件
    gatherer_sds804x.close_instrument()  # 断开示波器连接
