from trsfile import Header, SampleCoding
from trsfile.parametermap import TraceSetParameterMap, TraceParameterDefinitionMap, TraceParameterMap
from protocol.c51_aes128.target_c51_aes128 import TargetC51Aes128
from protocol.gatherer_sds804x import GathererSDS804X
from tools.sca import generate_random_hex_string

if __name__ == '__main__':
    # 目标设备初始化
    target_c51_aes128_key_hex = "2B 7E 15 16 28 AE D2 A6 AB F7 15 88 09 CF 4F 3C"
    target_c51_aes128 = TargetC51Aes128(port="COM3", baudrate=115200, timeout=1.0)

    # 示波器配置参数
    gatherer_sds804x_resource_name = "TCPIP0::169.254.114.206::inst0::INSTR"
    gatherer_sds804x_ref_channel_name = "C1"
    gatherer_sds804x_arm_timeout = 1.0
    gatherer_sds804x_acquisition_timeout = 5.0
    gatherer_sds804x_acquisition_times = 1000
    gatherer_sds804x_traceset_path = "D:\\traceset\\aes128_show.trs"

    # 示波器初始化
    gatherer_sds804x = GathererSDS804X()
    gatherer_sds804x.open_instrument(gatherer_sds804x_resource_name)
    gatherer_sds804x.arm(gatherer_sds804x_arm_timeout)
    gatherer_sds804x.snapshot_channels_parameters(gatherer_sds804x_acquisition_timeout)

    # 写入TraceSet参数
    traceset_parameter_map = TraceSetParameterMap()  # 创建参数映射

    # 定义Trace元数据
    trace_parameter_definition_map = TraceParameterDefinitionMap()

    # 构建TRS文件头
    ref_channel_parameters = gatherer_sds804x.get_channel_parameters(gatherer_sds804x_ref_channel_name)
    headers = {
        Header.TRS_VERSION: 2,
        Header.TITLE_SPACE: 255,
        Header.SAMPLE_CODING: SampleCoding.SHORT,
        Header.LABEL_X: "s",
        Header.SCALE_X: ref_channel_parameters["interval"],
        Header.LABEL_Y: "v",
        Header.SCALE_Y: ref_channel_parameters["vdiv"] / ref_channel_parameters["code"],
        Header.TRACE_SET_PARAMETERS: traceset_parameter_map,
        Header.TRACE_PARAMETER_DEFINITIONS: trace_parameter_definition_map
    }
    print(f"TRS文件头设置完成，参考通道: {gatherer_sds804x_ref_channel_name}")

    # 创建TRS文件
    gatherer_sds804x.open_traceset(traceset_path=gatherer_sds804x_traceset_path, headers=headers)

    # 示波器准备采集
    gatherer_sds804x.arm(gatherer_sds804x_arm_timeout)

    # 初始化设备
    target_c51_aes128.init(bytes.fromhex(target_c51_aes128_key_hex))

    # 设置Trace参数
    trace_parameter_map = TraceParameterMap()

    # 采集并保存数据
    gatherer_sds804x.acquisition(trace_parameter_map=trace_parameter_map,
                                 timeout=gatherer_sds804x_acquisition_timeout)

    # 示波器准备采集
    gatherer_sds804x.arm(gatherer_sds804x_arm_timeout)

    # 生成测试数据
    input_hex = generate_random_hex_string(16)  # 16字节随机数

    # 设置Trace参数
    trace_parameter_map = TraceParameterMap()
    input_arr = bytes.fromhex(input_hex)

    # 触发目标设备执行
    target_c51_aes128.process(0, input_arr)

    # 采集并保存数据
    gatherer_sds804x.acquisition(trace_parameter_map=trace_parameter_map,
                                 timeout=gatherer_sds804x_acquisition_timeout)

    # 示波器准备采集
    gatherer_sds804x.arm(gatherer_sds804x_arm_timeout)

    # 生成测试数据
    input_hex = generate_random_hex_string(16)  # 16字节随机数

    # 设置Trace参数
    trace_parameter_map = TraceParameterMap()
    input_arr = bytes.fromhex(input_hex)

    # 触发目标设备执行
    target_c51_aes128.process(1, input_arr)

    # 采集并保存数据
    gatherer_sds804x.acquisition(trace_parameter_map=trace_parameter_map,
                                 timeout=gatherer_sds804x_acquisition_timeout)

    # 资源清理
    target_c51_aes128.close()
    gatherer_sds804x.close_traceset()
    gatherer_sds804x.close_instrument()
