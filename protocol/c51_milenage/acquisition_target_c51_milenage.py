import time
import trsfile.traceparameter as tp
from trsfile import Header, SampleCoding
from trsfile.parametermap import TraceSetParameterMap, TraceParameterDefinitionMap, TraceParameterMap
from trsfile.traceparameter import ParameterType, TraceParameterDefinition
from protocol.c51_milenage.target_c51_milenage import TargetC51Milenage
from protocol.gatherer_sds804x import GathererSDS804X
from tools.sca import generate_random_hex_string

if __name__ == '__main__':
    # 目标设备初始化
    target_c51_milenage = TargetC51Milenage(port="COM3", baudrate=115200, timeout=1.0)
    target_c51_milenage.init()

    # 示波器配置参数
    gatherer_sds804x_resource_name = "TCPIP0::169.254.114.206::inst0::INSTR"
    gatherer_sds804x_ref_channel_name = "C1"
    gatherer_sds804x_arm_timeout = 0.1
    gatherer_sds804x_acquisition_timeout = 5.0
    gatherer_sds804x_acquisition_times = 10
    gatherer_sds804x_traceset_path = "D:\\traceset\\milenage.trs"

    # 异常处理配置
    max_exception_count = 10  # 允许的最大异常次数

    # 示波器初始化
    gatherer_sds804x = GathererSDS804X()
    gatherer_sds804x.open_instrument(gatherer_sds804x_resource_name)
    gatherer_sds804x.arm(gatherer_sds804x_arm_timeout)
    gatherer_sds804x.snapshot_channels_parameters(gatherer_sds804x_acquisition_timeout)

    # 写入TraceSet参数
    traceset_parameter_map = TraceSetParameterMap()

    # 定义Trace元数据
    trace_parameter_definition_map = TraceParameterDefinitionMap()
    trace_parameter_definition_map["INPUT"] = TraceParameterDefinition(ParameterType.BYTE, 32, 0)
    trace_parameter_definition_map["OUTPUT"] = TraceParameterDefinition(ParameterType.BYTE, 41, 32)

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

    # ============================ 开始采集 ============================
    successful_count = 0
    current_exception_count = 0
    exception_happened = False

    while True:
        # 检查成功条件
        if successful_count >= gatherer_sds804x_acquisition_times:
            print("已达到目标采集次数")
            break

        # 检查异常条件
        if current_exception_count >= max_exception_count:
            print(f"已达到最大异常次数 {max_exception_count}")
            break

        start_time = time.perf_counter()

        try:
            # 目标设备重新初始化
            if exception_happened:
                exception_happened = False
                target_c51_milenage.close()
                target_c51_milenage.init()

            # 示波器准备采集
            gatherer_sds804x.arm(gatherer_sds804x_arm_timeout)

            # 生成测试数据
            rand_hex = generate_random_hex_string(16)
            autn_hex = "FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF"

            # 设置Trace参数
            trace_parameter_map = TraceParameterMap()
            input_arr = bytes.fromhex(rand_hex + " " + autn_hex)
            trace_parameter_map["INPUT"] = tp.ByteArrayParameter(input_arr)

            # 触发目标设备执行
            command, data = target_c51_milenage.process(input_arr)
            output_arr = bytes([command]) + data
            trace_parameter_map["OUTPUT"] = tp.ByteArrayParameter(output_arr)

            # 采集并保存数据
            gatherer_sds804x.acquisition(trace_parameter_map=trace_parameter_map,
                                         timeout=gatherer_sds804x_acquisition_timeout)

            # 成功计数
            successful_count += 1

            # 输出采集耗时
            elapsed_time = time.perf_counter() - start_time
            print(f"第 {successful_count}/{gatherer_sds804x_acquisition_times} 次采集完成，"
                  f"用时 {elapsed_time:.3f} 秒")

        except Exception as e:
            # 异常处理
            current_exception_count += 1

            print(f"采集发生异常，异常次数: {current_exception_count}/{max_exception_count}")
            print(f"异常信息: {str(e)}")

            time.sleep(3)
            exception_happened = True
            continue

    print(f"采集完成，成功采集 {successful_count} 次，异常 {current_exception_count} 次")

    # 资源清理
    target_c51_milenage.close()
    gatherer_sds804x.close_traceset()
    gatherer_sds804x.close_instrument()
