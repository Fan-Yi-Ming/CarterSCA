import numpy as np
import trsfile
import trsfile.traceparameter as tp
from trsfile import Header, TracePadding, Trace, SampleCoding
from trsfile.parametermap import TraceParameterMap, TraceParameterDefinitionMap, TraceSetParameterMap
from trsfile.traceparameter import TraceParameterDefinition, ParameterType

from tools.aes import aes_keyexpansion, Aes
from tools.sca import generate_random_hex_string, hw

if __name__ == '__main__':
    # 配置参数
    traceset_path = "D:\\traceset\\aes128_simulation.trs"
    aes128_direction = 0  # 0:加密 1:解密
    aes128_key_hex = "2B 7E 15 16 28 AE D2 A6 AB F7 15 88 09 CF 4F 3C"
    simulate_times = 1000

    # 写入TraceSet参数
    traceset_parameter_map = TraceSetParameterMap()
    traceset_parameter_map["DIRECTION"] = tp.ByteArrayParameter(bytes([aes128_direction]))
    traceset_parameter_map['KEY'] = tp.ByteArrayParameter(bytes.fromhex(aes128_key_hex))

    # 定义Trace元数据
    trace_parameter_definition_map = TraceParameterDefinitionMap()
    trace_parameter_definition_map["INPUT"] = TraceParameterDefinition(ParameterType.BYTE, 16, 0)
    trace_parameter_definition_map["OUTPUT"] = TraceParameterDefinition(ParameterType.BYTE, 16, 16)

    # 构建TRS文件头
    headers = {
        Header.TRS_VERSION: 2,
        Header.TITLE_SPACE: 255,
        Header.LABEL_X: " ",
        Header.SCALE_X: 1.0,
        Header.LABEL_Y: " ",
        Header.SCALE_Y: 1.0,
        Header.TRACE_SET_PARAMETERS: traceset_parameter_map,
        Header.TRACE_PARAMETER_DEFINITIONS: trace_parameter_definition_map
    }

    # 创建TRS文件
    traceset = trsfile.trs_open(
        path=traceset_path,
        mode='w',
        engine='TrsEngine',
        headers=headers,
        padding_mode=TracePadding.AUTO,
        live_update=True
    )

    # AES初始化
    key = np.frombuffer(bytes.fromhex(aes128_key_hex), dtype=np.uint8)
    w = aes_keyexpansion(key)
    aes = Aes()
    rounds = 10

    if aes128_direction == 0:
        # AES加密模式
        for i in range(simulate_times):
            hw_intermediate_arr = []
            input_arr = np.frombuffer(bytes.fromhex(generate_random_hex_string(16)), dtype=np.uint8)

            aes.set_state(input_arr)

            # 初始轮
            aes.add_roundkey(w[0:4])

            # 主轮循环
            for j in range(1, rounds):
                aes.sub_state()
                intermediate_arr = aes.get_state()
                hw_intermediate_arr.extend(hw(byte) for byte in intermediate_arr)
                aes.shift_rows()
                aes.mix_columns()
                aes.add_roundkey(w[4 * j:4 * j + 4])

            # 最终轮
            aes.sub_state()
            intermediate_arr = aes.get_state()
            hw_intermediate_arr.extend(hw(byte) for byte in intermediate_arr)
            aes.shift_rows()
            aes.add_roundkey(w[4 * rounds:4 * rounds + 4])

            output_arr = aes.get_state()

            # 保存能量迹
            trace_parameter_map = TraceParameterMap()
            trace_parameter_map["INPUT"] = tp.ByteArrayParameter(input_arr)
            trace_parameter_map["OUTPUT"] = tp.ByteArrayParameter(output_arr)

            trace = Trace(
                sample_coding=SampleCoding.BYTE,
                samples=hw_intermediate_arr,
                parameters=trace_parameter_map,
                title=f"AES128 EN Simulate {i}"
            )
            traceset.append(trace)

    elif aes128_direction == 1:
        # AES解密模式
        for i in range(simulate_times):
            hw_intermediate_arr = []
            input_arr = np.frombuffer(bytes.fromhex(generate_random_hex_string(16)), dtype=np.uint8)

            aes.set_state(input_arr)

            # 初始轮
            aes.add_roundkey(w[4 * rounds:4 * rounds + 4])

            # 主轮循环
            for j in range(rounds - 1, 0, -1):
                aes.inv_shift_rows()
                aes.inv_sub_state()
                intermediate_arr = aes.get_state()
                hw_intermediate_arr.extend(hw(byte) for byte in intermediate_arr)
                aes.add_roundkey(w[4 * j:4 * j + 4])
                aes.inv_mix_columns()

            # 最终轮
            aes.inv_shift_rows()
            aes.inv_sub_state()
            intermediate_arr = aes.get_state()
            hw_intermediate_arr.extend(hw(byte) for byte in intermediate_arr)
            aes.add_roundkey(w[0:4])

            output_arr = aes.get_state()

            # 保存能量迹
            trace_parameter_map = TraceParameterMap()
            trace_parameter_map["INPUT"] = tp.ByteArrayParameter(input_arr)
            trace_parameter_map["OUTPUT"] = tp.ByteArrayParameter(output_arr)

            trace = Trace(
                sample_coding=SampleCoding.BYTE,
                samples=hw_intermediate_arr,
                parameters=trace_parameter_map,
                title=f"AES128 DE Simulate {i}"
            )
            traceset.append(trace)
    else:
        raise ValueError("AES128 direction must be 0 or 1.")

    # 清理资源
    traceset.close()
    print(f"仿真完成！共生成 {simulate_times} 条轨迹，保存至: {traceset_path}")
