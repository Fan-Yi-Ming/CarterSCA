import numpy as np
import trsfile
import trsfile.traceparameter as tp
from trsfile import Header, TracePadding, Trace, SampleCoding
from trsfile.parametermap import TraceParameterMap, TraceParameterDefinitionMap, TraceSetParameterMap
from trsfile.traceparameter import TraceParameterDefinition, ParameterType

from tools.aes import aes_keyexpansion, Aes
from tools.sca import generate_random_hex_string, hw

if __name__ == '__main__':
    # ============================ 配置文件路径和参数 ============================
    traceset_path = "..\\traceset\\usim_simulation.trs"
    key_hex = "2B 7E 15 16 28 AE D2 A6 AB F7 15 88 09 CF 4F 3C"  # AES-128加密密钥
    opc_hex = "00 11 22 33 44 55 66 77 88 99 AA BB CC DD EE FF"  # OPC运算参数
    simulate_times = 10000  # 仿真轨迹数量

    # ============================ TraceSet参数初始化 ============================
    # 初始化TraceSet级别的参数映射（所有轨迹共享）
    traceset_parameter_map = TraceSetParameterMap()
    traceset_parameter_map['KEY'] = tp.ByteArrayParameter(bytes.fromhex(key_hex))
    traceset_parameter_map['OPC'] = tp.ByteArrayParameter(bytes.fromhex(opc_hex))

    # 定义每条Trace的参数结构（每条轨迹特有的参数）
    trace_parameter_definition_map = TraceParameterDefinitionMap()
    # 定义RND参数：16字节的BYTE类型参数，偏移量为0（用于存储随机数）
    trace_parameter_definition_map["RND"] = TraceParameterDefinition(ParameterType.BYTE, 16, 0)

    # 设置TRS文件头部参数
    headers = {
        Header.TRS_VERSION: 2,  # TRS文件格式版本号
        Header.TITLE_SPACE: 255,  # 标题预留空间大小
        Header.LABEL_X: " ",  # X轴标签（通常表示时间采样点）
        Header.SCALE_X: 1.0,  # X轴缩放比例（采样时间间隔）
        Header.LABEL_Y: " ",  # Y轴标签（通常表示电压值或功耗）
        Header.SCALE_Y: 1.0,  # Y轴缩放比例
        Header.TRACE_SET_PARAMETERS: traceset_parameter_map,  # TraceSet级别参数
        Header.TRACE_PARAMETER_DEFINITIONS: trace_parameter_definition_map  # Trace参数定义
    }

    # 创建TraceSet文件（用于存储侧信道分析轨迹数据）
    traceset = trsfile.trs_open(
        path=traceset_path,  # 输出文件路径
        mode='w',  # 写入模式
        engine='TrsEngine',  # 存储引擎
        headers=headers,  # 文件头部信息
        padding_mode=TracePadding.AUTO,  # 自动填充模式
        live_update=True  # 启用实时更新（便于调试查看）
    )

    # ============================ AES加密初始化 ============================
    # 将十六进制密钥转换为字节数组并进行AES-128密钥扩展
    key = np.frombuffer(bytes.fromhex(key_hex), dtype=np.uint8)
    w = aes_keyexpansion(key)  # 生成轮密钥

    # 初始化AES加密对象
    aes = Aes()
    rounds = 10  # AES-128轮数

    # ============================ 轨迹仿真循环 ============================
    for i in range(simulate_times):
        # 存储中间值的汉明重量（用于模拟功耗轨迹）
        hw_intermediate_arr = []

        # 生成随机输入数据（16字节随机数）和OPC数据
        rnd_arr = np.frombuffer(bytes.fromhex(generate_random_hex_string(16)), dtype=np.uint8)
        opc_arr = np.frombuffer(bytes.fromhex(opc_hex), dtype=np.uint8)

        # 执行RND与OPC的异或操作（Milenage算法中的输入处理）
        xor_result = np.bitwise_xor(rnd_arr, opc_arr)
        aes.set_state(xor_result)  # 设置AES状态矩阵

        # ============================ AES加密轮处理 ============================
        # 初始轮：仅执行轮密钥加
        aes.add_roundkey(w[0:4])

        # 主轮循环（第1轮到第9轮）
        for j in range(1, rounds):
            aes.sub_state()  # 字节代换（S盒变换）
            intermediate_arr = aes.get_state()  # 获取S盒输出状态
            # 计算每个字节的汉明重量并添加到轨迹中（模拟功耗泄露）
            hw_intermediate_arr.extend(hw(byte) for byte in intermediate_arr)
            aes.shift_rows()  # 行移位
            aes.mix_columns()  # 列混合
            aes.add_roundkey(w[4 * j:4 * j + 4])  # 轮密钥加

        # 最终轮（第10轮）：不包含列混合操作
        aes.sub_state()
        intermediate_arr = aes.get_state()  # 获取最终S盒输出
        hw_intermediate_arr.extend(hw(byte) for byte in intermediate_arr)  # 计算汉明重量
        aes.shift_rows()  # 行移位
        aes.add_roundkey(w[4 * rounds:4 * rounds + 4])  # 最终轮密钥加

        # 获取加密输出结果
        output_data = aes.get_state()

        # ============================ 轨迹数据保存 ============================
        # 创建当前轨迹的参数映射
        trace_parameter_map = TraceParameterMap()
        trace_parameter_map["RND"] = tp.ByteArrayParameter(rnd_arr)  # 保存随机数参数

        # 创建轨迹对象
        trace = Trace(
            sample_coding=SampleCoding.BYTE,  # 样本编码方式（字节类型）
            samples=hw_intermediate_arr,  # 功耗轨迹样本（汉明重量序列）
            parameters=trace_parameter_map,  # 轨迹参数
            title=f"USIM Simulate {i}"  # 轨迹标题
        )

        # 将轨迹添加到TraceSet中
        traceset.append(trace)

    # ============================ 资源清理 ============================
    traceset.close()  # 关闭TraceSet文件，确保数据写入完成
    print(f"仿真完成！共生成 {simulate_times} 条轨迹，保存至: {traceset_path}")
