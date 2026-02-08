import time
import trsfile.traceparameter as tp
from trsfile import Header, SampleCoding
from trsfile.parametermap import TraceSetParameterMap, TraceParameterDefinitionMap, TraceParameterMap
from trsfile.traceparameter import ParameterType, TraceParameterDefinition

from protocol.gatherer_sds804x import GathererSDS804X
from protocol.usim.target_usim import TargetUsim
from tools.sca import generate_random_hex_string

if __name__ == '__main__':
    # ============================ USIM设备初始化 ============================
    target_uz_usim = TargetUsim(port="COM1", baudrate=115200, timeout=2.0, card_type=0)
    target_uz_usim.init()

    # ============================ 示波器配置 ============================
    # 示波器TCP/IP连接地址
    gatherer_sds804x_resource_name = "TCPIP0::169.254.114.206::inst0::INSTR"
    # 参考通道名称，用于构建TraceSet头部信息
    gatherer_sds804x_ref_channel_name = "C1"
    # 示波器触发准备延时（单位：秒）
    gatherer_sds804x_arm_delay = 0.5
    # 单次采集超时时间（单位：秒）
    gatherer_sds804x_acquisition_timeout = 5.0
    # 数据采集总次数
    gatherer_sds804x_acquisition_times = 5000
    # TraceSet数据文件保存路径
    gatherer_sds804x_traceset_path = "..\\..\\traceset\\test5001.trs"

    # ============================ 示波器初始化 ============================
    # 创建示波器控制实例
    gatherer_sds804x = GathererSDS804X()
    # 建立示波器网络连接
    gatherer_sds804x.open_instrument(gatherer_sds804x_resource_name)
    # 准备示波器进行数据采集（设置触发状态）
    gatherer_sds804x.arm(gatherer_sds804x_arm_delay)
    # 获取并更新示波器通道硬件参数（采样率、电压范围等）
    gatherer_sds804x.update_channels_parameters(timeout=gatherer_sds804x_acquisition_timeout)

    # ============================ TraceSet参数初始化 ============================
    # 初始化TraceSet级别参数映射容器
    traceset_parameter_map = TraceSetParameterMap()

    # 定义单条Trace参数结构（参数定义映射）
    trace_parameter_definition_map = TraceParameterDefinitionMap()
    # 定义RND参数：16字节随机数，BYTE数组类型，参数块偏移量0
    trace_parameter_definition_map["RND"] = TraceParameterDefinition(ParameterType.BYTE, 16, 0)
    # 定义ATUN参数：16字节认证令牌，BYTE数组类型，参数块偏移量16
    trace_parameter_definition_map["ATUN"] = TraceParameterDefinition(ParameterType.BYTE, 16, 16)

    # 基于参考通道参数构建TRS文件头信息
    ref_channel_parameters = gatherer_sds804x.get_channel_parameters(gatherer_sds804x_ref_channel_name)
    headers = {
        Header.TRS_VERSION: 2,  # TRS文件格式版本号
        Header.TITLE_SPACE: 255,  # 标题字段预留空间大小
        Header.SAMPLE_CODING: SampleCoding.SHORT,  # 采样数据编码格式
        Header.LABEL_X: "s",  # X轴物理量标签（时间，单位：秒）
        Header.SCALE_X: ref_channel_parameters["interval"],  # X轴缩放比例（采样时间间隔）
        Header.LABEL_Y: "v",  # Y轴物理量标签（电压，单位：伏特）
        Header.SCALE_Y: ref_channel_parameters["vdiv"] / ref_channel_parameters["code"],  # Y轴缩放比例（电压/编码值）
        Header.TRACE_SET_PARAMETERS: traceset_parameter_map,  # TraceSet级别参数映射
        Header.TRACE_PARAMETER_DEFINITIONS: trace_parameter_definition_map  # Trace参数结构定义
    }
    print(f"TraceSet文件头信息设置完成，参考通道: {gatherer_sds804x_ref_channel_name}")

    # 创建并打开TraceSet文件，准备写入采集数据
    gatherer_sds804x.open_traceset(traceset_path=gatherer_sds804x_traceset_path, headers=headers)

    # ============================ 主采集循环 ============================
    for i in range(gatherer_sds804x_acquisition_times):
        # 记录单次采集开始时间点
        start_time = time.monotonic()

        # 触发示波器进入采集准备状态
        gatherer_sds804x.arm(delay=gatherer_sds804x_arm_delay)

        # 生成随机认证参数
        rand_hex = generate_random_hex_string(16)  # 生成16字节真随机数
        autn_hex = "FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF FF"  # 16字节固定认证令牌
        rand = bytes.fromhex(rand_hex)
        atun = bytes.fromhex(autn_hex)

        # 构建当前Trace的参数数据块
        trace_parameter_map = TraceParameterMap()
        trace_parameter_map["RND"] = tp.ByteArrayParameter(rand)  # 设置RND参数值
        trace_parameter_map["ATUN"] = tp.ByteArrayParameter(atun)  # 设置ATUN参数值

        # 执行USIM设备认证流程（触发目标设备运行）
        target_uz_usim.process(rand_hex, autn_hex)

        # 启动示波器采集并保存Trace数据（包含波形数据和参数数据）
        gatherer_sds804x.acquisition(trace_parameter_map=trace_parameter_map,
                                     timeout=gatherer_sds804x_acquisition_timeout)

        # 计算并输出本次采集耗时
        end_time = time.monotonic()
        elapsed_time = end_time - start_time
        print(f"第 {i + 1} 次采集耗时: {elapsed_time:.2f} 秒，共 {gatherer_sds804x_acquisition_times} 次")

    # ============================ 资源清理 ============================
    # 关闭USIM目标设备连接
    target_uz_usim.close()
    # 设置示波器为停止状态（退出采集模式）
    gatherer_sds804x.de_arm()
    # 关闭TraceSet文件（确保数据完整写入磁盘）
    gatherer_sds804x.close_traceset()
    # 关闭与示波器的网络连接
    gatherer_sds804x.close_instrument()
