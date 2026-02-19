import serial
import serial.tools.list_ports
from typing import Tuple

from uart.apdu import CommandAPDU
from uart.uart_frame import create_uart_frame, parse_uart_frame, UART_FRAME_MIN_LENGTH


def list_available_serial_ports():
    """
    列出系统中所有可用的串口端口。

    :return: 可用串口端口名称的列表
    """
    ports = serial.tools.list_ports.comports()

    if not ports:
        print("未找到任何可用的串口设备")
        return []

    available_ports = []
    print("可用的串口端口:")
    print("-" * 60)
    print(f"{'端口':<10} {'描述':<25} {'硬件ID'}")
    print("-" * 60)

    for port in ports:
        available_ports.append(port.device)
        description = port.description if port.description else "N/A"
        hwid = port.hwid if port.hwid else "N/A"
        print(f"{port.device:<10} {description:<25} {hwid}")

    print("-" * 60)
    print(f"总共找到 {len(available_ports)} 个串口设备")
    print("")

    return available_ports


class SerialCommunicator:
    def __init__(self):
        """
        初始化串口通信对象。
        """
        self.serial_port = None

    def open_connection(self, port: str, baudrate: int = 9600, bytesize: int = serial.EIGHTBITS,
                        parity: str = serial.PARITY_NONE, stopbits: float = serial.STOPBITS_ONE,
                        timeout: float = 1.0, write_timeout: float = 1.0):
        """
        打开串口连接。

        :param port: 串口名称（如 'COM1' 或 '/dev/ttyUSB0'）
        :param baudrate: 波特率，默认 9600
        :param bytesize: 数据位，默认 8 位
        :param parity: 校验位，默认无校验
        :param stopbits: 停止位，默认 1 位
        :param timeout: 读超时时间，默认 1.0 秒
        :param write_timeout: 写超时时间，默认 1.0 秒
        """
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits,
                timeout=timeout,  # 读超时
                write_timeout=write_timeout  # 写超时
            )
        except serial.SerialException as e:
            raise serial.SerialException(f"无法打开串口 {port}: {e}")

    def close_connection(self):
        """
        关闭串口连接。
        """
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()

    def is_connected(self) -> bool:
        """
        检查串口是否已连接。

        :return: 如果串口打开返回 True，否则返回 False
        """
        return self.serial_port is not None and self.serial_port.is_open

    def send_data(self, data: bytes):
        """
        向串口发送数据。

        :param data: 需要发送的字节数据
        :raises SerialException: 如果发送数据失败或串口未打开
        """
        if not self.is_connected():
            raise serial.SerialException("串口未打开")

        # 打印发送的数据
        hex_string = ' '.join(f'{byte:02X}' for byte in data)
        print(f"--->: {hex_string}")
        bytes_sent = self.serial_port.write(data)
        if bytes_sent != len(data):
            raise serial.SerialException(f"发送数据不完整：预期发送 {len(data)} 字节，实际发送 {bytes_sent} 字节")

    def receive_data(self, expect_length: int) -> bytes:
        """
        从串口读取指定长度的数据，使用串口内置的超时设置。

        :param expect_length: 预期读取的数据长度
        :return: 读取到的字节数据
        :raises SerialException: 串口未打开
        :raises SerialTimeoutException: 读取超时（由PySerial自动抛出）
        """
        if not self.is_connected():
            raise serial.SerialException("串口未打开")

        if expect_length <= 0:
            return bytes()

        # 直接使用PySerial的read方法
        received_bytes = self.serial_port.read(expect_length)

        # 检查是否读取到足够的数据
        if len(received_bytes) < expect_length:
            raise serial.SerialTimeoutException(
                f"接收数据超时：预期长度 {expect_length} 字节，实际收到 {len(received_bytes)} 字节")

        # 打印接收的数据
        hex_string = ' '.join(f'{byte:02X}' for byte in received_bytes)
        print(f"<---: {hex_string}")

        return received_bytes

    def send_and_receive(self, data: bytes, expect_length: int) -> bytes:
        """
        发送数据并接收返回数据。

        :param data: 需要发送的字节数据
        :param expect_length: 预期接收的数据长度
        :return: 接收到的字节数据
        """
        self.send_data(data)
        return self.receive_data(expect_length)

    def transmit_apdu(self, command: int, command_apdu_bytes: bytes) -> Tuple[int, bytes, int, int]:
        """
        发送APDU命令到设备并接收响应数据。

        该方法采用两步通信协议：
        1. 首先发送命令并接收响应数据长度信息
        2. 然后根据长度信息接收完整的响应数据

        通信流程：
        - 将APDU命令封装为UART帧并发送
        - 接收第一帧响应（包含数据长度信息）
        - 解析长度后接收第二帧响应（包含实际数据）
        - 从第二帧响应中提取状态字和响应数据

        :param command: 命令码，用于标识要执行的APDU操作
        :param command_apdu_bytes: APDU命令的字节序列数据
        :return: 四元组包含：
                 - 命令码 (int)
                 - 响应数据字节序列 (bytes)，不包含状态字
                 - SW1状态字节 (int)
                 - SW2状态字节 (int)
        :raises SerialException: 当发生以下情况时抛出：
                                - 命令APDU格式解析失败
                                - UART帧解析失败
                                - 响应数据长度不足，缺少状态字
                                - 通信过程中发生其他串口错误
        """
        # 解析命令APDU数据，验证格式正确性
        command_apdu = CommandAPDU()
        if command_apdu.check(command_apdu_bytes) != command_apdu.SUCCESS:
            raise serial.SerialException("CommandAPDU格式错误")

        # 将命令码和APDU数据封装为UART帧格式
        frame = create_uart_frame(command, command_apdu_bytes)

        # 发送UART帧并接收第一帧响应（包含响应数据长度信息）
        # 第一帧响应长度为：UART帧最小长度 + 2字节长度字段
        received_bytes = self.send_and_receive(frame, UART_FRAME_MIN_LENGTH + 2)

        # 解析第一帧UART响应，获取响应数据长度信息
        try:
            command, data = parse_uart_frame(received_bytes)
        except ValueError as e:
            raise serial.SerialException("UART第一帧解析失败") from e

        # 从响应数据中提取APDU数据长度（前两个字节，大端序）
        # data[0]为高字节，data[1]为低字节
        response_apdu_bytes_length = (data[0] << 8) + data[1]

        # 根据数据长度接收完整的APDU响应数据（第二帧）
        # 接收长度为：数据长度 + UART帧最小长度（包含帧头等开销）
        received_bytes = self.receive_data(response_apdu_bytes_length + UART_FRAME_MIN_LENGTH)

        # 解析第二帧UART响应，获取实际的响应数据
        try:
            command, data = parse_uart_frame(received_bytes)
        except ValueError as e:
            raise serial.SerialException("UART第二帧解析失败") from e

        # 验证响应数据是否包含足够的状态字（至少2字节）
        if len(data) < 2:
            raise serial.SerialException("响应数据长度不足，缺少状态字")

        # 从响应数据中提取状态字和实际响应数据
        sw1 = data[-2]  # 倒数第二个字节为SW1状态字
        sw2 = data[-1]  # 最后一个字节为SW2状态字
        response_data = data[:-2]  # 除状态字外的其余字节为实际响应数据

        return command, response_data, sw1, sw2

    def transmit(self, command: int, data: bytes) -> Tuple[int, bytes]:
        """
        发送通用命令并接收响应数据。

        这是一个简化的通信方法，与transmit_apdu类似但处理更通用的数据格式。

        :param command: 命令码，用于标识要执行的操作
        :param data: 要发送的数据字节序列
        :return: 二元组包含：
                 - 命令码 (int)
                 - 接收到的完整响应数据字节序列 (bytes)
        :raises SerialException: 当UART帧解析失败时抛出
        """
        frame = create_uart_frame(command, data)

        received_bytes = self.send_and_receive(frame, 10)

        try:
            command, data = parse_uart_frame(received_bytes)
        except ValueError as e:
            raise serial.SerialException("UART第一帧解析失败") from e

        expect_length = (data[0] << 8) + data[1] + 8
        received_bytes = self.receive_data(expect_length)

        try:
            command, data = parse_uart_frame(received_bytes)
        except ValueError as e:
            raise serial.SerialException("UART第二帧解析失败") from e

        return command, data

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持上下文管理器，自动关闭串口"""
        self.close_connection()


if __name__ == '__main__':
    list_available_serial_ports()

    # 使用上下文管理器，确保串口正确关闭
    with SerialCommunicator() as serial_communicator:
        # 设置超时时间为1秒
        serial_communicator.open_connection(port="COM3", baudrate=115200, timeout=1.0)

        # 示例1：使用原始的send_and_receive方法
        hex_payload = ("A5 5A 00 00 10 "
                       "00 11 22 33 44 55 66 77 88 99 AA BB CC DD EE FF "
                       "26 39 FF")
        payload = bytes.fromhex(hex_payload)
        serial_communicator.send_and_receive(payload, len(payload))
