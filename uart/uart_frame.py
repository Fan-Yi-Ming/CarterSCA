UART_FRAME_MIN_LENGTH = 8


def calculate_crc16_ibm(data: bytes) -> int:
    """
    计算IBM CRC-16校验值

    Args:
        data: 需要计算CRC的字节数据

    Returns:
        int: 16位CRC校验值
    """
    crc16 = 0x0000  # 初始化CRC值为0
    for byte in data:
        crc16 ^= byte  # 当前字节与CRC值进行异或
        for _ in range(8):  # 处理每个字节的8位
            if crc16 & 0x0001:  # 检查最低位是否为1
                crc16 >>= 1  # 右移一位
                crc16 ^= 0x8005  # 与多项式0x8005进行异或
            else:
                crc16 >>= 1  # 直接右移一位
    return crc16 & 0xFFFF  # 确保返回16位无符号整数


def create_uart_frame(command: int, data: bytes) -> bytes:
    """
    创建UART通信帧

    Args:
        command: 命令字节
        data: 数据字节串

    Returns:
        bytes: 完整的UART帧数据
    """
    data_length = len(data)

    # 初始化帧数据
    frame = bytearray()

    # 帧头 (2字节: 0xA5, 0x5A)
    frame.append(0xA5)
    frame.append(0x5A)

    # 命令字节
    frame.append(command)

    # 数据长度 (2字节，大端序)
    frame.append((data_length >> 8) & 0xFF)  # 长度高字节
    frame.append(data_length & 0xFF)  # 长度低字节

    # 数据内容
    if data_length > 0:
        frame.extend(data)

    # 计算CRC16校验（从命令字节开始到数据结束）
    crc16 = calculate_crc16_ibm(frame[2:])
    # 添加CRC校验值 (2字节，高字节在前)
    frame.append(crc16 >> 8)  # CRC高字节
    frame.append(crc16 & 0xFF)  # CRC低字节

    # 帧尾
    frame.append(0xFF)

    return bytes(frame)


def validate_uart_frame(frame: bytes) -> bool:
    """
    验证UART帧的完整性

    Args:
        frame: 待验证的帧数据

    Returns:
        bool: 帧数据有效返回True，否则返回False
    """
    # 检查最小帧长度：帧头2 + 命令1 + 长度2 + CRC2 + 帧尾1 = 8字节
    if len(frame) < 8:
        return False  # 帧长度不足

    # 检查帧头是否正确
    if frame[0] != 0xA5 or frame[1] != 0x5A:
        return False  # 帧头不匹配

    # 提取数据长度 (2字节，高字节在前)
    data_length = (frame[3] << 8) | frame[4]

    # 验证数据长度是否与实际帧长度一致
    # 总帧长度 = 帧头2 + 命令1 + 长度2 + 数据data_length + CRC2 + 帧尾1
    expected_length = 8 + data_length
    if len(frame) != expected_length:
        return False  # 数据长度不匹配

    # 提取接收到的CRC校验值
    crc_received = (frame[5 + data_length] << 8) | frame[6 + data_length]

    # 计算实际的CRC校验值（从命令字节到数据结束）
    crc_calculated = calculate_crc16_ibm(frame[2:5 + data_length])

    # 验证CRC校验值
    if crc_received != crc_calculated:
        return False  # CRC校验失败

    # 检查帧尾是否正确
    if frame[7 + data_length] != 0xFF:
        return False  # 帧尾不匹配

    # 所有检查通过，帧有效
    return True


def parse_uart_frame(frame: bytes) -> tuple[int, bytes]:
    """
    解析UART帧并提取命令和数据

    Args:
        frame: 完整的UART帧数据

    Returns:
        tuple: (command, data) - 命令字节和数据字节串
    """
    # 首先验证帧的完整性
    if not validate_uart_frame(frame):
        raise ValueError(f'Invalid UART frame: {frame}')

    # 提取数据长度
    data_length = (frame[3] << 8) | frame[4]

    # 提取命令字节
    command = frame[2]

    # 提取数据内容
    data = frame[5:5 + data_length] if data_length > 0 else b''

    return command, data


if __name__ == '__main__':
    # 测试用例：创建并验证UART帧
    hex_data = "00 11 22 33 44 55 66 77 88 99 AA BB CC DD EE FF "
    frame1 = create_uart_frame(0x00, bytes.fromhex(hex_data))

    # 将帧数据转换为十六进制字符串显示
    hex_frame = ' '.join(f'{byte:02X}' for byte in frame1)
    print(f"生成的帧: {hex_frame}")

    # 验证帧的完整性
    if validate_uart_frame(frame1):
        print("帧检查: 通过")
    else:
        print("帧检查: 失败")

    command1, data1 = parse_uart_frame(frame1)
    if command1 is None:
        print("帧解析: 失败")
    else:
        print("帧解析: 通过")
