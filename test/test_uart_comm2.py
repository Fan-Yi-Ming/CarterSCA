import time

from uart.serial_reader import list_serial_ports, SerialCommunicator
from uart.uart_frame import create_uart_frame, parse_uart_frame

# 15836141022
if __name__ == '__main__':
    # 列出系统中所有可用的串口端口。
    list_serial_ports()

    # 使用上下文管理器，确保串口正确关闭
    with SerialCommunicator() as serial_communicator:
        # 设置超时时间为1秒
        serial_communicator.open_connection(port="COM1", baudrate=115200, timeout=1.0)
        # 发送复位指令
        frame = create_uart_frame(0x01, bytes.fromhex(""))
        received_bytes = serial_communicator.send_and_receive(frame, 8 + 0)
        command, data = parse_uart_frame(received_bytes)
        if command is None:
            raise ValueError("Frame parse Err")

        command_apdu_hex = "00 A4 00 04 02 2F 00"
        serial_communicator.transmit_apdu(0x02, bytes.fromhex(command_apdu_hex))

        command_apdu_hex = "00 B2 01 04 26"
        serial_communicator.transmit_apdu(0x02, bytes.fromhex(command_apdu_hex))

        command_apdu_hex = "00 A4 04 04 10 A0 00 00 00 87 10 02 FF 86 FF FF 89 FF FF FF FF"
        serial_communicator.transmit_apdu(0x02, bytes.fromhex(command_apdu_hex))

        command_apdu_hex = "00 C0 00 00 3C"
        serial_communicator.transmit_apdu(0x02, bytes.fromhex(command_apdu_hex))

        for i in range(10):
            command_apdu_hex = ("00 88 00 81 22 "
                                "10 A2 46 FE BA 13 12 FA 90 EC E1 12 C4 B6 AF B1 66 "
                                "10 9B 52 FC 43 9E DF 72 4C FD 08 68 4B DC AC FF D7")
            serial_communicator.transmit_apdu(0x03, bytes.fromhex(command_apdu_hex))
            time.sleep(0.2)
