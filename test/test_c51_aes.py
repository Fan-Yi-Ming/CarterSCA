from uart.serial_reader import list_available_serial_ports, SerialCommunicator
from uart.uart_frame import create_uart_frame

if __name__ == '__main__':
    list_available_serial_ports()

    # 使用上下文管理器，确保串口正确关闭
    with SerialCommunicator() as serial_communicator:
        # 设置超时时间为1秒
        serial_communicator.open_connection(port="COM3", baudrate=115200, timeout=1.0)

        hex_data = "2B 7E 15 16 28 AE D2 A6 AB F7 15 88 09 CF 4F 3C"
        frame = create_uart_frame(0x01, bytes.fromhex(hex_data))
        serial_communicator.send_and_receive(frame, 8)

        hex_data = "32 43 F6 A8 88 5A 30 8D 31 31 98 A2 E0 37 07 34"
        frame = create_uart_frame(0x02, bytes.fromhex(hex_data))
        serial_communicator.send_and_receive(frame, 24)

        hex_data = "39 25 84 1D 02 DC 09 FB DC 11 85 97 19 6A 0B 32"
        frame = create_uart_frame(0x03, bytes.fromhex(hex_data))
        serial_communicator.send_and_receive(frame, 24)

