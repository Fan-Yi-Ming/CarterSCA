from uart.serial_reader import list_serial_ports, SerialCommunicator
from uart.uart_frame import create_uart_frame

if __name__ == '__main__':
    list_serial_ports()

    # 使用上下文管理器，确保串口正确关闭
    with SerialCommunicator() as serial_communicator:
        # 设置超时时间为1秒
        serial_communicator.open_connection(port="COM3", baudrate=115200, timeout=1.0)

        frame = create_uart_frame(0x04, bytes.fromhex(""))
        serial_communicator.send_and_receive(frame, 8)

        rand_and_autn = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
                         0x19, 0xEB, 0xE6, 0xCA, 0x36, 0xF7, 0x80, 0x00, 0x62, 0xA6, 0x39, 0x33, 0xF8, 0x39, 0xC0, 0x19]
        serial_communicator.transmit(0x05, bytes(rand_and_autn))
