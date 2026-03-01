from uart.serial_reader import SerialCommunicator
from uart.uart_frame import create_uart_frame, parse_uart_frame
from typing import Tuple


class TargetC51Milenage:

    def __init__(self, port="COM3", baudrate=115200, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_communicator = SerialCommunicator()

    def init(self):
        self.serial_communicator.open_connection(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout
        )

        frame = create_uart_frame(0x04, bytes())
        received_bytes = self.serial_communicator.send_and_receive(frame, 8)
        parse_uart_frame(received_bytes)
        print(f"设备初始化完成: TargetC51Milenage")

    def process(self, input_arr: bytes) -> Tuple[int, bytes]:
        frame = create_uart_frame(0x05, input_arr)
        received_bytes = self.serial_communicator.send_and_receive(frame, 48)
        command, data = parse_uart_frame(received_bytes)
        return command, data

    def close(self):
        self.serial_communicator.close_connection()
        print(f"设备已关闭")
