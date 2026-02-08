from uart.serial_reader import SerialCommunicator
from uart.uart_frame import create_uart_frame, parse_uart_frame
from typing import Tuple


class TargetC51Aes128:

    def __init__(self, port="COM3", baudrate=115200, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_communicator = SerialCommunicator()

    def init(self, key_arr: bytes):
        self.serial_communicator.open_connection(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout
        )

        frame = create_uart_frame(0x01, key_arr)
        received_bytes = self.serial_communicator.send_and_receive(frame, 8)
        parse_uart_frame(received_bytes)

    def process(self, direction: int, input_arr: bytes) -> Tuple[int, bytes]:
        if direction == 0:
            frame = create_uart_frame(0x02, input_arr)
            received_bytes = self.serial_communicator.send_and_receive(frame, 8 + 16)

        elif direction == 1:
            frame = create_uart_frame(0x03, input_arr)
            received_bytes = self.serial_communicator.send_and_receive(frame, 8 + 16)

        else:
            raise ValueError("Direction must be 0 or 1")

        command, data = parse_uart_frame(received_bytes)
        return command, data

    def close(self):
        self.serial_communicator.close_connection()
