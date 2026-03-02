from uart.serial_reader import SerialCommunicator
from uart.uart_frame import create_uart_frame, parse_uart_frame
from typing import Tuple


class TargetUsimUZ:

    def __init__(self, port="COM1", baudrate=115200, timeout=2.0, card_type=0):
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

        # 发送智能卡复位指令
        frame = create_uart_frame(0x01, bytes.fromhex(""))
        received_bytes = self.serial_communicator.send_and_receive(frame, 8 + 0)
        command, data = parse_uart_frame(received_bytes)
        if command is None:
            raise ValueError("Frame parse Err")

        # 发送APDU命令序列进行USIM初始化
        command_apdu_hex = "00 A4 00 04 02 2F 00"
        self.serial_communicator.transmit_apdu(0x02, bytes.fromhex(command_apdu_hex))

        command_apdu_hex = "00 B2 01 04 26"
        self.serial_communicator.transmit_apdu(0x02, bytes.fromhex(command_apdu_hex))

        command_apdu_hex = "00 A4 04 04 10 A0 00 00 00 87 10 02 FF 49 FF FF 89 04 0B 00 FF"
        self.serial_communicator.transmit_apdu(0x02, bytes.fromhex(command_apdu_hex))

        command_apdu_hex = "00 C0 00 00 39"
        self.serial_communicator.transmit_apdu(0x02, bytes.fromhex(command_apdu_hex))

    def process(self, input_arr: bytes) -> Tuple[int, bytes, int, int]:
        # 分割前16字节作为rand，后16字节作为autn
        rand = input_arr[:16]
        autn = input_arr[16:]

        # 使用字符串格式构建命令
        auth_command_hex = f"00 88 00 81 22 10 {rand.hex()} 10 {autn.hex()}"
        return self.serial_communicator.transmit_apdu(0x03, bytes.fromhex(auth_command_hex))

    def close(self):
        self.serial_communicator.close_connection()
