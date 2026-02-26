from uart.serial_reader import SerialCommunicator
from uart.uart_frame import create_uart_frame, parse_uart_frame
from typing import Tuple


class TargetUsim:

    def __init__(self, port="COM1", baudrate=115200, timeout=2.0, card_type=0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.card_type = card_type
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

        if self.card_type == 0:  # 772057371(旧) 990395338
            # 发送APDU命令序列进行USIM初始化
            command_apdu_hex = "00 A4 00 04 02 2F 00"
            self.serial_communicator.transmit_apdu(0x02, bytes.fromhex(command_apdu_hex))

            command_apdu_hex = "00 B2 01 04 26"
            self.serial_communicator.transmit_apdu(0x02, bytes.fromhex(command_apdu_hex))

            command_apdu_hex = "00 A4 04 04 10 A0 00 00 00 87 10 02 FF 49 FF FF 89 04 0B 00 FF"
            self.serial_communicator.transmit_apdu(0x02, bytes.fromhex(command_apdu_hex))

            command_apdu_hex = "00 C0 00 00 39"
            self.serial_communicator.transmit_apdu(0x02, bytes.fromhex(command_apdu_hex))

        elif self.card_type == 1:  # 15836141022
            # 发送APDU命令序列进行USIM初始化
            command_apdu_hex = "00 A4 00 04 02 2F 00"
            self.serial_communicator.transmit_apdu(0x02, bytes.fromhex(command_apdu_hex))

            command_apdu_hex = "00 B2 01 04 26"
            self.serial_communicator.transmit_apdu(0x02, bytes.fromhex(command_apdu_hex))

            command_apdu_hex = "00 A4 04 04 10 A0 00 00 00 87 10 02 FF 86 FF FF 89 FF FF FF FF"
            self.serial_communicator.transmit_apdu(0x02, bytes.fromhex(command_apdu_hex))

            command_apdu_hex = "00 C0 00 00 3C"
            self.serial_communicator.transmit_apdu(0x02, bytes.fromhex(command_apdu_hex))

        else:
            raise ValueError("Card Type Err")

    def process(self, rnd_hex: str, autn_hex: str) -> Tuple[int, bytes, int, int]:
        # 构建并发送认证APDU命令
        command_apdu_hex = f"00 88 00 81 22 10 {rnd_hex} 10 {autn_hex}"
        return self.serial_communicator.transmit_apdu(0x03, bytes.fromhex(command_apdu_hex))

    def close(self):
        self.serial_communicator.close_connection()
