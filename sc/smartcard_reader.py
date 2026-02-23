from smartcard.System import readers
from smartcard.util import toHexString
import time


def list_smartcard_readers():
    reader_list = [str(reader) for reader in readers()]

    if reader_list:
        print(f"找到 {len(reader_list)} 个智能卡读卡器:")
        for i, reader in enumerate(reader_list, 1):
            print(f"{i}. {reader}")
    else:
        print("未找到读卡器")

    return reader_list


class SmartcardCommunicator:
    def __init__(self):
        self.connection = None

    def open_connection(self, reader_index: int = 0):
        reader = readers()[reader_index]
        self.connection = reader.createConnection()
        self.connection.connect()

    def close_connection(self):
        if self.connection:
            self.connection.disconnect()

    def transmit_apdu(self, apdu: bytes):
        hex_string = ' '.join(f'{byte:02X}' for byte in apdu)
        print(f"发送: {hex_string}")

        start_time = time.perf_counter()
        response, sw1, sw2 = self.connection.transmit(list(apdu))
        elapsed_time = time.perf_counter() - start_time

        print(f"响应: {sw1:02X} {sw2:02X}")
        if response:
            response_hex = toHexString(response).upper()
            print(f"数据: {response_hex}")

        hex_string = toHexString(response + [sw1, sw2]).upper()
        print(f"接收: {hex_string}")

        print(f"耗时: {(elapsed_time) * 1000:.3f} ms")
        return response, sw1, sw2
