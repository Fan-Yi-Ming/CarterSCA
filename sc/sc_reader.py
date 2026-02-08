from smartcard.System import readers
from smartcard.util import toBytes, toHexString
import time


def list_smartcard_readers():
    """列出所有读卡器"""
    reader_list = [str(reader) for reader in readers()]
    print("找到读卡器:", reader_list)
    return reader_list


def connect_smartcard_reader(reader_list, reader_index=0):
    """连接读卡器"""
    reader = readers()[reader_index]
    connection = reader.createConnection()
    connection.connect()
    print(f"已连接: {reader_list[reader_index]}")
    return connection


def transmit_apdu(connection, apdu_command):
    """发送APDU命令"""
    apdu_bytes = toBytes(apdu_command.replace(" ", ""))
    print(f"发送: {apdu_command}")

    start_time = time.time()
    response, sw1, sw2 = connection.transmit(apdu_bytes)
    end_time = time.time()

    print(f"响应: {sw1:02X} {sw2:02X}")

    if response:
        response_hex = toHexString(response).upper()
        print(f"数据: {response_hex}")

    full_response = response + [sw1, sw2]
    full_response_hex = toHexString(full_response).upper()
    print(f"接收: {full_response_hex}")

    print(f"耗时: {(end_time - start_time) * 1000:.2f} ms")
    return response, sw1, sw2
