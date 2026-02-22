from sc.smartcard_reader import list_smartcard_readers, SmartcardCommunicator
from tools.sca import generate_random_hex_string

# 772057371(旧) 990395338
if __name__ == '__main__':
    # 列出所有读卡器
    reader_list = list_smartcard_readers()

    # 连接读卡器
    smartcard_communicator = SmartcardCommunicator()
    smartcard_communicator.open_connection(0)

    # 1. 选择电信根目录
    apdu_hex = "00 A4 00 04 02 2F 00"
    smartcard_communicator.transmit_apdu(bytes.fromhex(apdu_hex))

    # 2. 读取目录记录
    apdu_hex = "00 B2 01 04 26"
    smartcard_communicator.transmit_apdu(bytes.fromhex(apdu_hex))

    # 3. 选择USIM应用
    apdu_hex = "00 A4 04 04 10 A0 00 00 00 87 10 02 FF 49 FF FF 89 04 0B 00 FF"
    smartcard_communicator.transmit_apdu(bytes.fromhex(apdu_hex))

    # 4. 获取响应数据
    apdu_hex = "00 C0 00 00 39"
    smartcard_communicator.transmit_apdu(bytes.fromhex(apdu_hex))

    # 5. 内部认证
    rand_hex = generate_random_hex_string(16)
    autn_hex = generate_random_hex_string(16)
    apdu_hex = "00 88 00 81 22" + " 10 " + rand_hex + " 10 " + autn_hex
    smartcard_communicator.transmit_apdu(bytes.fromhex(apdu_hex))

    smartcard_communicator.close_connection()