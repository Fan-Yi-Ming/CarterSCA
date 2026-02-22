import numpy as np

from sc.smartcard_reader import list_smartcard_readers, SmartcardCommunicator
from lib3gpp.milenage_auc import MilenageAuc
from tools.sca import generate_random_hex_string

# 18695802003 （旧白卡）
if __name__ == '__main__':
    milenageAuc = MilenageAuc()

    milenageAuc.K = np.array(
        [0xD3, 0x1D, 0xF5, 0xE3, 0xD8, 0xB4, 0x17, 0x01, 0x2E, 0x2F, 0x7C, 0x93, 0x6D, 0x75, 0x57, 0xA7],
        dtype=np.uint8)
    milenageAuc.OPc = np.array(
        [0xEF, 0x87, 0xDE, 0xEF, 0x2D, 0xC1, 0xD5, 0xC2, 0x71, 0xE7, 0x7D, 0x15, 0x80, 0x20, 0xA1, 0x35],
        dtype=np.uint8)
    milenageAuc.SQN = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00], dtype=np.uint8)
    milenageAuc.AMF = np.array([0x80, 0x00], dtype=np.uint8)

    # 列出所有读卡器
    reader_list = list_smartcard_readers()

    # 连接读卡器
    smartcard_communicator = SmartcardCommunicator()
    smartcard_communicator.open_connection(0)

    # 1. 选择电信根目录
    apdu_hex = "00 A4 00 04 02 2F 00"
    smartcard_communicator.transmit_apdu(bytes.fromhex(apdu_hex))

    # 2. 读取目录记录
    apdu_hex = "00 B2 01 04 30"
    smartcard_communicator.transmit_apdu(bytes.fromhex(apdu_hex))

    # 3. 选择USIM应用 A0 00 00 00 87 10 02 FF 86 FF 03 89 FF FF FF FF
    apdu_hex = "00 A4 04 04 10 A0 00 00 00 87 10 02 FF 86 FF 03 89 FF FF FF FF"
    smartcard_communicator.transmit_apdu(bytes.fromhex(apdu_hex))

    # 4. 获取响应数据
    apdu_hex = "00 C0 00 00 2E"
    smartcard_communicator.transmit_apdu(bytes.fromhex(apdu_hex))

    for i in range(3):
        print("Auc SQN: " + " ".join(f"0x{value:02X}" for value in milenageAuc.SQN))
        rand_hex = generate_random_hex_string(16)
        milenageAuc.RAND = np.frombuffer(bytes.fromhex(rand_hex), dtype=np.uint8)
        print("Generated RAND: " + " ".join(f"0x{value:02X}" for value in milenageAuc.RAND))
        milenageAuc.execute_OPc()
        print("Generated RES: " + " ".join(f"0x{value:02X}" for value in milenageAuc.RES))
        print("Generated CK: " + " ".join(f"0x{value:02X}" for value in milenageAuc.CK))
        print("Generated IK: " + " ".join(f"0x{value:02X}" for value in milenageAuc.IK))
        print("Generated AUTN: " + " ".join(f"0x{value:02X}" for value in milenageAuc.AUTN))

        # 5. 内部认证
        rand_hex = " ".join(f"{value:02X}" for value in milenageAuc.RAND)
        autn_hex = " ".join(f"{value:02X}" for value in milenageAuc.AUTN)
        apdu_hex = "00 88 00 81 22" + " 10 " + rand_hex + " 10 " + autn_hex
        response, sw1, sw2 = smartcard_communicator.transmit_apdu(bytes.fromhex(apdu_hex))

        if sw1 == 0x61 and sw2 == 0x10:
            # 获取ATUS响应数据
            apdu_hex = "00 C0 00 00 10"
            response, sw1, sw2 = smartcard_communicator.transmit_apdu(bytes.fromhex(apdu_hex))
            result = milenageAuc.verifyAUTS(np.array(response[2:16], dtype=np.uint8))
            if result:
                print("AUTS verification successful")
            else:
                print("AUTS verification failed")

        elif sw1 == 0x61 and sw2 == 0x35:
            # 获取RES
            apdu_hex = "00 C0 00 00 35"
            response, sw1, sw2 = smartcard_communicator.transmit_apdu(bytes.fromhex(apdu_hex))
            print("Authentication successful")
            print("Generated RES: " + " ".join(f"0x{value:02X}" for value in response[2:10]))
            print("Generated CK: " + " ".join(f"0x{value:02X}" for value in response[11:27]))
            print("Generated IK: " + " ".join(f"0x{value:02X}" for value in response[28:44]))
            print("Generated Kc: " + " ".join(f"0x{value:02X}" for value in response[45:53]))  # GSM兼容密钥

        elif sw1 == 0x98 and sw2 == 0x62:
            print("Illegal access attempted")

        else:
            print("Uknown attempted")

    smartcard_communicator.close_connection()
