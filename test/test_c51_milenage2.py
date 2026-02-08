import numpy as np
from uart.serial_reader import list_available_serial_ports, SerialCommunicator
from uart.uart_frame import create_uart_frame
from lib3gpp.milenage_auc import MilenageAuc
from tools.sca import generate_random_hex_string

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

    list_available_serial_ports()

    # 使用上下文管理器，确保串口正确关闭
    with SerialCommunicator() as serial_communicator:
        # 设置超时时间为1秒
        serial_communicator.open_connection(port="COM3", baudrate=115200, timeout=1.0)

        frame = create_uart_frame(0x04, bytes.fromhex(""))
        serial_communicator.send_and_receive(frame, 8)

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
            RAND_hex = " ".join(f"{value:02X}" for value in milenageAuc.RAND)
            AUTN_hex = " ".join(f"{value:02X}" for value in milenageAuc.AUTN)

            cmd, data = serial_communicator.transmit(0x05, bytes.fromhex(RAND_hex + " " + AUTN_hex))
            if cmd == 5:
                print("Authentication successful")
                print("Generated RES: " + " ".join(f"0x{value:02X}" for value in data[0:8]))
                print("Generated CK: " + " ".join(f"0x{value:02X}" for value in data[8:24]))
                print("Generated IK: " + " ".join(f"0x{value:02X}" for value in data[24:40]))

            elif cmd == 6:
                print("Illegal access attempted")

            elif cmd == 7:
                result = milenageAuc.verifyAUTS(np.frombuffer(data[0:14], dtype=np.uint8))
                if result:
                    print("AUTS verification successful")
                else:
                    print("AUTS verification failed")

            else:
                print("Uknown attempted")

            print()
