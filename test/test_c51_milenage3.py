import numpy as np
from uart.serial_reader import list_available_serial_ports, SerialCommunicator
from uart.uart_frame import create_uart_frame, parse_uart_frame
from lib3gpp.milenage_auc import MilenageAuc
from tools.sca import generate_random_hex_string

if __name__ == '__main__':
    milenageAuc = MilenageAuc()

    milenageAuc.K = np.array(
        [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F],
        dtype=np.uint8)
    milenageAuc.OPc = np.array(
        [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF],
        dtype=np.uint8)
    milenageAuc.SQN = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0xFF], dtype=np.uint8)
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
            rand_hex = " ".join(f"{value:02X}" for value in milenageAuc.RAND)
            autn_hex = " ".join(f"{value:02X}" for value in milenageAuc.AUTN)

            frame = create_uart_frame(0x05, bytes.fromhex(rand_hex + " " + autn_hex))
            received_bytes = serial_communicator.send_and_receive(frame, 48)
            command, data = parse_uart_frame(received_bytes)

            if command == 5:
                print("Authentication successful")
                print("Generated RES: " + " ".join(f"0x{value:02X}" for value in data[0:8]))
                print("Generated CK: " + " ".join(f"0x{value:02X}" for value in data[8:24]))
                print("Generated IK: " + " ".join(f"0x{value:02X}" for value in data[24:40]))

            elif command == 6:
                print("Illegal access attempted")

            elif command == 7:
                result = milenageAuc.verifyAUTS(np.frombuffer(data[0:14], dtype=np.uint8))
                if result:
                    print("AUTS verification successful")
                else:
                    print("AUTS verification failed")

            else:
                print("Uknown attempted")

            print()
