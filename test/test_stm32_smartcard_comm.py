import time

from uart.serial_reader import list_serial_ports, SerialCommunicator
from uart.uart_frame import create_uart_frame

# 772057371(旧) 990395338
if __name__ == '__main__':
    list_serial_ports()

    serial_communicator = SerialCommunicator()
    serial_communicator.open_connection(port="COM1", baudrate=115200, timeout=1.0)

    # 发送复位指令
    frame = create_uart_frame(0x01, bytes.fromhex(""))
    serial_communicator.send_and_receive(frame, 8)

    apdu_hex = "00 A4 00 04 02 2F 00"
    serial_communicator.transmit_apdu(0x02, bytes.fromhex(apdu_hex))

    apdu_hex = "00 B2 01 04 26"
    serial_communicator.transmit_apdu(0x02, bytes.fromhex(apdu_hex))

    apdu_hex = "00 A4 04 04 10 A0 00 00 00 87 10 02 FF 49 FF FF 89 04 0B 00 FF"
    serial_communicator.transmit_apdu(0x02, bytes.fromhex(apdu_hex))

    apdu_hex = "00 C0 00 00 39"
    serial_communicator.transmit_apdu(0x02, bytes.fromhex(apdu_hex))

    for i in range(10):
        apdu_hex = ("00 88 00 81 22 "
                    "10 A2 46 FE BA 13 12 FA 90 EC E1 12 C4 B6 AF B1 66 "
                    "10 9B 52 FC 43 9E DF 72 4C FD 08 68 4B DC AC FF D7")
        serial_communicator.transmit_apdu(0x03, bytes.fromhex(apdu_hex))
        time.sleep(0.1)
