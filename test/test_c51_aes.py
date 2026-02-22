from uart.serial_reader import list_serial_ports, SerialCommunicator
from uart.uart_frame import create_uart_frame

if __name__ == '__main__':
    list_serial_ports()

    serial_communicator = SerialCommunicator()
    serial_communicator.open_connection(port="COM3", baudrate=115200, timeout=1.0)

    data_hex = "2B 7E 15 16 28 AE D2 A6 AB F7 15 88 09 CF 4F 3C"
    frame = create_uart_frame(0x01, bytes.fromhex(data_hex))
    serial_communicator.send_and_receive(frame, 8)

    data_hex = "32 43 F6 A8 88 5A 30 8D 31 31 98 A2 E0 37 07 34"
    frame = create_uart_frame(0x02, bytes.fromhex(data_hex))
    serial_communicator.send_and_receive(frame, 24)

    data_hex = "39 25 84 1D 02 DC 09 FB DC 11 85 97 19 6A 0B 32"
    frame = create_uart_frame(0x03, bytes.fromhex(data_hex))
    serial_communicator.send_and_receive(frame, 24)

    serial_communicator.close_connection()
