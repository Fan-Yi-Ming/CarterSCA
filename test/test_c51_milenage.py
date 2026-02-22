from tools.sca import generate_random_hex_string
from uart.serial_reader import list_serial_ports, SerialCommunicator
from uart.uart_frame import create_uart_frame

if __name__ == '__main__':
    list_serial_ports()

    serial_communicator = SerialCommunicator()
    serial_communicator.open_connection(port="COM3", baudrate=115200, timeout=1.0)

    frame = create_uart_frame(0x04, bytes.fromhex(""))
    serial_communicator.send_and_receive(frame, 8)

    rand_hex = generate_random_hex_string(16)
    autn_hex = generate_random_hex_string(16)
    frame = create_uart_frame(0x05, bytes.fromhex(rand_hex + " " + autn_hex))
    received_bytes = serial_communicator.send_and_receive(frame, 48)
