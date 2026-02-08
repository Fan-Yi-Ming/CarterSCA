import numpy as np
from tools.aes import aes_keyexpansion, Aes


class Aes128:
    def __init__(self):
        self.Nr = 10
        self.aes = Aes()
        self.w = None

    def keyexpansion(self, key: np.array):
        if len(key) != 16:
            raise ValueError("key must be 16 bytes")
        self.w = aes_keyexpansion(key)

    def encrypt(self, input_arr: np.array):
        self.aes.set_state(input_arr)

        # 初始轮
        self.aes.add_roundkey(self.w[0:4])

        # 主轮循环
        for i in range(1, self.Nr):
            self.aes.sub_state()
            self.aes.shift_rows()
            self.aes.mix_columns()
            self.aes.add_roundkey(self.w[4 * i:4 * i + 4])

        # 最终轮
        self.aes.sub_state()
        self.aes.shift_rows()
        self.aes.add_roundkey(self.w[4 * self.Nr:4 * self.Nr + 4])

        output_arr = self.aes.get_state()
        return output_arr

    def decrypt(self, input_arr: np.array):
        self.aes.set_state(input_arr)

        # 初始轮
        self.aes.add_roundkey(self.w[4 * self.Nr:4 * self.Nr + 4])

        # 主轮循环
        for i in range(self.Nr - 1, 0, -1):
            self.aes.inv_shift_rows()
            self.aes.inv_sub_state()
            self.aes.add_roundkey(self.w[4 * i:4 * i + 4])
            self.aes.inv_mix_columns()

        # 最终轮
        self.aes.inv_shift_rows()
        self.aes.inv_sub_state()
        self.aes.add_roundkey(self.w[0:4])

        output_arr = self.aes.get_state()
        return output_arr


if __name__ == '__main__':
    aes128 = Aes128()
    key128 = np.frombuffer(bytes.fromhex("2B 7E 15 16 28 AE D2 A6 AB F7 15 88 09 CF 4F 3C"), dtype=np.uint8)
    aes128.keyexpansion(key128)

    input_arr = [0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34]
    output_arr = aes128.encrypt(input_arr)
    hex_str = " ".join(f"0x{value:02X}" for value in output_arr)
    print(hex_str)

    input_arr = [0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb, 0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32]
    output_arr = aes128.decrypt(input_arr)
    hex_str = " ".join(f"0x{value:02X}" for value in output_arr)
    print(hex_str)
