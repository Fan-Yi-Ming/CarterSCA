import numpy as np

from tools.aes128 import Aes128


class MilenageAuc:
    def __init__(self):
        self.aes128 = Aes128()
        self.r1 = 64
        self.r2 = 0
        self.r3 = 32
        self.r4 = 64
        self.r5 = 96
        self.c1 = np.array(
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            dtype=np.uint8)
        self.c2 = np.array(
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01],
            dtype=np.uint8)
        self.c3 = np.array(
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02],
            dtype=np.uint8)
        self.c4 = np.array(
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04],
            dtype=np.uint8)
        self.c5 = np.array(
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08],
            dtype=np.uint8)

        self.K = None  # 用户密钥，128位，是函数 f1、f1、f2、f3、f4、f5 和 f5 的输入
        self.OP = None  # 运营商变体算法配置字段，128位，是函数 f1、f1、f2、f3、f4、f5 和 f5 的组成部分
        self.OPc = None  # 由 OP 和 K 推导得到的128位值，用于函数的计算过程中
        self.SQN = None  # 序列号，48位，是函数 f1 的输入
        self.AMF = None  # 认证管理字段，16位，是函数 f1 和 f1* 的输入
        self.RAND = None  # 随机数，128位

        self.TEMP = None  # 128位中间值，在函数的计算过程中使用

        self.AK = None  # 匿名密钥，48位，由函数 f5 或 f5* 输出
        self.MAC_A = None  # 网络认证码，64位，由函数 f1 输出
        self.MAC_S = None  # 再同步认证码，64位，由函数 f1* 输出

        self.RES = None  # 签名响应，64位，由函数 f2 输出
        self.CK = None  # 保密性密钥，128位，由函数 f3 输出
        self.IK = None  # 完整性密钥，128位，由函数 f4 输出

        self.AUTN = None  # 128位网络认证令牌AUTN

    def computeK(self):
        self.aes128.keyexpansion(self.K)

    def computeOPc(self):
        encrypt_result = self.aes128.encrypt(self.OP)
        self.OPc = np.bitwise_xor(self.OP, encrypt_result)

    def computeTEMP(self):
        xor_result = np.bitwise_xor(self.RAND, self.OPc)
        self.TEMP = self.aes128.encrypt(xor_result)

    def increaseSQN(self):
        for i in range(5, -1, -1):
            if self.SQN[i] < 0xFF:
                self.SQN[i] += 1
                for j in range(i + 1, 6):
                    self.SQN[j] = 0
                return
        self.SQN[:] = 0

    def f1(self):
        # MAC_A
        IN1 = np.zeros(16, dtype=np.uint8)
        IN1[0:6] = self.SQN
        IN1[6:8] = self.AMF
        IN1[8:14] = self.SQN
        IN1[14:16] = self.AMF

        xor_result = np.bitwise_xor(IN1, self.OPc)
        rot_result = np.roll(xor_result, -int(self.r1 / 8))
        xor_result = np.bitwise_xor(self.TEMP, rot_result)
        xor_result = np.bitwise_xor(xor_result, self.c1)
        encrypt_result = self.aes128.encrypt(xor_result)
        OUT1 = np.bitwise_xor(encrypt_result, self.OPc)
        self.MAC_A = OUT1[0:8]

    def f2345(self):
        # AK RES
        xor_result = np.bitwise_xor(self.TEMP, self.OPc)
        rot_result = np.roll(xor_result, -int(self.r2 / 8))
        xor_result = np.bitwise_xor(rot_result, self.c2)
        encrypt_result = self.aes128.encrypt(xor_result)
        OUT2 = np.bitwise_xor(encrypt_result, self.OPc)
        self.AK = OUT2[0:6]
        self.RES = OUT2[8:16]

        # CK
        xor_result = np.bitwise_xor(self.TEMP, self.OPc)
        rot_result = np.roll(xor_result, -int(self.r3 / 8))
        xor_result = np.bitwise_xor(rot_result, self.c3)
        encrypt_result = self.aes128.encrypt(xor_result)
        OUT3 = np.bitwise_xor(encrypt_result, self.OPc)
        self.CK = OUT3

        # IK
        xor_result = np.bitwise_xor(self.TEMP, self.OPc)
        rot_result = np.roll(xor_result, -int(self.r4 / 8))
        xor_result = np.bitwise_xor(rot_result, self.c4)
        encrypt_result = self.aes128.encrypt(xor_result)
        OUT4 = np.bitwise_xor(encrypt_result, self.OPc)
        self.IK = OUT4

    def f1star(self, SQNms: np.array, AMF: np.array):
        # MAC_S
        IN1 = np.zeros(16, dtype=np.uint8)
        IN1[0:6] = SQNms
        IN1[6:8] = AMF
        IN1[8:14] = SQNms
        IN1[14:16] = AMF

        xor_result = np.bitwise_xor(IN1, self.OPc)
        rot_result = np.roll(xor_result, -int(self.r1 / 8))
        xor_result = np.bitwise_xor(self.TEMP, rot_result)
        xor_result = np.bitwise_xor(xor_result, self.c1)
        encrypt_result = self.aes128.encrypt(xor_result)
        OUT1 = np.bitwise_xor(encrypt_result, self.OPc)
        self.MAC_S = OUT1[8:16]

    def f5star(self):
        # AK
        xor_result = np.bitwise_xor(self.TEMP, self.OPc)
        rot_result = np.roll(xor_result, -int(self.r5 / 8))
        xor_result = np.bitwise_xor(rot_result, self.c5)
        encrypt_result = self.aes128.encrypt(xor_result)
        OUT5 = np.bitwise_xor(encrypt_result, self.OPc)
        self.AK = OUT5[0:6]

    def computeAUTN(self):
        self.AUTN = np.zeros(16, dtype=np.uint8)
        self.AUTN[0:6] = np.bitwise_xor(self.SQN, self.AK)
        self.AUTN[6:8] = self.AMF
        self.AUTN[8:16] = self.MAC_A

    def execute(self):
        self.computeK()
        self.computeOPc()
        self.computeTEMP()
        self.f1()
        self.f2345()
        self.computeAUTN()
        self.increaseSQN()

    def execute_OPc(self):
        self.computeK()
        self.computeTEMP()
        self.f1()
        self.f2345()
        self.computeAUTN()
        self.increaseSQN()

    def verifyAUTS(self, ATUS: np.ndarray) -> bool:
        ATUS_AMF = np.array([0x00, 0x00], dtype=np.uint8)
        SQN_xor_AK = ATUS[0:6]
        MAC_S = ATUS[6:14]

        self.f5star()
        SQNms = np.bitwise_xor(SQN_xor_AK, self.AK)
        self.f1star(SQNms, ATUS_AMF)
        checkMAC_S_result = np.array_equal(self.MAC_S, MAC_S)
        if checkMAC_S_result:
            self.SQN = SQNms
            for i in range(32):
                self.increaseSQN()
            return True
        else:
            return False


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
    milenageAuc.RAND = np.array(
        [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F],
        dtype=np.uint8)

    milenageAuc.execute_OPc()
    print("Generated RES: " + " ".join(f"0x{value:02X}" for value in milenageAuc.RES))
    print("Generated CK: " + " ".join(f"0x{value:02X}" for value in milenageAuc.CK))
    print("Generated IK: " + " ".join(f"0x{value:02X}" for value in milenageAuc.IK))
    print("Generated AUTN: " + ", ".join(f"0x{value:02X}" for value in milenageAuc.AUTN))
    print()

    ATUS = np.array([0x82, 0x3F, 0x13, 0x95, 0xEA, 0x2A, 0xD3, 0x05, 0x34, 0xAD, 0x35, 0x4B, 0x46, 0x94],
                    dtype=np.uint8)
    result = milenageAuc.verifyAUTS(ATUS)
    if result:
        print("AUTS verification successful.")
    else:
        print("AUTS verification failed.")
    print()

    # 打印所有参数的数据类型
    print("参数数据类型:")
    print("=" * 50)
    params = [
        ("K", "用户密钥，128位"),
        ("OP", "运营商变体算法配置字段，128位"),
        ("OPc", "由 OP 和 K 推导得到的128位值"),
        ("SQN", "序列号，48位"),
        ("AMF", "认证管理字段，16位"),
        ("RAND", "随机数，128位"),
        ("TEMP", "128位中间值"),
        ("AK", "匿名密钥，48位"),
        ("MAC_A", "网络认证码，64位"),
        ("MAC_S", "再同步认证码，64位"),
        ("RES", "签名响应，64位"),
        ("CK", "保密性密钥，128位"),
        ("IK", "完整性密钥，128位"),
        ("AUTN", "128位网络认证令牌AUTN"),
    ]

    for param_name, description in params:
        param_value = getattr(milenageAuc, param_name)

        if hasattr(param_value, 'dtype'):
            dtype_info = str(param_value.dtype)
        else:
            dtype_info = f"No dtype attribute (type: {type(param_value).__name__})"

        # Now using fixed width works perfectly with English text
        print(f"{param_name:8} - dtype info: {dtype_info:40} | Description: {description}")
