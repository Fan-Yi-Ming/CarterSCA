import numpy as np

from tools.aes128 import Aes128


class MilenageUsim:
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
        self.AUTS = None  # 112位同步令牌AUTS

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

    def f1(self, SQN: np.ndarray):
        # MAC_A
        IN1 = np.zeros(16, dtype=np.uint8)
        IN1[0:6] = SQN
        IN1[6:8] = self.AMF
        IN1[8:14] = SQN
        IN1[14:16] = self.AMF

        xor_result = np.bitwise_xor(IN1, self.OPc)
        rot_result = np.roll(xor_result, -int(self.r1 / 8))
        xor_result = np.bitwise_xor(self.TEMP, rot_result)
        xor_result = np.bitwise_xor(xor_result, self.c1)
        encrypt_result = self.aes128.encrypt(xor_result)
        OUT1 = np.bitwise_xor(encrypt_result, self.OPc)
        self.MAC_A = OUT1[0:8]

    def checkSQN(self, SQN: np.array):
        # 检查接收到的SQN是否在正向64窗口内
        received_num = int.from_bytes(SQN.tobytes(), 'big')
        current_num = int.from_bytes(self.SQN.tobytes(), 'big')

        # 完全相等
        if received_num == current_num:
            print("SQN is OK")
            return True

        max_value = 0xFFFFFFFFFFFF  # 6字节最大值（48位）

        # 计算正向距离
        if received_num >= current_num:
            # 没有回绕或刚好回绕
            forward_distance = received_num - current_num
        else:
            # 发生回绕
            forward_distance = (max_value - current_num + 1) + received_num

        # 检查是否在正向64窗口内
        if forward_distance <= 64:
            print(f"SQN is OK")
            return True
        else:
            print(f"SQN is Error")
            return False

    def checkMAC_A(self, MAC_A: np.array):
        if np.array_equal(self.MAC_A, MAC_A):
            print("MAC_A is OK")
            return True
        else:
            print("MAC_A is Error")
            return False

    def kernel(self):
        # AK RES
        xor_result = np.bitwise_xor(self.TEMP, self.OPc)
        rot_result = np.roll(xor_result, -int(self.r2 / 8))
        xor_result = np.bitwise_xor(rot_result, self.c2)
        encrypt_result = self.aes128.encrypt(xor_result)
        OUT2 = np.bitwise_xor(encrypt_result, self.OPc)
        self.AK = OUT2[0:6]
        self.RES = OUT2[8:16]

        # check SQN
        SQN_xor_AK = self.AUTN[0:6]
        SQN = np.bitwise_xor(SQN_xor_AK, self.AK)
        checkSQN_result = self.checkSQN(SQN)

        # MAC_A
        self.AMF = self.AUTN[6:8]
        self.f1(SQN)

        # check MAC_A
        MAC_A = self.AUTN[8:16]
        checkMAC_A_result = self.checkMAC_A(MAC_A)

        print("MAC_A: " + " ".join(f"0x{value:02X}" for value in self.MAC_A))

        # final check
        if not checkMAC_A_result:
            return 1
        if not checkSQN_result:
            self.computeAUTS()
            self.increaseSQN()
            return 2
        else:
            self.SQN = SQN

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

        self.increaseSQN()
        return 0

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

    def execute(self):
        self.computeK()
        self.computeOPc()
        self.computeTEMP()
        return self.kernel()

    def execute_OPc(self):
        self.computeK()
        self.computeTEMP()
        return self.kernel()

    def computeAUTS(self):
        AUTS_AMF = np.array([0x00, 0x00], dtype=np.uint8)
        self.AUTS = np.zeros(14, dtype=np.uint8)
        self.f5star()
        self.f1star(self.SQN, AUTS_AMF)
        self.AUTS[0:6] = np.bitwise_xor(self.SQN, self.AK)
        self.AUTS[6:14] = self.MAC_S


if __name__ == '__main__':
    milenageUsim = MilenageUsim()

    milenageUsim.K = np.array(
        [0xD3, 0x1D, 0xF5, 0xE3, 0xD8, 0xB4, 0x17, 0x01, 0x2E, 0x2F, 0x7C, 0x93, 0x6D, 0x75, 0x57, 0xA7],
        dtype=np.uint8)
    milenageUsim.OPc = np.array(
        [0xEF, 0x87, 0xDE, 0xEF, 0x2D, 0xC1, 0xD5, 0xC2, 0x71, 0xE7, 0x7D, 0x15, 0x80, 0x20, 0xA1, 0x35],
        dtype=np.uint8)
    milenageUsim.SQN = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00], dtype=np.uint8)
    milenageUsim.RAND = np.array(
        [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F],
        dtype=np.uint8)
    milenageUsim.AUTN = np.array(
        [0x19, 0xEB, 0xE6, 0xCA, 0x36, 0xF7, 0x80, 0x00, 0x62, 0xA6, 0x39, 0x33, 0xF8, 0x39, 0xC0, 0x19],
        dtype=np.uint8)

    result = milenageUsim.execute_OPc()
    if result == 0:
        print("Generated RES: " + " ".join(f"0x{value:02X}" for value in milenageUsim.RES))
        print("Generated CK: " + " ".join(f"0x{value:02X}" for value in milenageUsim.CK))
        print("Generated IK: " + " ".join(f"0x{value:02X}" for value in milenageUsim.IK))
    if result == 2:
        print("Generated AUTS: " + " ".join(f"0x{value:02X}" for value in milenageUsim.AUTS))
    if result == 1:
        print("Illegal access attempted")
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
        ("AUTS", "112位同步令牌AUTS")
    ]

    for param_name, description in params:
        param_value = getattr(milenageUsim, param_name)

        if hasattr(param_value, 'dtype'):
            dtype_info = str(param_value.dtype)
        else:
            dtype_info = f"No dtype attribute (type: {type(param_value).__name__})"

        # Now using fixed width works perfectly with English text
        print(f"{param_name:8} - dtype info: {dtype_info:40} | Description: {description}")
