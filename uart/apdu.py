class CommandAPDU:
    # 错误码定义
    ERR_NULL = 1
    ERR_LENGTH = 2
    SUCCESS = 0

    def __init__(self):
        # APDU字段定义
        self.cla = 0
        self.ins = 0
        self.p1 = 0
        self.p2 = 0
        self.lc = 0
        self.le = 0
        self.has_data = False
        self.has_le = False
        self.data = bytes()

    def check(self, command_apdu_bytes: bytes) -> int:
        """
        检查APDU字节数据

        Args:
            command_apdu_bytes: APDU字节数据 (bytes类型)

        Returns:
            int: 返回码 (SUCCESS, ERR_NULL, ERR_LENGTH)
        """
        # 参数合法性检查
        if command_apdu_bytes is None or len(command_apdu_bytes) < 4:
            return self.ERR_NULL

        # 清空现有数据
        self.cla = 0
        self.ins = 0
        self.p1 = 0
        self.p2 = 0
        self.lc = 0
        self.le = 0
        self.has_data = False
        self.has_le = False
        self.data = bytes()

        # 解析APDU固定头部
        self.cla = command_apdu_bytes[0]
        self.ins = command_apdu_bytes[1]
        self.p1 = command_apdu_bytes[2]
        self.p2 = command_apdu_bytes[3]

        # 根据APDU长度判断Case类型
        if len(command_apdu_bytes) == 4:
            # Case 1: 仅有头部，无数据域和Le
            self.has_data = False
            self.has_le = False
        elif len(command_apdu_bytes) == 5:
            # Case 2: 有Le字段，无数据域
            self.has_data = False
            self.has_le = True
            self.le = command_apdu_bytes[4]
        else:
            # Case 3/4: 包含数据域
            self.lc = command_apdu_bytes[4]  # 数据长度
            offset = 5

            # 验证APDU总长度是否匹配
            if offset + self.lc <= len(command_apdu_bytes):
                # Case 3: 有数据域，无Le字段
                if offset + self.lc == len(command_apdu_bytes):
                    self.has_data = True
                    self.has_le = False
                    self.data = command_apdu_bytes[offset:offset + self.lc]
                # Case 4: 有数据域和Le字段
                elif offset + self.lc + 1 == len(command_apdu_bytes):
                    self.has_data = True
                    self.has_le = True
                    self.data = command_apdu_bytes[offset:offset + self.lc]
                    self.le = command_apdu_bytes[offset + self.lc]
                else:
                    return self.ERR_LENGTH
            else:
                return self.ERR_LENGTH

        return self.SUCCESS
