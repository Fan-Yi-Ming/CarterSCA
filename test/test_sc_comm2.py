from sc.sc_reader import list_smartcard_readers, connect_smartcard_reader, transmit_apdu

# 15836141022
if __name__ == '__main__':
    # 列出所有读卡器
    reader_list = list_smartcard_readers()

    # 连接读卡器（默认使用第一个）
    connection = connect_smartcard_reader(reader_list, 0)

    # 如果连接成功，执行APDU命令
    if connection:
        # 1. 选择电信根目录
        transmit_apdu(connection, "00 A4 00 04 02 2F 00")

        # 2. 读取目录记录
        transmit_apdu(connection, "00 B2 01 04 26")

        # 3. 选择USIM应用
        transmit_apdu(connection, "00 A4 04 04 10 A0 00 00 00 87 10 02 FF 86 FF FF 89 FF FF FF FF")

        # 4. 获取响应数据
        transmit_apdu(connection, "00 C0 00 00 3C")

        # 5. 内部认证
        transmit_apdu(connection, "00 88 00 81 22 "
                                  "10 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F "  # RND
                                  "10 19 EB E6 AC DD AC 80 00 A4 A5 AE 92 5D D6 ED 45")  # AUTN

        print("操作完成!")
