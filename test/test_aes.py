import numpy as np
from tools.aes import Aes, aes_keyexpansion


def aes_test():
    """
    AES-128加密解密过程完整演示
    包含完整的加密流程和对应的解密流程
    """
    # 初始化AES加密对象
    aes = Aes()
    rounds = 10  # AES-128加密轮数

    # ============================ 加密参数配置 ============================
    # AES-128加密密钥（16字节）
    key_hex = "2B 7E 15 16 28 AE D2 A6 AB F7 15 88 09 CF 4F 3C"
    # 输入明文数据（16字节）
    plaintext_hex = "32 43 F6 A8 88 5A 30 8D 31 31 98 A2 E0 37 07 34"

    # ============================ 数据预处理 ============================
    # 移除空格后转换为字节数组
    key = np.frombuffer(bytes.fromhex(key_hex.replace(" ", "")), dtype=np.uint8)
    plaintext_data = np.frombuffer(bytes.fromhex(plaintext_hex.replace(" ", "")), dtype=np.uint8)

    print("=" * 60)
    print("AES-128 加密过程")
    print("=" * 60)
    print(f"加密密钥: {key_hex}")
    print(f"输入明文: {plaintext_hex}")

    # ============================ 密钥扩展 ============================
    # 执行AES密钥扩展，生成轮密钥
    w = aes_keyexpansion(key)

    # ============================ 加密过程 ============================
    # 设置AES状态矩阵（初始明文数据）
    aes.set_state(plaintext_data)

    # 第0轮：初始轮密钥加
    aes.add_roundkey(w[0:4])

    # 主轮循环：第1轮到第9轮（共9轮）
    for j in range(1, rounds):
        aes.sub_state()  # 字节代换（S盒变换）
        aes.shift_rows()  # 行移位操作
        aes.mix_columns()  # 列混合操作
        aes.add_roundkey(w[4 * j:4 * j + 4])  # 轮密钥加

    # 最终轮：第10轮（不包含列混合操作）
    aes.sub_state()  # 字节代换（S盒变换）
    aes.shift_rows()  # 行移位操作
    aes.add_roundkey(w[4 * rounds:4 * rounds + 4])  # 最终轮密钥加

    # 获取加密后的密文数据
    ciphertext_data = aes.get_state()
    ciphertext_hex = ' '.join(f'{byte:02X}' for byte in ciphertext_data)

    print(f"输出密文: {ciphertext_hex}")
    print("加密完成!")

    # ============================ 解密过程 ============================
    print("\n" + "=" * 60)
    print("AES-128 解密过程")
    print("=" * 60)
    print(f"解密密钥: {key_hex}")
    print(f"输入密文: {ciphertext_hex}")

    # 重新设置AES状态为密文数据
    aes.set_state(ciphertext_data)

    # 第0轮：初始轮密钥加（使用最后一轮轮密钥）
    aes.add_roundkey(w[4 * rounds:4 * rounds + 4])

    # 主轮循环：第9轮到第1轮（逆序）
    for j in range(rounds - 1, 0, -1):
        aes.inv_shift_rows()  # 逆行移位
        aes.inv_sub_state()  # 逆字节代换（逆S盒）
        aes.add_roundkey(w[4 * j:4 * j + 4])  # 轮密钥加
        aes.inv_mix_columns()  # 逆列混合

    # 最终轮：第0轮（不包含逆列混合操作）
    aes.inv_shift_rows()  # 逆行移位
    aes.inv_sub_state()  # 逆字节代换（逆S盒）
    aes.add_roundkey(w[0:4])  # 初始轮密钥加

    # 获取解密后的明文数据
    decrypted_data = aes.get_state()
    decrypted_hex = ' '.join(f'{byte:02X}' for byte in decrypted_data)

    print(f"解密明文: {decrypted_hex}")
    print("解密完成!")


if __name__ == "__main__":
    # 执行完整的加解密演示
    aes_test()
