import numpy as np

from milenage_auc import MilenageAuc
from milenage_usim import MilenageUsim
from tools.sca import generate_random_hex_string

if __name__ == '__main__':
    milenageAuc = MilenageAuc()
    milenageAuc.K = np.array(
        [0x46, 0x5b, 0x5c, 0xe8, 0xb1, 0x99, 0xb4, 0x9f, 0xaa, 0x5f, 0x0a, 0x2e, 0xe2, 0x38, 0xa6, 0xbc],
        dtype=np.uint8)
    milenageAuc.OP = np.array(
        [0xcd, 0xc2, 0x02, 0xd5, 0x12, 0x3e, 0x20, 0xf6, 0x2b, 0x6d, 0x67, 0x6a, 0xc7, 0x2c, 0xb3, 0x18],
        dtype=np.uint8)
    milenageAuc.SQN = np.array([0xff, 0x9b, 0xb4, 0xd0, 0xb6, 0x07], dtype=np.uint8)
    milenageAuc.AMF = np.array([0xb9, 0xb9], dtype=np.uint8)

    milenageUsim = MilenageUsim()
    milenageUsim.K = np.array(
        [0x46, 0x5b, 0x5c, 0xe8, 0xb1, 0x99, 0xb4, 0x9f, 0xaa, 0x5f, 0x0a, 0x2e, 0xe2, 0x38, 0xa6, 0xbc],
        dtype=np.uint8)
    milenageUsim.OP = np.array(
        [0xcd, 0xc2, 0x02, 0xd5, 0x12, 0x3e, 0x20, 0xf6, 0x2b, 0x6d, 0x67, 0x6a, 0xc7, 0x2c, 0xb3, 0x18],
        dtype=np.uint8)
    milenageUsim.SQN = np.array([0xff, 0x9b, 0xb4, 0xd0, 0xb6, 0x08], dtype=np.uint8)

    for i in range(3):
        print("Auc SQN: " + " ".join(f"0x{value:02X}" for value in milenageAuc.SQN))
        rand_hex = generate_random_hex_string(16)
        milenageAuc.RAND = np.frombuffer(bytes.fromhex(rand_hex), dtype=np.uint8)
        print("Generated RAND: " + " ".join(f"0x{value:02X}" for value in milenageAuc.RAND))
        milenageAuc.execute()
        print("Generated RES: " + " ".join(f"0x{value:02X}" for value in milenageAuc.RES))
        print("Generated CK: " + " ".join(f"0x{value:02X}" for value in milenageAuc.CK))
        print("Generated IK: " + " ".join(f"0x{value:02X}" for value in milenageAuc.IK))
        print("Generated AUTN: " + " ".join(f"0x{value:02X}" for value in milenageAuc.AUTN))
        print()

        print("USIM SQN: " + " ".join(f"0x{value:02X}" for value in milenageUsim.SQN))
        milenageUsim.RAND = milenageAuc.RAND
        milenageUsim.AUTN = milenageAuc.AUTN

        result = milenageUsim.execute()
        if result == 0:
            print("Generated RES: " + " ".join(f"0x{value:02X}" for value in milenageUsim.RES))
            print("Generated CK: " + " ".join(f"0x{value:02X}" for value in milenageUsim.CK))
            print("Generated IK: " + " ".join(f"0x{value:02X}" for value in milenageUsim.IK))
        elif result == 2:
            print("Generated AUTS: " + " ".join(f"0x{value:02X}" for value in milenageUsim.AUTS))
            result = milenageAuc.verifyAUTS(milenageUsim.AUTS)
            if result:
                print("AUTS verification successful.")
            else:
                print("AUTS verification failed.")
        elif result == 1:
            print("Illegal access attempted")
        print()
