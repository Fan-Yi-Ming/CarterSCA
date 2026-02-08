import numpy as np

from milenage_auc import MilenageAuc
from milenage_usim import MilenageUsim
from tools.sca import generate_random_hex_string

if __name__ == '__main__':
    milenageAuc = MilenageAuc()
    milenageAuc.K = np.array(
        [0xD3, 0x1D, 0xF5, 0xE3, 0xD8, 0xB4, 0x17, 0x01, 0x2E, 0x2F, 0x7C, 0x93, 0x6D, 0x75, 0x57, 0xA7],
        dtype=np.uint8)
    milenageAuc.OPc = np.array(
        [0xEF, 0x87, 0xDE, 0xEF, 0x2D, 0xC1, 0xD5, 0xC2, 0x71, 0xE7, 0x7D, 0x15, 0x80, 0x20, 0xA1, 0x35],
        dtype=np.uint8)
    milenageAuc.SQN = np.array([0x00, 0x00, 0x00, 0x66, 0xEB, 0x5C], dtype=np.uint8)
    milenageAuc.AMF = np.array([0x80, 0x00], dtype=np.uint8)

    milenageUsim = MilenageUsim()
    milenageUsim.K = np.array(
        [0xD3, 0x1D, 0xF5, 0xE3, 0xD8, 0xB4, 0x17, 0x01, 0x2E, 0x2F, 0x7C, 0x93, 0x6D, 0x75, 0x57, 0xA7],
        dtype=np.uint8)
    milenageUsim.OPc = np.array(
        [0xEF, 0x87, 0xDE, 0xEF, 0x2D, 0xC1, 0xD5, 0xC2, 0x71, 0xE7, 0x7D, 0x15, 0x80, 0x20, 0xA1, 0x35],
        dtype=np.uint8)
    milenageUsim.SQN = np.array([0x00, 0x00, 0x00, 0x66, 0xEB, 0x5D], dtype=np.uint8)

    for i in range(3):
        print("Auc SQN: " + " ".join(f"0x{value:02X}" for value in milenageAuc.SQN))
        rand_hex = generate_random_hex_string(16)
        milenageAuc.RAND = np.frombuffer(bytes.fromhex(rand_hex), dtype=np.uint8)
        print("Generated RAND: " + " ".join(f"0x{value:02X}" for value in milenageAuc.RAND))
        milenageAuc.execute_OPc()
        print("Generated RES: " + " ".join(f"0x{value:02X}" for value in milenageAuc.RES))
        print("Generated CK: " + " ".join(f"0x{value:02X}" for value in milenageAuc.CK))
        print("Generated IK: " + " ".join(f"0x{value:02X}" for value in milenageAuc.IK))
        print("Generated AUTN: " + " ".join(f"0x{value:02X}" for value in milenageAuc.AUTN))
        print()

        print("USIM SQN: " + " ".join(f"0x{value:02X}" for value in milenageUsim.SQN))
        milenageUsim.RAND = milenageAuc.RAND
        milenageUsim.AUTN = milenageAuc.AUTN

        result = milenageUsim.execute_OPc()
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
