import gc
import math
import struct
import time
import pyvisa
from pyvisa import constants
from trsfile import trs_open
from trsfile import Trace, SampleCoding, TracePadding
from trsfile.parametermap import TraceParameterMap

if __name__ == '__main__':
    gatherer_sds804x_resource_name = "TCPIP0::169.254.114.206::inst0::INSTR"
    gatherer_sds804x_ref_channel_name = "C1"
    gatherer_sds804x_arm_delay = 0.1
    gatherer_sds804x_acquisition_timeout = 5.0
    gatherer_sds804x_acquisition_times = 10
    gatherer_sds804x_traceset_path = "D:\\traceset\\aes128_en.trs"

    rm = pyvisa.ResourceManager()

    sds804x = rm.open_resource(
        resource_name=gatherer_sds804x_resource_name,
        access_mode=constants.AccessModes.exclusive_lock,
        open_timeout=5000,
        timeout=5000,
        chunk_size=20 * 1024 * 1024,
    )

    trigger_status = sds804x.query(":TRIGger:STATus?").strip()
    if trigger_status != "Stop":

        sds804x.write(":TRIGger:MODE SINGle")
        while True:
            trigger_mode = sds804x.query(":TRIGger:MODE?").strip()
            if trigger_mode == "SINGle":
                break
            time.sleep(0.01)

        sds804x.write(":TRIGger:MODE FTRIG")

        while True:
            trigger_status = sds804x.query(":TRIGger:STATus?").strip()
            if trigger_status == "Stop":
                break
            time.sleep(0.01)

    sds804x.write(":TRIGger:MODE SINGle")

    # sds804x.write(":TRIGger:RUN")
    # sds804x.write(":TRIGger:MODE SINGle")
    # sds804x.write(":WAVeform:BYTeorder LSB")
    # temp = sds804x.query(":WAVeform:BYTeorder?").strip()
    # print(temp)
    #
    # sds804x.write(":ACQuire:TYPE NORMal")
    # temp = sds804x.query(":ACQuire:TYPE?").strip()
    # print(temp)
    #
    # sds804x.write(":ACQuire:RESolution 8Bits")
    # temp = sds804x.query(":ACQuire:RESolution?").strip()
    # print(temp)
    #
    # sds804x.write("ACQ:MODE YT")
    # temp = sds804x.query("ACQ:MODE?").strip()
    # print(temp)
    #
    # sds804x.write(":WAVeform:SOURce C1")
    # temp = sds804x.query(":WAVeform:SOURce?").strip()
    # print(temp)
    #
    # temp = sds804x.query(f":ACQuire:POINts?").strip()
    # print(float(temp))
    #
    # sds804x.write(":WAVeform:PREamble?")
    # received_bytes = sds804x.read_raw()
    # hex_dump(received_bytes)
    #
    # temp = sds804x.query(":WAVeform:MAXPoint?").strip()
    # print(temp)
    #
    # sds804x.write(":ACQuire:MDEPth 10k")
    # temp = sds804x.query(":ACQuire:MDEPth?").strip()
    # print(temp)
    #
    # sds804x.write(":WAVeform:STARt 0")
    # temp = sds804x.query(":WAVeform:STARt?").strip()
    # print(temp)
    #
    # temp = sds804x.query(":TRIGger:STATus?").strip()
    # print(temp)
    #
    # sds804x.write(":WAVeform:INTerval 1")
    # temp = sds804x.query(":WAVeform:INTerval?").strip()
    # print(temp)
    #
    # sds804x.write(":WAVeform:POINt 100000")
    # temp = sds804x.query(":WAVeform:POINt?").strip()
    # print(temp)
    #
    # sds804x.write(":WAVeform:WIDTh WORD")
    # temp = sds804x.query(":WAVeform:WIDTh?").strip()
    # print(temp)

    sds804x.close()
