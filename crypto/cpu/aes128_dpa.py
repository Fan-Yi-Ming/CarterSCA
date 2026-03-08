import os
import numpy as np
import trsfile
import time
from datetime import datetime
from trsfile import SampleCoding, Trace, Header, TracePadding
from trsfile.parametermap import TraceParameterMap, TraceSetParameterMap, TraceParameterDefinitionMap
from typing import Tuple
from multiprocessing import Pool, cpu_count
from tools.aes import Aes, aes_inv_keyexpansion
from tools.sca import d_func, index_str_to_range, rank_sbox_key_guesses, report_sbox_key_guesses


def process_single_trace(traceset_path, trace_index,
                         sample_first_pos, sample_number,
                         sbox_num, sbox_size,
                         d_fuc_mode, crypto_direction) -> Tuple[np.array, np.array]:
    traceset = trsfile.open(traceset_path, 'r')
    trace = traceset[trace_index]
    traceset.close()

    sample_arr = np.array(trace[sample_first_pos:sample_first_pos + sample_number], dtype=np.float32)
    data_arr_2d = np.zeros((sbox_num, sbox_size), dtype=np.uint8)

    aes = Aes()
    input_arr = np.frombuffer(bytes(trace.parameters["INPUT"].value), dtype=np.uint8)[0:16]
    Nb = 4
    wi = np.zeros(Nb, dtype=np.uint32)

    if crypto_direction == 0:
        for i in range(sbox_size):
            aes.set_state(input_arr)

            # 构造猜测轮密钥
            wi[0] = (i << 24) | (i << 16) | (i << 8) | i
            wi[1] = wi[0]
            wi[2] = wi[0]
            wi[3] = wi[0]

            # 执行第一轮加密操作
            aes.add_roundkey(wi)
            aes.sub_state()

            # 获取中间值并计算差分分组
            intermediate_arr = aes.get_state()
            d_func_intermediate_arr = np.array([d_func(byte, d_fuc_mode) for byte in intermediate_arr], dtype=np.uint8)
            data_arr_2d[:, i] = d_func_intermediate_arr

    if crypto_direction == 1:
        for i in range(sbox_size):
            aes.set_state(input_arr)

            # 构造猜测轮密钥
            wi[0] = (i << 24) | (i << 16) | (i << 8) | i
            wi[1] = wi[0]
            wi[2] = wi[0]
            wi[3] = wi[0]

            # 执行第一轮解密操作
            aes.add_roundkey(wi)
            aes.inv_shift_rows()
            aes.inv_sub_state()

            # 获取中间值并计算差分分组
            intermediate_arr = aes.get_state()
            d_func_intermediate_arr = np.array([d_func(byte, d_fuc_mode) for byte in intermediate_arr], dtype=np.uint8)
            data_arr_2d[:, i] = d_func_intermediate_arr

    return sample_arr, data_arr_2d


class Aes128DPA:

    def __init__(self):
        self.candidates: int = 4
        self.process_number: int = cpu_count()

        self.trace_number: int = 0
        self.sample_first_pos: int = 0
        self.sample_number: int = 0

        self.sbox_num: int = 16
        self.sbox_size: int = 256
        self.sbox_index_arr = None

        self.d_fuc_mode: int = 0  # 差分分析D函数工作模式
        self.crypto_direction: int = 0  # 0加密 1解密

        self.sbox_key_result_path = "./aes128_dpa_sbox_key_result.npz"
        # 攻击结果数组: [sbox_num, candidates]
        self.sbox_key_arr_2d = np.zeros((self.sbox_num, self.candidates), dtype=np.uint8)
        self.sbox_keyvalue_arr_2d = np.zeros((self.sbox_num, self.candidates), dtype=np.float32)
        self.sbox_keypos_arr_2d = np.zeros((self.sbox_num, self.candidates), dtype=np.int32)

        self.traceset = None
        self.traceset_path: str = ""
        self.traceset2_switch: bool = True
        self.traceset2 = None

        # 分析数据数组
        self.data_arr_3d = None  # 中间值数组: [trace_number, sbox_num, sbox_size]
        self.sample_arr_2d = None  # 样本点数据: [trace_number, sample_number]

    def init_process(self):
        self.open_traceset()
        self.open_traceset2()
        self.load_sbox_key_result()
        self.data_arr_3d = np.zeros((self.trace_number, self.sbox_num, self.sbox_size), dtype=np.uint8)
        self.sample_arr_2d = np.zeros((self.trace_number, self.sample_number), dtype=np.float32)

    def finish_process(self):
        self.save_sbox_key_result()
        self.report()
        self.recovery_key()
        self.close_traceset()
        self.close_traceset2()

    def open_traceset(self):
        self.traceset = trsfile.open(self.traceset_path, 'r')
        self.trace_number = self.traceset.get_headers()[Header.NUMBER_TRACES]
        number_samples_per_trace = self.traceset.get_headers()[Header.NUMBER_SAMPLES]

        if self.sample_first_pos < 0:
            self.sample_first_pos = 0
            print(f"自动调整样本点起始位置为: 0")

        if self.sample_first_pos + self.sample_number > number_samples_per_trace:
            self.sample_number = number_samples_per_trace - self.sample_first_pos
            print(f"自动调整样本点数量为: {self.sample_number}")

    def open_traceset2(self):
        if self.traceset2_switch:
            traceset_parameter_map = TraceSetParameterMap()

            trace_parameter_definition_map = TraceParameterDefinitionMap()

            headers = {
                Header.TRS_VERSION: 2,
                Header.TITLE_SPACE: 255,
                Header.SAMPLE_CODING: SampleCoding.FLOAT,
                Header.LABEL_X: " ",
                Header.SCALE_X: 1.0,
                Header.LABEL_Y: " ",
                Header.SCALE_Y: 1.0,
                Header.TRACE_SET_PARAMETERS: traceset_parameter_map,
                Header.TRACE_PARAMETER_DEFINITIONS: trace_parameter_definition_map
            }

            base_name = os.path.splitext(os.path.basename(self.traceset_path))[0]
            timestamp = datetime.now().strftime("%H%M%S")
            new_base_name = f"{base_name}+Aes128DPA({timestamp})"
            traceset2_path = os.path.join(os.path.dirname(self.traceset_path),
                                          new_base_name + os.path.splitext(self.traceset_path)[1])

            self.traceset2 = trsfile.trs_open(
                path=traceset2_path,
                mode='w',
                engine='TrsEngine',
                headers=headers,
                padding_mode=TracePadding.AUTO,
                live_update=True
            )

    def close_traceset(self):
        if self.traceset:
            self.traceset.close()

    def close_traceset2(self):
        if self.traceset2_switch:
            if self.traceset2:
                self.traceset2.close()

    def report(self):
        for sbox_index in self.sbox_index_arr:
            report_sbox_key_guesses(sbox_index,
                                    self.sbox_key_arr_2d[sbox_index],
                                    self.sbox_keyvalue_arr_2d[sbox_index],
                                    self.sbox_keypos_arr_2d[sbox_index],
                                    self.sample_first_pos)

    def save_sbox_key_result(self):
        try:
            np.savez(
                self.sbox_key_result_path,
                sbox_key_arr_2d=self.sbox_key_arr_2d,
                sbox_keyvalue_arr_2d=self.sbox_keyvalue_arr_2d,
                sbox_keypos_arr_2d=self.sbox_keypos_arr_2d
            )
            print(f"Sbox分析结果已保存")
        except Exception as e:
            print(f"Sbox分析结果保存失败: {e}")

    def load_sbox_key_result(self):
        if not os.path.exists(self.sbox_key_result_path):
            return

        try:
            data = np.load(self.sbox_key_result_path)
            self.sbox_key_arr_2d = data['sbox_key_arr_2d']
            self.sbox_keyvalue_arr_2d = data['sbox_keyvalue_arr_2d']
            self.sbox_keypos_arr_2d = data['sbox_keypos_arr_2d']
            print(f"Sbox分析结果已加载")
            if self.sbox_key_arr_2d.shape[1] != self.candidates:
                self.candidates = self.sbox_key_arr_2d.shape[1]
                print(f"candidates 已调整为 {self.candidates} 以匹配历史数据。"
                      f"如需重置candidates，请删除: {self.sbox_key_result_path}")
        except Exception as e:
            print(f"Sbox分析结果加载失败: {e}")
            try:
                os.remove(self.sbox_key_result_path)
                print(f"已删除损坏文件: {self.sbox_key_result_path}")
            except Exception as delete_error:
                print(f"删除损坏文件失败: {delete_error}")

    def load_samples_and_creat_intermediates(self):
        start_time = time.perf_counter()
        print(f"开始加载 {self.trace_number} 条迹线")

        batch_size = 100
        batch_number = (self.trace_number + batch_size - 1) // batch_size

        # 多进程并行加载
        with Pool(processes=self.process_number) as pool:
            for batch_idx in range(batch_number):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, self.trace_number)
                current_batch_size = end_idx - start_idx
                print(f"批{batch_idx + 1}/{batch_number} ({current_batch_size}条)")

                # 准备任务参数
                batch_tasks = [(self.traceset_path, start_idx + i,
                                self.sample_first_pos, self.sample_number,
                                self.sbox_num, self.sbox_size,
                                self.d_fuc_mode, self.crypto_direction) for i in range(current_batch_size)]

                # 并行处理
                batch_results = pool.starmap(process_single_trace, batch_tasks)

                # 写入结果
                for i, (sample_arr, data_arr_2d) in enumerate(batch_results):
                    trace_index = start_idx + i
                    self.sample_arr_2d[trace_index] = sample_arr
                    self.data_arr_3d[trace_index, :, :] = data_arr_2d

        elapsed_time = time.perf_counter() - start_time
        print(f"所有迹线加载完成 用时 {elapsed_time:.3f} 秒")

    def analyze(self):
        self.init_process()

        self.load_samples_and_creat_intermediates()

        print(f"开始分析Sbox")
        start_time = time.perf_counter()

        for i in self.sbox_index_arr:
            diff_arr_2d = np.zeros((self.sbox_size, self.sample_number))
            for j in range(self.sbox_size):
                group = self.data_arr_3d[:, i, j]
                mean_0 = np.mean(self.sample_arr_2d[group == 0], axis=0)
                mean_1 = np.mean(self.sample_arr_2d[group == 1], axis=0)
                diff_arr_2d[j, :] = mean_1 - mean_0

            (self.sbox_key_arr_2d[i],
             self.sbox_keyvalue_arr_2d[i],
             self.sbox_keypos_arr_2d[i]) = rank_sbox_key_guesses(diff_arr_2d, self.candidates)

            if self.traceset2_switch:
                for j in range(self.sbox_size):
                    trace_parameter_map = TraceParameterMap()
                    trace = Trace(
                        sample_coding=SampleCoding.FLOAT,
                        samples=diff_arr_2d[j],
                        parameters=trace_parameter_map,
                        title=f"Sbox{i}-KeyGuess_0x{j:02X}")
                    self.traceset2.append(trace)
            print(f"Sbox{i} 分析完成")

        elapsed_time = time.perf_counter() - start_time
        print(f"所有Sbox分析完成 用时 {elapsed_time:.3f} 秒")

        self.finish_process()

    def recovery_key(self):
        temp = self.sbox_key_arr_2d[:, 0]

        if self.crypto_direction == 0:
            temp = self.sbox_key_arr_2d[:, 0]
            print("KEY:", ' '.join(f'{x:02X}' for x in temp))

        if self.crypto_direction == 1:
            aes = Aes()
            aes.set_state(temp)
            aes.shift_rows()
            temp = aes.get_state()
            Nb = 4
            wi = np.zeros(Nb, dtype=np.uint32)
            for i in range(4):
                wi[i] = ((np.uint32(temp[i * Nb + 0]) << 24) |
                         (np.uint32(temp[i * Nb + 1]) << 16) |
                         (np.uint32(temp[i * Nb + 2]) << 8) |
                         np.uint32(temp[i * Nb + 3]))
            Nk = 4
            temp = aes_inv_keyexpansion(wi, 10, None, Nk)
            print("KEY:", ' '.join(f'{x:02X}' for x in temp))


if __name__ == '__main__':
    aes128_dpa = Aes128DPA()
    aes128_dpa.process_number = 8
    aes128_dpa.d_fuc_mode = 11  # 汉明重量模型 阈值 < 4

    # 第一轮攻击配置（加密）
    aes128_dpa.traceset_path = "D:\\traceset\\aes128_en+LowPass(203439)+StaticAlign(203755).trs"
    aes128_dpa.traceset2_switch = False
    aes128_dpa.sample_first_pos = 450000
    aes128_dpa.sample_number = 20000
    aes128_dpa.crypto_direction = 0
    aes128_dpa.sbox_index_arr = index_str_to_range("0-15")
    aes128_dpa.analyze()

    # 第二轮攻击配置（解密）
    aes128_dpa.traceset_path = "D:\\traceset\\aes128_de+LowPass(211850)+StaticAlign(212310).trs"
    aes128_dpa.traceset2_switch = False
    aes128_dpa.sample_first_pos = 470000
    aes128_dpa.sample_number = 30000
    aes128_dpa.crypto_direction = 1
    aes128_dpa.sbox_index_arr = index_str_to_range("0-15")
    aes128_dpa.analyze()
