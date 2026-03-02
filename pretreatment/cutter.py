import os
import copy
import trsfile
import time
from datetime import datetime
from trsfile import Header, TracePadding, Trace


class Cutter:
    def __init__(self):
        self.sample_first_pos: int = 0
        self.sample_number: int = 0

        self.traceset = None
        self.traceset_path: str = ""
        self.traceset2 = None

    def open_traceset(self):
        self.traceset = trsfile.open(self.traceset_path, 'r')
        number_samples_per_trace = self.traceset.get_headers()[Header.NUMBER_SAMPLES]

        if self.sample_first_pos < 0:
            self.sample_first_pos = 0
            print(f"自动调整样本点起始位置为: 0")

        if self.sample_first_pos + self.sample_number > number_samples_per_trace:
            self.sample_number = number_samples_per_trace - self.sample_first_pos
            print(f"自动调整样本点数量为: {self.sample_number}")

        headers = copy.deepcopy(self.traceset.get_headers())
        headers[Header.NUMBER_TRACES] = 0
        headers[Header.NUMBER_SAMPLES] = self.sample_number

        base_name = os.path.splitext(os.path.basename(self.traceset_path))[0]
        timestamp = datetime.now().strftime("%H%M%S")
        new_base_name = f"{base_name}+Cut({timestamp})"
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
        if self.traceset2:
            self.traceset2.close()

    def cut(self):
        self.open_traceset()
        trace_number = self.traceset.get_headers()[Header.NUMBER_TRACES]
        start_time = time.perf_counter()
        print(f"开始剪切 {trace_number} 条迹线")

        for trace_index in range(trace_number):
            trace = self.traceset[trace_index]
            cutted_trace = Trace(
                sample_coding=trace.sample_coding,
                samples=trace[self.sample_first_pos:self.sample_first_pos + self.sample_number],
                parameters=trace.parameters,
                title=trace.title,
                headers=trace.headers)
            self.traceset2.append(cutted_trace)

            if (trace_index + 1) % 100 == 0:
                print(f"已处理 {trace_index + 1}/{trace_number} 条迹线")

        elapsed_time = time.perf_counter() - start_time
        print(f"剪切完成 用时:{elapsed_time:.3f}秒")
        self.close_traceset()


if __name__ == '__main__':
    cutter = Cutter()

    cutter.traceset_path = "D:\\traceset\\aes128_en+LowPass(203439)+StaticAlign(203755).trs"
    cutter.sample_first_pos = 400000
    cutter.sample_number = 200000

    cutter.cut()
