import json
import os
from script import run_simulation  # 确保 script.py 在当前目录


def start_batch_work(count):
    save_dir = "knet_verify_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(count):
        # 每次传入不同的种子，保证 CSSW 模型的随机跳变在每份数据中都不同
        print(f"正在生成第 {i + 1}/{count} 组数据...")
        data = run_simulation(seed=i)

        file_name = os.path.join(save_dir, f"track_data_{i:03d}.json")
        with open(file_name, 'w') as f:
            json.dump(data, f)


if __name__ == "__main__":
    start_batch_work(10)