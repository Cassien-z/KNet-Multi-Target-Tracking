import os
import scipy.io
import traceback

from script_single_CA import run_simulation as run_ca, _random_scenario as rand_ca
from script_single_CV import run_simulation as run_cv, _random_scenario as rand_cv
from script_single_CSSW import run_simulation as run_cssw, _random_scenario as rand_cssw
from script_single_CT import run_simulation as run_ct, _random_scenario as rand_ct

def generate_mode(name, run_fn, rand_fn, num=30):
    save_dir = f"../test_data/test_data_{name.lower()}"
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    attempts = 0
    print(f"🚀 开始生成 {num} 条 {name} 模式的全新测试轨迹...")
    while count < num:
        attempts += 1
        try:
            data = run_fn(*rand_fn())
            if data is None: continue
            scipy.io.savemat(os.path.join(save_dir, f"track_data_{name.lower()}_{count:04d}.mat"), data)
            count += 1
            if count % 10 == 0:
                print(f"  ✅ {count}/{num}  (尝试 {attempts})")
        except Exception as e:
            pass
    print(f"🎉 完成！数据保存于 {save_dir}/")

if __name__ == "__main__":
    import numpy as np
    # 使用完全不同的种子确保不和训练集重叠
    np.random.seed(9999) 
    generate_mode("CA", run_ca, rand_ca)
    generate_mode("CV", run_cv, rand_cv)
    generate_mode("CSSW", run_cssw, rand_cssw)
    generate_mode("CT", run_ct, rand_ct)
