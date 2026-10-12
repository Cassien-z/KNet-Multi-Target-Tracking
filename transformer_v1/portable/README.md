# MATLAB 与 C 算法交付版

本目录采用同一份固定尺寸 ONNX 模型，提供 MATLAB 和 C 两个完整推理入口。MATLAB
入口通过预编译 MEX 调用与 C 版本相同的 ONNX Runtime 推理核心，保证两种交付入口
使用同一模型、同一输入定义和同一数值实现。训练仍在原 Python/PyTorch 工程中完成，
不纳入 MATLAB/C 交付范围。

## 固定验收接口

- 批量大小：1
- 序列长度：100 帧
- 协同平台数量：5
- 输出状态：`[px,vx,py,vy,pz,vz]`
- 模型：`portable/artifacts/coop_tracker_v7.onnx`
- 测试向量：`portable/artifacts/golden_case`

固定尺寸是为了在结题期限内降低跨语言动态内存、动态形状和代码生成风险。如需支持
其他序列长度，应重新运行导出脚本产生对应模型。

## 重新导出并校验

```powershell
conda run --no-capture-output -n pytorch python portable/python/export_onnx.py `
  --seq-len 100 --num-uavs 5 --output-dir portable/artifacts
```

脚本会加载 V7 权重和首个 V7 验证场景，导出 ONNX，执行 PyTorch/ONNX Runtime
数值对照，并生成 MATLAB/C 共用的 float32 二进制测试向量。

不依赖 PyTorch 工程的独立验证命令：

```powershell
conda run --no-capture-output -n pytorch python portable/python/verify_onnx.py
```

## MATLAB 版本

已提供 Windows x64、MATLAB R2026a 编译的 `coop_tracker_mex.mexw64`，不需要 Deep
Learning Toolbox，也不需要 ONNX 模型转换插件。执行：

```matlab
addpath("portable/matlab")
verify_golden_case("portable/artifacts/coop_tracker_v7.onnx", ...
    "portable/artifacts/golden_case")
```

业务调用入口为 `run_tracker.m`。输入是包含 18 个固定字段的结构体，字段和尺寸见
`golden_case/manifest.json`。

如果更换 MATLAB 大版本或编译环境，可在 MATLAB 中重新生成 MEX：

```matlab
addpath("portable/matlab")
mex -setup C
build_mex
```

MEX 运行所需的 `onnxruntime.dll` 已放在 `portable/matlab` 中。R2026a 进程内若已经加载
自带的 ONNX Runtime 1.17，接口会自动协商兼容版本；独立运行时使用随包的 1.18.1。

## C 版本

C 版本使用 ONNX Runtime C API。交付目录已经包含官方 Windows x64 CPU 运行包，执行：

```powershell
cmake -S portable/c -B portable/c/build `
  -DONNXRUNTIME_ROOT=portable/third_party/onnxruntime-win-x64-1.18.1
cmake --build portable/c/build --config Release
```

公共接口位于 `portable/c/include/tracker.h`。调用方创建 `Tracker`、填充
`TrackerInputs` 的 18 个连续 float32 缓冲区，然后调用 `tracker_run` 获取 600 个状态值。
编译后可用共用测试向量验证：

```powershell
portable/c/build/Release/verify_golden_case.exe `
  portable/artifacts/coop_tracker_v7.onnx portable/artifacts/golden_case
```

## 一致性门槛

- MATLAB/C 与 Python 的最大位置向量差：不超过 1 m
- 最大速度分量差：不超过 0.2 m/s
- 最终合同指标仍使用完整验证集单独测试，本测试向量只证明跨语言实现一致

## 一键验证

在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File portable/verify_delivery.ps1
```

脚本依次执行 C 和 MATLAB 黄金样例；两项均通过时返回退出码 0。

## 生成和校验交付压缩包

```powershell
conda run --no-capture-output -n pytorch python portable/python/build_delivery_package.py
conda run --no-capture-output -n pytorch python portable/python/verify_delivery_package.py `
  artifacts/delivery/coop_tracker_matlab_c_delivery_20260921.zip
```

压缩包内含文件级 SHA-256 清单、四项指标原始证据和模型来源信息。日期文件名可按
实际生成日期调整；外部的 `.zip.sha256` 文件用于在传输后校验整个压缩包。
