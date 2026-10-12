# 基于多源异构数据融合的端到端适变跟踪项目

这是研究、训练、验证和结题交付的总工程。当前结题阶段以 `portable` 中的 MATLAB/C
版本为软件交付入口；`tracking_system` 是早期前后端演示系统，不是本次算法交付主入口。

## 结题阶段从这里开始

| 要找的内容 | 位置 |
|---|---|
| 最终 MATLAB/C 交付压缩包 | `artifacts/delivery/coop_tracker_matlab_c_delivery_20260921.zip` |
| MATLAB/C 使用说明 | `portable/README.md` |
| MATLAB 入口 | `portable/matlab/run_tracker.m` |
| C 公共接口 | `portable/c/include/tracker.h` |
| ONNX 模型 | `portable/artifacts/coop_tracker_v7.onnx` |
| 一键验证 | `portable/verify_delivery.ps1` |
| 四项指标正式证据 | `artifacts/reports/v7_final_validation/` |
| 最终文档 | `docs/` 和 `docs/software_delivery/` |
| 模型训练与研究代码 | `train.py`、`src/`、`scripts/` |

交付前验证：

```powershell
powershell -ExecutionPolicy Bypass -File portable/verify_delivery.ps1
```

## 目录结构

```text
transformer_v1/
├─ train.py                    # 唯一训练入口：注册 PKL 数据
├─ src/                        # 模型、数据集与可复用算法模块
│  ├─ data_association/        # 时间对齐、关联、补全、异常剔除与配准
│  ├─ utils/                   # 坐标转换等基础工具
│  └─ wingman_fusion/          # 僚机本地融合
├─ scripts/
│  ├─ data_generation/         # 仿真数据生成与外部数据导入
│  ├─ data_association/        # MATLAB 数据关联演示和批处理
│  ├─ pipelines/               # 数据生成/注册流水线入口
│  ├─ evaluation/              # 精度、基线、时延及消融评估
│  └─ visualization/           # 论文和实验图表生成
├─ data/                       # 原始、训练及验证数据
├─ results/                    # 模型权重和训练运行结果
├─ artifacts/
│  ├─ delivery/                # 最终 MATLAB/C 交付压缩包和哈希
│  ├─ figures/                 # 汇总图表
│  ├─ metrics/                 # 独立评估指标 JSON
│  ├─ reports/                 # 评估及时延报告
│  └─ （其他构建目录均属于可重新生成的中间产物）
├─ portable/                   # 当前正式 MATLAB/C 算法交付源码与运行文件
├─ tracking_system/            # 早期 Web 前后端演示系统（非当前交付主入口）
└─ docs/
   ├─ paper/                   # 论文、技术方案和专利文档
   ├─ extracted/               # 文档提取文本
   ├─ paper_outline.md         # 论文提纲
   ├─ training_commands.txt    # 历史训练命令
   └─ 技术指标.txt
```

## 常用命令

以下命令均从项目根目录执行。

```powershell
# 训练注册后的 PKL 数据
python train.py --data_dir data/train_data_v5_registered `
  --val_data_dir data/val_data_v5_registered --epochs 40 --batch_size 32

# 生成统一数据集
python scripts/data_generation/script_generate_dataset_unified.py `
  --mode val --count_per_model 500 --num_targets 3 --save_dir data/val_data_v5

# MAT 数据注册为 PKL
python scripts/pipelines/run_register_pipeline.py `
  --in_dir data/val_data_v5 --out_dir data/val_data_v5_registered

# 通用数据集评估
python scripts/evaluation/eval_dataset.py `
  --model results/<run>/best_model.pt --data data/val_data_v5_registered

# 一键运行 V5 数据流水线
powershell -ExecutionPolicy Bypass -File scripts/pipelines/run_pipeline_v5.ps1
```

评估指标默认写入 `artifacts/metrics`，报告写入 `artifacts/reports`，独立绘图脚本统一写入 `artifacts/figures`。

## 保留原则

- `data/`：原始、训练和验证数据，不随意删除。
- `results/`：训练权重和历史运行结果，不随意删除。
- `artifacts/delivery`、`artifacts/metrics`、`artifacts/reports`：交付及验收证据。
- `portable/`：MATLAB/C 最终交付工作区。
- `.codex_work/`、`__pycache__/`、文档渲染目录、`node_modules/`、调试符号：均为可重新生成的临时内容。
