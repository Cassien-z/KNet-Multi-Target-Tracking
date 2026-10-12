# 端到端融合跟踪原型软件

该目录包含从 `Tracking_System` 迁移的 Vue 前端框架，以及直接调用本工程
`src.model.CoopTrackerModel` 的 FastAPI 后端。旧 KNet 模型不在此原型中使用。

## 端到端技术边界

本项目区分“软件全流程”和“模型端到端范围”。软件全流程从原始雷达、光电、
电子侦察及数据链观测开始，依次完成量测驱动的目标关联、源时间戳对齐、按目标
组织序列、神经网络融合跟踪和航迹输出。模型所称“端到端适变跟踪”特指：以已经
完成时空对准和目标关联的多源时序观测为输入，直接输出连续位置与速度状态。

因此，当前神经网络不包含原始检测点提取，也不联合学习目标身份关联；这两部分是
完整软件链中的确定性前处理。评审、验收及对外材料不得将模型范围表述为“从原始
检测点到航迹身份输出的单一可训练网络”。详细口径见 `docs/end_to_end_scope.md`。

## 启动后端

推荐直接使用项目现有的 Conda `pytorch` 环境。在 Anaconda Prompt 或已初始化
Conda 的 PowerShell 中执行：

```powershell
conda activate pytorch
cd D:\workstation\End-to-End-Tracking-System\transformer_v1
python -m tracking_system.backend.server
```

如果不想激活环境，也可以在 `transformer_v1` 根目录直接执行：

```powershell
conda run --no-capture-output -n pytorch python -m tracking_system.backend.server
```

启动成功后，后端地址为 `http://127.0.0.1:8000`，可在浏览器访问
`http://127.0.0.1:8000/api/health` 检查模型是否加载成功，访问
`http://127.0.0.1:8000/docs` 查看和直接调用全部接口。当前 `pytorch`
环境已经包含 PyTorch、CUDA、FastAPI、Uvicorn、WebSocket、SciPy 和 h5py，
无需再次安装后端依赖。

默认加载 v7 最佳权重和 `data/val_data_v7_registered`。可通过
`TRACKER_MODEL_PATH`、`TRACKER_DATA_DIR`、`TRACKER_DEVICE` 和
`TRACKER_SEQ_LEN` 环境变量覆盖。

后端先对一个完整序列执行推理，再通过 WebSocket 按时间轴回放结果。
查询参数 `dataset=100` 对应当前数据目录按文件名排序后的第一个 PKL，
`target=0` 对应该文件中的第一个目标。健康检查和场景列表接口分别为
`/api/health` 与 `/api/scenarios`。

## 模型管理接口

- `GET /api/models`：列出注册的 V5、V6、V7 三个续训最佳模型，同时返回
  run 编号、训练/验证数据集、文件信息及可用的历史评估摘要。
- `GET /api/models/active`：查询当前激活模型、设备和加载耗时。
- `GET /api/devices`：查询可用的 CPU/CUDA 设备。
- `POST /api/models/refresh`：重新检查三个注册权重是否可用。
- `POST /api/models/activate`：加载并切换模型，例如：

```json
{
  "model_id": "v7",
  "device": "cuda:0"
}
```

模型切换采用先加载、后替换策略。新权重加载失败时当前模型保持不变；已经开始
的 WebSocket 回放继续使用其原模型，新连接使用切换后的模型。

## 批量测试接口

- `GET /api/datasets`：查询 V5、V6、V7 和 V8 强干扰验证集。
- `POST /api/batch-tests`：提交后台批测任务并立即返回任务 ID。
- `GET /api/batch-tests`：查询批测历史及结果摘要。
- `GET /api/batch-tests/{job_id}`：查询任务进度和完整结果。
- `POST /api/batch-tests/{job_id}/cancel`：取消排队中或运行中的任务。

提交任务示例：

```json
{
  "name": "V7 模型标准批测",
  "model_id": "v7",
  "dataset_id": "v7",
  "device": "cuda:0",
  "warmup_frames": 5,
  "batch_size": 64,
  "workers": 0,
  "max_scenarios": null
}
```

`max_scenarios` 为空时评估完整数据集，开发调试时可设置为较小数值。任务状态包括
`queued`、`running`、`completed`、`failed` 和 `cancelled`。位置精度仅统计
`target_valid_mask=true` 的跟踪稳定阶段样本，并排除每条航迹配置的前置预热帧；结果包含
稳定阶段位置 RMSE、MAE、中位数、P95、最大误差、平均航迹 RMSE、批次/航迹/实测单帧时延、
吞吐率、关联正确率（数据集包含关联审计信息时）及逐航迹指标。报告持久化到
`artifacts/reports/batch_tests/{job_id}.json`，后端重启后仍可查询历史结果。

数据集接口同时返回与指标绑定的`quality_profile`。V7配置明确传感器噪声、视场、
更新周期、随机丢测、数据链延迟和适用边界；V8配置明确复合干扰强度并标记为鲁棒性
证据，不与V7的8 m验收结论混用。可运行以下命令复核输入质量、机动强度和协议经费
科目，结果写入`artifacts/metrics/midterm_review/items_10_12_evidence.json`：

```powershell
python scripts/evaluation/review_items_10_12.py
```

## 多维评估与报告接口

- `GET /api/config/acceptance-criteria`：查询任务书技术指标及判定阈值。
- `GET /api/batch-tests/{job_id}/assessment`：对已完成任务进行多维评估。
- `GET /api/batch-tests/{job_id}/assessment?baseline_rmse_m=...`：同时计算相对
  同测试条件 IMM-JPDA 基线的 RMSE 降幅。
- `GET /api/batch-tests/{job_id}/report.json`：下载完整、可复现的 JSON 报告。
- `GET /api/batch-tests/{job_id}/tracks.csv`：下载逐航迹指标 CSV。
- `GET /api/evaluations/cross-dataset`：查询已有 V5/V6/V7 模型在四类数据集上的
  3×4 RMSE 交叉评估结果。
- `GET /api/evaluations/compare`：将已完成批测归一化为模型/数据集对比表；也可用
  `job_ids=id1,id2` 只比较指定任务。
- `GET /api/evaluations/references`：查询已有 IMM-JPDA 对比和数据关联审计结果。
- `GET /api/reports`：查询全部持久化任务、任务书判定摘要及可下载制品目录。
- `GET /api/reports/{job_id}/assessment.json`：下载包含原始任务、四项判定和基线
  来源的综合评估 JSON；可附加 `baseline_rmse_m` 查询参数。
- `GET /api/reports/comparison.csv?job_ids=...`：下载最多 50 个已完成任务的统一
  横向对比 CSV。
- `GET /api/system`：查询 Python、PyTorch、CUDA、GPU、当前模型和批测队列状态。

任务书中的判定条件为：跟踪稳定阶段位置 RMSE 不超过 8 m、数据关联正确率不低于 85%、
相对 IMM-JPDA 的 RMSE 降幅不低于 15%、单帧跟踪结果生成时间（模型前向）不超过 100 ms。
本软件不生成飞行控制指令，因此该时延不包含控制律、执行机构和通信链路。评估结果
采用 `passed`、`failed`、`unavailable` 三态；缺少关联审计信息或同条件基线时会
标记为数据不足，不会被错误地当成通过。总体状态为 `passed`、`failed` 或
`incomplete`。

关联正确率和同条件基线降幅由两个独立脚本补充。先完成批量测试，然后在已激活
`pytorch` 环境、项目根目录中将下面的报告文件替换为待评估任务：

```powershell
python scripts/evaluation/test_association_accuracy.py --batch-report artifacts/reports/batch_tests/任务ID.json
python scripts/evaluation/test_rmse_improvement.py --batch-report artifacts/reports/batch_tests/任务ID.json
```

第一个脚本按相同场景范围审计观测与真值目标的关联结果；第二个脚本在相同数据集、
场景数、序列长度和预热帧设置下运行 NumPy IMM-JPDA 基线，并计算 Transformer 的
RMSE 降幅。证据写入 `artifacts/metrics/acceptance/{job_id}/`。评估接口会校验任务
签名后自动加载，配置或批测结果发生变化时旧证据会被拒绝。性能评估页会显示两份
证据的状态，并可复制缺失证据对应的运行命令。完整数据集脚本可能耗时较长。

## 后端验收

保持后端运行，再打开另一个已激活 `pytorch` 环境的终端，在项目根目录执行：

```powershell
python -m tracking_system.backend.acceptance
```

该命令会按黑盒方式检查健康状态、运行环境、三个模型、数据集、模型激活、场景
读取、真实 WebSocket 推理、四项指标配置、历史对比、3×4 交叉评估、真实小批量
CUDA 测试、多维评估，以及 JSON/CSV 报告下载。默认使用 V7 模型、V7 数据集的
第一个场景。最后出现“后端功能验收通过”且进程退出码为 0，表示软件链路验收
通过。

完整算法指标验收应将 `max_scenarios` 设为空提交完整批测，再针对该报告执行上述
两个指标脚本。也可在评估接口手动传入同条件 IMM-JPDA 基线 RMSE 作为临时覆盖值。
自动验收中的单场景结果只用于验证软件链路，不能代替完整数据集上的算法结论。

后端单元测试命令为：

```powershell
python -m unittest discover -s tracking_system\backend\tests -v
```

## 启动前端

```powershell
cd tracking_system/frontend
npm install
npm run dev
```

前端默认连接 `http://127.0.0.1:8000`。当前已完成系统总览、模型管理、跟踪验证、
批量测试、性能评估和报告中心六个页面：
总览页展示服务、模型、CUDA、批测队列、运行环境和任务书指标；模型管理页支持
V5/V6/V7 权重查看、部署设备选择、仓库刷新和带二次确认的模型切换；跟踪验证页
支持 CA/CT/CV/CSSW 机动类型筛选、场景/目标选择、三维真实与预测航迹、实时误差、
播放暂停、时间轴跳转、重播和当前序列 CSV 导出。三维图以首帧真实位置为可视化
原点并采用统一空间比例尺，避免百万米量级绝对坐标压缩局部航迹；导出的 CSV 和
图下方读数仍保留原始绝对坐标。批量测试页支持模型/数据集/设备配置、抽样或完整测试、后台
进度、持久化历史、任务取消、结果详情及 JSON/CSV 报告下载；性能评估页支持对
已完成批测按任务书四项指标自动判定、录入同条件 IMM-JPDA 基线、展示精度/时延/
航迹分布/机动类型等维度、查看 3×4 跨数据集鲁棒性矩阵、历史基线证据以及批测结果
横向对比；报告中心支持归档检索、状态和模型筛选、报告详情复核、同条件基线重判、
原始 JSON、综合评估 JSON、逐航迹 CSV、勾选任务对比 CSV 以及浏览器打印/PDF。
侧栏设置按钮可查看 API、模型、PyTorch、CUDA、报告目录和后端启动命令。页面地址
分别为 `#overview`、`#models`、`#tracking`、`#batch`、`#evaluation` 和
`#reports`。
