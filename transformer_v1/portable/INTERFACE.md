# 固定推理接口说明

交付接口以完成时间对齐、航迹关联和空间注册后的单目标序列为输入。Python 训练、原始
MAT 文件解析以及数据集生成不属于 MATLAB/C 在线推理接口。MATLAB 与 C 使用完全相同
的 float32 张量、ONNX 权重和输出定义。

维度符号：`B=1`，`T=100`，`N=5`。

| 输入 | 尺寸 | 含义 |
|---|---:|---|
| radar | B×T×N×3 | 机体系 `[方位角, 俯仰角, 距离]` |
| eo | B×T×N×2 | 光电 `[方位角, 俯仰角]` |
| esm | B×T×N×2 | 电子战方向量，当前第2通道保留为0 |
| radar_mask eo_mask esm_mask | B×T×N | 对应传感器有效标志 |
| ref_pos | B×3 | 序列首帧目标参考位置 |
| rotmat_b2e | B×T×N×3×3 | 机体系到ECEF的旋转矩阵 |
| uav_pos | B×T×N×3 | 相对 `ref_pos` 的平台位置 |
| fast_link slow_link | B×T×N×3 | 相对 `ref_pos` 的快慢链路位置量测 |
| fast_update slow_update | B×T×N | 链路更新标志 |
| imu | B×T×N×6 | 平台姿态/运动输入 |
| fast_age slow_age | B×T×N | 链路量测年龄 |
| esm_mode | B×T×N | ESM工作模式 |
| dt | B | 采样周期，单位秒 |

输出 `prediction` 尺寸为 `B×T×6`，状态顺序为
`[px,vx,py,vy,pz,vz]`，位置为ECEF绝对坐标，速度单位为 m/s。

所有数组必须连续存储。C接口按行优先顺序解释缓冲区；MATLAB加载函数负责在二进制
文件的行优先顺序与MATLAB列优先顺序之间转换。
