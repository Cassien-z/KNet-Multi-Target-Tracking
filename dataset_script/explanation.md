**三种传感器详解：**

1. **雷达传感器（Radar）**
   使用 CartesianToElevationBearingRange 模型
   能测量三维：方位角（az）、仰角（el）、距离（range）
   不限制 FOV（全向）
   中等探测噪声
   杂波率 0.5
2. **光电传感器（EO / Electro-Optical）**
   测量模型仍然是 CartesianToElevationBearingRange
   只关心角度信息（距离噪声非常大，相当于无法测量距离）
   水平方向 FOV：±90°
   垂直方向 FOV：±90°
   噪声比雷达更小（高精度）
   杂波率 0.2
3. **被动传感器（Passive）**

 	只能测 方位角（az）

 	仰角噪声非常大 → 实际相当于测不到

 	距离噪声无穷大 → 不能测距离

 	类似于：

 		无线电方向测向仪

 		ESM/ELINT

 		被动雷达（TDOA 方位）

 	杂波率 0.1



**干扰手段详解：**

**1. RGPO — Range Gate Pull-Off（距离门拖拽）**

 	雷达在跟踪目标时，会对回波的\*\*距离门（Range Gate）\*\*进行锁定。

 	干扰机在目标回波基础上叠加一个假回波（比真实回波稍微靠后）。

 	假回波的延迟不断增加，使跟踪雷达的距离门被“拉走”。

 	雷达会锁定假目标，从而丢失真实目标。

 	RGPO = 制造一个不断后退的假目标，让雷达跟丢。

**2. Clutter — 杂波（环境回波）**

 	Clutter 是雷达探测中的非目标回波背景，不是干扰措施，而是一种物理现象。

 	Clutter = 雷达环境中的“噪声背景”，但也可以被伪造用来迷惑雷达。

**3. Spoof/Spoofing — 欺骗式干扰**

 	Spoof 或 spoofing 是电子战中对雷达进行假目标欺骗的总称。包括：

 	距离欺骗（RGPO、VGPO）

 	角度欺骗（Angle Deception）

 	多假目标生成（False Targets / Multiple Spoofed Targets）

 	数字射频记忆（DRFM）欺骗（现代主流）



**具体实现：**

 	RGPO（雷达干扰）：Z方向加入周期性干扰

 	SPOOF（欺骗）：随机偏移方位/仰角/距离

 	CLUTTER（杂波）：随机生成测量

