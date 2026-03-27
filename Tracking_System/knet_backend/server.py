# server.py
import asyncio
import json
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from inference import KNet_Engine, NORM_CFG 

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/tracking")
async def tracking_endpoint(websocket: WebSocket, dataset: str = "006"):
    await websocket.accept()
    
    engine = KNet_Engine("knet_weights_finetuned.pth")

    # 控制状态字典，用于在发送和接收协程间共享状态
    ctrl = {
        "paused": False,
        "idx": 0,
        "seek_target": None
    }

    try:
        file_path = f"knet_verify_data/track_data_{dataset}.json"
        with open(file_path, 'r') as f:
            data = json.load(f)

        total_frames = len(data)
        max_time = data[-1]['time_s'] if total_frames > 0 else 0
        errors_pos = []

        # 🌟 定义子协程：专门负责接收前端的控制指令
        async def receive_commands():
            try:
                while True:
                    msg = await websocket.receive_json()
                    if msg.get("action") == "pause":
                        ctrl["paused"] = True
                    elif msg.get("action") == "play":
                        ctrl["paused"] = False
                    elif msg.get("action") == "seek":
                        ctrl["seek_target"] = float(msg.get("time"))
            except Exception:
                pass # 客户端断开等异常会退出循环

        # 启动接收协程
        recv_task = asyncio.create_task(receive_commands())

        # 主发送循环
        while ctrl["idx"] < total_frames:
            # 1. 处理跳转 (Seek) 请求
            if ctrl["seek_target"] is not None:
                target_t = ctrl["seek_target"]
                # 寻找最接近的时间戳索引
                ctrl["idx"] = 0
                for i, frame in enumerate(data):
                    if frame['time_s'] >= target_t:
                        ctrl["idx"] = i
                        break
                ctrl["seek_target"] = None
                
                # 🌟 修复 3：统一使用 reset() 彻底重置所有状态（包括速度初始化缓存）
                engine.reset()
                errors_pos.clear()

            # 2. 处理暂停 (Pause) 逻辑
            if ctrl["paused"]:
                await asyncio.sleep(0.1) # 暂停时让出 CPU
                continue

            # 3. 正常读取与推理
            frame = data[ctrl["idx"]]
            t = frame['time_s']
            
            # 维持相对坐标系逻辑
            gt_w = np.array(frame['ground_truth_state'])
            p_w = np.array(frame['platform_state'])
            gt_rel = gt_w[[0, 2, 4]] - p_w[[0, 2, 4]]

            obs = frame['sensors']['sensor_0']['observations']
            if obs:
                meas_sph = obs[0]['meas']
                pred_norm = engine.predict_step(meas_sph, t)
                pred_pos = pred_norm[[0, 2, 4]] * NORM_CFG['pos']

                err_pos = np.linalg.norm(pred_pos - gt_rel)
                errors_pos.append(err_pos)
                rmse_current = np.sqrt(np.mean(np.square(errors_pos)))

                # 🌟 修复 1：打包数据恢复为相对坐标变量
                payload = {
                    "time": float(t),
                    "max_time": float(max_time),
                    "is_reset": len(errors_pos) == 1,
                    "truth_pos": gt_rel.tolist(),
                    "pred_pos": pred_pos.tolist(),
                    "metrics": {
                        "err_pos": float(err_pos),
                        "rmse_current": float(rmse_current)
                    }
                }
                
                # 🌟 修复 2：去掉多余的 await send，仅在 try 中发送
                try:
                    await websocket.send_json(payload)
                except RuntimeError as e:
                    if "websocket.close" in str(e) or "already completed" in str(e):
                        print("检测到前端主动断开连接，安全终止当前推送。")
                        break # 跳出 while 循环，结束当前协程
                    else:
                        raise e # 其他未知错误继续抛出

            ctrl["idx"] += 1
            await asyncio.sleep(0.05) 

    except Exception as e:
        print(f"WebSocket 运行异常或断开: {e}")
    finally:
        # 确保清理任务和状态
        if 'recv_task' in locals():
            recv_task.cancel()
        # 🌟 修复 3：统一使用 reset()
        engine.reset()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)