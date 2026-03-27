<template>
  <div class="dashboard">
    <header class="header">
      <h1>基于多源异构数据融合的端对端适变跟踪系统</h1>
      
      <div class="controls">
        <label for="dataset-select" class="control-label">测试场景：</label>
        <select id="dataset-select" v-model="selectedDataset" @change="handleDatasetChange" class="dark-input">
          <option v-for="i in 10" :key="i" :value="String(i-1).padStart(3, '0')">
            track_data_{{ String(i-1).padStart(3, '0') }}
          </option>
        </select>

        <button @click="togglePlay" class="control-btn" :class="{ paused: isPaused }">
          {{ isPaused ? '▶ 播放' : '⏸ 暂停' }}
        </button>

        <div class="slider-container">
          <span class="time-text">{{ latestData.time.toFixed(1) }}s</span>
          <input 
            type="range" 
            class="time-slider" 
            :min="0" 
            :max="latestData.max_time || 100" 
            step="0.5" 
            :value="isDraggingSlider ? sliderTempValue : latestData.time"
            @input="handleSliderDrag"
            @change="handleSliderRelease"
          >
          <span class="time-text">{{ latestData.max_time.toFixed(1) }}s</span>
        </div>

        <div class="status" :class="{ connected: isConnected }">
          {{ isConnected ? '系统在线' : '连接断开' }}
        </div>
      </div>
    </header>

    <main class="main-content">
      <div class="panel map-panel">
        <div class="panel-header">三维动态态势底图 (3D 空间)</div>
        <div ref="trajectoryChartRef" class="chart-container"></div>
      </div>

      <div class="right-panels">
        <div class="panel stats-panel">
          <div class="stat-box">
            <div class="label">实时位置误差 (m)</div>
            <div class="value highlight">{{ latestData.err_pos.toFixed(2) }}</div>
          </div>
          <div class="stat-box">
            <div class="label">当前 RMSE (m)</div>
            <div class="value warning">{{ latestData.rmse.toFixed(2) }}</div>
          </div>
        </div>

        <div class="panel chart-panel">
          <div class="panel-header">跟踪误差评估 (RMSE & 绝对误差)</div>
          <div ref="errorChartRef" class="chart-container"></div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, reactive } from 'vue';
import * as echarts from 'echarts';
import 'echarts-gl'; 

const isConnected = ref(false);
const selectedDataset = ref('006'); 
const isPaused = ref(false); // 播放状态
const isDraggingSlider = ref(false); // 是否正在拖拽滑动条
const sliderTempValue = ref(0); // 拖拽时的临时时间

const latestData = reactive({
  time: 0,
  max_time: 0,
  err_pos: 0,
  rmse: 0,
});

const trajectoryChartRef = ref<HTMLElement | null>(null);
const errorChartRef = ref<HTMLElement | null>(null);
let trajectoryChart: echarts.ECharts | null = null;
let errorChart: echarts.ECharts | null = null;

const truthPath: number[][] = [];
const predPath: number[][] = [];
const timeData: string[] = [];
const rmseData: number[] = [];
const errPosData: number[] = [];

let ws: WebSocket | null = null;
let manualClose = false; 

// 清空图表数据辅助函数
const clearChartData = () => {
  truthPath.length = 0; predPath.length = 0;
  timeData.length = 0; rmseData.length = 0; errPosData.length = 0;
  updateCharts();
};

const connectWebSocket = () => {
  ws = new WebSocket(`ws://127.0.0.1:8000/ws/tracking?dataset=${selectedDataset.value}`);

  ws.onopen = () => {
    isConnected.value = true;
    isPaused.value = false;
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    // 🚨 收到重置信号（说明发生了时间轴跳转），立即清空前端缓存！
    if (data.is_reset) {
      clearChartData();
    }

    latestData.time = data.time;
    latestData.max_time = data.max_time;
    latestData.err_pos = data.metrics.err_pos;
    latestData.rmse = data.metrics.rmse_current;

    const truthX = data.truth_pos[0]; const truthY = data.truth_pos[1]; const truthZ = data.truth_pos[2];
    const predX = data.pred_pos[0]; const predY = data.pred_pos[1]; const predZ = data.pred_pos[2];

    const MAX_POINTS = 300;
    
    truthPath.push([truthX, truthY, truthZ]);
    predPath.push([predX, predY, predZ]);
    if (truthPath.length > MAX_POINTS) { truthPath.shift(); predPath.shift(); }

    timeData.push(data.time.toFixed(1));
    rmseData.push(data.metrics.rmse_current);
    errPosData.push(data.metrics.err_pos);
    if (timeData.length > MAX_POINTS) {
      timeData.shift(); rmseData.shift(); errPosData.shift();
    }

    // 如果正在拖动进度条，暂停图表刷新以防闪烁
    if (!isDraggingSlider.value) {
      updateCharts();
    }
  };

  ws.onclose = () => {
    isConnected.value = false;
    if (!manualClose) setTimeout(connectWebSocket, 3000);
    manualClose = false; 
  };
};

// 🌟 交互：播放/暂停切换
const togglePlay = () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  isPaused.value = !isPaused.value;
  ws.send(JSON.stringify({ action: isPaused.value ? 'pause' : 'play' }));
};

// 🌟 交互：正在拖拽滑动条 (不触发后端 seek，只暂停前端显示)
const handleSliderDrag = (e: Event) => {
  isDraggingSlider.value = true;
  sliderTempValue.value = Number((e.target as HTMLInputElement).value);
};

// 🌟 交互：松开滑动条 (向后端发送 seek 指令)
const handleSliderRelease = (e: Event) => {
  isDraggingSlider.value = false;
  const targetTime = Number((e.target as HTMLInputElement).value);
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: 'seek', time: targetTime }));
    // 若原先是暂停状态，跳转后自动恢复播放体验更好
    if (isPaused.value) {
      isPaused.value = false;
      ws.send(JSON.stringify({ action: 'play' }));
    }
  }
};

const handleDatasetChange = () => {
  if (ws) {
    manualClose = true; 
    ws.close();
  }
  clearChartData();
  latestData.time = 0; latestData.err_pos = 0; latestData.rmse = 0;
  connectWebSocket();
};

// --- Echarts 初始化 (无改变) ---
const initCharts = () => {
  if (trajectoryChartRef.value && errorChartRef.value) {
    trajectoryChart = echarts.init(trajectoryChartRef.value);
    errorChart = echarts.init(errorChartRef.value);

    trajectoryChart.setOption({
      tooltip: {},
      legend: { data: ['真实轨迹 (Truth)', '预测航迹 (Pred)'], textStyle: { color: '#ccc' } },
      grid3D: {
        viewControl: { projection: 'perspective', autoRotate: true, autoRotateSpeed: 10, distance: 150 },
        boxWidth: 100, boxHeight: 60, boxDepth: 100,
        environment: '#161b22', axisPointer: { lineStyle: { color: '#fff' } }
      },
      xAxis3D: { type: 'value', name: 'X (m)', nameTextStyle: { color: '#8b949e' } },
      yAxis3D: { type: 'value', name: 'Y (m)', nameTextStyle: { color: '#8b949e' } },
      zAxis3D: { type: 'value', name: 'Z (m)', nameTextStyle: { color: '#8b949e' } },
      series: [
        { name: '真实轨迹 (Truth)', type: 'line3D', data: [], lineStyle: { color: '#00ff00', width: 3, opacity: 0.8 } },
        { name: '预测航迹 (Pred)', type: 'line3D', data: [], lineStyle: { color: '#ff3333', width: 3, opacity: 0.8 } }
      ]
    });

    errorChart.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: ['位置误差 (m)', 'RMSE (m)'], textStyle: { color: '#ccc' } },
      grid: { left: '8%', right: '5%', bottom: '15%', top: '15%', containLabel: true },
      xAxis: { type: 'category', name: '时间 (s)', data: [] },
      yAxis: { type: 'value', name: '误差值', splitLine: { lineStyle: { type: 'dashed', color: '#333' } } },
      series: [
        { name: '位置误差 (m)', type: 'line', showSymbol: false, itemStyle: { color: '#ff9900' }, data: [] },
        { name: 'RMSE (m)', type: 'line', showSymbol: false, itemStyle: { color: '#00ccff' }, data: [] }
      ]
    });
  }
};

const updateCharts = () => {
  trajectoryChart?.setOption({ series: [{ data: truthPath }, { data: predPath }] });
  errorChart?.setOption({ xAxis: { data: timeData }, series: [{ data: errPosData }, { data: rmseData }] });
};

const handleResize = () => { trajectoryChart?.resize(); errorChart?.resize(); };

onMounted(() => { initCharts(); connectWebSocket(); window.addEventListener('resize', handleResize); });
onUnmounted(() => { ws?.close(); window.removeEventListener('resize', handleResize); trajectoryChart?.dispose(); errorChart?.dispose(); });
</script>

<style scoped>
/* 继承之前的风格，强化控件 UI */
.dashboard {
  height: 100vh; width: 100vw; background-color: #0d1117; color: #e6edf3; display: flex; flex-direction: column; font-family: 'Segoe UI', Tahoma, sans-serif;
}
.header {
  padding: 15px 20px; background-color: #161b22; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center;
}
.header h1 { margin: 0; font-size: 1.5rem; color: #58a6ff; }
.controls { display: flex; align-items: center; gap: 15px; }

.control-label { color: #8b949e; font-size: 0.9rem; }
.dark-input {
  background-color: #0d1117; color: #e6edf3; border: 1px solid #30363d; padding: 5px 10px; border-radius: 4px; outline: none; font-family: monospace;
}
.dark-input:focus { border-color: #58a6ff; }

/* 按钮样式 */
.control-btn {
  background-color: #238636; color: white; border: none; padding: 6px 16px; border-radius: 4px; cursor: pointer; font-weight: bold; transition: background 0.2s;
}
.control-btn:hover { background-color: #2ea043; }
.control-btn.paused { background-color: #d29922; }
.control-btn.paused:hover { background-color: #e3b341; }

/* 滑动条样式 */
.slider-container { display: flex; align-items: center; gap: 10px; background: #0d1117; padding: 4px 12px; border-radius: 20px; border: 1px solid #30363d; }
.time-text { font-family: monospace; font-size: 0.85rem; color: #8b949e; width: 40px; text-align: center; }
.time-slider { -webkit-appearance: none; width: 150px; height: 4px; background: #30363d; border-radius: 2px; outline: none; cursor: pointer; }
.time-slider::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 12px; height: 12px; border-radius: 50%; background: #58a6ff; cursor: pointer; transition: transform 0.1s; }
.time-slider::-webkit-slider-thumb:hover { transform: scale(1.3); }

.status { padding: 5px 12px; border-radius: 4px; font-size: 0.9rem; background-color: #3b2323; color: #ff7b72; }
.status.connected { background-color: #233624; color: #3fb950; }

.main-content { display: flex; flex: 1; padding: 20px; gap: 20px; overflow: hidden; }
.panel { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; }
.panel-header { font-weight: bold; margin-bottom: 15px; color: #8b949e; font-size: 1.1rem; }
.map-panel { flex: 2; }
.right-panels { flex: 1; display: flex; flex-direction: column; gap: 20px; }
.stats-panel { display: flex; flex-direction: row; justify-content: space-around; padding: 20px 10px; }
.stat-box { text-align: center; }
.stat-box .label { font-size: 0.9rem; color: #8b949e; margin-bottom: 8px; }
.stat-box .value { font-size: 1.8rem; font-weight: bold; color: #e6edf3; }
.stat-box .value.highlight { color: #ff7b72; }
.stat-box .value.warning { color: #00ccff; }
.chart-panel { flex: 1; }
.chart-container { flex: 1; width: 100%; min-height: 250px; }
</style>