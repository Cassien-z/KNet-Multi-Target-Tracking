<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, reactive, ref } from 'vue'
import * as echarts from 'echarts'
import 'echarts-gl'
import { listScenarios, type ScenarioInfo, type TrackingFrame, wsBase } from '../api'
import { formatNumber } from '../format'
import AppIcon from '../components/AppIcon.vue'

type ConnectionState = 'idle' | 'connecting' | 'live' | 'paused' | 'completed' | 'error'

const scenarios = ref<ScenarioInfo[]>([])
const selectedScenario = ref('')
const maneuverFilter = ref('CT')
const targetIndex = ref(0)
const loadingScenarios = ref(true)
const connectionState = ref<ConnectionState>('idle')
const connectionMessage = ref('等待选择场景')
const isSeeking = ref(false)
const seekValue = ref(0)
const webglAvailable = ref(true)
const renderedPoints = ref(0)
const pathLengthM = ref(0)
const trajectoryElement = ref<HTMLElement | null>(null)
const errorElement = ref<HTMLElement | null>(null)

const latest = reactive<TrackingFrame>({
  time: 0,
  max_time: 0,
  is_reset: false,
  scenario: '',
  target_index: 0,
  truth_pos: [0, 0, 0],
  pred_pos: [0, 0, 0],
  metrics: {
    err_pos: 0,
    rmse_current: 0,
    inference_time_ms: 0,
    sequence_inference_time_ms: 0,
  },
})

const truthPath: number[][] = []
const predictionPath: number[][] = []
const timeSeries: number[] = []
const errorSeries: number[] = []
const rmseSeries: number[] = []
let trajectoryChart: echarts.ECharts | null = null
let errorChart: echarts.ECharts | null = null
let socket: WebSocket | null = null
let connectionToken = 0

const selectedScenarioInfo = computed(() => scenarios.value.find((item) => item.id === selectedScenario.value))
const maneuverOptions = computed(() => ['CA', 'CT', 'CV', 'CSSW'].map((id) => ({
  id,
  count: scenarios.value.filter((item) => item.name.toUpperCase().includes(`TRACK_${id}_`)).length,
})).filter((item) => item.count > 0))
const filteredScenarios = computed(() => maneuverFilter.value === 'ALL'
  ? scenarios.value
  : scenarios.value.filter((item) => item.name.toUpperCase().includes(`TRACK_${maneuverFilter.value}_`)))
const progress = computed(() => latest.max_time > 0 ? Math.min(100, latest.time / latest.max_time * 100) : 0)
const canControl = computed(() => socket?.readyState === WebSocket.OPEN)
const statusLabel = computed(() => ({
  idle: '待机', connecting: '模型推理中', live: '实时回放', paused: '已暂停', completed: '回放完成', error: '连接异常',
})[connectionState.value])

function clearSeries() {
  truthPath.length = 0
  predictionPath.length = 0
  timeSeries.length = 0
  errorSeries.length = 0
  rmseSeries.length = 0
  renderedPoints.value = 0
  pathLengthM.value = 0
  updateCharts()
}

function relativePaths(): { truth: number[][]; prediction: number[][] } {
  if (!truthPath.length) return { truth: [], prediction: [] }
  const origin = truthPath[0] ?? [0, 0, 0]
  const translate = (point: number[]) => point.map((value, axis) => value - (origin[axis] ?? 0))
  return {
    truth: truthPath.map(translate),
    prediction: predictionPath.map(translate),
  }
}

function equalScaleBounds(paths: number[][][]): Array<{ min: number; max: number }> {
  const points = paths.flat()
  if (!points.length) return [{ min: -10, max: 10 }, { min: -10, max: 10 }, { min: -10, max: 10 }]
  const minima = [0, 1, 2].map((axis) => Math.min(...points.map((point) => point[axis] ?? 0)))
  const maxima = [0, 1, 2].map((axis) => Math.max(...points.map((point) => point[axis] ?? 0)))
  const commonSpan = Math.max(40, ...maxima.map((value, axis) => value - (minima[axis] ?? 0))) * 1.12
  const step = 10 ** Math.floor(Math.log10(commonSpan / 5))
  const roundedSpan = Math.ceil(commonSpan / step) * step
  return minima.map((minimum, axis) => {
    const center = (minimum + (maxima[axis] ?? minimum)) / 2
    return {
      min: Math.floor((center - roundedSpan / 2) / step) * step,
      max: Math.ceil((center + roundedSpan / 2) / step) * step,
    }
  })
}

function initCharts() {
  const probe = document.createElement('canvas')
  webglAvailable.value = Boolean(probe.getContext('webgl2') || probe.getContext('webgl'))
  if (trajectoryElement.value && webglAvailable.value) {
    trajectoryChart = echarts.init(trajectoryElement.value)
    trajectoryChart.setOption({
      animation: false,
      tooltip: { show: false },
      grid3D: {
        boxWidth: 105,
        boxHeight: 105,
        boxDepth: 105,
        environment: '#111917',
        axisLine: { lineStyle: { color: '#4d5a56' } },
        axisPointer: { show: false },
        splitLine: { lineStyle: { color: '#25312e', opacity: 0.9 } },
        splitArea: { show: false },
        viewControl: { projection: 'perspective', alpha: 24, beta: 38, distance: 170, autoRotate: false },
      },
      xAxis3D: { type: 'value', scale: true, name: 'ΔX / m', nameTextStyle: { color: '#83908c', fontSize: 10 }, axisLabel: { color: '#687570', fontSize: 9, formatter: (value: number) => Math.round(value).toLocaleString('zh-CN') } },
      yAxis3D: { type: 'value', scale: true, name: 'ΔY / m', nameTextStyle: { color: '#83908c', fontSize: 10 }, axisLabel: { color: '#687570', fontSize: 9, formatter: (value: number) => Math.round(value).toLocaleString('zh-CN') } },
      zAxis3D: { type: 'value', scale: true, name: 'ΔZ / m', nameTextStyle: { color: '#83908c', fontSize: 10 }, axisLabel: { color: '#687570', fontSize: 9, formatter: (value: number) => Math.round(value).toLocaleString('zh-CN') } },
      series: [
        { name: '真实轨迹', type: 'line3D', data: [], lineStyle: { color: '#e1aa45', width: 5, opacity: 1 } },
        { name: '预测轨迹', type: 'line3D', data: [], lineStyle: { color: '#e7654b', width: 4, opacity: 0.9 } },
        { name: '真实位置', type: 'scatter3D', data: [], symbolSize: 10, itemStyle: { color: '#f0bd5b' } },
        { name: '预测位置', type: 'scatter3D', data: [], symbolSize: 9, itemStyle: { color: '#e7654b' } },
      ],
    })
  }
  if (errorElement.value) {
    errorChart = echarts.init(errorElement.value)
    errorChart.setOption({
      animation: false,
      grid: { left: 48, right: 20, top: 32, bottom: 32 },
      tooltip: { trigger: 'axis', backgroundColor: '#17201f', borderWidth: 0, textStyle: { color: '#fff', fontSize: 10 } },
      legend: { top: 2, right: 5, itemWidth: 14, itemHeight: 2, textStyle: { color: '#69736f', fontSize: 9 } },
      xAxis: { type: 'category', boundaryGap: false, data: [], axisLine: { lineStyle: { color: '#c5c6bf' } }, axisTick: { show: false }, axisLabel: { color: '#7c837f', fontSize: 9 } },
      yAxis: { type: 'value', name: '误差 / m', nameTextStyle: { color: '#7c837f', fontSize: 9 }, axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: '#7c837f', fontSize: 9 }, splitLine: { lineStyle: { color: '#e4e1d9' } } },
      series: [
        { name: '位置误差', type: 'line', showSymbol: false, data: [], lineStyle: { color: '#e7654b', width: 1.6 }, areaStyle: { color: 'rgba(231,101,75,.08)' } },
        { name: '当前 RMSE', type: 'line', showSymbol: false, data: [], lineStyle: { color: '#2f7d64', width: 1.6 } },
      ],
    })
  }
}

function updateCharts() {
  const relative = relativePaths()
  const bounds = equalScaleBounds([relative.truth, relative.prediction])
  const truthPoint = relative.truth.length ? [relative.truth[relative.truth.length - 1]] : []
  const predictionPoint = relative.prediction.length ? [relative.prediction[relative.prediction.length - 1]] : []
  trajectoryChart?.setOption({
    xAxis3D: { min: bounds[0]?.min, max: bounds[0]?.max },
    yAxis3D: { min: bounds[1]?.min, max: bounds[1]?.max },
    zAxis3D: { min: bounds[2]?.min, max: bounds[2]?.max },
    series: [
      { data: relative.truth }, { data: relative.prediction }, { data: truthPoint }, { data: predictionPoint },
    ],
  })
  errorChart?.setOption({
    xAxis: { data: timeSeries.map((value) => value.toFixed(1)) },
    series: [{ data: errorSeries }, { data: rmseSeries }],
  })
}

function receiveFrame(frame: TrackingFrame) {
  if (frame.is_reset) clearSeries()
  Object.assign(latest, frame)
  latest.metrics = { ...frame.metrics }
  const previousTruth = truthPath[truthPath.length - 1]
  truthPath.push([...frame.truth_pos])
  predictionPath.push([...frame.pred_pos])
  timeSeries.push(frame.time)
  errorSeries.push(frame.metrics.err_pos)
  rmseSeries.push(frame.metrics.rmse_current)
  const maxPoints = 500
  if (truthPath.length > maxPoints) {
    truthPath.shift(); predictionPath.shift(); timeSeries.shift(); errorSeries.shift(); rmseSeries.shift()
  }
  if (previousTruth) {
    pathLengthM.value += Math.hypot(
      frame.truth_pos[0] - (previousTruth[0] ?? frame.truth_pos[0]),
      frame.truth_pos[1] - (previousTruth[1] ?? frame.truth_pos[1]),
      frame.truth_pos[2] - (previousTruth[2] ?? frame.truth_pos[2]),
    )
  }
  renderedPoints.value = truthPath.length
  if (!isSeeking.value) seekValue.value = frame.time
  updateCharts()
}

function disconnect() {
  connectionToken += 1
  if (socket) {
    socket.close()
    socket = null
  }
}

function connect() {
  if (!selectedScenario.value) return
  disconnect()
  clearSeries()
  latest.time = 0
  latest.max_time = 0
  seekValue.value = 0
  connectionState.value = 'connecting'
  connectionMessage.value = '后端正在执行完整序列推理'
  const token = connectionToken
  const url = `${wsBase}/ws/tracking?dataset=${encodeURIComponent(selectedScenario.value)}&target=${targetIndex.value}`
  const nextSocket = new WebSocket(url)
  socket = nextSocket

  nextSocket.onopen = () => {
    if (token !== connectionToken) return
    connectionState.value = 'live'
    connectionMessage.value = '正在接收跟踪帧'
  }
  nextSocket.onmessage = (event) => {
    if (token !== connectionToken) return
    try {
      receiveFrame(JSON.parse(event.data) as TrackingFrame)
    } catch {
      connectionState.value = 'error'
      connectionMessage.value = '收到无法解析的跟踪数据'
    }
  }
  nextSocket.onerror = () => {
    if (token !== connectionToken) return
    connectionState.value = 'error'
    connectionMessage.value = 'WebSocket 连接失败'
  }
  nextSocket.onclose = (event) => {
    if (token !== connectionToken) return
    socket = null
    const sequenceFinished = latest.max_time > 0 && latest.time >= latest.max_time
    connectionState.value = sequenceFinished ? 'completed' : 'error'
    connectionMessage.value = sequenceFinished ? '序列回放已经完成' : (event.reason || '连接已关闭，请检查目标编号')
  }
}

function changeManeuver() {
  if (!filteredScenarios.value.some((item) => item.id === selectedScenario.value)) {
    selectedScenario.value = filteredScenarios.value[0]?.id ?? ''
  }
  connect()
}

function togglePlayback() {
  if (!canControl.value || !socket) return
  const pause = connectionState.value !== 'paused'
  socket.send(JSON.stringify({ action: pause ? 'pause' : 'play' }))
  connectionState.value = pause ? 'paused' : 'live'
  connectionMessage.value = pause ? '回放已暂停，可拖动时间轴' : '正在接收跟踪帧'
}

function finishSeek() {
  isSeeking.value = false
  if (!canControl.value || !socket) return
  socket.send(JSON.stringify({ action: 'seek', time: seekValue.value }))
}

function setCamera(view: 'top' | 'side' | 'free') {
  const settings = view === 'top'
    ? { alpha: 90, beta: 0, distance: 190 }
    : view === 'side'
      ? { alpha: 8, beta: 0, distance: 170 }
      : { alpha: 24, beta: 38, distance: 170 }
  trajectoryChart?.setOption({ grid3D: { viewControl: settings } })
}

function exportFrames() {
  if (!timeSeries.length) return
  const rows = ['time_s,position_error_m,rmse_m,truth_x,truth_y,truth_z,pred_x,pred_y,pred_z']
  for (let index = 0; index < timeSeries.length; index += 1) {
    rows.push([
      timeSeries[index], errorSeries[index], rmseSeries[index],
      ...(truthPath[index] ?? []), ...(predictionPath[index] ?? []),
    ].join(','))
  }
  const blob = new Blob([`\ufeff${rows.join('\n')}`], { type: 'text/csv;charset=utf-8' })
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = `tracking-${selectedScenario.value}-target-${targetIndex.value}.csv`
  link.click()
  URL.revokeObjectURL(link.href)
}

function resizeCharts() {
  trajectoryChart?.resize()
  errorChart?.resize()
}

onMounted(async () => {
  await nextTick()
  initCharts()
  window.addEventListener('resize', resizeCharts)
  try {
    scenarios.value = await listScenarios()
    if (scenarios.value.length) {
      if (!maneuverOptions.value.some((item) => item.id === maneuverFilter.value)) {
        maneuverFilter.value = maneuverOptions.value[0]?.id ?? 'ALL'
      }
      selectedScenario.value = filteredScenarios.value[0]?.id ?? scenarios.value[0]?.id ?? ''
      connect()
    } else {
      connectionState.value = 'error'
      connectionMessage.value = '没有可用的验证场景'
    }
  } catch (error) {
    connectionState.value = 'error'
    connectionMessage.value = error instanceof Error ? error.message : '场景读取失败'
  } finally {
    loadingScenarios.value = false
  }
})

onUnmounted(() => {
  disconnect()
  trajectoryChart?.dispose()
  errorChart?.dispose()
  window.removeEventListener('resize', resizeCharts)
})
</script>

<template>
  <section class="page-shell tracking-page">
    <header class="page-heading tracking-heading">
      <div>
        <p class="eyebrow">TRACKING LAB / 跟踪验证</p>
        <h1>单场景跟踪验证</h1>
        <p class="page-description">选择注册场景与目标，观察端到端模型的三维航迹和逐帧误差。</p>
      </div>
      <div class="tracking-status" :class="connectionState">
        <i></i><span><b>{{ statusLabel }}</b><small>{{ connectionMessage }}</small></span>
      </div>
    </header>

    <div class="tracking-toolbar surface">
      <label class="maneuver-input">
        <span>机动类型</span>
        <select v-model="maneuverFilter" :disabled="loadingScenarios" @change="changeManeuver">
          <option value="ALL">全部 · {{ scenarios.length }}</option>
          <option v-for="option in maneuverOptions" :key="option.id" :value="option.id">{{ option.id }} · {{ option.count }}</option>
        </select>
      </label>
      <label>
        <span>验证场景</span>
        <select v-model="selectedScenario" :disabled="loadingScenarios" @change="connect">
          <option v-for="scenario in filteredScenarios" :key="scenario.id" :value="scenario.id">{{ scenario.id }} · {{ scenario.name }}</option>
        </select>
      </label>
      <label class="target-input">
        <span>目标编号</span>
        <input v-model.number="targetIndex" type="number" min="0" step="1" @change="connect" />
      </label>
      <div class="scenario-file"><span>数据文件</span><b>{{ selectedScenarioInfo?.file ?? '—' }}</b></div>
      <div class="toolbar-actions">
        <button class="button button-secondary" :disabled="!timeSeries.length" @click="exportFrames">导出当前数据</button>
        <button class="button button-primary" @click="connect"><AppIcon name="replay" :size="17" />重新推理</button>
      </div>
    </div>

    <div class="tracking-stage">
      <article class="trajectory-panel">
        <div class="instrument-head">
          <div><span>3D TRAJECTORY · RELATIVE TO FIRST TRUTH POINT</span><b>{{ latest.scenario || selectedScenarioInfo?.name || '等待数据' }}</b></div>
          <div class="trajectory-legend"><span class="truth"><i></i>真实轨迹</span><span class="prediction"><i></i>模型预测</span></div>
          <div class="camera-controls"><button @click="setCamera('top')">俯视</button><button @click="setCamera('side')">侧视</button><button @click="setCamera('free')">自由</button></div>
        </div>
        <div class="trajectory-canvas-wrap">
          <div ref="trajectoryElement" class="trajectory-canvas"></div>
          <div v-if="!webglAvailable" class="webgl-fallback">
            <AppIcon name="tracking" :size="30" />
            <b>三维渲染不可用</b>
            <span>当前浏览器未启用 WebGL，跟踪数据和误差曲线仍会正常接收。</span>
          </div>
        </div>
        <div class="canvas-readout">
          <div><span>TRUTH XYZ</span><b>{{ latest.truth_pos.map((value) => formatNumber(value, 1)).join(' / ') }}</b></div>
          <div><span>PRED XYZ</span><b>{{ latest.pred_pos.map((value) => formatNumber(value, 1)).join(' / ') }}</b></div>
          <div><span>POINTS / PATH</span><b>{{ renderedPoints }} / {{ formatNumber(pathLengthM, 1) }} m</b></div>
        </div>
      </article>

      <aside class="tracking-readout">
        <div class="readout-title"><p class="section-kicker">LIVE METRICS</p><h2>实时指标</h2></div>
        <div class="primary-reading"><span>当前位置误差</span><b>{{ formatNumber(latest.metrics.err_pos) }}<em>m</em></b><small>预测位置与真实位置的欧氏距离</small></div>
        <div class="secondary-readings">
          <div><span>当前 RMSE</span><b>{{ formatNumber(latest.metrics.rmse_current) }} <em>m</em></b></div>
          <div><span>估算单帧耗时</span><b>{{ formatNumber(latest.metrics.inference_time_ms, 3) }} <em>ms</em></b></div>
          <div><span>序列推理耗时</span><b>{{ formatNumber(latest.metrics.sequence_inference_time_ms, 1) }} <em>ms</em></b></div>
          <div><span>目标编号</span><b># {{ latest.target_index }}</b></div>
        </div>
        <div class="metric-gates">
          <div><span>RMSE ≤ 8 m</span><b :class="latest.metrics.rmse_current <= 8 ? 'pass' : 'fail'">{{ latest.metrics.rmse_current <= 8 ? '通过' : '观察中' }}</b></div>
          <div><span>单帧 ≤ 100 ms</span><b :class="latest.metrics.inference_time_ms <= 100 ? 'pass' : 'fail'">{{ latest.metrics.inference_time_ms <= 100 ? '通过' : '超限' }}</b></div>
        </div>
      </aside>
    </div>

    <article class="playback-panel surface">
      <div class="playback-topline">
        <div><p class="section-kicker">PLAYBACK CONTROL</p><h2>误差时间序列</h2></div>
        <div class="playback-buttons">
          <button class="round-control" :disabled="!canControl" :aria-label="connectionState === 'paused' ? '播放' : '暂停'" @click="togglePlayback">
            <AppIcon :name="connectionState === 'paused' ? 'play' : 'pause'" :size="18" />
          </button>
          <span>{{ formatNumber(isSeeking ? seekValue : latest.time, 1) }} s</span>
          <input
            v-model.number="seekValue"
            class="timeline-range"
            type="range"
            min="0"
            :max="latest.max_time || 1"
            step="0.1"
            :style="{ '--progress': `${isSeeking ? (seekValue / Math.max(latest.max_time, 1) * 100) : progress}%` }"
            :disabled="!latest.max_time"
            @input="isSeeking = true"
            @change="finishSeek"
          />
          <span>{{ formatNumber(latest.max_time, 1) }} s</span>
        </div>
      </div>
      <div ref="errorElement" class="error-canvas"></div>
    </article>
  </section>
</template>
