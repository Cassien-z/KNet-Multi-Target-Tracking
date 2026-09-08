<script setup lang="ts">
import { computed } from 'vue'
import type { DashboardData } from '../api'
import { formatBytes, formatCompact, formatDuration, formatNumber } from '../format'
import AppIcon from '../components/AppIcon.vue'

const props = defineProps<{
  data: DashboardData | null
  loading: boolean
  lastUpdated: Date | null
}>()

const emit = defineEmits<{
  refresh: []
  navigate: [page: 'models']
}>()

const activeModel = computed(() => props.data?.health.active_model)
const gpu = computed(() => props.data?.system.devices.find((device) => device.id.startsWith('cuda') && device.available))
const jobs = computed(() => props.data?.system.batch_testing.jobs ?? {})
const completedJobs = computed(() => jobs.value.completed ?? 0)
const runningJobs = computed(() => (jobs.value.running ?? 0) + (jobs.value.queued ?? 0))

function criterionDisplay(item: NonNullable<DashboardData['criteria']>[number]): string {
  const operator = item.operator === '<=' ? '≤' : '≥'
  if (item.display_threshold) return `${operator} ${item.display_threshold}`
  return `${operator} ${item.threshold} ${item.unit}`
}
</script>

<template>
  <section class="page-shell">
    <header class="page-heading overview-heading">
      <div>
        <p class="eyebrow">OPERATIONS / 总览</p>
        <h1>融合跟踪工作台</h1>
        <p class="page-description">掌握模型、算力与批测队列状态，快速进入下一项验证工作。</p>
      </div>
      <div class="heading-actions">
        <span v-if="lastUpdated" class="last-updated">更新于 {{ lastUpdated.toLocaleTimeString('zh-CN', { hour12: false }) }}</span>
        <button class="button button-secondary" :disabled="loading" @click="emit('refresh')">
          <AppIcon name="refresh" :size="17" />
          {{ loading ? '同步中' : '刷新数据' }}
        </button>
      </div>
    </header>

    <div class="status-ribbon" :class="{ muted: !data }">
      <div class="status-ribbon-main">
        <span class="live-mark"><i></i>{{ data ? '服务运行正常' : '等待后端连接' }}</span>
        <span v-if="data">API {{ data.health.service_version }}</span>
        <span v-if="data">运行 {{ formatDuration(data.health.uptime_seconds) }}</span>
      </div>
      <div v-if="data" class="status-ribbon-side">
        <span>{{ activeModel?.device }}</span>
        <span class="separator"></span>
        <span>{{ activeModel?.scenario_count }} 个可用场景</span>
      </div>
    </div>

    <div class="metric-grid">
      <article class="metric-card accent-ink">
        <div class="metric-topline"><span>当前部署</span><AppIcon name="cube" /></div>
        <strong>{{ activeModel?.model_id?.toUpperCase() ?? '—' }}</strong>
        <p>{{ activeModel?.model ?? '尚未连接模型服务' }}</p>
        <div class="metric-foot"><span class="tiny-dot green"></span>{{ data ? '已加载，可推理' : '状态未知' }}</div>
      </article>
      <article class="metric-card accent-coral">
        <div class="metric-topline"><span>模型参数量</span><AppIcon name="cpu" /></div>
        <strong>{{ formatCompact(activeModel?.parameters) }}</strong>
        <p>{{ activeModel?.architecture ?? '—' }}</p>
        <div class="metric-foot">序列长度 {{ activeModel?.sequence_length ?? '—' }}</div>
      </article>
      <article class="metric-card accent-amber">
        <div class="metric-topline"><span>批测队列</span><AppIcon name="batch" /></div>
        <strong>{{ runningJobs }}</strong>
        <p>正在运行或等待执行</p>
        <div class="metric-foot">累计完成 {{ completedJobs }} 项</div>
      </article>
      <article class="metric-card accent-green">
        <div class="metric-topline"><span>加速设备</span><AppIcon name="server" /></div>
        <strong>{{ gpu ? 'CUDA' : 'CPU' }}</strong>
        <p>{{ gpu?.name ?? '当前未发现可用 GPU' }}</p>
        <div class="metric-foot">{{ gpu?.total_memory_bytes ? formatBytes(gpu.total_memory_bytes) : '系统计算设备' }}</div>
      </article>
    </div>

    <div class="overview-layout">
      <article class="surface runtime-card">
        <div class="surface-heading">
          <div>
            <p class="section-kicker">RUNTIME PROFILE</p>
            <h2>运行环境</h2>
          </div>
          <span class="outline-badge">{{ data?.system.runtime.cuda_available ? 'GPU READY' : 'CPU MODE' }}</span>
        </div>
        <div class="runtime-diagram">
          <div class="runtime-node primary-node">
            <AppIcon name="model" :size="24" />
            <div><small>推理核心</small><b>{{ activeModel?.model ?? '—' }}</b></div>
          </div>
          <div class="runtime-link"><i></i><i></i><i></i></div>
          <div class="runtime-node-stack">
            <div class="runtime-node"><span>PY</span><div><small>Python</small><b>{{ data?.system.runtime.python ?? '—' }}</b></div></div>
            <div class="runtime-node"><span>PT</span><div><small>PyTorch</small><b>{{ data?.system.runtime.pytorch ?? '—' }}</b></div></div>
            <div class="runtime-node"><span>CU</span><div><small>CUDA Runtime</small><b>{{ data?.system.runtime.cuda_runtime ?? '不可用' }}</b></div></div>
          </div>
        </div>
        <dl class="detail-list">
          <div><dt>运行设备</dt><dd>{{ activeModel?.device ?? '—' }}</dd></div>
          <div><dt>协同节点</dt><dd>{{ activeModel?.num_uavs ?? '—' }} UAVs</dd></div>
          <div><dt>模型载入耗时</dt><dd>{{ formatNumber(activeModel?.load_time_ms, 1) }} ms</dd></div>
          <div><dt>批测工作线程</dt><dd>{{ data?.system.batch_testing.worker_count ?? '—' }}</dd></div>
        </dl>
      </article>

      <article class="surface criteria-card">
        <div class="surface-heading">
          <div>
            <p class="section-kicker">ACCEPTANCE GATE</p>
            <h2>任务书指标</h2>
          </div>
          <span class="count-mark">{{ data?.criteria.length ?? 0 }}/4</span>
        </div>
        <p class="surface-intro">后端已将四项指标固化为统一判定规则，批测完成后自动给出结论。</p>
        <ul class="criteria-list">
          <li v-for="(criterion, index) in data?.criteria ?? []" :key="criterion.id">
            <span class="criterion-index">0{{ index + 1 }}</span>
            <div><b>{{ criterion.name }}</b><small>{{ criterion.source }}</small></div>
            <strong>{{ criterionDisplay(criterion) }}</strong>
          </li>
        </ul>
        <div v-if="!data" class="empty-lines"><i></i><i></i><i></i><i></i></div>
      </article>
    </div>

    <article class="surface model-overview">
      <div class="surface-heading">
        <div>
          <p class="section-kicker">MODEL REGISTRY</p>
          <h2>模型仓库</h2>
        </div>
        <button class="text-button" @click="emit('navigate', 'models')">管理模型 <AppIcon name="arrow" :size="16" /></button>
      </div>
      <div class="model-table-wrap">
        <table class="data-table">
          <thead><tr><th>模型版本</th><th>训练数据</th><th>历史 RMSE</th><th>平均时延</th><th>权重大小</th><th>状态</th></tr></thead>
          <tbody>
            <tr v-for="model in data?.models ?? []" :key="model.id">
              <td><div class="model-cell"><span>{{ model.version }}</span><div><b>{{ model.name }}</b><small>{{ model.run_id }}</small></div></div></td>
              <td>{{ model.training_dataset.replace('_registered', '') }}</td>
              <td class="numeric">{{ formatNumber(model.metrics?.rmse_m) }} <small>m</small></td>
              <td class="numeric">{{ formatNumber(model.metrics?.mean_latency_ms, 1) }} <small>ms</small></td>
              <td>{{ formatNumber(model.size_mb, 1) }} MB</td>
              <td><span class="state-chip" :class="model.active ? 'active' : ''"><i></i>{{ model.active ? '当前部署' : '可用' }}</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </article>
  </section>
</template>
