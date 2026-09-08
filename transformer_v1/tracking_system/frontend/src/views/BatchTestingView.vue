<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import type { BatchJob, BatchJobStatus, DashboardData, DatasetInfo } from '../api'
import { batchReportUrl, cancelBatchJob, createBatchJob, getBatchJob, listBatchJobs, listDatasets } from '../api'
import { formatDate, formatDuration, formatNumber } from '../format'
import AppIcon from '../components/AppIcon.vue'

const props = defineProps<{
  data: DashboardData | null
}>()

const emit = defineEmits<{
  notice: [message: string, tone: 'success' | 'error']
}>()

const datasets = ref<DatasetInfo[]>([])
const jobs = ref<BatchJob[]>([])
const selectedJob = ref<BatchJob | null>(null)
const loading = ref(true)
const submitting = ref(false)
const cancelling = ref(false)
const historyFilter = ref<'all' | 'active' | 'completed' | 'failed'>('all')
const scopeMode = ref<'sample' | 'full'>('sample')
let pollTimer: number | undefined

const form = reactive({
  name: '',
  model_id: 'v7',
  dataset_id: 'scenario_cooperative',
  device: 'cuda:0',
  warmup_frames: 5,
  batch_size: 64,
  workers: 0,
  max_scenarios: 10,
})

const activeJobs = computed(() => jobs.value.filter((job) => ['queued', 'running'].includes(job.status)))
const completedCount = computed(() => jobs.value.filter((job) => job.status === 'completed').length)
const failedCount = computed(() => jobs.value.filter((job) => ['failed', 'cancelled'].includes(job.status)).length)
const filteredJobs = computed(() => jobs.value.filter((job) => {
  if (historyFilter.value === 'all') return true
  if (historyFilter.value === 'active') return ['queued', 'running'].includes(job.status)
  if (historyFilter.value === 'failed') return ['failed', 'cancelled'].includes(job.status)
  return job.status === historyFilter.value
}))
const selectedDataset = computed(() => datasets.value.find((item) => item.id === form.dataset_id))
const datasetGroups = computed(() => [
  { id: 'scenario', title: '验收作战想定', subtitle: '验证层配置，不参与模型训练', items: datasets.value.filter((item) => item.kind === 'acceptance_scenario') },
  { id: 'technical', title: '技术验证数据集', subtitle: '用于模型版本与噪声条件对比', items: datasets.value.filter((item) => item.kind !== 'acceptance_scenario') },
].filter((group) => group.items.length))
const selectedModel = computed(() => props.data?.models.find((item) => item.id === form.model_id))
const availableDevices = computed(() => props.data?.devices.filter((item) => item.available) ?? [])

watch(() => props.data, (data) => {
  if (!data) return
  const active = data.models.find((model) => model.active)
  if (active && !submitting.value) form.model_id = active.id
  const cuda = data.devices.find((device) => device.available && device.id.startsWith('cuda'))
  if (cuda && form.device === 'cuda:0') form.device = cuda.id
}, { immediate: true })

function statusText(status: BatchJobStatus): string {
  return { queued: '排队中', running: '运行中', completed: '已完成', failed: '失败', cancelled: '已取消' }[status]
}

function statusTone(status: BatchJobStatus): string {
  if (status === 'completed') return 'success'
  if (status === 'running') return 'running'
  if (status === 'queued') return 'queued'
  return 'danger'
}

function elapsed(job: BatchJob): string {
  if (!job.started_at) return '—'
  const end = job.finished_at ? new Date(job.finished_at).getTime() : Date.now()
  return formatDuration((end - new Date(job.started_at).getTime()) / 1000)
}

function datasetCode(dataset: DatasetInfo): string {
  if (dataset.id === 'scenario_cooperative') return 'S1'
  if (dataset.id === 'scenario_maneuver_jamming') return 'S2'
  return dataset.id.replace('_interference', '').toUpperCase()
}

function frameLatency(job: BatchJob): number {
  return job.result?.latency.measured_frame_ms ?? job.result?.latency.estimated_frame_ms ?? 0
}

function manifestSummary(job: BatchJob): string {
  const counts = job.result?.scenario_manifest?.class_counts
  if (!counts) return '历史任务（未记录）'
  return Object.entries(counts).map(([name, count]) => `${name} ${count}`).join(' · ')
}

async function loadPage(silent = false) {
  if (!silent) loading.value = true
  try {
    let jobItems: BatchJob[]
    if (silent) {
      jobItems = await listBatchJobs()
    } else {
      const [datasetItems, initialJobs] = await Promise.all([listDatasets(), listBatchJobs()])
      datasets.value = datasetItems
      jobItems = initialJobs
    }
    jobs.value = jobItems
    if (!datasets.value.some((item) => item.id === form.dataset_id && item.available)) {
      form.dataset_id = datasets.value.find((item) => item.available)?.id ?? ''
    }
    if (selectedJob.value) {
      const latest = jobItems.find((item) => item.id === selectedJob.value?.id)
      if (latest) selectedJob.value = await getBatchJob(latest.id)
    }
  } catch (error) {
    if (!silent) emit('notice', error instanceof Error ? error.message : '批测数据加载失败', 'error')
  } finally {
    loading.value = false
  }
}

async function submitJob() {
  if (!form.model_id || !form.dataset_id || !form.device) return
  submitting.value = true
  try {
    const job = await createBatchJob({
      name: form.name.trim() || null,
      model_id: form.model_id,
      dataset_id: form.dataset_id,
      device: form.device,
      warmup_frames: form.warmup_frames,
      batch_size: form.batch_size,
      workers: form.workers,
      max_scenarios: scopeMode.value === 'full' ? null : form.max_scenarios,
    })
    jobs.value = [job, ...jobs.value.filter((item) => item.id !== job.id)]
    selectedJob.value = job
    form.name = ''
    emit('notice', `批测任务 ${job.id.slice(0, 8)} 已进入队列`, 'success')
  } catch (error) {
    emit('notice', error instanceof Error ? error.message : '批测任务提交失败', 'error')
  } finally {
    submitting.value = false
  }
}

async function openJob(job: BatchJob) {
  try {
    selectedJob.value = await getBatchJob(job.id)
  } catch (error) {
    emit('notice', error instanceof Error ? error.message : '任务详情读取失败', 'error')
  }
}

async function cancelSelectedJob() {
  if (!selectedJob.value || !['queued', 'running'].includes(selectedJob.value.status)) return
  cancelling.value = true
  try {
    selectedJob.value = await cancelBatchJob(selectedJob.value.id)
    emit('notice', '已提交任务取消请求', 'success')
    await loadPage(true)
  } catch (error) {
    emit('notice', error instanceof Error ? error.message : '任务取消失败', 'error')
  } finally {
    cancelling.value = false
  }
}

onMounted(async () => {
  await loadPage()
  pollTimer = window.setInterval(() => loadPage(true), 2500)
})

onUnmounted(() => {
  if (pollTimer) window.clearInterval(pollTimer)
})
</script>

<template>
  <section class="page-shell batch-page">
    <header class="page-heading batch-heading">
      <div>
        <p class="eyebrow">BATCH WORKBENCH / 批量测试</p>
        <h1>批测任务中心</h1>
        <p class="page-description">编排模型与数据集组合，监控后台推理进度，并沉淀可复现的评估报告。</p>
      </div>
      <div class="batch-summary-strip">
        <div><span>活动任务</span><b>{{ activeJobs.length }}</b></div>
        <div><span>已完成</span><b>{{ completedCount }}</b></div>
        <div><span>异常/取消</span><b>{{ failedCount }}</b></div>
      </div>
    </header>

    <div class="batch-compose-layout">
      <article class="surface job-composer">
        <div class="surface-heading">
          <div><p class="section-kicker">NEW EXPERIMENT</p><h2>创建批测任务</h2></div>
          <span class="composer-step">01 / CONFIGURE</span>
        </div>

        <div class="compose-grid">
          <label class="field full-field"><span>任务名称 <small>可选</small></span><input v-model="form.name" maxlength="100" placeholder="例如：V7 低噪完整验证" /></label>
          <label class="field"><span>测试模型</span><select v-model="form.model_id"><option v-for="model in data?.models ?? []" :key="model.id" :value="model.id">{{ model.version }} · {{ model.name }}</option></select></label>
          <label class="field"><span>计算设备</span><select v-model="form.device"><option v-for="device in availableDevices" :key="device.id" :value="device.id">{{ device.id.toUpperCase() }} · {{ device.name }}</option></select></label>
        </div>

        <div class="dataset-picker-title"><span>选择验证范围</span><small>{{ selectedDataset?.scenario_count ?? 0 }} 个场景可用</small></div>
        <div v-for="group in datasetGroups" :key="group.id" class="dataset-group">
          <div class="dataset-group-title"><b>{{ group.title }}</b><span>{{ group.subtitle }}</span></div>
          <div class="dataset-picker" :class="`dataset-picker-${group.id}`">
            <button
              v-for="dataset in group.items"
              :key="dataset.id"
              :disabled="!dataset.available"
              :class="{ selected: form.dataset_id === dataset.id }"
              @click="form.dataset_id = dataset.id"
            >
              <span class="dataset-code">{{ datasetCode(dataset) }}</span>
              <span><b>{{ dataset.name }}</b><small>{{ dataset.description }}</small></span>
              <i></i>
            </button>
          </div>
        </div>

        <div v-if="selectedDataset?.kind === 'acceptance_scenario'" class="scenario-profile">
          <div><span>想定说明</span><p>{{ selectedDataset.scenario_note }}</p></div>
          <dl>
            <div><dt>数据分层</dt><dd><em v-for="item in selectedDataset.scenario_classes" :key="item">{{ item }}</em></dd></div>
            <div><dt>融合信源</dt><dd><em v-for="item in selectedDataset.sensor_modalities" :key="item">{{ item }}</em></dd></div>
            <div><dt>干扰条件</dt><dd><em v-for="item in selectedDataset.interference_modes" :key="item">{{ item }}</em></dd></div>
          </dl>
        </div>

        <div class="scope-selector">
          <div><span>测试范围</span><small>抽样测试会在不同机动类别间均衡取样，正式评估可选择完整数据集。</small></div>
          <div class="scope-options">
            <button :class="{ active: scopeMode === 'sample' }" @click="scopeMode = 'sample'">抽样测试</button>
            <button :class="{ active: scopeMode === 'full' }" @click="scopeMode = 'full'">完整测试</button>
          </div>
          <label v-if="scopeMode === 'sample'"><input v-model.number="form.max_scenarios" type="number" min="1" :max="selectedDataset?.scenario_count || 2000" /><span>个场景</span></label>
          <strong v-else>{{ selectedDataset?.scenario_count ?? '—' }} 个场景</strong>
        </div>

        <details class="advanced-config">
          <summary>高级推理参数 <span>序列长度由后端固定为 100</span></summary>
          <div>
            <label class="field"><span>预热帧数</span><input v-model.number="form.warmup_frames" type="number" min="0" max="99" /></label>
            <label class="field"><span>批大小</span><input v-model.number="form.batch_size" type="number" min="1" max="256" /></label>
            <label class="field"><span>数据线程</span><input v-model.number="form.workers" type="number" min="0" max="8" /></label>
          </div>
        </details>

        <div class="compose-review">
          <div><span>MODEL</span><b>{{ selectedModel?.version ?? '—' }}</b></div>
          <AppIcon name="arrow" :size="16" />
          <div><span>DATASET</span><b>{{ selectedDataset ? datasetCode(selectedDataset) : '—' }}</b></div>
          <AppIcon name="arrow" :size="16" />
          <div><span>SCOPE</span><b>{{ scopeMode === 'full' ? 'FULL' : form.max_scenarios }}</b></div>
          <button class="submit-job" :disabled="submitting || !form.dataset_id" @click="submitJob">
            <span>{{ submitting ? '正在提交…' : '提交批测任务' }}</span><AppIcon name="arrow" :size="17" />
          </button>
        </div>
      </article>

      <aside class="batch-queue-panel">
        <div class="queue-heading"><div><p class="section-kicker">LIVE QUEUE</p><h2>运行队列</h2></div><span>{{ activeJobs.length }}</span></div>
        <div v-if="activeJobs.length" class="active-job-list">
          <button v-for="job in activeJobs" :key="job.id" @click="openJob(job)">
            <div class="active-job-top"><span class="job-status" :class="statusTone(job.status)"><i></i>{{ statusText(job.status) }}</span><small>{{ job.id.slice(0, 8) }}</small></div>
            <b>{{ job.name }}</b>
            <div class="job-pair"><span>{{ job.config.model_id.toUpperCase() }}</span><AppIcon name="arrow" :size="13" /><span>{{ job.config.dataset_id.toUpperCase() }}</span></div>
            <div class="progress-track"><i :style="{ width: `${job.progress.percent}%` }"></i></div>
            <div class="progress-meta"><span>{{ job.progress.message }}</span><strong>{{ formatNumber(job.progress.percent, 0) }}%</strong></div>
          </button>
        </div>
        <div v-else class="empty-queue">
          <AppIcon name="batch" :size="28" />
          <b>队列空闲</b>
          <span>新任务提交后将在此显示实时进度。</span>
        </div>
        <div class="queue-policy"><span>执行策略</span><b>单 GPU 串行队列</b><small>避免并发批测争抢显存</small></div>
      </aside>
    </div>

    <article class="surface job-history">
      <div class="history-heading">
        <div><p class="section-kicker">PERSISTED RUNS</p><h2>任务历史</h2></div>
        <div class="history-filters">
          <button v-for="item in [{id:'all',label:'全部'},{id:'active',label:'活动'},{id:'completed',label:'已完成'},{id:'failed',label:'异常'}]" :key="item.id" :class="{ active: historyFilter === item.id }" @click="historyFilter = item.id as typeof historyFilter">{{ item.label }}</button>
        </div>
      </div>
      <div class="history-table-wrap">
        <table class="batch-table">
          <thead><tr><th>任务</th><th>组合</th><th>范围</th><th>进度/结果</th><th>耗时</th><th>创建时间</th><th>状态</th><th></th></tr></thead>
          <tbody>
            <tr v-for="job in filteredJobs" :key="job.id">
              <td><b>{{ job.name }}</b><small>{{ job.id }}</small></td>
              <td><span class="pair-chip">{{ job.config.model_id.toUpperCase() }}</span><span class="pair-arrow">×</span><span class="pair-chip pale">{{ job.config.dataset_id.toUpperCase() }}</span></td>
              <td>{{ job.config.max_scenarios ?? '完整' }} <small>{{ job.config.max_scenarios ? '场景' : '数据集' }}</small></td>
              <td v-if="job.result" class="result-cell"><b>{{ formatNumber(job.result.metrics.position_rmse_m) }} m</b><small>RMSE · {{ job.result.metrics.evaluated_tracks }} 航迹</small></td>
              <td v-else class="progress-cell"><div><i :style="{ width: `${job.progress.percent}%` }"></i></div><small>{{ formatNumber(job.progress.percent, 0) }}%</small></td>
              <td>{{ elapsed(job) }}</td>
              <td>{{ formatDate(job.created_at) }}</td>
              <td><span class="job-status" :class="statusTone(job.status)"><i></i>{{ statusText(job.status) }}</span></td>
              <td><button class="row-action" @click="openJob(job)"><AppIcon name="arrow" :size="16" /></button></td>
            </tr>
            <tr v-if="!filteredJobs.length"><td colspan="8" class="empty-history">{{ loading ? '正在读取任务…' : '暂无符合条件的任务' }}</td></tr>
          </tbody>
        </table>
      </div>
    </article>

    <div v-if="selectedJob" class="drawer-backdrop" @click.self="selectedJob = null">
      <aside class="job-drawer" role="dialog" aria-modal="true" aria-labelledby="job-detail-title">
        <div class="drawer-head">
          <div><p class="section-kicker">RUN DETAIL / {{ selectedJob.id.slice(0, 8) }}</p><h2 id="job-detail-title">{{ selectedJob.name }}</h2></div>
          <button aria-label="关闭详情" @click="selectedJob = null"><AppIcon name="close" /></button>
        </div>
        <div class="drawer-status-line">
          <span class="job-status" :class="statusTone(selectedJob.status)"><i></i>{{ statusText(selectedJob.status) }}</span>
          <span>{{ selectedJob.progress.message }}</span>
          <b>{{ formatNumber(selectedJob.progress.percent, 0) }}%</b>
        </div>
        <div class="drawer-progress"><i :style="{ width: `${selectedJob.progress.percent}%` }"></i></div>

        <section class="drawer-section"><p>任务配置</p><dl class="drawer-config">
          <div><dt>模型</dt><dd>{{ selectedJob.config.model_id.toUpperCase() }}</dd></div><div><dt>数据集</dt><dd>{{ selectedJob.config.dataset_id }}</dd></div>
          <div><dt>设备</dt><dd>{{ selectedJob.config.device }}</dd></div><div><dt>测试范围</dt><dd>{{ selectedJob.config.max_scenarios ?? '完整数据集' }}</dd></div>
          <div><dt>批大小</dt><dd>{{ selectedJob.config.batch_size }}</dd></div><div><dt>预热帧</dt><dd>{{ selectedJob.config.warmup_frames }}</dd></div>
          <div><dt>抽样策略</dt><dd>{{ selectedJob.config.sampling_strategy === 'stratified_by_scenario_type' ? '机动类型分层' : '历史顺序抽样' }}</dd></div><div><dt>场景构成</dt><dd>{{ manifestSummary(selectedJob) }}</dd></div>
        </dl></section>

        <section v-if="selectedJob.result" class="drawer-section"><p>核心结果</p>
          <div class="drawer-metrics">
            <div class="featured"><span>位置 RMSE</span><b>{{ formatNumber(selectedJob.result.metrics.position_rmse_m) }}<em>m</em></b></div>
            <div><span>位置 MAE</span><b>{{ formatNumber(selectedJob.result.metrics.position_mae_m) }} m</b></div>
            <div><span>P95 误差</span><b>{{ formatNumber(selectedJob.result.metrics.position_p95_m) }} m</b></div>
            <div><span>单帧耗时</span><b>{{ formatNumber(frameLatency(selectedJob), 3) }} ms</b></div>
            <div><span>评估航迹</span><b>{{ selectedJob.result.metrics.evaluated_tracks }}</b></div>
            <div><span>吞吐率</span><b>{{ formatNumber(selectedJob.result.latency.throughput_tracks_per_second, 1) }}/s</b></div>
          </div>
        </section>

        <section v-if="selectedJob.error" class="drawer-error"><b>任务错误</b><span>{{ selectedJob.error }}</span></section>

        <div class="drawer-timeline"><div><span>创建</span><b>{{ formatDate(selectedJob.created_at) }}</b></div><div><span>开始</span><b>{{ formatDate(selectedJob.started_at) }}</b></div><div><span>结束</span><b>{{ formatDate(selectedJob.finished_at) }}</b></div></div>

        <div class="drawer-actions">
          <button v-if="['queued', 'running'].includes(selectedJob.status)" class="button cancel-button" :disabled="cancelling" @click="cancelSelectedJob">{{ cancelling ? '正在取消…' : '取消任务' }}</button>
          <template v-if="selectedJob.status === 'completed'">
            <a class="button button-secondary" :href="batchReportUrl(selectedJob.id, 'json')" target="_blank">JSON 报告</a>
            <a class="button button-primary" :href="batchReportUrl(selectedJob.id, 'csv')">逐航迹 CSV <AppIcon name="arrow" :size="16" /></a>
          </template>
        </div>
      </aside>
    </div>
  </section>
</template>
