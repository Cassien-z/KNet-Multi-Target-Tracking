<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import type { BatchAssessment, BatchJob, BatchJobStatus, DashboardData, ReportCatalog, ReportRecord } from '../api'
import {
  assessmentReportUrl,
  batchReportUrl,
  comparisonReportUrl,
  getBatchAssessment,
  getBatchJob,
  getReportCatalog,
} from '../api'
import { formatBytes, formatDate, formatNumber } from '../format'
import AppIcon from '../components/AppIcon.vue'

defineProps<{ data: DashboardData | null }>()

const emit = defineEmits<{
  notice: [message: string, tone: 'success' | 'error']
}>()

const emptyCatalog: ReportCatalog = {
  total: 0,
  summary: { completed: 0, failed_or_cancelled: 0, assessable: 0, total_size_bytes: 0 },
  items: [],
}

const catalog = ref<ReportCatalog>(emptyCatalog)
const loading = ref(true)
const query = ref('')
const statusFilter = ref<'all' | 'completed' | 'exceptions'>('all')
const modelFilter = ref('all')
const selectedIds = ref<string[]>([])
const detailRecord = ref<ReportRecord | null>(null)
const detailJob = ref<BatchJob | null>(null)
const detailAssessment = ref<BatchAssessment | null>(null)
const detailLoading = ref(false)
const assessing = ref(false)
const baselineRmse = ref<number | null>(null)

const modelOptions = computed(() => [...new Set(catalog.value.items.map((item) => item.model_id).filter(Boolean))].sort())
const filteredReports = computed(() => {
  const keyword = query.value.trim().toLowerCase()
  return catalog.value.items.filter((item) => {
    if (statusFilter.value === 'completed' && item.status !== 'completed') return false
    if (statusFilter.value === 'exceptions' && !['failed', 'cancelled'].includes(item.status)) return false
    if (modelFilter.value !== 'all' && item.model_id !== modelFilter.value) return false
    if (!keyword) return true
    return `${item.name} ${item.id} ${item.model_id} ${item.dataset_id}`.toLowerCase().includes(keyword)
  })
})
const selectableReports = computed(() => filteredReports.value.filter((item) => item.status === 'completed'))
const allVisibleSelected = computed(() => selectableReports.value.length > 0 && selectableReports.value.every((item) => selectedIds.value.includes(item.id)))
const bestRmse = computed(() => {
  const values = catalog.value.items.map((item) => item.metrics?.position_rmse_m).filter((value): value is number => value !== null && value !== undefined)
  return values.length ? Math.min(...values) : null
})

function statusText(status: BatchJobStatus): string {
  return { queued: '排队中', running: '运行中', completed: '已归档', failed: '失败', cancelled: '已取消' }[status]
}

function verdictText(status?: BatchAssessment['overall_status']): string {
  if (!status) return '不可评估'
  return { passed: '全部通过', failed: '存在未达标', incomplete: '证据不完整' }[status]
}

function toggleSelection(id: string) {
  selectedIds.value = selectedIds.value.includes(id)
    ? selectedIds.value.filter((item) => item !== id)
    : [...selectedIds.value, id]
}

function toggleVisible() {
  const visibleIds = selectableReports.value.map((item) => item.id)
  if (allVisibleSelected.value) {
    selectedIds.value = selectedIds.value.filter((id) => !visibleIds.includes(id))
  } else {
    selectedIds.value = [...new Set([...selectedIds.value, ...visibleIds])]
  }
}

async function loadPage() {
  loading.value = true
  try {
    catalog.value = await getReportCatalog()
    selectedIds.value = selectedIds.value.filter((id) => catalog.value.items.some((item) => item.id === id && item.status === 'completed'))
  } catch (error) {
    emit('notice', error instanceof Error ? error.message : '报告目录读取失败', 'error')
  } finally {
    loading.value = false
  }
}

async function openDetail(record: ReportRecord) {
  detailRecord.value = record
  detailJob.value = null
  detailAssessment.value = null
  baselineRmse.value = null
  detailLoading.value = true
  try {
    const jobPromise = getBatchJob(record.id)
    const assessmentPromise = record.status === 'completed' ? getBatchAssessment(record.id) : Promise.resolve(null)
    const [job, assessment] = await Promise.all([jobPromise, assessmentPromise])
    detailJob.value = job
    detailAssessment.value = assessment
  } catch (error) {
    emit('notice', error instanceof Error ? error.message : '报告详情读取失败', 'error')
  } finally {
    detailLoading.value = false
  }
}

async function reassess() {
  if (!detailRecord.value || detailRecord.value.status !== 'completed') return
  assessing.value = true
  try {
    detailAssessment.value = await getBatchAssessment(detailRecord.value.id, baselineRmse.value ?? undefined)
    emit('notice', baselineRmse.value ? '已使用同条件基线重新生成判定' : '已恢复无基线判定', 'success')
  } catch (error) {
    emit('notice', error instanceof Error ? error.message : '评估判定失败', 'error')
  } finally {
    assessing.value = false
  }
}

function exportComparison() {
  if (!selectedIds.value.length) return
  window.location.href = comparisonReportUrl(selectedIds.value)
}

function printReport() {
  if (!detailAssessment.value) return
  window.print()
}

onMounted(loadPage)
</script>

<template>
  <section class="page-shell report-page">
    <header class="page-heading report-heading">
      <div>
        <p class="eyebrow">REPORT ARCHIVE / 报告中心</p>
        <h1>验证成果归档</h1>
        <p class="page-description">集中检索批测记录、复核任务书结论，并导出可追溯的实验制品。</p>
      </div>
      <button class="button button-secondary" :disabled="loading" @click="loadPage"><AppIcon name="refresh" :size="17" />{{ loading ? '同步中' : '同步归档' }}</button>
    </header>

    <div class="report-summary-grid">
      <article><span>归档记录</span><b>{{ catalog.total }}</b><small>持久化批测任务</small></article>
      <article><span>完整报告</span><b>{{ catalog.summary.completed }}</b><small>可下载原始与评估制品</small></article>
      <article><span>可判定</span><b>{{ catalog.summary.assessable }}</b><small>已生成任务书结论</small></article>
      <article><span>归档体积</span><b>{{ formatBytes(catalog.summary.total_size_bytes) }}</b><small>原始 JSON 合计</small></article>
    </div>

    <div class="report-library-layout">
      <article class="surface report-library">
        <div class="report-toolbar">
          <div><p class="section-kicker">ARTIFACT LIBRARY</p><h2>报告档案</h2></div>
          <label class="report-search"><span>搜索</span><input v-model="query" placeholder="任务名称、ID、模型或数据集" /></label>
          <label><span>模型</span><select v-model="modelFilter"><option value="all">全部模型</option><option v-for="model in modelOptions" :key="model" :value="model">{{ model.toUpperCase() }}</option></select></label>
          <div class="report-filter-tabs"><button v-for="item in [{id:'all',label:'全部'},{id:'completed',label:'已完成'},{id:'exceptions',label:'异常'}]" :key="item.id" :class="{ active: statusFilter === item.id }" @click="statusFilter = item.id as typeof statusFilter">{{ item.label }}</button></div>
        </div>

        <div class="report-selection-bar">
          <label><input type="checkbox" :checked="allVisibleSelected" :disabled="!selectableReports.length" @change="toggleVisible" /><span>选择当前可导出记录</span></label>
          <span>已选择 <b>{{ selectedIds.length }}</b> 项</span>
          <button :disabled="!selectedIds.length" @click="exportComparison"><AppIcon name="report" :size="15" />导出对比 CSV</button>
        </div>

        <div class="report-table-wrap">
          <table class="report-table">
            <thead><tr><th></th><th>报告</th><th>实验组合</th><th>核心结果</th><th>任务书结论</th><th>完成时间</th><th>制品</th><th></th></tr></thead>
            <tbody>
              <tr v-for="record in filteredReports" :key="record.id">
                <td><input type="checkbox" :checked="selectedIds.includes(record.id)" :disabled="record.status !== 'completed'" :aria-label="`选择 ${record.name}`" @change="toggleSelection(record.id)" /></td>
                <td><b>{{ record.name }}</b><small>{{ record.id }}</small></td>
                <td><span class="report-pair">{{ record.model_id?.toUpperCase() || '—' }}</span><i>×</i><span>{{ record.dataset_id?.toUpperCase() || '—' }}</span><small>{{ record.scope ?? '完整数据集' }}{{ record.scope ? ' 场景' : '' }}</small></td>
                <td v-if="record.metrics"><b :class="{ 'best-report': record.metrics.position_rmse_m === bestRmse }">{{ formatNumber(record.metrics.position_rmse_m) }} m</b><small>RMSE · {{ record.metrics.evaluated_tracks ?? '—' }} 航迹</small></td><td v-else><b>—</b><small>无结果数据</small></td>
                <td><span class="archive-verdict" :class="record.assessment?.overall_status || record.status">{{ record.status === 'completed' ? verdictText(record.assessment?.overall_status) : statusText(record.status) }}</span><small v-if="record.assessment">{{ record.assessment.summary.passed }}/{{ record.assessment.summary.total }} 项通过</small></td>
                <td>{{ formatDate(record.finished_at) }}<small>{{ formatBytes(record.artifacts.report_size_bytes) }}</small></td>
                <td><span class="artifact-count">{{ [record.artifacts.raw_json, record.artifacts.tracks_csv, record.artifacts.assessment_json].filter(Boolean).length }}</span><small>种格式</small></td>
                <td><button class="row-action" aria-label="打开报告" @click="openDetail(record)"><AppIcon name="arrow" :size="16" /></button></td>
              </tr>
              <tr v-if="!filteredReports.length"><td colspan="8" class="empty-history">{{ loading ? '正在建立报告索引…' : '没有符合当前筛选条件的报告' }}</td></tr>
            </tbody>
          </table>
        </div>
      </article>

      <aside class="report-guidance">
        <div><p class="section-kicker">EXPORT POLICY</p><h2>制品说明</h2></div>
        <ol><li><span>01</span><div><b>原始 JSON</b><small>保留任务配置、模型路径、完整指标与逐航迹结果。</small></div></li><li><span>02</span><div><b>综合评估 JSON</b><small>加入任务书四项判定及同条件基线来源，适合归档复核。</small></div></li><li><span>03</span><div><b>逐航迹 CSV</b><small>便于在 Excel、MATLAB 或 Python 中继续统计。</small></div></li><li><span>04</span><div><b>对比 CSV</b><small>勾选多项已完成报告后生成统一横向结果表。</small></div></li></ol>
        <div class="report-integrity"><AppIcon name="check" :size="20" /><div><b>本地可追溯归档</b><span>所有制品均来自后端持久化任务，不在前端伪造指标。</span></div></div>
      </aside>
    </div>

    <div v-if="detailRecord" class="drawer-backdrop report-backdrop" @click.self="detailRecord = null">
      <aside class="job-drawer report-drawer" role="dialog" aria-modal="true" aria-labelledby="report-detail-title">
        <div class="drawer-head">
          <div><p class="section-kicker">ASSESSMENT REPORT / {{ detailRecord.id.slice(0, 8) }}</p><h2 id="report-detail-title">{{ detailRecord.name }}</h2></div>
          <button class="no-print" aria-label="关闭报告" @click="detailRecord = null"><AppIcon name="close" /></button>
        </div>

        <div v-if="detailLoading" class="report-detail-loading">正在装载完整报告…</div>
        <template v-else-if="detailJob">
          <div class="report-document-meta"><div><span>模型</span><b>{{ detailJob.config.model_id.toUpperCase() }}</b></div><div><span>数据集</span><b>{{ detailJob.config.dataset_id.toUpperCase() }}</b></div><div><span>测试范围</span><b>{{ detailJob.config.max_scenarios ?? '完整数据集' }}</b></div><div><span>完成时间</span><b>{{ formatDate(detailJob.finished_at) }}</b></div></div>

          <section v-if="detailAssessment" class="report-verdict-block" :class="detailAssessment.overall_status">
            <div><span>任务书综合结论</span><h3>{{ verdictText(detailAssessment.overall_status) }}</h3><small>{{ detailAssessment.summary.passed }} 通过 · {{ detailAssessment.summary.failed }} 未通过 · {{ detailAssessment.summary.unavailable }} 数据不足</small></div>
            <strong>{{ detailAssessment.summary.passed }}<small>/{{ detailAssessment.summary.total }}</small></strong>
          </section>

          <section v-if="detailJob.result" class="report-result-grid"><div><span>位置 RMSE</span><b>{{ formatNumber(detailJob.result.metrics.position_rmse_m) }} m</b></div><div><span>位置 MAE</span><b>{{ formatNumber(detailJob.result.metrics.position_mae_m) }} m</b></div><div><span>P95 误差</span><b>{{ formatNumber(detailJob.result.metrics.position_p95_m) }} m</b></div><div><span>估算单帧时延</span><b>{{ formatNumber(detailJob.result.latency.estimated_frame_ms, 3) }} ms</b></div><div><span>场景 / 航迹</span><b>{{ detailJob.result.metrics.evaluated_scenarios }} / {{ detailJob.result.metrics.evaluated_tracks }}</b></div><div><span>吞吐率</span><b>{{ formatNumber(detailJob.result.latency.throughput_tracks_per_second, 1) }}/s</b></div></section>

          <section v-if="detailAssessment" class="report-criteria-section"><div class="mini-section-title"><span>四项技术指标</span><small>任务书自动判定</small></div><div class="report-criteria-list"><div v-for="criterion in detailAssessment.criteria" :key="criterion.id" :class="criterion.status"><i></i><span><b>{{ criterion.name }}</b><small>{{ criterion.note || `${criterion.operator} ${criterion.display_threshold || `${criterion.threshold} ${criterion.unit}`}` }}</small></span><strong>{{ criterion.value === null ? 'N/A' : `${formatNumber(criterion.unit === 'ratio' ? criterion.value * 100 : criterion.value, 2)}${criterion.unit === 'ratio' ? '%' : ` ${criterion.unit}`}` }}</strong></div></div></section>

          <section v-if="detailRecord.status === 'completed'" class="report-baseline-control no-print"><label><span>同条件 IMM-JPDA RMSE <small>可选</small></span><div><input v-model.number="baselineRmse" type="number" min="0.0001" step="0.01" placeholder="输入基线后重新判定" /><em>m</em></div></label><button :disabled="assessing" @click="reassess">{{ assessing ? '判定中…' : '应用基线' }}</button></section>
          <section v-if="detailJob.error" class="drawer-error"><b>任务错误</b><span>{{ detailJob.error }}</span></section>

          <div class="report-downloads no-print">
            <a v-if="detailRecord.artifacts.raw_json" class="button button-secondary" :href="batchReportUrl(detailRecord.id, 'json')">原始 JSON</a>
            <a v-if="detailRecord.artifacts.tracks_csv" class="button button-secondary" :href="batchReportUrl(detailRecord.id, 'csv')">逐航迹 CSV</a>
            <a v-if="detailRecord.artifacts.assessment_json" class="button button-primary" :href="assessmentReportUrl(detailRecord.id, baselineRmse)">综合评估 JSON</a>
            <button v-if="detailAssessment" class="button print-button" @click="printReport"><AppIcon name="report" :size="16" />打印 / PDF</button>
          </div>
        </template>
      </aside>
    </div>
  </section>
</template>
