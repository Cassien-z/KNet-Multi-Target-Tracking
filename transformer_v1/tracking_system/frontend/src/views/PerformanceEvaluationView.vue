<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import type { BatchAssessment, BatchJob, ComparisonRow, CrossDatasetEvaluation, DashboardData, HistoricalReferences, MetricEvidenceStatus } from '../api'
import { generateMetricEvidence, getBatchAssessment, getCrossDatasetEvaluation, getEvaluationComparison, getHistoricalReferences, listBatchJobs } from '../api'
import { formatDate, formatNumber } from '../format'
import AppIcon from '../components/AppIcon.vue'

const props = defineProps<{
  data: DashboardData | null
}>()

const emit = defineEmits<{
  notice: [message: string, tone: 'success' | 'error']
}>()

const loading = ref(true)
const assessing = ref(false)
const completedJobs = ref<BatchJob[]>([])
const selectedJobId = ref('')
const baselineRmse = ref<number | null>(null)
const assessment = ref<BatchAssessment | null>(null)
const crossDataset = ref<CrossDatasetEvaluation | null>(null)
const references = ref<HistoricalReferences | null>(null)
const comparisons = ref<ComparisonRow[]>([])
const runningEvidenceMetric = ref<MetricEvidenceStatus['metric_id'] | null>(null)

const modelIds = ['v5', 'v6', 'v7']
const datasetIds = ['v5', 'v6', 'v7', 'v8_interference']
const datasetLabels: Record<string, string> = { v5: 'V5 标准', v6: 'V6 -30% 噪声', v7: 'V7 -40% 噪声', v8_interference: 'V8 强干扰' }
const globalReference = computed(() => references.value?.imm_jpda_comparison?.global)
const associationReference = computed(() => references.value?.association_audit)
const bestComparisonRmse = computed(() => comparisons.value.length ? Math.min(...comparisons.value.map((item) => item.position_rmse_m)) : null)
const metricEvidence = computed(() => {
  const evidence = assessment.value?.evidence
  if (!evidence) return []
  return [
    { name: '数据关联正确率脚本', ...evidence.association_accuracy },
    { name: 'IMM-JPDA 对比脚本', ...evidence.rmse_improvement_pct },
  ]
})

function matrixValue(modelId: string, datasetId: string): number | null {
  return crossDataset.value?.matrix.find((cell) => cell.model_id === modelId && cell.dataset_id === datasetId)?.position_rmse_m ?? null
}

function isBestCell(modelId: string, datasetId: string): boolean {
  const values = modelIds.map((id) => matrixValue(id, datasetId)).filter((value): value is number => value !== null)
  const value = matrixValue(modelId, datasetId)
  return value !== null && value === Math.min(...values)
}

function heatClass(value: number | null): string {
  if (value === null) return 'empty'
  if (value <= 8) return 'excellent'
  if (value <= 12) return 'good'
  if (value <= 30) return 'watch'
  return 'poor'
}

function thresholdText(operator: string, threshold: number, unit: string, display?: string): string {
  return `${operator === '<=' ? '≤' : '≥'} ${display ?? `${threshold} ${unit}`}`
}

function criterionValue(criterion: BatchAssessment['criteria'][number]): string {
  if (criterion.value === null) return '数据不足'
  if (criterion.unit === 'ratio') return `${formatNumber(criterion.value * 100, 2)}%`
  return `${formatNumber(criterion.value, criterion.id === 'inference_time_ms' ? 3 : 2)} ${criterion.unit}`
}

function assessmentLabel(status: BatchAssessment['overall_status']): string {
  return { passed: '全部通过', failed: '存在未达标项', incomplete: '证据不完整' }[status]
}

async function copyEvidenceCommand(command: string) {
  try {
    await navigator.clipboard.writeText(command)
    emit('notice', '测试命令已复制', 'success')
  } catch {
    emit('notice', '无法访问剪贴板，请手动复制命令', 'error')
  }
}

async function runEvidence(metricId: MetricEvidenceStatus['metric_id']) {
  if (!selectedJobId.value || runningEvidenceMetric.value) return
  runningEvidenceMetric.value = metricId
  try {
    const result = await generateMetricEvidence(selectedJobId.value, metricId)
    assessment.value = result.assessment
    emit('notice', '独立测试完成，评估结果已刷新', 'success')
  } catch (error) {
    emit('notice', error instanceof Error ? error.message : '独立测试执行失败', 'error')
  } finally {
    runningEvidenceMetric.value = null
  }
}

async function loadAssessment() {
  if (!selectedJobId.value) {
    assessment.value = null
    return
  }
  assessing.value = true
  try {
    assessment.value = await getBatchAssessment(selectedJobId.value, baselineRmse.value ?? undefined)
  } catch (error) {
    emit('notice', error instanceof Error ? error.message : '任务评估失败', 'error')
  } finally {
    assessing.value = false
  }
}

async function loadPage() {
  loading.value = true
  try {
    const [jobs, matrix, historical, rows] = await Promise.all([
      listBatchJobs(200), getCrossDatasetEvaluation(), getHistoricalReferences(), getEvaluationComparison(),
    ])
    completedJobs.value = jobs.filter((job) => job.status === 'completed')
    crossDataset.value = matrix
    references.value = historical
    comparisons.value = rows
    if (!selectedJobId.value && completedJobs.value.length) {
      selectedJobId.value = completedJobs.value[0]?.id ?? ''
      await loadAssessment()
    }
  } catch (error) {
    emit('notice', error instanceof Error ? error.message : '性能评估数据加载失败', 'error')
  } finally {
    loading.value = false
  }
}

onMounted(loadPage)
</script>

<template>
  <section class="page-shell evaluation-page">
    <header class="page-heading evaluation-heading">
      <div>
        <p class="eyebrow">EVALUATION / 性能评估</p>
        <h1>多维性能评估</h1>
        <p class="page-description">从单次批测、跨数据集鲁棒性和历史基线三个层级审视模型表现。</p>
      </div>
      <div class="evidence-summary">
        <div><span>批测证据</span><b>{{ completedJobs.length }}</b></div>
        <div><span>交叉组合</span><b>{{ crossDataset?.matrix.length ?? 0 }}</b></div>
        <div><span>指标门槛</span><b>{{ data?.criteria.length ?? 0 }}/4</b></div>
      </div>
    </header>

    <div class="criteria-gate-grid">
      <article v-for="(criterion, index) in data?.criteria ?? []" :key="criterion.id">
        <div><span>GATE 0{{ index + 1 }}</span><AppIcon :name="index === 0 ? 'target' : index === 1 ? 'tracking' : index === 2 ? 'evaluation' : 'clock'" :size="20" /></div>
        <b>{{ criterion.name }}</b>
        <strong>{{ thresholdText(criterion.operator, criterion.threshold, criterion.unit, criterion.display_threshold) }}</strong>
        <small>{{ criterion.source }} · 自动判定</small>
      </article>
    </div>

    <article class="surface assessment-workbench">
      <div class="assessment-toolbar">
        <div><p class="section-kicker">RUN ASSESSMENT</p><h2>批测任务判定</h2></div>
        <label><span>已完成任务</span><select v-model="selectedJobId" @change="baselineRmse = null; loadAssessment()"><option v-for="job in completedJobs" :key="job.id" :value="job.id">{{ job.name }} · {{ job.config.model_id.toUpperCase() }} × {{ job.config.dataset_id.toUpperCase() }}</option></select></label>
        <label class="baseline-field"><span>同条件 IMM-JPDA RMSE <small>仅供技术复核</small></span><div><input v-model.number="baselineRmse" type="number" min="0.0001" step="0.01" placeholder="输入复核值" /><em>m</em></div></label>
        <button class="button button-primary" :disabled="assessing || !selectedJobId" @click="loadAssessment"><AppIcon name="evaluation" :size="17" />{{ assessing ? '计算中' : '重新判定' }}</button>
      </div>

      <div v-if="assessment" class="assessment-body">
        <div class="assessment-verdict" :class="assessment.overall_status">
          <div class="verdict-mark"><span>{{ assessment.summary.passed }}</span><small>/ {{ assessment.summary.total }}</small></div>
          <div><span>OVERALL VERDICT</span><h3>{{ assessmentLabel(assessment.overall_status) }}</h3><p>{{ assessment.job_name }} · {{ assessment.model_id.toUpperCase() }} × {{ assessment.dataset_id.toUpperCase() }}</p><p v-if="!assessment.formal_evidence_complete">{{ assessment.formal_evidence_requirement }}</p></div>
          <dl><div><dt>通过</dt><dd>{{ assessment.summary.passed }}</dd></div><div><dt>未通过</dt><dd>{{ assessment.summary.failed }}</dd></div><div><dt>数据不足</dt><dd>{{ assessment.summary.unavailable }}</dd></div></dl>
        </div>

        <div class="assessment-criteria">
          <article v-for="criterion in assessment.criteria" :key="criterion.id" :class="criterion.status">
            <div class="criterion-state"><i></i><span>{{ criterion.status === 'passed' ? 'PASS' : criterion.status === 'failed' ? 'FAIL' : 'N/A' }}</span></div>
            <b>{{ criterion.name }}</b>
            <strong>{{ criterionValue(criterion) }}</strong>
            <small>门槛 {{ thresholdText(criterion.operator, criterion.threshold, criterion.unit, criterion.display_threshold) }}</small>
            <p v-if="criterion.note">{{ criterion.note }}</p>
          </article>
        </div>

        <div v-if="metricEvidence.length" class="metric-evidence-grid">
          <article v-for="item in metricEvidence" :key="item.metric_id" :class="{ accepted: item.accepted }">
            <div><span>{{ item.accepted ? 'EVIDENCE READY' : 'ACTION REQUIRED' }}</span><i></i></div>
            <b>{{ item.name }}</b>
            <p>{{ item.message }}</p>
            <code>{{ item.command }}</code>
            <div class="evidence-actions">
              <button type="button" class="evidence-run" :disabled="runningEvidenceMetric !== null" @click="runEvidence(item.metric_id)">{{ runningEvidenceMetric === item.metric_id ? '测试运行中…' : item.accepted ? '重新运行' : '运行测试' }}</button>
              <button type="button" @click="copyEvidenceCommand(item.command)">复制命令</button>
            </div>
          </article>
        </div>

        <div class="dimension-layout">
          <div class="dimension-metrics">
            <div><span>中位误差</span><b>{{ formatNumber(assessment.dimensions.accuracy.position_median_m) }} m</b></div>
            <div><span>P95 误差</span><b>{{ formatNumber(assessment.dimensions.accuracy.position_p95_m) }} m</b></div>
            <div><span>最大误差</span><b>{{ formatNumber(assessment.dimensions.accuracy.position_max_m) }} m</b></div>
            <div><span>航迹数</span><b>{{ assessment.dimensions.accuracy.evaluated_tracks }}</b></div>
            <div><span>≤8m 航迹占比</span><b>{{ assessment.dimensions.distribution ? formatNumber(assessment.dimensions.distribution.tracks_within_8m_ratio * 100, 1) : '—' }}%</b></div>
            <div><span>单帧实测</span><b>{{ formatNumber(assessment.dimensions.latency.measured_frame_ms ?? assessment.dimensions.latency.estimated_frame_ms, 3) }} ms</b></div>
          </div>
          <div class="maneuver-table">
            <div class="mini-section-title"><span>机动类型表现</span><small>逐航迹 RMSE</small></div>
            <div v-if="assessment.dimensions.by_maneuver.length" class="maneuver-rows">
              <div v-for="item in assessment.dimensions.by_maneuver" :key="item.maneuver"><b>{{ item.maneuver }}</b><span>{{ item.tracks }} 航迹</span><i><em :style="{ width: `${Math.min(100, item.mean_track_rmse_m / Math.max(assessment.dimensions.accuracy.position_p95_m, 1) * 100)}%` }"></em></i><strong>{{ formatNumber(item.mean_track_rmse_m) }} m</strong></div>
            </div>
            <div v-else class="maneuver-empty">当前任务的航迹名称不包含机动类型信息</div>
          </div>
        </div>
      </div>
      <div v-else class="assessment-empty">{{ loading ? '正在载入评估证据…' : '请选择一个已完成的批测任务' }}</div>
    </article>

    <div class="evaluation-lower-grid">
      <article class="surface cross-matrix-panel">
        <div class="surface-heading"><div><p class="section-kicker">ROBUSTNESS MATRIX</p><h2>跨数据集 RMSE</h2></div><div class="heat-legend"><span><i class="excellent"></i>≤8</span><span><i class="good"></i>≤12</span><span><i class="watch"></i>≤30</span><span><i class="poor"></i>&gt;30 m</span></div></div>
        <p class="surface-intro">每一行代表模型训练版本，每一列代表独立验证条件；标记表示该列最优结果。</p>
        <div class="cross-matrix">
          <div class="matrix-corner">MODEL \ DATA</div><div v-for="datasetId in datasetIds" :key="datasetId" class="matrix-head">{{ datasetLabels[datasetId] }}</div>
          <template v-for="modelId in modelIds" :key="modelId">
            <div class="matrix-model"><b>{{ modelId.toUpperCase() }}</b><span>Model</span></div>
            <div v-for="datasetId in datasetIds" :key="`${modelId}-${datasetId}`" class="matrix-cell" :class="heatClass(matrixValue(modelId, datasetId))"><span v-if="isBestCell(modelId, datasetId)" class="best-mark">BEST</span><b>{{ formatNumber(matrixValue(modelId, datasetId)) }}</b><small>m</small></div>
          </template>
        </div>
      </article>

      <article class="surface reference-panel">
        <div class="surface-heading"><div><p class="section-kicker">HISTORICAL EVIDENCE</p><h2>基线参考</h2></div><span class="reference-tag">REFERENCE ONLY</span></div>
        <div v-if="globalReference" class="imm-summary">
          <div><span>Transformer</span><b>{{ formatNumber(globalReference.tf_mean_rmse) }} m</b></div><AppIcon name="arrow" :size="18" /><div><span>IMM-JPDA</span><b>{{ formatNumber(globalReference.imm_mean_rmse) }} m</b></div><strong>↓ {{ formatNumber(globalReference.improvement_pct, 1) }}%</strong>
        </div>
        <div v-if="references?.imm_jpda_comparison" class="mode-bars">
          <div v-for="(item, mode) in references.imm_jpda_comparison.per_mode" :key="mode"><b>{{ mode }}</b><i><em :style="{ width: `${Math.min(100, item.improvement_pct)}%` }"></em></i><span>{{ formatNumber(item.improvement_pct, 1) }}%</span></div>
        </div>
        <div v-if="associationReference" class="association-reference">
          <div class="association-score"><span>关联审计正确率</span><b>{{ formatNumber(associationReference.accuracy * 100, 2) }}%</b><small>{{ associationReference.audited_files }} 文件 · {{ associationReference.total.toLocaleString('zh-CN') }} 观测</small></div>
          <div class="sensor-scores"><div v-for="(item, sensor) in associationReference.per_sensor" :key="sensor"><span>{{ sensor }}</span><b>{{ formatNumber(item.accuracy * 100, 2) }}%</b></div></div>
        </div>
        <p class="reference-notice">{{ references?.notice }}</p>
      </article>
    </div>

    <article class="surface comparison-panel">
      <div class="history-heading"><div><p class="section-kicker">RUN COMPARISON</p><h2>已完成任务对比</h2></div><span class="best-run-note">当前最低 RMSE · {{ formatNumber(bestComparisonRmse) }} m</span></div>
      <div class="history-table-wrap"><table class="comparison-table"><thead><tr><th>任务</th><th>模型 × 数据集</th><th>场景/航迹</th><th>RMSE</th><th>MAE</th><th>P95</th><th>单帧耗时</th><th>吞吐率</th><th>完成时间</th></tr></thead><tbody>
        <tr v-for="row in comparisons" :key="row.job_id"><td><b>{{ row.name }}</b><small>{{ row.job_id.slice(0, 8) }}</small></td><td><span>{{ row.model_id?.toUpperCase() || '—' }}</span> × {{ row.dataset_id?.toUpperCase() || '—' }}</td><td>{{ row.evaluated_scenarios }} / {{ row.evaluated_tracks }}</td><td :class="{ best: row.position_rmse_m === bestComparisonRmse }">{{ formatNumber(row.position_rmse_m) }} m</td><td>{{ formatNumber(row.position_mae_m) }} m</td><td>{{ formatNumber(row.position_p95_m) }} m</td><td>{{ formatNumber(row.estimated_frame_ms, 3) }} ms</td><td>{{ formatNumber(row.throughput_tracks_per_second, 1) }}/s</td><td>{{ formatDate(row.finished_at) }}</td></tr>
        <tr v-if="!comparisons.length"><td colspan="9" class="empty-history">暂无已完成批测</td></tr>
      </tbody></table></div>
    </article>
  </section>
</template>
