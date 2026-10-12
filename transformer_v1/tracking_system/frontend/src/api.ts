export interface ActiveModel {
  model_id: string
  model: string
  architecture: string
  weights: string
  data_dir: string
  device: string
  cuda_available: boolean
  sequence_length: number
  num_uavs: number
  parameters: number
  scenario_count: number
  activated_at: string
  load_time_ms: number
}

export interface BatchStatus {
  worker_count: number
  report_dir?: string
  jobs: Record<string, number>
  active_job_ids: string[]
}

export interface HealthResponse {
  status: string
  service_version: string
  uptime_seconds: number
  active_model: ActiveModel
  batch_testing: BatchStatus
}

export interface DeviceInfo {
  id: string
  name: string
  available: boolean
  total_memory_bytes?: number
  compute_capability?: string
}

export interface SystemResponse {
  service: {
    name: string
    version: string
    uptime_seconds: number
  }
  runtime: {
    python: string
    pytorch: string
    platform: string
    cuda_available: boolean
    cuda_runtime: string | null
  }
  devices: DeviceInfo[]
  active_model: ActiveModel
  batch_testing: BatchStatus
}

export interface ModelMetrics {
  rmse_m: number | null
  mae_m: number | null
  p95_m: number | null
  max_error_m: number | null
  mean_latency_ms: number | null
  p95_latency_ms: number | null
  epochs: number | null
  d_model: number | null
}

export interface ModelInfo {
  id: string
  name: string
  version: string
  run_id: string
  training_dataset: string
  validation_dataset: string
  file_name: string
  path: string
  size_bytes: number
  size_mb: number
  modified_at: string
  active: boolean
  metrics: ModelMetrics | null
}

export interface CriterionInfo {
  id: string
  name: string
  operator: '<=' | '>='
  threshold: number
  display_threshold?: string
  unit: string
  source: string
}

export interface ScenarioInfo {
  id: string
  name: string
  file: string
  size_bytes: number
}

export interface TrackingFrame {
  time: number
  max_time: number
  is_reset: boolean
  scenario: string
  target_index: number
  truth_pos: [number, number, number]
  pred_pos: [number, number, number]
  metrics: {
    err_pos: number
    rmse_current: number
    inference_time_ms: number
    sequence_inference_time_ms: number
  }
}

export interface DatasetInfo {
  id: string
  name: string
  description: string
  path: string
  available: boolean
  scenario_count: number
  kind: 'acceptance_scenario' | 'technical_dataset'
  scenario_classes: string[]
  sensor_modalities: string[]
  interference_modes: string[]
  scenario_note: string | null
}

export type BatchJobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface BatchJobConfig {
  model_id: string
  dataset_id: string
  device: string
  warmup_frames: number
  batch_size: number
  workers: number
  max_scenarios: number | null
  sequence_length: number
  sampling_strategy?: 'leading_sorted' | 'stratified_by_scenario_type'
}

export interface BatchProgress {
  stage: string
  message: string
  completed_batches: number
  total_batches: number
  processed_tracks: number
  total_tracks: number
  percent: number
}

export interface BatchMetrics {
  position_rmse_m: number
  position_mae_m: number
  position_median_m: number
  position_p95_m: number
  position_max_m: number
  mean_track_rmse_m: number
  evaluated_scenarios: number
  evaluated_tracks: number
  evaluated_points: number
}

export interface BatchLatency {
  total_inference_ms: number
  mean_batch_ms: number
  p95_batch_ms: number
  mean_track_ms: number
  estimated_frame_ms: number
  measured_frame_ms?: number
  p95_measured_frame_ms?: number
  mean_single_sequence_ms?: number
  latency_method?: string
  throughput_tracks_per_second: number
  cold_start_batch_excluded: boolean
}

export interface AssociationSummary {
  audited_files: number
  correct: number
  total: number
  accuracy: number | null
  per_sensor: Record<string, { correct: number; total: number; accuracy: number | null }>
}

export interface TrackMetric {
  track: string
  valid_frames: number
  rmse_m: number
  mae_m: number
  p95_m: number
  max_error_m: number
}

export interface BatchResult {
  model_id: string
  model_path?: string
  dataset_id: string
  dataset_path?: string
  metrics: BatchMetrics
  latency: BatchLatency
  association: AssociationSummary | null
  tracks?: TrackMetric[]
  report_path: string
  scenario_manifest?: {
    sampling_strategy: string
    files: string[]
    class_counts: Record<string, number>
  }
}

export interface BatchJob {
  id: string
  name: string
  status: BatchJobStatus
  created_at: string
  started_at: string | null
  finished_at: string | null
  config: BatchJobConfig
  progress: BatchProgress
  result: BatchResult | null
  error: string | null
}

export interface CreateBatchJobPayload {
  name: string | null
  model_id: string
  dataset_id: string
  device: string
  warmup_frames: number
  batch_size: number
  workers: number
  max_scenarios: number | null
}

export type CriterionStatus = 'passed' | 'failed' | 'unavailable'

export interface AssessmentCriterion extends CriterionInfo {
  value: number | null
  status: CriterionStatus
  passed: boolean | null
  note: string | null
}

export interface TrackDistribution {
  minimum_rmse_m: number
  median_rmse_m: number
  p75_rmse_m: number
  p90_rmse_m: number
  p95_rmse_m: number
  maximum_rmse_m: number
  tracks_within_8m: number
  tracks_within_8m_ratio: number
}

export interface ManeuverAssessment {
  maneuver: string
  tracks: number
  mean_track_rmse_m: number
  median_track_rmse_m: number
  p95_track_rmse_m: number
  max_track_rmse_m: number
}

export interface MetricEvidenceStatus {
  metric_id: 'association_accuracy' | 'rmse_improvement_pct'
  available: boolean
  accepted: boolean
  path: string
  command: string
  message: string
}

export interface BatchAssessment {
  job_id: string
  job_name: string
  model_id: string
  dataset_id: string
  overall_status: 'passed' | 'failed' | 'incomplete'
  formal_evidence_complete: boolean
  formal_evidence_requirement: string
  summary: { total: number; passed: number; failed: number; unavailable: number }
  criteria: AssessmentCriterion[]
  evidence?: {
    association_accuracy: MetricEvidenceStatus
    rmse_improvement_pct: MetricEvidenceStatus
  }
  dimensions: {
    accuracy: BatchMetrics
    latency: BatchLatency
    association: AssociationSummary | null
    distribution: TrackDistribution | null
    by_maneuver: ManeuverAssessment[]
  }
}

export interface CrossDatasetCell {
  model_id: string
  dataset_id: string
  position_rmse_m: number
}

export interface CrossDatasetEvaluation {
  available: boolean
  metric: string
  source?: string
  matrix: CrossDatasetCell[]
}

export interface HistoricalReferences {
  kind: string
  notice: string
  imm_jpda_comparison: {
    global: { tf_mean_rmse: number; imm_mean_rmse: number; improvement_pct: number }
    per_mode: Record<string, { tf_mean_rmse: number; imm_mean_rmse: number; improvement_pct: number; n_samples: number }>
  } | null
  association_audit: {
    method: string
    audited_files: number
    truth_used_for_observation_grouping: boolean
    correct: number
    total: number
    accuracy: number
    per_sensor: Record<string, { correct: number; total: number; accuracy: number }>
  } | null
  sources: Record<string, string>
}

export interface ComparisonRow {
  job_id: string
  name: string
  model_id: string
  dataset_id: string
  created_at: string
  finished_at: string
  position_rmse_m: number
  position_mae_m: number
  position_p95_m: number
  estimated_frame_ms: number
  throughput_tracks_per_second: number
  association_accuracy: number | null
  evaluated_scenarios: number
  evaluated_tracks: number
}

export interface ReportRecord {
  id: string
  name: string
  status: BatchJobStatus
  created_at: string
  finished_at: string | null
  model_id: string
  dataset_id: string
  device: string
  scope: number | null
  metrics: {
    position_rmse_m: number | null
    position_mae_m: number | null
    position_p95_m: number | null
    estimated_frame_ms: number | null
    evaluated_scenarios: number | null
    evaluated_tracks: number | null
  } | null
  assessment: {
    overall_status: BatchAssessment['overall_status']
    formal_evidence_complete: BatchAssessment['formal_evidence_complete']
    summary: BatchAssessment['summary']
  } | null
  error: string | null
  artifacts: {
    raw_json: boolean
    tracks_csv: boolean
    assessment_json: boolean
    report_size_bytes: number
  }
}

export interface ReportCatalog {
  total: number
  summary: {
    completed: number
    failed_or_cancelled: number
    assessable: number
    total_size_bytes: number
  }
  items: ReportRecord[]
}

export interface DashboardData {
  health: HealthResponse
  system: SystemResponse
  models: ModelInfo[]
  devices: DeviceInfo[]
  criteria: CriterionInfo[]
}

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

const backendHost = `${window.location.hostname || '127.0.0.1'}:8000`
export const apiBase = `${window.location.protocol}//${backendHost}`
export const wsBase = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${backendHost}`

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  })
  if (!response.ok) {
    let detail = `请求失败（HTTP ${response.status}）`
    try {
      const payload = await response.json() as { detail?: string }
      if (payload.detail) detail = payload.detail
    } catch {
      // Keep the HTTP status fallback when the response is not JSON.
    }
    throw new ApiError(detail, response.status)
  }
  return response.json() as Promise<T>
}

export async function loadDashboard(): Promise<DashboardData> {
  // Health initializes the backend model singleton. Waiting for it avoids sending
  // several cold-start requests while the model weights are still loading.
  const health = await request<HealthResponse>('/api/health')
  const [system, modelPayload, devicePayload, criteriaPayload] = await Promise.all([
    request<SystemResponse>('/api/system'),
    request<{ items: ModelInfo[] }>('/api/models'),
    request<{ items: DeviceInfo[] }>('/api/devices'),
    request<{ items: CriterionInfo[] }>('/api/config/acceptance-criteria'),
  ])
  return {
    health,
    system,
    models: modelPayload.items,
    devices: devicePayload.items,
    criteria: criteriaPayload.items,
  }
}

export function refreshModels(): Promise<{ items: ModelInfo[] }> {
  return request<{ items: ModelInfo[] }>('/api/models/refresh', { method: 'POST' })
}

export function activateModel(modelId: string, device: string): Promise<ActiveModel & { changed: boolean }> {
  return request<ActiveModel & { changed: boolean }>('/api/models/activate', {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId, device }),
  })
}

export async function listScenarios(limit?: number): Promise<ScenarioInfo[]> {
  const query = limit ? `?limit=${limit}` : ''
  const payload = await request<{ items: ScenarioInfo[] }>(`/api/scenarios${query}`)
  return payload.items
}

export async function listDatasets(): Promise<DatasetInfo[]> {
  const payload = await request<{ items: DatasetInfo[] }>('/api/datasets')
  return payload.items
}

export async function listBatchJobs(limit = 100): Promise<BatchJob[]> {
  const payload = await request<{ items: BatchJob[] }>(`/api/batch-tests?limit=${limit}`)
  return payload.items
}

export function getBatchJob(jobId: string): Promise<BatchJob> {
  return request<BatchJob>(`/api/batch-tests/${encodeURIComponent(jobId)}`)
}

export function createBatchJob(payload: CreateBatchJobPayload): Promise<BatchJob> {
  return request<BatchJob>('/api/batch-tests', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function cancelBatchJob(jobId: string): Promise<BatchJob> {
  return request<BatchJob>(`/api/batch-tests/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' })
}

export function batchReportUrl(jobId: string, format: 'json' | 'csv'): string {
  const suffix = format === 'json' ? 'report.json' : 'tracks.csv'
  return `${apiBase}/api/batch-tests/${encodeURIComponent(jobId)}/${suffix}`
}

export function getBatchAssessment(jobId: string, baselineRmse?: number): Promise<BatchAssessment> {
  const baseline = baselineRmse && baselineRmse > 0 ? `?baseline_rmse_m=${baselineRmse}` : ''
  return request<BatchAssessment>(`/api/batch-tests/${encodeURIComponent(jobId)}/assessment${baseline}`)
}

export function generateMetricEvidence(
  jobId: string,
  metricId: MetricEvidenceStatus['metric_id'],
): Promise<{ metric_id: string; status: MetricEvidenceStatus; assessment: BatchAssessment }> {
  return request(`/api/batch-tests/${encodeURIComponent(jobId)}/evidence/${encodeURIComponent(metricId)}`, {
    method: 'POST',
  })
}

export function getCrossDatasetEvaluation(): Promise<CrossDatasetEvaluation> {
  return request<CrossDatasetEvaluation>('/api/evaluations/cross-dataset')
}

export function getHistoricalReferences(): Promise<HistoricalReferences> {
  return request<HistoricalReferences>('/api/evaluations/references')
}

export async function getEvaluationComparison(jobIds?: string[]): Promise<ComparisonRow[]> {
  const query = jobIds?.length ? `?job_ids=${encodeURIComponent(jobIds.join(','))}` : ''
  const payload = await request<{ items: ComparisonRow[] }>(`/api/evaluations/compare${query}`)
  return payload.items
}

export function getReportCatalog(limit = 200): Promise<ReportCatalog> {
  return request<ReportCatalog>(`/api/reports?limit=${limit}`)
}

export function assessmentReportUrl(jobId: string, baselineRmse?: number | null): string {
  const baseline = baselineRmse && baselineRmse > 0 ? `?baseline_rmse_m=${baselineRmse}` : ''
  return `${apiBase}/api/reports/${encodeURIComponent(jobId)}/assessment.json${baseline}`
}

export function comparisonReportUrl(jobIds: string[]): string {
  return `${apiBase}/api/reports/comparison.csv?job_ids=${encodeURIComponent(jobIds.join(','))}`
}
