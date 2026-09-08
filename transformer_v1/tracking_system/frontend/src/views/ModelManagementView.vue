<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { DashboardData, DeviceInfo, ModelInfo } from '../api'
import { activateModel, refreshModels } from '../api'
import { formatDate, formatNumber } from '../format'
import AppIcon from '../components/AppIcon.vue'

const props = defineProps<{
  data: DashboardData | null
  loading: boolean
}>()

const emit = defineEmits<{
  refresh: []
  notice: [message: string, tone: 'success' | 'error']
}>()

const selectedDevice = ref('auto')
const pendingModel = ref<ModelInfo | null>(null)
const activating = ref(false)
const syncing = ref(false)

const availableDevices = computed<DeviceInfo[]>(() => props.data?.devices.filter((item) => item.available) ?? [])
const activeModel = computed(() => props.data?.models.find((model) => model.active) ?? null)

watch(availableDevices, (devices) => {
  if (selectedDevice.value === 'auto' && devices.some((item) => item.id.startsWith('cuda'))) {
    selectedDevice.value = devices.find((item) => item.id.startsWith('cuda'))?.id ?? 'cpu'
  }
}, { immediate: true })

async function syncRegistry() {
  syncing.value = true
  try {
    await refreshModels()
    emit('refresh')
    emit('notice', '模型仓库已重新扫描', 'success')
  } catch (error) {
    emit('notice', error instanceof Error ? error.message : '模型仓库刷新失败', 'error')
  } finally {
    syncing.value = false
  }
}

async function confirmActivation() {
  if (!pendingModel.value) return
  activating.value = true
  try {
    const result = await activateModel(pendingModel.value.id, selectedDevice.value)
    emit('notice', result.changed ? `${pendingModel.value.version} 已完成部署` : `${pendingModel.value.version} 已处于当前设备`, 'success')
    pendingModel.value = null
    emit('refresh')
  } catch (error) {
    emit('notice', error instanceof Error ? error.message : '模型部署失败', 'error')
  } finally {
    activating.value = false
  }
}
</script>

<template>
  <section class="page-shell">
    <header class="page-heading">
      <div>
        <p class="eyebrow">DEPLOYMENT / 模型管理</p>
        <h1>模型仓库</h1>
        <p class="page-description">管理三个续训最佳权重，并以原子切换方式部署到指定计算设备。</p>
      </div>
      <button class="button button-secondary" :disabled="syncing || loading" @click="syncRegistry">
        <AppIcon name="refresh" :size="17" />
        {{ syncing ? '扫描中' : '重新扫描仓库' }}
      </button>
    </header>

    <div class="deployment-banner">
      <div class="deployment-symbol"><AppIcon name="model" :size="27" /></div>
      <div class="deployment-copy">
        <span>当前在线模型</span>
        <h2>{{ activeModel?.version ?? '—' }} <small>{{ activeModel?.name ?? '等待后端连接' }}</small></h2>
      </div>
      <dl>
        <div><dt>设备</dt><dd>{{ data?.health.active_model.device ?? '—' }}</dd></div>
        <div><dt>加载时间</dt><dd>{{ formatNumber(data?.health.active_model.load_time_ms, 1) }} ms</dd></div>
        <div><dt>激活于</dt><dd>{{ formatDate(data?.health.active_model.activated_at) }}</dd></div>
      </dl>
      <span class="deployment-state"><i></i> SERVING</span>
    </div>

    <div class="device-selector surface">
      <div>
        <p class="section-kicker">TARGET DEVICE</p>
        <h2>部署设备</h2>
        <p>选择下次模型切换使用的计算设备。</p>
      </div>
      <div class="device-options" role="radiogroup" aria-label="部署设备">
        <button
          v-for="device in availableDevices"
          :key="device.id"
          :class="{ selected: selectedDevice === device.id }"
          role="radio"
          :aria-checked="selectedDevice === device.id"
          @click="selectedDevice = device.id"
        >
          <AppIcon :name="device.id.startsWith('cuda') ? 'cpu' : 'server'" :size="20" />
          <span><b>{{ device.id.toUpperCase() }}</b><small>{{ device.name }}</small></span>
          <i class="radio-mark"></i>
        </button>
      </div>
    </div>

    <div class="registry-heading">
      <div><h2>已注册权重</h2><span>{{ data?.models.length ?? 0 }} 个版本</span></div>
      <p>每个版本仅保留对应数据集上的续训最佳结果。</p>
    </div>

    <div class="model-card-grid">
      <article v-for="model in data?.models ?? []" :key="model.id" class="model-deploy-card" :class="{ current: model.active }">
        <div class="model-card-head">
          <div class="version-tile">{{ model.version }}</div>
          <div><span>{{ model.active ? 'ACTIVE MODEL' : 'AVAILABLE' }}</span><h3>{{ model.name }}</h3></div>
          <span v-if="model.active" class="current-flag"><i></i> 当前</span>
        </div>
        <div class="model-scoreboard">
          <div><small>历史 RMSE</small><b>{{ formatNumber(model.metrics?.rmse_m) }}<em>m</em></b></div>
          <div><small>平均时延</small><b>{{ formatNumber(model.metrics?.mean_latency_ms, 1) }}<em>ms</em></b></div>
          <div><small>P95 误差</small><b>{{ formatNumber(model.metrics?.p95_m) }}<em>m</em></b></div>
        </div>
        <dl class="model-metadata">
          <div><dt>Run ID</dt><dd>{{ model.run_id }}</dd></div>
          <div><dt>训练数据</dt><dd>{{ model.training_dataset }}</dd></div>
          <div><dt>验证数据</dt><dd>{{ model.validation_dataset }}</dd></div>
          <div><dt>权重文件</dt><dd>{{ model.file_name }} · {{ formatNumber(model.size_mb, 1) }} MB</dd></div>
        </dl>
        <button class="deploy-button" :disabled="model.active || loading" @click="pendingModel = model">
          <span>{{ model.active ? '正在提供推理服务' : `部署到 ${selectedDevice.toUpperCase()}` }}</span>
          <AppIcon :name="model.active ? 'check' : 'arrow'" :size="17" />
        </button>
      </article>
    </div>

    <div v-if="pendingModel" class="modal-backdrop" @click.self="pendingModel = null">
      <div class="confirm-modal" role="dialog" aria-modal="true" aria-labelledby="deploy-title">
        <button class="modal-close" aria-label="关闭" @click="pendingModel = null"><AppIcon name="close" /></button>
        <div class="modal-emblem"><AppIcon name="cube" :size="28" /></div>
        <p class="section-kicker">ATOMIC DEPLOYMENT</p>
        <h2 id="deploy-title">部署 {{ pendingModel.version }}？</h2>
        <p>后端会先在 <b>{{ selectedDevice.toUpperCase() }}</b> 完整加载并校验权重，成功后再替换当前模型。正在进行的回放不会中断。</p>
        <div class="deploy-summary"><span>目标模型 <b>{{ pendingModel.run_id }}</b></span><span>目标设备 <b>{{ selectedDevice }}</b></span></div>
        <div class="modal-actions">
          <button class="button button-secondary" :disabled="activating" @click="pendingModel = null">取消</button>
          <button class="button button-primary" :disabled="activating" @click="confirmActivation">
            {{ activating ? '正在加载权重…' : '确认部署' }}
            <AppIcon v-if="!activating" name="arrow" :size="17" />
          </button>
        </div>
      </div>
    </div>
  </section>
</template>
