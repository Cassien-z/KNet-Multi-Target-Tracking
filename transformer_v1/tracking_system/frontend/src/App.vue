<script setup lang="ts">
import { defineAsyncComponent, onMounted, onUnmounted, ref } from 'vue'
import { apiBase, loadDashboard, type DashboardData } from './api'
import AppIcon from './components/AppIcon.vue'
import ModelManagementView from './views/ModelManagementView.vue'
import OverviewView from './views/OverviewView.vue'

const TrackingValidationView = defineAsyncComponent(() => import('./views/TrackingValidationView.vue'))
const BatchTestingView = defineAsyncComponent(() => import('./views/BatchTestingView.vue'))
const PerformanceEvaluationView = defineAsyncComponent(() => import('./views/PerformanceEvaluationView.vue'))
const ReportCenterView = defineAsyncComponent(() => import('./views/ReportCenterView.vue'))

type PageId = 'overview' | 'models' | 'tracking' | 'batch' | 'evaluation' | 'reports'

interface NavItem {
  id: PageId
  label: string
  description: string
  icon: 'overview' | 'model' | 'tracking' | 'batch' | 'evaluation' | 'report'
  available: boolean
}

const navItems: NavItem[] = [
  { id: 'overview', label: '系统总览', description: '运行态势', icon: 'overview', available: true },
  { id: 'models', label: '模型管理', description: '部署与切换', icon: 'model', available: true },
  { id: 'tracking', label: '跟踪验证', description: '单场景回放', icon: 'tracking', available: true },
  { id: 'batch', label: '批量测试', description: '任务与队列', icon: 'batch', available: true },
  { id: 'evaluation', label: '性能评估', description: '指标与对比', icon: 'evaluation', available: true },
  { id: 'reports', label: '报告中心', description: '归档与导出', icon: 'report', available: true },
]

const currentPage = ref<PageId>('overview')
const dashboard = ref<DashboardData | null>(null)
const loading = ref(true)
const errorMessage = ref('')
const lastUpdated = ref<Date | null>(null)
const mobileNavOpen = ref(false)
const settingsOpen = ref(false)
const toast = ref<{ message: string; tone: 'success' | 'error' } | null>(null)
let toastTimer: number | undefined
let refreshTimer: number | undefined

async function refreshData(silent = false) {
  if (!silent) loading.value = true
  try {
    dashboard.value = await loadDashboard()
    errorMessage.value = ''
    lastUpdated.value = new Date()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '无法连接后端服务'
  } finally {
    loading.value = false
  }
}

function navigate(page: PageId) {
  const item = navItems.find((entry) => entry.id === page)
  if (!item?.available) return
  currentPage.value = page
  if (window.location.hash !== `#${page}`) window.history.replaceState(null, '', `#${page}`)
  mobileNavOpen.value = false
}

function syncPageFromHash() {
  const page = window.location.hash.slice(1) as PageId
  const item = navItems.find((entry) => entry.id === page && entry.available)
  if (item) currentPage.value = item.id
}

function showNotice(message: string, tone: 'success' | 'error') {
  toast.value = { message, tone }
  if (toastTimer) window.clearTimeout(toastTimer)
  toastTimer = window.setTimeout(() => { toast.value = null }, 3600)
}

async function copyStartCommand() {
  const command = 'conda run --no-capture-output -n pytorch python -m tracking_system.backend.server'
  try {
    await navigator.clipboard.writeText(command)
    showNotice('后端启动命令已复制', 'success')
  } catch {
    showNotice('浏览器未允许写入剪贴板', 'error')
  }
}

function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape') {
    settingsOpen.value = false
    mobileNavOpen.value = false
  }
}

onMounted(() => {
  syncPageFromHash()
  refreshData()
  refreshTimer = window.setInterval(() => refreshData(true), 30_000)
  window.addEventListener('hashchange', syncPageFromHash)
  window.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  if (toastTimer) window.clearTimeout(toastTimer)
  if (refreshTimer) window.clearInterval(refreshTimer)
  window.removeEventListener('hashchange', syncPageFromHash)
  window.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <div class="application">
    <aside class="sidebar" :class="{ open: mobileNavOpen }">
      <div class="brand-block">
        <div class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></div>
        <div><b>COOP TRACK</b><span>融合跟踪原型系统</span></div>
      </div>

      <div class="sidebar-context">
        <span>WORKSPACE</span>
        <b>端到端验证平台</b>
        <small>Prototype · API {{ dashboard?.health.service_version ?? '—' }}</small>
      </div>

      <nav class="main-nav" aria-label="主导航">
        <p>工作区</p>
        <button
          v-for="item in navItems"
          :key="item.id"
          :class="{ active: currentPage === item.id, unavailable: !item.available }"
          :aria-current="currentPage === item.id ? 'page' : undefined"
          :disabled="!item.available"
          @click="navigate(item.id)"
        >
          <span class="nav-icon"><AppIcon :name="item.icon" :size="20" /></span>
          <span class="nav-copy"><b>{{ item.label }}</b><small>{{ item.description }}</small></span>
          <span v-if="!item.available" class="soon-tag">下一阶段</span>
          <span v-else-if="currentPage === item.id" class="nav-active-mark"></span>
        </button>
      </nav>

      <div class="sidebar-footer">
        <div class="backend-state" :class="{ offline: !!errorMessage }">
          <span class="pulse-dot"></span>
          <div><b>{{ errorMessage ? '后端未连接' : '后端服务在线' }}</b><small>127.0.0.1:8000</small></div>
        </div>
        <button class="settings-button" aria-label="系统设置" @click="settingsOpen = true"><AppIcon name="settings" :size="19" /></button>
      </div>
    </aside>

    <div v-if="mobileNavOpen" class="nav-scrim" @click="mobileNavOpen = false"></div>

    <main class="workspace">
      <div class="mobile-bar">
        <button class="menu-button" aria-label="打开导航" @click="mobileNavOpen = true"><span></span><span></span></button>
        <b>COOP TRACK</b>
        <span class="mobile-status" :class="{ offline: !!errorMessage }"></span>
      </div>

      <div v-if="errorMessage" class="connection-alert">
        <div><AppIcon name="server" /><span><b>无法读取后端状态</b>{{ errorMessage }}。请确认 `pytorch` 环境中的后端已经启动。</span></div>
        <button @click="refreshData()">重新连接</button>
      </div>

      <OverviewView
        v-if="currentPage === 'overview'"
        :data="dashboard"
        :loading="loading"
        :last-updated="lastUpdated"
        @refresh="refreshData()"
        @navigate="navigate"
      />
      <ModelManagementView
        v-else-if="currentPage === 'models'"
        :data="dashboard"
        :loading="loading"
        @refresh="refreshData(true)"
        @notice="showNotice"
      />
      <TrackingValidationView v-else-if="currentPage === 'tracking'" />
      <BatchTestingView
        v-else-if="currentPage === 'batch'"
        :data="dashboard"
        @notice="showNotice"
      />
      <PerformanceEvaluationView
        v-else-if="currentPage === 'evaluation'"
        :data="dashboard"
        @notice="showNotice"
      />
      <ReportCenterView
        v-else-if="currentPage === 'reports'"
        :data="dashboard"
        @notice="showNotice"
      />

      <footer class="workspace-footer">
        <span>End-to-End Cooperative Tracking Prototype</span>
        <span>Transformer V1 · Local Runtime</span>
      </footer>
    </main>

    <div v-if="settingsOpen" class="drawer-backdrop settings-backdrop" @click.self="settingsOpen = false">
      <aside class="settings-panel" role="dialog" aria-modal="true" aria-labelledby="settings-title">
        <div class="drawer-head"><div><p class="section-kicker">LOCAL RUNTIME</p><h2 id="settings-title">运行配置</h2></div><button aria-label="关闭设置" @click="settingsOpen = false"><AppIcon name="close" /></button></div>
        <p class="settings-intro">当前原型采用本机 Conda 环境与独立前端开发服务，运行参数由后端统一管理。</p>
        <dl class="settings-list">
          <div><dt>API 地址</dt><dd>{{ apiBase }}</dd></div>
          <div><dt>服务版本</dt><dd>{{ dashboard?.health.service_version ?? '—' }}</dd></div>
          <div><dt>当前模型</dt><dd>{{ dashboard?.health.active_model.model_id?.toUpperCase() ?? '—' }}</dd></div>
          <div><dt>推理设备</dt><dd>{{ dashboard?.health.active_model.device ?? '—' }}</dd></div>
          <div><dt>PyTorch</dt><dd>{{ dashboard?.system.runtime.pytorch ?? '—' }}</dd></div>
          <div><dt>CUDA Runtime</dt><dd>{{ dashboard?.system.runtime.cuda_runtime ?? '不可用' }}</dd></div>
          <div class="settings-wide"><dt>报告目录</dt><dd>{{ dashboard?.system.batch_testing.report_dir ?? '—' }}</dd></div>
        </dl>
        <div class="settings-command"><span>后端启动命令</span><code>conda run --no-capture-output -n pytorch python -m tracking_system.backend.server</code><button @click="copyStartCommand">复制命令</button></div>
        <div class="settings-actions"><button class="button button-secondary" @click="refreshData()">刷新后端状态</button><button class="button button-primary" @click="settingsOpen = false">完成</button></div>
      </aside>
    </div>

    <Transition name="toast">
      <div v-if="toast" class="toast-message" :class="toast.tone">
        <AppIcon :name="toast.tone === 'success' ? 'check' : 'server'" :size="18" />
        {{ toast.message }}
      </div>
    </Transition>
  </div>
</template>
