<script setup lang="ts">
defineProps<{
  name: 'overview' | 'model' | 'tracking' | 'batch' | 'evaluation' | 'report' | 'settings' | 'refresh' | 'cpu' | 'database' | 'clock' | 'check' | 'arrow' | 'cube' | 'server' | 'close' | 'play' | 'pause' | 'replay' | 'target'
  size?: number
}>()
</script>

<template>
  <svg class="app-icon" :width="size ?? 20" :height="size ?? 20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path v-if="name === 'overview'" d="M4 13h6V4H4v9Zm0 7h6v-4H4v4Zm10 0h6v-9h-6v9Zm0-16v4h6V4h-6Z" />
    <path v-else-if="name === 'model'" d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Zm0 0v9m8-4.5-8 4.5m-8-4.5 8 4.5m0 9v-9" />
    <path v-else-if="name === 'tracking'" d="M12 3a9 9 0 1 0 9 9M12 7a5 5 0 1 0 5 5m-5-2a2 2 0 1 0 2 2m0-2 7-7m0 0v5m0-5h-5" />
    <path v-else-if="name === 'batch'" d="M5 4h14v5H5V4Zm0 11h14v5H5v-5Zm3-6v6m8-6v6" />
    <path v-else-if="name === 'evaluation'" d="M4 19V9m5 10V5m5 14v-7m5 7V3" />
    <path v-else-if="name === 'report'" d="M6 3h9l4 4v14H6V3Zm8 0v5h5M9 12h7m-7 4h7" />
    <path v-else-if="name === 'settings'" d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7Zm8 3.5-2.1-1.1.1-2.4-2.5-1.5-2 1.3L11.5 7 9 8.5l.1 2.4L7 12l.1 2.4L9 15.5l-.1 2.4 2.5 1.5 2-1.3 2 1.3 2.5-1.5-.1-2.4L20 14.4V12Z" />
    <path v-else-if="name === 'refresh'" d="M20 7v5h-5M4 17v-5h5m10.2-3A8 8 0 0 0 6.3 6.4L4 9m16 6-2.3 2.6A8 8 0 0 1 4.8 15" />
    <path v-else-if="name === 'cpu'" d="M8 8h8v8H8V8Zm2-5v3m4-3v3m-4 12v3m4-3v3M3 10h3m-3 4h3m12-4h3m-3 4h3" />
    <path v-else-if="name === 'database'" d="M5 6c0-1.7 3.1-3 7-3s7 1.3 7 3-3.1 3-7 3-7-1.3-7-3Zm0 0v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6m-14 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" />
    <path v-else-if="name === 'clock'" d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Zm0-13v4l3 2" />
    <path v-else-if="name === 'check'" d="m5 12 4 4L19 6" />
    <path v-else-if="name === 'arrow'" d="M5 12h14m-5-5 5 5-5 5" />
    <path v-else-if="name === 'cube'" d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Zm0 9 8-4.5M12 12 4 7.5m8 4.5v9" />
    <path v-else-if="name === 'server'" d="M4 4h16v6H4V4Zm0 10h16v6H4v-6Zm3-7h.01M7 17h.01" />
    <path v-else-if="name === 'close'" d="m6 6 12 12M18 6 6 18" />
    <path v-else-if="name === 'play'" d="m8 5 11 7-11 7V5Z" />
    <path v-else-if="name === 'pause'" d="M8 5v14m8-14v14" />
    <path v-else-if="name === 'replay'" d="M4 8V4m0 0h4M4.6 4.7A9 9 0 1 1 3 14" />
    <path v-else-if="name === 'target'" d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Zm0-13a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm0 3v2m-1-1h2" />
  </svg>
</template>

<style scoped>
.app-icon {
  display: block;
  flex: 0 0 auto;
  stroke: currentColor;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}
</style>
