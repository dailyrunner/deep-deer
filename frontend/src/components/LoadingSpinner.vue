<template>
  <div class="flex flex-col items-center justify-center p-8">
    <div class="relative mb-4">
      <!-- Main spinner -->
      <div class="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>

      <!-- Inner spinner -->
      <div class="absolute top-2 left-2 w-12 h-12 border-4 border-purple-200 border-t-purple-600 rounded-full animate-spin"
           style="animation-direction: reverse; animation-duration: 1.5s;"></div>
    </div>

    <div class="text-center space-y-2">
      <h3 class="text-lg font-semibold text-gray-800">
        {{ title }}
      </h3>
      <p class="text-sm text-gray-600 max-w-md">
        {{ subtitle }}
      </p>

      <!-- Progress steps -->
      <div class="mt-4 space-y-1 text-xs text-gray-500">
        <div v-for="step in steps" :key="step" class="flex items-center justify-center space-x-2">
          <div class="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
          <span>{{ step }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    default: '데이터를 분석하고 있습니다...'
  },
  subtitle: {
    type: String,
    default: '잠시만 기다려주세요. AI가 최적의 비즈니스 아이디어를 생성 중입니다.'
  },
  steps: {
    type: Array,
    default: () => [
      '📊 내부 DB 데이터 분석 중...',
      '📁 문서 검색 수행 중...',
      '🌐 웹 트렌드 조사 중...',
      '🤖 AI 아이디어 생성 중...'
    ]
  }
})
</script>