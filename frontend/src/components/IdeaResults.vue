<template>
  <div class="space-y-6">
    <!-- Success Message -->
    <div class="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center space-x-3">
      <div class="flex-shrink-0">
        <svg class="h-5 w-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
          <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
        </svg>
      </div>
      <div>
        <h3 class="text-sm font-medium text-green-800">
          ✅ 아이디어가 성공적으로 생성되었습니다!
        </h3>
        <p class="text-sm text-green-600 mt-1">
          내부 데이터와 시장 트렌드를 분석하여 맞춤형 비즈니스 아이디어를 생성했습니다.
        </p>
      </div>
    </div>

    <!-- Results Card -->
    <div class="bg-white rounded-xl shadow-lg overflow-hidden">
      <!-- Header -->
      <div class="card-gradient p-6 text-white">
        <div class="flex items-center space-x-3">
          <span class="text-2xl">🎯</span>
          <div>
            <h2 class="text-xl font-bold">생성된 비즈니스 아이디어</h2>
            <p class="text-blue-100 text-sm mt-1">
              데이터 기반 AI 분석 결과
            </p>
          </div>
        </div>
      </div>

      <!-- Content -->
      <div class="p-6">
        <div class="prose prose-sm max-w-none">
          <div
            v-html="formattedIdeas"
            class="text-gray-700 leading-relaxed whitespace-pre-wrap"
          ></div>
        </div>
      </div>

      <!-- Footer Actions -->
      <div class="px-6 py-4 bg-gray-50 border-t border-gray-200">
        <div class="flex flex-wrap gap-3">
          <button
            @click="copyToClipboard"
            class="flex items-center space-x-2 px-4 py-2 bg-blue-100 hover:bg-blue-200 text-blue-800 rounded-lg transition-colors"
          >
            <svg class="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
              <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" />
              <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z" />
            </svg>
            <span>복사하기</span>
          </button>

          <button
            @click="exportToPdf"
            class="flex items-center space-x-2 px-4 py-2 bg-green-100 hover:bg-green-200 text-green-800 rounded-lg transition-colors"
          >
            <svg class="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M6 2a2 2 0 00-2 2v12a2 2 0 002 2h8a2 2 0 002-2V7.414A2 2 0 0015.414 6L12 2.586A2 2 0 0010.586 2H6zm5 6a1 1 0 10-2 0v3.586l-1.293-1.293a1 1 0 10-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L11 11.586V8z" clip-rule="evenodd" />
            </svg>
            <span>PDF 내보내기</span>
          </button>

          <button
            @click="shareResults"
            class="flex items-center space-x-2 px-4 py-2 bg-purple-100 hover:bg-purple-200 text-purple-800 rounded-lg transition-colors"
          >
            <svg class="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
              <path d="M15 8a3 3 0 10-2.977-2.63l-4.94 2.47a3 3 0 100 4.319l4.94 2.47a3 3 0 10.895-1.789l-4.94-2.47a3.027 3.027 0 000-.74l4.94-2.47C13.456 7.68 14.19 8 15 8z" />
            </svg>
            <span>공유하기</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Generate Another Button -->
    <div class="text-center">
      <button
        @click="$emit('generateAnother')"
        class="px-6 py-2 bg-white hover:bg-gray-50 text-gray-700 border border-gray-300 rounded-lg transition-colors"
      >
        🔄 다른 아이디어 생성하기
      </button>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  ideas: {
    type: String,
    required: true
  }
})

const emit = defineEmits(['generateAnother'])

const formattedIdeas = computed(() => {
  let formatted = props.ideas

  // Add basic formatting for better readability
  formatted = formatted
    .replace(/### (.*)/g, '<h3 class="text-lg font-bold text-gray-900 mt-6 mb-3">$1</h3>')
    .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-gray-900">$1</strong>')
    .replace(/💡|📊|🎯|⚙️|💰|🚧|📈/g, '<span class="text-lg">$&</span>')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>')

  return formatted
})

const copyToClipboard = async () => {
  try {
    await navigator.clipboard.writeText(props.ideas)
    // You could show a toast notification here
    console.log('아이디어가 클립보드에 복사되었습니다!')
  } catch (err) {
    console.error('복사 실패:', err)
  }
}

const exportToPdf = () => {
  // Basic PDF export implementation
  const printWindow = window.open('', '_blank')
  printWindow.document.write(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>Deep Deer - 비즈니스 아이디어</title>
      <style>
        body { font-family: Arial, sans-serif; padding: 20px; line-height: 1.6; }
        h1 { color: #2563eb; }
        h3 { color: #1f2937; margin-top: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .content { white-space: pre-wrap; }
      </style>
    </head>
    <body>
      <div class="header">
        <h1>🦌 Deep Deer</h1>
        <p>데이터 기반 비즈니스 아이디어</p>
      </div>
      <div class="content">${props.ideas}</div>
    </body>
    </html>
  `)
  printWindow.document.close()
  printWindow.print()
}

const shareResults = async () => {
  if (navigator.share) {
    try {
      await navigator.share({
        title: 'Deep Deer - 비즈니스 아이디어',
        text: props.ideas,
        url: window.location.href
      })
    } catch (err) {
      console.log('공유가 취소되었습니다.')
    }
  } else {
    // Fallback to copy to clipboard
    await copyToClipboard()
  }
}
</script>