<template>
  <div class="min-h-screen gradient-bg">
    <div class="container mx-auto px-4 py-8">
      <!-- Header -->
      <header class="text-center mb-12">
        <div class="bg-white bg-opacity-20 backdrop-blur-sm rounded-2xl p-8 text-white">
          <h1 class="text-5xl font-bold mb-4">
            <span class="text-6xl">🦌</span> Deep Deer
          </h1>
          <p class="text-xl opacity-90 max-w-2xl mx-auto">
            기업 데이터 기반 비즈니스 아이디어 발굴 플랫폼
          </p>
          <div class="mt-4 text-sm opacity-75">
            내부 데이터 + 시장 트렌드 + AI 분석 = 혁신적인 비즈니스 아이디어
          </div>
        </div>
      </header>

      <!-- Main Content -->
      <main class="max-w-4xl mx-auto">
        <!-- Server Status Indicator -->
        <div v-if="serverStatus !== null" class="mb-6">
          <div v-if="serverStatus" class="bg-green-100 border border-green-300 text-green-800 px-4 py-2 rounded-lg text-sm flex items-center">
            <div class="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
            API 서버 연결됨
          </div>
          <div v-else class="bg-red-100 border border-red-300 text-red-800 px-4 py-2 rounded-lg text-sm flex items-center">
            <div class="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
            API 서버 연결 실패 - 서버가 실행 중인지 확인해주세요
          </div>
        </div>

        <!-- Error Display -->
        <ErrorAlert
          :error="error"
          @retry="handleRetry"
          @clear="clearError"
        />

        <!-- Loading State -->
        <LoadingSpinner v-if="loading" />

        <!-- Form (when not loading and no results) -->
        <IdeaForm
          v-else-if="!results"
          :loading="loading"
          @submit="generateIdeas"
        />

        <!-- Results Display -->
        <IdeaResults
          v-else
          :ideas="results.ideas"
          @generateAnother="resetForm"
        />
      </main>

      <!-- Footer -->
      <footer class="text-center mt-16 text-white text-opacity-75">
        <div class="max-w-md mx-auto">
          <p class="text-sm">
            Deep Deer는 기업의 내부 데이터와 외부 시장 트렌드를 AI가 분석하여<br>
            실행 가능한 비즈니스 아이디어를 제시합니다.
          </p>
          <div class="mt-4 text-xs opacity-60">
            Powered by Vue.js + FastAPI + Ollama
          </div>
        </div>
      </footer>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useApi } from './composables/useApi'

// Components
import IdeaForm from './components/IdeaForm.vue'
import IdeaResults from './components/IdeaResults.vue'
import LoadingSpinner from './components/LoadingSpinner.vue'
import ErrorAlert from './components/ErrorAlert.vue'

// Composables
const { loading, error, clearError, generateIdeas: apiGenerateIdeas, checkHealth } = useApi()

// Reactive state
const results = ref(null)
const serverStatus = ref(null)
const lastRequest = ref('')

// Methods
const generateIdeas = async (userRequest) => {
  lastRequest.value = userRequest
  results.value = null

  try {
    const response = await apiGenerateIdeas(userRequest)

    if (response.success) {
      results.value = response
      // Scroll to results
      setTimeout(() => {
        const resultsElement = document.querySelector('[data-results]')
        if (resultsElement) {
          resultsElement.scrollIntoView({ behavior: 'smooth', block: 'start' })
        }
      }, 100)
    } else {
      throw new Error(response.message || '아이디어 생성에 실패했습니다.')
    }
  } catch (err) {
    console.error('아이디어 생성 오류:', err)
    // Error is handled by useApi composable
  }
}

const handleRetry = () => {
  if (lastRequest.value) {
    generateIdeas(lastRequest.value)
  }
}

const resetForm = () => {
  results.value = null
  clearError()
  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

// Check server health on mount
onMounted(async () => {
  try {
    const health = await checkHealth()
    serverStatus.value = health !== null
  } catch (err) {
    serverStatus.value = false
    console.warn('서버 상태 확인 실패:', err)
  }
})
</script>