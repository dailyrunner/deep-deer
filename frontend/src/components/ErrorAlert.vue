<template>
  <Transition name="fade">
    <div v-if="error" class="mb-6">
      <div class="bg-red-50 border border-red-200 rounded-lg p-4">
        <div class="flex items-start space-x-3">
          <div class="flex-shrink-0">
            <svg class="h-5 w-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
            </svg>
          </div>

          <div class="flex-1 min-w-0">
            <h3 class="text-sm font-medium text-red-800">
              오류가 발생했습니다
            </h3>
            <p class="text-sm text-red-700 mt-1">
              {{ error }}
            </p>

            <!-- Error details (if available) -->
            <div v-if="showDetails && errorDetails" class="mt-3 p-3 bg-red-100 rounded text-xs text-red-600 font-mono">
              {{ errorDetails }}
            </div>

            <!-- Action buttons -->
            <div class="mt-3 flex flex-wrap gap-2">
              <button
                @click="$emit('retry')"
                class="text-sm bg-red-100 hover:bg-red-200 text-red-800 px-3 py-1 rounded transition-colors"
              >
                🔄 다시 시도
              </button>

              <button
                v-if="errorDetails"
                @click="showDetails = !showDetails"
                class="text-sm bg-gray-100 hover:bg-gray-200 text-gray-800 px-3 py-1 rounded transition-colors"
              >
                {{ showDetails ? '상세내용 숨기기' : '상세내용 보기' }}
              </button>

              <button
                @click="$emit('clear')"
                class="text-sm text-red-600 hover:text-red-800 px-3 py-1 transition-colors"
              >
                ✕ 닫기
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Troubleshooting tips -->
      <div v-if="showTroubleshooting" class="mt-4 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <h4 class="text-sm font-medium text-yellow-800 mb-2">💡 문제 해결 팁:</h4>
        <ul class="text-sm text-yellow-700 space-y-1">
          <li>• 네트워크 연결 상태를 확인해주세요</li>
          <li>• API 서버가 정상 동작하는지 확인해주세요 (http://localhost:8000/health)</li>
          <li>• 요청 내용이 너무 길거나 복잡하지 않은지 확인해주세요</li>
          <li>• 잠시 후 다시 시도해주세요</li>
        </ul>
      </div>
    </div>
  </Transition>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  error: {
    type: String,
    default: null
  },
  details: {
    type: String,
    default: null
  }
})

const emit = defineEmits(['retry', 'clear'])

const showDetails = ref(false)
const showTroubleshooting = ref(true)

const errorDetails = computed(() => {
  return props.details || null
})
</script>