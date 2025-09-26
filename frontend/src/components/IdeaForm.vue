<template>
  <div class="bg-white rounded-xl shadow-lg p-8">
    <form @submit.prevent="handleSubmit" class="space-y-6">
      <div>
        <label for="userRequest" class="block text-sm font-medium text-gray-700 mb-2">
          💡 어떤 비즈니스 아이디어를 찾고 계신가요?
        </label>

        <div class="relative">
          <textarea
            id="userRequest"
            v-model="userRequest"
            rows="6"
            class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors resize-none"
            placeholder="예: AI 기술을 활용한 교육 서비스 아이디어를 찾고 싶어요. 우리 회사의 학습 데이터를 바탕으로 새로운 서비스를 개발하고 싶습니다."
            required
            :disabled="loading"
          ></textarea>

          <!-- Character counter -->
          <div class="absolute bottom-3 right-3 text-xs text-gray-400">
            {{ userRequest.length }} / 2000
          </div>
        </div>

        <div class="mt-2 text-xs text-gray-500">
          구체적으로 설명해 주실수록 더 정확한 아이디어를 생성할 수 있습니다.
        </div>
      </div>

      <!-- Submit Button -->
      <div class="flex justify-center">
        <button
          type="submit"
          :disabled="loading || !userRequest.trim()"
          class="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-full
                 hover:from-blue-700 hover:to-purple-700 transform hover:scale-105 transition-all duration-200
                 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none
                 flex items-center space-x-2 shadow-lg"
        >
          <span v-if="!loading">🚀 아이디어 생성하기</span>
          <span v-else class="flex items-center space-x-2">
            <svg class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span>생성 중...</span>
          </span>
        </button>
      </div>

      <!-- Example prompts -->
      <div class="mt-6">
        <h4 class="text-sm font-medium text-gray-700 mb-3">💡 예시 질문:</h4>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <button
            v-for="example in examples"
            :key="example"
            type="button"
            @click="userRequest = example"
            :disabled="loading"
            class="text-left p-3 bg-gray-50 hover:bg-blue-50 rounded-lg border border-gray-200 hover:border-blue-300 transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {{ example }}
          </button>
        </div>
      </div>
    </form>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const emit = defineEmits(['submit'])
const props = defineProps({
  loading: {
    type: Boolean,
    default: false
  }
})

const userRequest = ref('')

const examples = [
  "AI 기술을 활용한 교육 서비스 아이디어를 찾고 싶어요",
  "고객 데이터를 활용한 개인화 마케팅 서비스를 개발하고 싶습니다",
  "IoT 센서 데이터를 활용한 스마트 시티 솔루션 아이디어",
  "헬스케어 데이터 기반 새로운 의료 서비스 모델"
]

const handleSubmit = () => {
  if (userRequest.value.trim() && !props.loading) {
    emit('submit', userRequest.value.trim())
  }
}
</script>