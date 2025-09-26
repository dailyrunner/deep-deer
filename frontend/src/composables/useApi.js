import axios from 'axios'
import { ref, reactive } from 'vue'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes for idea generation
  headers: {
    'Content-Type': 'application/json',
  }
})

export function useApi() {
  const loading = ref(false)
  const error = ref(null)

  const clearError = () => {
    error.value = null
  }

  const generateIdeas = async (userRequest) => {
    loading.value = true
    error.value = null

    try {
      const response = await api.post('/api/v1/idea/generate', {
        request: userRequest
      })

      return response.data
    } catch (err) {
      console.error('API Error:', err)
      error.value = err.response?.data?.detail || err.message || '서버 오류가 발생했습니다.'
      throw err
    } finally {
      loading.value = false
    }
  }

  const checkHealth = async () => {
    try {
      const response = await api.get('/health')
      return response.data
    } catch (err) {
      console.error('Health check failed:', err)
      return null
    }
  }

  return {
    loading,
    error,
    clearError,
    generateIdeas,
    checkHealth
  }
}