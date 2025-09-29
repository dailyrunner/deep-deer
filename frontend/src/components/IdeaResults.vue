<template>
  <div class="space-y-6" data-results>
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

      <!-- Content with Markdown Rendering -->
      <div class="p-6">
        <div
          class="markdown-content prose prose-sm max-w-none
                 prose-headings:text-gray-900 prose-h1:text-2xl prose-h2:text-xl prose-h3:text-lg
                 prose-p:text-gray-700 prose-p:leading-relaxed
                 prose-strong:text-gray-900 prose-strong:font-semibold
                 prose-ul:my-4 prose-li:my-1
                 prose-ol:my-4
                 prose-blockquote:border-l-4 prose-blockquote:border-blue-500 prose-blockquote:pl-4 prose-blockquote:italic
                 prose-code:bg-gray-100 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-sm
                 prose-pre:bg-gray-900 prose-pre:text-gray-100"
          v-html="formattedIdeas"
        ></div>
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
            <span>{{ copied ? '복사됨!' : '복사하기' }}</span>
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
import { computed, ref, onMounted } from 'vue'
import { marked } from 'marked'
import hljs from 'highlight.js'
import 'highlight.js/styles/github-dark.css'

const props = defineProps({
  ideas: {
    type: String,
    required: true
  }
})

const emit = defineEmits(['generateAnother'])
const copied = ref(false)

// Configure marked with syntax highlighting
onMounted(() => {
  marked.setOptions({
    breaks: true,
    gfm: true,
    headerIds: false,
    highlight: function(code, lang) {
      if (lang && hljs.getLanguage(lang)) {
        try {
          return hljs.highlight(code, { language: lang }).value
        } catch (err) {
          console.error('Highlight error:', err)
        }
      }
      return hljs.highlightAuto(code).value
    }
  })
})

const formattedIdeas = computed(() => {
  try {
    // Parse markdown content
    let html = marked.parse(props.ideas)

    // Add custom styling for specific elements
    html = html
      // Style emoji icons
      .replace(/(💡|📊|🎯|⚙️|💰|🚧|📈|🔍|📱|🏢|🚀|✨|📝|🔧|🛠️|🎨|📌|⭐|🔑|💼|📦|🌐|🔒|⚡|🎖️|🏆|💎)/g,
        '<span class="inline-block text-xl align-middle mr-1">$1</span>')
      // Style numbered lists
      .replace(/<ol>/g, '<ol class="list-decimal list-inside space-y-2">')
      // Style bullet lists
      .replace(/<ul>/g, '<ul class="list-disc list-inside space-y-2">')
      // Add spacing to paragraphs
      .replace(/<p>/g, '<p class="mb-4">')
      // Style headers
      .replace(/<h1>/g, '<h1 class="text-2xl font-bold text-gray-900 mt-6 mb-4">')
      .replace(/<h2>/g, '<h2 class="text-xl font-bold text-gray-900 mt-6 mb-3">')
      .replace(/<h3>/g, '<h3 class="text-lg font-semibold text-gray-900 mt-4 mb-2">')
      .replace(/<h4>/g, '<h4 class="text-base font-semibold text-gray-800 mt-3 mb-2">')
      // Style blockquotes
      .replace(/<blockquote>/g, '<blockquote class="border-l-4 border-blue-500 pl-4 my-4 italic text-gray-700">')
      // Style tables
      .replace(/<table>/g, '<table class="min-w-full divide-y divide-gray-200 my-4">')
      .replace(/<thead>/g, '<thead class="bg-gray-50">')
      .replace(/<th>/g, '<th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">')
      .replace(/<td>/g, '<td class="px-4 py-2 whitespace-nowrap text-sm text-gray-700">')
      // Style horizontal rules
      .replace(/<hr>/g, '<hr class="my-6 border-t border-gray-300">')

    return html
  } catch (error) {
    console.error('Markdown parsing error:', error)
    // Fallback to basic formatting
    return props.ideas
      .replace(/\n/g, '<br>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
  }
})

const copyToClipboard = async () => {
  try {
    await navigator.clipboard.writeText(props.ideas)
    copied.value = true
    setTimeout(() => {
      copied.value = false
    }, 2000)
  } catch (err) {
    console.error('복사 실패:', err)
    alert('클립보드 복사에 실패했습니다.')
  }
}

const exportToPdf = () => {
  const printWindow = window.open('', '_blank')
  const htmlContent = marked.parse(props.ideas)

  printWindow.document.write(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>Deep Deer - 비즈니스 아이디어</title>
      <style>
        @page { margin: 2cm; }
        body {
          font-family: 'Noto Sans KR', Arial, sans-serif;
          padding: 20px;
          line-height: 1.8;
          color: #1f2937;
        }
        h1 {
          color: #2563eb;
          font-size: 28px;
          margin-top: 24px;
          margin-bottom: 16px;
        }
        h2 {
          color: #1e40af;
          font-size: 22px;
          margin-top: 20px;
          margin-bottom: 12px;
        }
        h3 {
          color: #1f2937;
          font-size: 18px;
          margin-top: 16px;
          margin-bottom: 8px;
        }
        .header {
          text-align: center;
          margin-bottom: 40px;
          padding-bottom: 20px;
          border-bottom: 2px solid #e5e7eb;
        }
        .content {
          max-width: 800px;
          margin: 0 auto;
        }
        ul, ol {
          margin: 12px 0;
          padding-left: 24px;
        }
        li {
          margin: 8px 0;
        }
        blockquote {
          border-left: 4px solid #3b82f6;
          padding-left: 16px;
          margin: 16px 0;
          font-style: italic;
          color: #4b5563;
        }
        code {
          background: #f3f4f6;
          padding: 2px 6px;
          border-radius: 4px;
          font-family: 'Courier New', monospace;
          font-size: 14px;
        }
        pre {
          background: #1f2937;
          color: #f9fafb;
          padding: 16px;
          border-radius: 8px;
          overflow-x: auto;
        }
        strong {
          font-weight: 600;
          color: #111827;
        }
        .footer {
          margin-top: 40px;
          padding-top: 20px;
          border-top: 1px solid #e5e7eb;
          text-align: center;
          color: #6b7280;
          font-size: 12px;
        }
      </style>
    </head>
    <body>
      <div class="header">
        <h1>🦌 Deep Deer</h1>
        <p style="color: #4b5563;">데이터 기반 비즈니스 아이디어 생성 플랫폼</p>
        <p style="color: #9ca3af; font-size: 14px;">생성일: ${new Date().toLocaleDateString('ko-KR')}</p>
      </div>
      <div class="content">${htmlContent}</div>
      <div class="footer">
        <p>Generated by Deep Deer - AI-Powered Business Idea Platform</p>
      </div>
    </body>
    </html>
  `)
  printWindow.document.close()

  // Wait for content to load before printing
  printWindow.onload = () => {
    printWindow.print()
  }
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
      if (err.name !== 'AbortError') {
        console.error('공유 실패:', err)
        await copyToClipboard()
      }
    }
  } else {
    // Fallback to copy to clipboard
    await copyToClipboard()
  }
}
</script>

<style scoped>
/* Additional custom styles for markdown content */
.markdown-content :deep(pre) {
  @apply rounded-lg overflow-x-auto my-4;
}

.markdown-content :deep(code) {
  @apply font-mono text-sm;
}

.markdown-content :deep(table) {
  @apply border-collapse w-full my-4;
}

.markdown-content :deep(th) {
  @apply bg-gray-100 font-semibold;
}

.markdown-content :deep(td),
.markdown-content :deep(th) {
  @apply border border-gray-300 px-4 py-2;
}

.markdown-content :deep(tr:nth-child(even)) {
  @apply bg-gray-50;
}

/* Card gradient background */
.card-gradient {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
</style>