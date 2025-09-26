# 🦌 Deep Deer Frontend

Vue.js 기반 Deep Deer 프론트엔드 애플리케이션입니다.

## 🚀 시작하기

### 1. 의존성 설치

```bash
cd frontend
npm install
```

### 2. 개발 서버 실행

```bash
npm run dev
```

브라우저에서 http://localhost:3000 으로 접속하세요.

### 3. 빌드 (프로덕션)

```bash
npm run build
```

빌드된 파일은 `dist/` 폴더에 생성됩니다.

## 🛠️ 기술 스택

- **Vue.js 3** - 프론트엔드 프레임워크
- **Vite** - 빌드 도구
- **Tailwind CSS** - 스타일링
- **Axios** - HTTP 클라이언트
- **Heroicons** - 아이콘

## 📁 프로젝트 구조

```
frontend/
├── src/
│   ├── components/           # Vue 컴포넌트
│   │   ├── IdeaForm.vue     # 아이디어 생성 폼
│   │   ├── IdeaResults.vue  # 결과 표시
│   │   ├── LoadingSpinner.vue # 로딩 스피너
│   │   └── ErrorAlert.vue   # 오류 알림
│   ├── composables/         # Vue 컴포저블
│   │   └── useApi.js       # API 통신 로직
│   ├── App.vue             # 메인 앱 컴포넌트
│   ├── main.js             # 앱 진입점
│   └── style.css           # 전역 스타일
├── package.json
├── vite.config.js
├── tailwind.config.js
└── README.md
```

## ✨ 주요 기능

- 📝 **직관적인 입력 폼**: 사용자 요청 입력 및 예시 제공
- ⚡ **실시간 로딩 상태**: 단계별 처리 과정 표시
- 📊 **결과 시각화**: 생성된 아이디어를 보기 좋게 포맷팅
- 📋 **결과 활용**: 복사, PDF 내보내기, 공유 기능
- 🔄 **오류 처리**: 친화적인 오류 메시지 및 재시도 기능
- 📱 **반응형 디자인**: 모바일, 태블릿, 데스크톱 지원

## 🔧 환경 설정

`.env` 파일에서 API 서버 URL을 설정할 수 있습니다:

```bash
VITE_API_URL=http://localhost:8000
```

## 🌐 API 연동

프론트엔드는 다음 API 엔드포인트와 통신합니다:

- `GET /health` - 서버 상태 확인
- `POST /api/v1/idea/generate` - 아이디어 생성

API 서버가 `http://localhost:8000`에서 실행되고 있어야 합니다.

## 🎨 커스터마이징

### 스타일 수정
- `src/style.css`: 전역 스타일
- `tailwind.config.js`: Tailwind CSS 설정

### 컴포넌트 수정
각 컴포넌트는 독립적으로 수정 가능하며, Vue 3의 Composition API를 사용합니다.

## 🔍 디버깅

브라우저 개발자 도구의 Console 탭에서 API 호출 로그를 확인할 수 있습니다.