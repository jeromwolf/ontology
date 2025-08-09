// 온톨로지 전문서 - 메인 JavaScript

// 전역 변수
let currentChapter = 'intro';
let darkMode = localStorage.getItem('darkMode') === 'true';
let sidebarCollapsed = false;

// DOM 로드 완료 시 실행
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    // intro 페이지는 이미 로드되어 있으므로 loadChapter를 호출하지 않음
    if (currentChapter !== 'intro') {
        loadChapter(currentChapter);
    }
});

// 앱 초기화
function initializeApp() {
    // 다크모드 적용
    if (darkMode) {
        document.body.classList.add('dark-mode');
    }
    
    // 프로그레스 바 생성
    createProgressBar();
    
    // 툴팁 초기화
    initializeTooltips();
    
    // 스크롤 애니메이션
    initializeScrollAnimations();
}

// 이벤트 리스너 설정
function setupEventListeners() {
    // 챕터 네비게이션
    document.querySelectorAll('.chapter-item').forEach(item => {
        item.addEventListener('click', function() {
            const chapter = this.getAttribute('data-chapter');
            loadChapter(chapter);
        });
    });
    
    // 다크모드 토글
    const darkModeToggle = document.getElementById('darkModeToggle');
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', toggleDarkMode);
    }
    
    // 사이드바 토글
    const sidebarToggle = document.getElementById('sidebarToggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', toggleSidebar);
    }
    
    // 스크롤 이벤트
    window.addEventListener('scroll', updateProgressBar);
}

// 챕터 로드
async function loadChapter(chapterId) {
    // intro 페이지는 이미 로드되어 있으므로 스킵
    if (chapterId === 'intro') {
        // 현재 챕터 저장
        currentChapter = chapterId;
        localStorage.setItem('currentChapter', chapterId);
        
        // 활성 챕터 표시
        document.querySelectorAll('.chapter-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-chapter="${chapterId}"]`)?.classList.add('active');
        return;
    }
    
    showLoader();
    
    try {
        const response = await fetch(`chapters/${chapterId}.html`);
        const content = await response.text();
        
        // 컨텐츠 업데이트
        const mainContent = document.getElementById('mainContent');
        mainContent.innerHTML = content;
        
        // 활성 챕터 표시
        document.querySelectorAll('.chapter-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-chapter="${chapterId}"]`)?.classList.add('active');
        
        // 현재 챕터 저장
        currentChapter = chapterId;
        localStorage.setItem('currentChapter', chapterId);
        
        // 그래프 초기화
        if (typeof initializeGraphs === 'function') {
            initializeGraphs();
        }
        
        // 코드 하이라이팅
        highlightCode();
        
        // 스크롤 상단으로
        window.scrollTo(0, 0);
        
    } catch (error) {
        console.error('챕터 로드 실패:', error);
        showError('챕터를 불러올 수 없습니다.');
    } finally {
        hideLoader();
    }
}

// 다크모드 토글
function toggleDarkMode() {
    darkMode = !darkMode;
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', darkMode);
    
    // 그래프 재렌더링
    if (window.currentGraphs) {
        window.currentGraphs.forEach(graph => graph.update());
    }
}

// 사이드바 토글
function toggleSidebar() {
    sidebarCollapsed = !sidebarCollapsed;
    document.querySelector('.sidebar').classList.toggle('collapsed');
    document.querySelector('.main-content').classList.toggle('full-width');
}

// 프로그레스 바 생성
function createProgressBar() {
    const progressBar = document.createElement('div');
    progressBar.className = 'progress-bar';
    progressBar.innerHTML = '<div class="progress-fill" id="progressFill"></div>';
    document.body.appendChild(progressBar);
}

// 프로그레스 바 업데이트
function updateProgressBar() {
    const scrollTop = window.pageYOffset;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const scrollPercent = (scrollTop / docHeight) * 100;
    
    const progressFill = document.getElementById('progressFill');
    if (progressFill) {
        progressFill.style.width = scrollPercent + '%';
    }
}

// 로더 표시
function showLoader() {
    const loader = document.createElement('div');
    loader.className = 'loader';
    loader.id = 'pageLoader';
    document.body.appendChild(loader);
}

// 로더 숨기기
function hideLoader() {
    const loader = document.getElementById('pageLoader');
    if (loader) {
        loader.remove();
    }
}

// 에러 표시
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    document.getElementById('mainContent').appendChild(errorDiv);
}

// 툴팁 초기화
function initializeTooltips() {
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    document.body.appendChild(tooltip);
    
    document.addEventListener('mouseover', function(e) {
        if (e.target.hasAttribute('data-tooltip')) {
            const text = e.target.getAttribute('data-tooltip');
            tooltip.textContent = text;
            tooltip.classList.add('show');
            
            const rect = e.target.getBoundingClientRect();
            tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
            tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';
        }
    });
    
    document.addEventListener('mouseout', function(e) {
        if (e.target.hasAttribute('data-tooltip')) {
            tooltip.classList.remove('show');
        }
    });
}

// 스크롤 애니메이션
function initializeScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);
    
    document.querySelectorAll('.animate-on-scroll').forEach(el => {
        observer.observe(el);
    });
}

// 코드 하이라이팅 (간단한 버전)
function highlightCode() {
    document.querySelectorAll('.code-block').forEach(block => {
        const lang = block.getAttribute('data-lang') || 'text';
        block.setAttribute('data-lang', lang);
        
        // 간단한 신택스 하이라이팅
        let code = block.textContent;
        
        // 키워드
        const keywords = ['class', 'function', 'const', 'let', 'var', 'if', 'else', 'for', 'while', 'return'];
        keywords.forEach(keyword => {
            const regex = new RegExp(`\\b${keyword}\\b`, 'g');
            code = code.replace(regex, `<span class="keyword">${keyword}</span>`);
        });
        
        // 문자열
        code = code.replace(/(["'])(?:(?=(\\?))\2.)*?\1/g, '<span class="string">$&</span>');
        
        // 숫자
        code = code.replace(/\b\d+\b/g, '<span class="number">$&</span>');
        
        // 주석
        code = code.replace(/\/\/.*/g, '<span class="comment">$&</span>');
        
        block.innerHTML = code;
    });
}

// 유틸리티 함수들
const utils = {
    // 디바운스
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // 쓰로틀
    throttle: function(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },
    
    // 랜덤 색상 생성
    getRandomColor: function() {
        const colors = [
            '#6366f1', '#8b5cf6', '#ec4899', '#f43f5e', 
            '#f59e0b', '#10b981', '#14b8a6', '#3b82f6'
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    },
    
    // 포맷팅
    formatNumber: function(num) {
        return new Intl.NumberFormat('ko-KR').format(num);
    }
};

// 전역으로 노출
window.OntologyBook = {
    loadChapter,
    toggleDarkMode,
    toggleSidebar,
    utils
};