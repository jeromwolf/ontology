#!/bin/bash

# CLAUDE.md 파일 크기 체크 스크립트
# 1000줄 이상 파일을 자동으로 감지하여 경고

echo "🔍 KSS 모듈 파일 크기 체크 시작..."
echo "================================"

CRITICAL_LIMIT=1000
WARNING_LIMIT=700

# ChapterContent.tsx 파일들 찾기
find src/app/modules -name "ChapterContent.tsx" | while read file; do
    lines=$(wc -l < "$file")
    module=$(echo "$file" | cut -d'/' -f5)
    
    if [ $lines -ge $CRITICAL_LIMIT ]; then
        echo "🔴 CRITICAL: $module - $lines lines (>${CRITICAL_LIMIT})"
    elif [ $lines -ge $WARNING_LIMIT ]; then
        echo "🟡 WARNING: $module - $lines lines (>${WARNING_LIMIT})"
    else
        echo "✅ OK: $module - $lines lines"
    fi
done

echo "================================"
echo "📋 요약:"
echo "- CRITICAL (>$CRITICAL_LIMIT lines): 즉시 리팩토링 필요"
echo "- WARNING (>$WARNING_LIMIT lines): 곧 리팩토링 필요"
echo "- OK: 가이드라인 준수 중"