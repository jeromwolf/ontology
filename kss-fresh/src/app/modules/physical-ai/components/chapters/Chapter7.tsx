'use client';

import React from 'react';

export default function Chapter7() {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <h2>엣지 AI와 실시간 처리</h2>
      
      <h3>1. 엣지 컴퓨팅의 중요성</h3>
      <p>
        Physical AI는 밀리초 단위의 반응이 필요하므로, 
        클라우드가 아닌 엣지에서 처리해야 합니다.
      </p>

      <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-lg my-6">
        <h4 className="font-semibold mb-3">모델 경량화 기법</h4>
        <pre className="bg-white dark:bg-gray-900 p-3 rounded text-sm overflow-x-auto">
{`# TensorFlow Lite 변환
import tensorflow as tf

def quantize_model(model_path):
    # 모델 로드
    model = tf.keras.models.load_model(model_path)
    
    # 대표 데이터셋 생성
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, 224, 224, 3).astype(np.float32)
            yield [data]
    
    # 변환기 설정
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    
    # INT8 양자화
    tflite_model = converter.convert()
    
    return tflite_model`}
        </pre>
      </div>
    </div>
  )
}