'use client'

import { useState } from 'react'
import Link from 'next/link'
import { 
  Database,
  Shield,
  Cloud,
  Lock,
  FileText,
  Server,
  GitBranch,
  ChevronRight,
  ArrowLeft,
  ArrowRight,
  CheckCircle,
  AlertCircle,
  BarChart3,
  Users,
  Key,
  Globe,
  Layers,
  Search,
  Clock
} from 'lucide-react'

export default function MedicalDataPage() {
  const [activeSection, setActiveSection] = useState('overview')

  const sections = [
    { id: 'overview', title: '개요', icon: Database },
    { id: 'standards', title: '데이터 표준', icon: FileText },
    { id: 'integration', title: '통합 관리', icon: GitBranch },
    { id: 'security', title: '보안과 규제', icon: Shield },
    { id: 'analytics', title: '분석 플랫폼', icon: BarChart3 }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <header className="sticky top-0 z-30 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/medical-ai"
                className="flex items-center gap-2 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                <span>목록으로</span>
              </Link>
              <div className="h-6 w-px bg-gray-300 dark:bg-gray-700"></div>
              <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
                Chapter 7: 의료 데이터 관리
              </h1>
            </div>
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400 rounded-full text-sm font-medium">
                중급
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid lg:grid-cols-4 gap-8">
          {/* Sidebar Navigation */}
          <aside className="lg:col-span-1">
            <div className="sticky top-24 space-y-2">
              {sections.map((section) => {
                const Icon = section.icon
                return (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                      activeSection === section.id
                        ? 'bg-gradient-to-r from-orange-500 to-red-600 text-white shadow-lg'
                        : 'text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800'
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                    <span className="font-medium">{section.title}</span>
                  </button>
                )
              })}
            </div>
          </aside>

          {/* Main Content */}
          <main className="lg:col-span-3 space-y-8">
            {/* Overview Section */}
            {activeSection === 'overview' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    의료 데이터 관리와 AI
                  </h2>
                  
                  <div className="prose prose-lg dark:prose-invert max-w-none">
                    <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                      의료 데이터 관리는 방대한 양의 의료 정보를 효율적으로 수집, 저장, 처리, 분석하는 
                      종합적인 프로세스입니다. AI와 빅데이터 기술을 활용하여 의료 데이터의 가치를 
                      극대화하고 환자 치료 개선에 활용합니다.
                    </p>

                    <div className="grid md:grid-cols-2 gap-6 my-8">
                      <div className="bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg p-6">
                        <Database className="w-10 h-10 text-orange-600 dark:text-orange-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          데이터 유형
                        </h3>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-orange-500 mt-0.5" />
                            <span>구조화 데이터 (검사 결과)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-orange-500 mt-0.5" />
                            <span>비구조화 데이터 (의무기록)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-orange-500 mt-0.5" />
                            <span>영상 데이터 (DICOM)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-orange-500 mt-0.5" />
                            <span>시계열 데이터 (생체신호)</span>
                          </li>
                        </ul>
                      </div>

                      <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
                        <Server className="w-10 h-10 text-blue-600 dark:text-blue-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          데이터 규모
                        </h3>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>2.3 엑사바이트/년 (전세계)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>연평균 36% 증가율</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>80% 비구조화 데이터</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <ChevronRight className="w-5 h-5 text-blue-500 mt-0.5" />
                            <span>97% 미활용 데이터</span>
                          </li>
                        </ul>
                      </div>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 my-8">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        데이터 관리의 중요성
                      </h3>
                      <div className="grid grid-cols-3 gap-4">
                        <div className="text-center">
                          <div className="text-3xl font-bold text-orange-600 dark:text-orange-400">30%</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">진단 정확도 향상</div>
                        </div>
                        <div className="text-center">
                          <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">40%</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">치료 효율 증가</div>
                        </div>
                        <div className="text-center">
                          <div className="text-3xl font-bold text-green-600 dark:text-green-400">25%</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">비용 절감</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Standards Section */}
            {activeSection === 'standards' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    의료 데이터 표준
                  </h2>
                  
                  <div className="space-y-8">
                    {/* HL7 FHIR */}
                    <div className="border-l-4 border-orange-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        HL7 FHIR (Fast Healthcare Interoperability Resources)
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        의료 정보 교환을 위한 차세대 표준 프레임워크
                      </p>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <pre className="text-sm overflow-x-auto">
                          <code className="language-json">
{`{
  "resourceType": "Patient",
  "id": "example-patient",
  "meta": {
    "versionId": "1",
    "lastUpdated": "2024-01-15T10:30:00Z"
  },
  "identifier": [{
    "use": "official",
    "system": "http://hospital.org/patients",
    "value": "12345"
  }],
  "name": [{
    "use": "official",
    "family": "김",
    "given": ["철수"]
  }],
  "gender": "male",
  "birthDate": "1980-05-15",
  "address": [{
    "use": "home",
    "city": "서울",
    "country": "KR"
  }],
  "contact": [{
    "relationship": [{
      "coding": [{
        "system": "http://terminology.hl7.org/CodeSystem/v2-0131",
        "code": "E",
        "display": "Emergency Contact"
      }]
    }],
    "name": {
      "family": "김",
      "given": ["영희"]
    },
    "telecom": [{
      "system": "phone",
      "value": "010-1234-5678"
    }]
  }]
}`}
                          </code>
                        </pre>
                      </div>
                    </div>

                    {/* DICOM */}
                    <div className="border-l-4 border-blue-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        DICOM (Digital Imaging and Communications in Medicine)
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        의료 영상 저장 및 전송 표준
                      </p>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">주요 태그</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• (0010,0010) - Patient Name</li>
                            <li>• (0010,0020) - Patient ID</li>
                            <li>• (0008,0060) - Modality</li>
                            <li>• (0020,000D) - Study UID</li>
                          </ul>
                        </div>
                        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">영상 유형</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• CT - Computed Tomography</li>
                            <li>• MR - Magnetic Resonance</li>
                            <li>• US - Ultrasound</li>
                            <li>• XA - X-ray Angiography</li>
                          </ul>
                        </div>
                      </div>
                    </div>

                    {/* ICD & SNOMED */}
                    <div className="border-l-4 border-purple-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        의료 용어 표준
                      </h3>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">ICD-11</h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                            국제 질병 분류 체계
                          </p>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• 55,000+ 질병 코드</li>
                            <li>• 다국어 지원</li>
                            <li>• WHO 표준</li>
                          </ul>
                        </div>
                        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">SNOMED CT</h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                            체계적 의학 용어집
                          </p>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• 350,000+ 개념</li>
                            <li>• 1.5M+ 관계</li>
                            <li>• 온톨로지 기반</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Integration Section */}
            {activeSection === 'integration' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    데이터 통합 관리
                  </h2>
                  
                  <div className="space-y-8">
                    {/* Data Lake Architecture */}
                    <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        의료 데이터 레이크 아키텍처
                      </h3>
                      <div className="bg-white dark:bg-gray-800 rounded-lg p-4">
                        <pre className="text-sm overflow-x-auto">
                          <code className="language-python">
{`# Medical Data Lake Pipeline
from pyspark.sql import SparkSession
from delta import DeltaTable
import mlflow

class MedicalDataLake:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("MedicalDataLake") \
            .config("spark.sql.extensions", 
                   "io.delta.sql.DeltaSparkSessionExtension") \
            .getOrCreate()
        
    def ingest_ehr_data(self, source_path):
        """EHR 데이터 수집"""
        # Bronze Layer - Raw Data
        raw_data = self.spark.read \
            .format("json") \
            .option("multiline", "true") \
            .load(source_path)
        
        raw_data.write \
            .format("delta") \
            .mode("append") \
            .partitionBy("date", "hospital_id") \
            .save("/datalake/bronze/ehr")
        
        # Silver Layer - Cleaned Data
        cleaned_data = self.clean_and_validate(raw_data)
        cleaned_data.write \
            .format("delta") \
            .mode("overwrite") \
            .save("/datalake/silver/ehr")
        
        # Gold Layer - Analytics Ready
        analytics_data = self.create_features(cleaned_data)
        analytics_data.write \
            .format("delta") \
            .mode("overwrite") \
            .save("/datalake/gold/ehr")
    
    def process_medical_images(self, dicom_path):
        """DICOM 영상 처리"""
        import pydicom
        import numpy as np
        
        # 메타데이터 추출
        metadata = self.extract_dicom_metadata(dicom_path)
        
        # 영상 전처리
        processed_images = self.preprocess_images(dicom_path)
        
        # Feature extraction
        features = self.extract_image_features(processed_images)
        
        # Store in data lake
        self.store_image_data(metadata, features)
    
    def integrate_genomics_data(self, vcf_path):
        """유전체 데이터 통합"""
        genomics_df = self.spark.read \
            .format("com.databricks.vcf") \
            .load(vcf_path)
        
        # Annotate variants
        annotated = self.annotate_variants(genomics_df)
        
        # Join with clinical data
        integrated = annotated.join(
            self.clinical_data,
            on="patient_id"
        )
        
        return integrated`}
                          </code>
                        </pre>
                      </div>
                    </div>

                    {/* ETL Pipeline */}
                    <div className="border-l-4 border-green-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        ETL 파이프라인
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        실시간 데이터 수집, 변환, 적재 프로세스
                      </p>
                      <div className="grid md:grid-cols-3 gap-3">
                        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 text-center">
                          <Layers className="w-8 h-8 text-green-600 dark:text-green-400 mx-auto mb-2" />
                          <h4 className="font-semibold mb-1">Extract</h4>
                          <ul className="text-xs text-gray-600 dark:text-gray-400">
                            <li>• EMR/EHR</li>
                            <li>• PACS</li>
                            <li>• LIS</li>
                            <li>• IoT Devices</li>
                          </ul>
                        </div>
                        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 text-center">
                          <GitBranch className="w-8 h-8 text-blue-600 dark:text-blue-400 mx-auto mb-2" />
                          <h4 className="font-semibold mb-1">Transform</h4>
                          <ul className="text-xs text-gray-600 dark:text-gray-400">
                            <li>• 정규화</li>
                            <li>• 표준화</li>
                            <li>• 익명화</li>
                            <li>• 품질 검증</li>
                          </ul>
                        </div>
                        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 text-center">
                          <Database className="w-8 h-8 text-purple-600 dark:text-purple-400 mx-auto mb-2" />
                          <h4 className="font-semibold mb-1">Load</h4>
                          <ul className="text-xs text-gray-600 dark:text-gray-400">
                            <li>• Data Lake</li>
                            <li>• Data Warehouse</li>
                            <li>• ML Platform</li>
                            <li>• Analytics DB</li>
                          </ul>
                        </div>
                      </div>
                    </div>

                    {/* Master Data Management */}
                    <div className="border-l-4 border-purple-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        마스터 데이터 관리 (MDM)
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        환자, 의료진, 약물 등 핵심 데이터의 일관성 유지
                      </p>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <div className="grid md:grid-cols-2 gap-4">
                          <div>
                            <h4 className="font-semibold mb-2">환자 마스터</h4>
                            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                              <li>• 중복 제거</li>
                              <li>• ID 매핑</li>
                              <li>• 360도 뷰</li>
                              <li>• 이력 관리</li>
                            </ul>
                          </div>
                          <div>
                            <h4 className="font-semibold mb-2">데이터 거버넌스</h4>
                            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                              <li>• 품질 규칙</li>
                              <li>• 접근 제어</li>
                              <li>• 감사 추적</li>
                              <li>• 생명주기 관리</li>
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Security Section */}
            {activeSection === 'security' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    데이터 보안과 규제 준수
                  </h2>
                  
                  <div className="space-y-6">
                    {/* Privacy Regulations */}
                    <div className="grid md:grid-cols-2 gap-6">
                      <div className="bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 rounded-lg p-6">
                        <Shield className="w-10 h-10 text-red-600 dark:text-red-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          HIPAA (미국)
                        </h3>
                        <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                          <li>• PHI 보호 요구사항</li>
                          <li>• 최소 필요 원칙</li>
                          <li>• 암호화 의무</li>
                          <li>• 위반 시 최대 $2M 벌금</li>
                        </ul>
                      </div>
                      
                      <div className="bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-6">
                        <Globe className="w-10 h-10 text-blue-600 dark:text-blue-400 mb-4" />
                        <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                          GDPR (유럽)
                        </h3>
                        <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                          <li>• 명시적 동의 필요</li>
                          <li>• 삭제권 보장</li>
                          <li>• 72시간 내 침해 신고</li>
                          <li>• 위반 시 매출 4% 벌금</li>
                        </ul>
                      </div>
                    </div>

                    {/* Security Measures */}
                    <div className="border-l-4 border-orange-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        보안 조치
                      </h3>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <pre className="text-sm overflow-x-auto">
                          <code className="language-python">
{`# Medical Data Security Implementation
from cryptography.fernet import Fernet
import hashlib
import secrets

class MedicalDataSecurity:
    def __init__(self):
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
    def anonymize_patient_data(self, patient_data):
        """환자 데이터 익명화"""
        # Direct Identifiers 제거
        anonymized = patient_data.copy()
        
        # 이름, 주민번호 등 제거
        direct_identifiers = [
            'name', 'ssn', 'phone', 'email', 
            'address', 'medical_record_number'
        ]
        for field in direct_identifiers:
            if field in anonymized:
                del anonymized[field]
        
        # Quasi-identifiers 일반화
        if 'birth_date' in anonymized:
            # 연령대로 변환
            age = self.calculate_age(anonymized['birth_date'])
            anonymized['age_group'] = f"{(age // 10) * 10}대"
            del anonymized['birth_date']
        
        if 'zip_code' in anonymized:
            # 앞 3자리만 유지
            anonymized['zip_code'] = anonymized['zip_code'][:3] + '**'
        
        # 고유 ID 생성
        anonymized['patient_id'] = self.generate_pseudo_id(patient_data)
        
        return anonymized
    
    def encrypt_sensitive_data(self, data):
        """민감 데이터 암호화"""
        if isinstance(data, str):
            encrypted = self.cipher.encrypt(data.encode())
        else:
            encrypted = self.cipher.encrypt(str(data).encode())
        return encrypted.decode()
    
    def implement_access_control(self, user_role, data_type):
        """역할 기반 접근 제어"""
        access_matrix = {
            'physician': ['clinical', 'lab', 'imaging'],
            'nurse': ['vitals', 'medications'],
            'researcher': ['anonymized', 'aggregated'],
            'admin': ['administrative', 'billing']
        }
        
        return data_type in access_matrix.get(user_role, [])
    
    def audit_log(self, user_id, action, resource):
        """감사 로그 기록"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'ip_address': self.get_client_ip(),
            'session_id': self.get_session_id()
        }
        
        # 블록체인 또는 불변 저장소에 기록
        self.store_audit_log(log_entry)`}
                          </code>
                        </pre>
                      </div>
                    </div>

                    {/* De-identification */}
                    <div className="border-l-4 border-green-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        비식별화 기술
                      </h3>
                      <div className="grid md:grid-cols-3 gap-4">
                        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">Safe Harbor</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• 18개 식별자 제거</li>
                            <li>• HIPAA 준수</li>
                            <li>• 자동화 가능</li>
                          </ul>
                        </div>
                        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">Expert Determination</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• 통계적 분석</li>
                            <li>• 재식별 위험 평가</li>
                            <li>• 전문가 검증</li>
                          </ul>
                        </div>
                        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">Synthetic Data</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• AI 생성 데이터</li>
                            <li>• 통계적 유사성</li>
                            <li>• 프라이버시 보장</li>
                          </ul>
                        </div>
                      </div>
                    </div>

                    {/* Compliance Checklist */}
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        규제 준수 체크리스트
                      </h3>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div>
                          <h4 className="font-semibold mb-2">기술적 보호조치</h4>
                          <ul className="space-y-2 text-sm">
                            <li className="flex items-center gap-2">
                              <CheckCircle className="w-4 h-4 text-green-500" />
                              <span>전송 중 암호화 (TLS 1.3+)</span>
                            </li>
                            <li className="flex items-center gap-2">
                              <CheckCircle className="w-4 h-4 text-green-500" />
                              <span>저장 시 암호화 (AES-256)</span>
                            </li>
                            <li className="flex items-center gap-2">
                              <CheckCircle className="w-4 h-4 text-green-500" />
                              <span>접근 제어 (RBAC/ABAC)</span>
                            </li>
                            <li className="flex items-center gap-2">
                              <CheckCircle className="w-4 h-4 text-green-500" />
                              <span>감사 로그</span>
                            </li>
                          </ul>
                        </div>
                        <div>
                          <h4 className="font-semibold mb-2">관리적 보호조치</h4>
                          <ul className="space-y-2 text-sm">
                            <li className="flex items-center gap-2">
                              <CheckCircle className="w-4 h-4 text-green-500" />
                              <span>보안 정책 수립</span>
                            </li>
                            <li className="flex items-center gap-2">
                              <CheckCircle className="w-4 h-4 text-green-500" />
                              <span>직원 교육</span>
                            </li>
                            <li className="flex items-center gap-2">
                              <CheckCircle className="w-4 h-4 text-green-500" />
                              <span>사고 대응 계획</span>
                            </li>
                            <li className="flex items-center gap-2">
                              <CheckCircle className="w-4 h-4 text-green-500" />
                              <span>정기 보안 감사</span>
                            </li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Analytics Section */}
            {activeSection === 'analytics' && (
              <div className="space-y-6">
                <div className="bg-white dark:bg-gray-800 rounded-xl p-8 shadow-lg">
                  <h2 className="text-3xl font-bold mb-6 text-gray-900 dark:text-white">
                    의료 데이터 분석 플랫폼
                  </h2>
                  
                  <div className="space-y-8">
                    {/* Real-time Analytics */}
                    <div className="bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        실시간 분석 대시보드
                      </h3>
                      <div className="grid md:grid-cols-4 gap-4">
                        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
                          <BarChart3 className="w-8 h-8 text-purple-600 dark:text-purple-400 mx-auto mb-2" />
                          <div className="text-2xl font-bold">1,247</div>
                          <div className="text-sm text-gray-500">현재 입원 환자</div>
                        </div>
                        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
                          <Users className="w-8 h-8 text-blue-600 dark:text-blue-400 mx-auto mb-2" />
                          <div className="text-2xl font-bold">89%</div>
                          <div className="text-sm text-gray-500">병상 가동률</div>
                        </div>
                        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
                          <Clock className="w-8 h-8 text-green-600 dark:text-green-400 mx-auto mb-2" />
                          <div className="text-2xl font-bold">3.2h</div>
                          <div className="text-sm text-gray-500">평균 대기시간</div>
                        </div>
                        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
                          <AlertCircle className="w-8 h-8 text-red-600 dark:text-red-400 mx-auto mb-2" />
                          <div className="text-2xl font-bold">12</div>
                          <div className="text-sm text-gray-500">위험 환자</div>
                        </div>
                      </div>
                    </div>

                    {/* Predictive Analytics */}
                    <div className="border-l-4 border-blue-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        예측 분석 모델
                      </h3>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <pre className="text-sm overflow-x-auto">
                          <code className="language-python">
{`# Healthcare Predictive Analytics Platform
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

class HealthcareAnalytics:
    def predict_patient_flow(self):
        """환자 흐름 예측"""
        # Historical data
        historical = self.load_patient_flow_data()
        
        # Feature engineering
        features = self.create_temporal_features(historical)
        features['day_of_week'] = features['date'].dt.dayofweek
        features['month'] = features['date'].dt.month
        features['holiday'] = self.is_holiday(features['date'])
        
        # Train model
        model = RandomForestRegressor(n_estimators=100)
        model.fit(features[['day_of_week', 'month', 'holiday']], 
                 features['patient_count'])
        
        # Predict next 7 days
        future_dates = pd.date_range(start='today', periods=7)
        predictions = model.predict(self.prepare_features(future_dates))
        
        return predictions
    
    def analyze_treatment_outcomes(self, treatment_data):
        """치료 결과 분석"""
        # Cohort analysis
        cohorts = treatment_data.groupby(['treatment_type', 'patient_group'])
        
        outcomes = {
            'recovery_rate': cohorts['recovered'].mean(),
            'avg_recovery_time': cohorts['recovery_days'].mean(),
            'readmission_rate': cohorts['readmitted'].mean(),
            'cost_effectiveness': self.calculate_cost_effectiveness(cohorts)
        }
        
        # Statistical significance testing
        from scipy import stats
        for treatment_a, treatment_b in combinations(treatments, 2):
            p_value = stats.ttest_ind(
                treatment_data[treatment_data.treatment == treatment_a]['outcome'],
                treatment_data[treatment_data.treatment == treatment_b]['outcome']
            )[1]
            
            outcomes[f'{treatment_a}_vs_{treatment_b}_pvalue'] = p_value
        
        return outcomes
    
    def resource_optimization(self):
        """자원 최적화 분석"""
        # Linear programming for resource allocation
        from scipy.optimize import linprog
        
        # Objective: Minimize cost
        c = [staff_cost, equipment_cost, bed_cost]
        
        # Constraints: Service requirements
        A_ub = [[-1, -1, -1],  # Minimum resources
                [1, 0, 0],      # Staff limit
                [0, 1, 0],      # Equipment limit
                [0, 0, 1]]      # Bed limit
        b_ub = [-min_requirement, max_staff, max_equipment, max_beds]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
        
        return result.x  # Optimal resource allocation`}
                          </code>
                        </pre>
                      </div>
                    </div>

                    {/* Clinical Research Platform */}
                    <div className="border-l-4 border-green-500 pl-6">
                      <h3 className="text-xl font-semibold mb-3 text-gray-900 dark:text-white">
                        임상 연구 플랫폼
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 mb-4">
                        Real World Evidence 생성과 임상시험 데이터 분석
                      </p>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">RWE 분석</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• 치료 패턴 분석</li>
                            <li>• 비교효과 연구</li>
                            <li>• 안전성 모니터링</li>
                            <li>• 의료 경제성 평가</li>
                          </ul>
                        </div>
                        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                          <h4 className="font-semibold mb-2">임상시험 지원</h4>
                          <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                            <li>• 환자 코호트 선별</li>
                            <li>• 프로토콜 최적화</li>
                            <li>• 안전성 신호 감지</li>
                            <li>• 규제 보고서 생성</li>
                          </ul>
                        </div>
                      </div>
                    </div>

                    {/* Data Visualization */}
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
                      <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                        데이터 시각화 도구
                      </h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center">
                          <Search className="w-8 h-8 text-blue-600 dark:text-blue-400 mx-auto mb-2" />
                          <div className="font-semibold">Tableau</div>
                          <div className="text-xs text-gray-500">대시보드</div>
                        </div>
                        <div className="text-center">
                          <BarChart3 className="w-8 h-8 text-green-600 dark:text-green-400 mx-auto mb-2" />
                          <div className="font-semibold">Power BI</div>
                          <div className="text-xs text-gray-500">비즈니스 인텔리전스</div>
                        </div>
                        <div className="text-center">
                          <GitBranch className="w-8 h-8 text-purple-600 dark:text-purple-400 mx-auto mb-2" />
                          <div className="font-semibold">D3.js</div>
                          <div className="text-xs text-gray-500">인터랙티브</div>
                        </div>
                        <div className="text-center">
                          <Cloud className="w-8 h-8 text-orange-600 dark:text-orange-400 mx-auto mb-2" />
                          <div className="font-semibold">Plotly</div>
                          <div className="text-xs text-gray-500">과학적 시각화</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Navigation */}
            <div className="flex justify-between items-center pt-8">
              <Link
                href="/medical-ai/chapter/patient-monitoring"
                className="flex items-center gap-2 px-6 py-3 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
                이전 챕터
              </Link>
              <Link
                href="/medical-ai/chapter/ethics-regulation"
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-orange-600 to-red-600 text-white rounded-lg hover:shadow-lg transition-all"
              >
                다음 챕터
                <ArrowRight className="w-5 h-5" />
              </Link>
            </div>
          </main>
        </div>
      </div>
    </div>
  )
}