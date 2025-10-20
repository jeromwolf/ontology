'use client';

import { useState } from 'react';
import {
  Database, Layers, Zap, Shield,
  CheckCircle, XCircle, Info, TrendingUp
} from 'lucide-react';

interface Layer {
  name: string;
  technology: string;
  description: string;
  features: string[];
}

export default function DataLakehouseArchitect() {
  const [selectedArchitecture, setSelectedArchitecture] = useState<'delta' | 'iceberg' | 'hudi'>('delta');
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>(['acid', 'time-travel']);

  const architectures = {
    delta: {
      name: 'Delta Lake',
      vendor: 'Databricks',
      icon: '🔺',
      color: 'from-red-500 to-orange-600',
      description: 'Databricks가 개발한 오픈소스 스토리지 레이어',
      layers: [
        {
          name: 'Bronze Layer (Raw)',
          technology: 'Parquet + Delta Log',
          description: '원시 데이터를 있는 그대로 저장',
          features: ['Schema-on-read', '불변성', '감사 로그'],
        },
        {
          name: 'Silver Layer (Cleaned)',
          technology: 'Delta Tables',
          description: '정제되고 검증된 데이터',
          features: ['스키마 강제', '중복 제거', 'Data Quality 체크'],
        },
        {
          name: 'Gold Layer (Aggregated)',
          technology: 'Delta Tables',
          description: '비즈니스 로직이 적용된 집계 데이터',
          features: ['사전 집계', '비정규화', '성능 최적화'],
        },
      ],
      strengths: [
        'Spark와 완벽한 통합',
        'Z-Ordering으로 쿼리 최적화',
        'Auto Compaction',
        'Change Data Feed (CDC)',
      ],
      weaknesses: [
        'Databricks 생태계에 종속적',
        '다른 엔진 지원 제한적',
      ],
    },
    iceberg: {
      name: 'Apache Iceberg',
      vendor: 'Netflix (오픈소스)',
      icon: '🧊',
      color: 'from-blue-500 to-cyan-600',
      description: 'Netflix가 개발한 벤더 중립적 테이블 포맷',
      layers: [
        {
          name: 'Data Layer',
          technology: 'Parquet/ORC/Avro',
          description: '실제 데이터 파일',
          features: ['컬럼나 포맷', '압축', '파티셔닝'],
        },
        {
          name: 'Metadata Layer',
          technology: 'Manifest Files',
          description: '메타데이터 관리',
          features: ['스키마 진화', '파티션 진화', '스냅샷'],
        },
        {
          name: 'Catalog Layer',
          technology: 'Hive Metastore / REST',
          description: '테이블 카탈로그',
          features: ['네임스페이스', '버전 관리', '멀티 엔진 지원'],
        },
      ],
      strengths: [
        '엔진 중립적 (Spark, Flink, Presto, Trino)',
        '숨겨진 파티셔닝 (Hidden Partitioning)',
        '파티션 진화 (Partition Evolution)',
        '타임 트래블',
      ],
      weaknesses: [
        '상대적으로 복잡한 메타데이터 구조',
        '초기 설정 복잡도',
      ],
    },
    hudi: {
      name: 'Apache Hudi',
      vendor: 'Uber (오픈소스)',
      icon: '🚗',
      color: 'from-green-500 to-teal-600',
      description: 'Uber가 개발한 증분 처리 최적화 포맷',
      layers: [
        {
          name: 'Storage Layer',
          technology: 'Parquet + Hudi Metadata',
          description: '데이터 저장',
          features: ['Upsert/Delete 지원', 'Copy-on-Write', 'Merge-on-Read'],
        },
        {
          name: 'Index Layer',
          technology: 'Bloom Filter / HBase',
          description: '빠른 레코드 조회',
          features: ['키 기반 인덱싱', '증분 Upsert', '빠른 업데이트'],
        },
        {
          name: 'Timeline Layer',
          technology: 'Commit Timeline',
          description: '변경 이력 추적',
          features: ['증분 쿼리', 'CDC 지원', '롤백'],
        },
      ],
      strengths: [
        '증분 처리 (Incremental Processing)',
        'Upsert/Delete 성능 우수',
        'CDC 네이티브 지원',
        '스트리밍 수집 최적화',
      ],
      weaknesses: [
        '상대적으로 높은 메타데이터 오버헤드',
        'Merge-on-Read 모드 복잡도',
      ],
    },
  };

  const features = [
    { id: 'acid', name: 'ACID 트랜잭션', desc: '원자성, 일관성, 격리성, 지속성 보장' },
    { id: 'time-travel', name: '타임 트래블', desc: '과거 스냅샷 조회' },
    { id: 'schema-evolution', name: '스키마 진화', desc: '스키마 변경 시 기존 데이터 유지' },
    { id: 'upsert', name: 'Upsert/Delete', desc: '레코드 업데이트 및 삭제' },
    { id: 'partition-evolution', name: '파티션 진화', desc: '파티셔닝 전략 변경' },
    { id: 'incremental', name: '증분 처리', desc: '변경된 데이터만 읽기' },
  ];

  const current = architectures[selectedArchitecture];

  const toggleFeature = (featureId: string) => {
    setSelectedFeatures(prev =>
      prev.includes(featureId)
        ? prev.filter(id => id !== featureId)
        : [...prev, featureId]
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <Layers size={32} />
          <h2 className="text-2xl font-bold">데이터 레이크하우스 아키텍트</h2>
        </div>
        <p className="text-indigo-100">
          Delta Lake, Iceberg, Hudi 아키텍처를 비교하고 설계하세요
        </p>
      </div>

      {/* Architecture Selection */}
      <div className="grid md:grid-cols-3 gap-4">
        {Object.entries(architectures).map(([key, arch]) => (
          <button
            key={key}
            onClick={() => setSelectedArchitecture(key as any)}
            className={`p-6 rounded-xl border-2 transition-all text-left ${
              selectedArchitecture === key
                ? 'border-indigo-500 shadow-xl scale-105'
                : 'border-gray-200 dark:border-gray-700 hover:border-indigo-300'
            }`}
          >
            <div className="text-4xl mb-2">{arch.icon}</div>
            <h3 className="text-xl font-bold mb-1">{arch.name}</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{arch.vendor}</p>
            <p className="text-sm">{arch.description}</p>
          </button>
        ))}
      </div>

      {/* Architecture Layers */}
      <div className={`bg-gradient-to-r ${current.color} rounded-xl p-6 text-white`}>
        <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
          {current.icon} {current.name} 레이어 아키텍처
        </h3>
        <div className="grid md:grid-cols-3 gap-4">
          {current.layers.map((layer, idx) => (
            <div key={idx} className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
              <h4 className="font-bold mb-2">{layer.name}</h4>
              <p className="text-sm mb-3 text-white/80">{layer.description}</p>
              <div className="text-xs">
                <div className="font-semibold mb-1">기술: {layer.technology}</div>
                <ul className="space-y-1">
                  {layer.features.map((feature, i) => (
                    <li key={i} className="flex items-start gap-1">
                      <CheckCircle size={14} className="mt-0.5 flex-shrink-0" />
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Strengths & Weaknesses */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-green-600">
            <CheckCircle />
            강점
          </h3>
          <ul className="space-y-2">
            {current.strengths.map((strength, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-green-500 mt-0.5">✓</span>
                <span>{strength}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-orange-600">
            <Info />
            고려사항
          </h3>
          <ul className="space-y-2">
            {current.weaknesses.map((weakness, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-orange-500 mt-0.5">⚠</span>
                <span>{weakness}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Feature Comparison */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">🔧 기능 비교표</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-100 dark:bg-gray-700">
              <tr>
                <th className="p-3 text-left">기능</th>
                <th className="p-3 text-center">Delta Lake</th>
                <th className="p-3 text-center">Apache Iceberg</th>
                <th className="p-3 text-center">Apache Hudi</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {features.map((feature) => (
                <tr key={feature.id}>
                  <td className="p-3">
                    <div className="font-semibold">{feature.name}</div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">{feature.desc}</div>
                  </td>
                  <td className="p-3 text-center">
                    {['acid', 'time-travel', 'schema-evolution', 'upsert'].includes(feature.id) ? (
                      <CheckCircle className="inline text-green-500" size={20} />
                    ) : feature.id === 'incremental' ? (
                      <span className="text-yellow-600">△</span>
                    ) : (
                      <XCircle className="inline text-gray-300" size={20} />
                    )}
                  </td>
                  <td className="p-3 text-center">
                    {['acid', 'time-travel', 'schema-evolution', 'partition-evolution'].includes(feature.id) ? (
                      <CheckCircle className="inline text-green-500" size={20} />
                    ) : feature.id === 'upsert' ? (
                      <span className="text-yellow-600">△</span>
                    ) : (
                      <span className="text-yellow-600">△</span>
                    )}
                  </td>
                  <td className="p-3 text-center">
                    {['acid', 'time-travel', 'upsert', 'incremental'].includes(feature.id) ? (
                      <CheckCircle className="inline text-green-500" size={20} />
                    ) : feature.id === 'schema-evolution' ? (
                      <CheckCircle className="inline text-green-500" size={20} />
                    ) : (
                      <span className="text-yellow-600">△</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 flex gap-4 text-sm text-gray-600 dark:text-gray-400">
          <span><CheckCircle className="inline text-green-500" size={16} /> 완전 지원</span>
          <span><span className="text-yellow-600">△</span> 부분 지원</span>
          <span><XCircle className="inline text-gray-300" size={16} /> 미지원</span>
        </div>
      </div>

      {/* Code Example */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">📝 {current.name} 코드 예제</h3>
        {selectedArchitecture === 'delta' && (
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`# Delta Lake로 Bronze-Silver-Gold 파이프라인 구축
from delta.tables import DeltaTable
from pyspark.sql import SparkSession

spark = SparkSession.builder \\
    .appName("Lakehouse Pipeline") \\
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \\
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \\
    .getOrCreate()

# Bronze Layer: Raw 데이터 수집
df_raw = spark.read.json("s3://bucket/raw/events/")
df_raw.write.format("delta").mode("append").save("s3://lakehouse/bronze/events")

# Silver Layer: 정제
df_bronze = spark.read.format("delta").load("s3://lakehouse/bronze/events")
df_silver = df_bronze \\
    .dropDuplicates(["event_id"]) \\
    .filter("event_timestamp IS NOT NULL") \\
    .withColumn("event_date", to_date("event_timestamp"))

df_silver.write.format("delta") \\
    .mode("overwrite") \\
    .partitionBy("event_date") \\
    .option("overwriteSchema", "true") \\
    .save("s3://lakehouse/silver/events")

# Gold Layer: 집계
df_gold = df_silver.groupBy("event_date", "event_type") \\
    .agg(count("*").alias("event_count"))

df_gold.write.format("delta") \\
    .mode("overwrite") \\
    .save("s3://lakehouse/gold/daily_events")

# Time Travel 쿼리
df_yesterday = spark.read.format("delta") \\
    .option("versionAsOf", 0) \\
    .load("s3://lakehouse/silver/events")

# Z-Ordering 최적화
deltaTable = DeltaTable.forPath(spark, "s3://lakehouse/silver/events")
deltaTable.optimize().executeZOrderBy("user_id", "event_type")`}
          </pre>
        )}

        {selectedArchitecture === 'iceberg' && (
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`# Apache Iceberg로 Lakehouse 구축
from pyspark.sql import SparkSession

spark = SparkSession.builder \\
    .appName("Iceberg Lakehouse") \\
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \\
    .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog") \\
    .config("spark.sql.catalog.spark_catalog.type", "hive") \\
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \\
    .config("spark.sql.catalog.local.type", "hadoop") \\
    .config("spark.sql.catalog.local.warehouse", "s3://lakehouse/") \\
    .getOrCreate()

# 테이블 생성 (Iceberg 포맷)
spark.sql("""
    CREATE TABLE local.db.events (
        event_id STRING,
        user_id STRING,
        event_type STRING,
        event_timestamp TIMESTAMP,
        value DOUBLE
    )
    USING iceberg
    PARTITIONED BY (days(event_timestamp))
""")

# 데이터 쓰기
df.writeTo("local.db.events").append()

# Hidden Partitioning (사용자가 파티션 신경 안써도 됨)
spark.sql("SELECT * FROM local.db.events WHERE event_timestamp >= '2024-01-01'")

# 스키마 진화
spark.sql("ALTER TABLE local.db.events ADD COLUMN country STRING")

# Time Travel
spark.sql("SELECT * FROM local.db.events VERSION AS OF 123456789")

# 파티션 진화 (기존 데이터 유지하면서 파티셔닝 변경)
spark.sql("ALTER TABLE local.db.events DROP PARTITION FIELD days(event_timestamp)")
spark.sql("ALTER TABLE local.db.events ADD PARTITION FIELD bucket(16, user_id)")`}
          </pre>
        )}

        {selectedArchitecture === 'hudi' && (
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
{`# Apache Hudi로 Upsert 최적화 Lakehouse
from pyspark.sql import SparkSession

spark = SparkSession.builder \\
    .appName("Hudi Lakehouse") \\
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \\
    .getOrCreate()

# Hudi 옵션
hudi_options = {
    'hoodie.table.name': 'events',
    'hoodie.datasource.write.recordkey.field': 'event_id',
    'hoodie.datasource.write.partitionpath.field': 'event_date',
    'hoodie.datasource.write.precombine.field': 'event_timestamp',
    'hoodie.datasource.write.operation': 'upsert',  # insert/upsert/bulk_insert
    'hoodie.datasource.write.table.type': 'COPY_ON_WRITE',  # or MERGE_ON_READ
    'hoodie.upsert.shuffle.parallelism': 100,
}

# 초기 데이터 쓰기
df_initial.write.format("hudi") \\
    .options(**hudi_options) \\
    .mode("overwrite") \\
    .save("s3://lakehouse/hudi/events")

# Upsert (기존 레코드 업데이트)
df_updates.write.format("hudi") \\
    .options(**hudi_options) \\
    .mode("append") \\
    .save("s3://lakehouse/hudi/events")

# Delete (Hudi는 실제 삭제 지원)
df_deletes.write.format("hudi") \\
    .options(**{**hudi_options, 'hoodie.datasource.write.operation': 'delete'}) \\
    .mode("append") \\
    .save("s3://lakehouse/hudi/events")

# 증분 쿼리 (변경된 데이터만 읽기)
incremental_df = spark.read.format("hudi") \\
    .option("hoodie.datasource.query.type", "incremental") \\
    .option("hoodie.datasource.read.begin.instanttime", "20240101000000") \\
    .load("s3://lakehouse/hudi/events")

# Time Travel
snapshot_df = spark.read.format("hudi") \\
    .option("as.of.instant", "20240101120000") \\
    .load("s3://lakehouse/hudi/events")`}
          </pre>
        )}
      </div>

      {/* Decision Guide */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl">
        <h3 className="text-xl font-bold mb-4">💡 선택 가이드</h3>
        <div className="space-y-3">
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-red-600">Delta Lake 선택:</strong> Databricks/Spark 생태계 사용, Z-Ordering 필요, Auto Compaction 중요
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-blue-600">Apache Iceberg 선택:</strong> 멀티 엔진 지원 필요, Hidden Partitioning, 파티션 진화
          </div>
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
            <strong className="text-green-600">Apache Hudi 선택:</strong> 빈번한 Upsert/Delete, CDC 파이프라인, 증분 처리 최적화
          </div>
        </div>
      </div>
    </div>
  );
}
