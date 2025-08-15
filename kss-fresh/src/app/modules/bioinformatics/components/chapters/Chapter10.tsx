'use client';

import { useState } from 'react';
import { Copy, CheckCircle } from 'lucide-react';

export default function Chapter10() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const clinicalCode = `# 임상 유전체 분석 파이프라인
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lifelines import KaplanMeierFitter, CoxPHFitter

class ClinicalGenomicsAnalyzer:
    def __init__(self):
        self.variant_db = self.load_variant_database()
        self.drug_db = self.load_pharmacogenomics_db()
        
    def analyze_tumor_mutations(self, vcf_file, bam_file):
        """종양 변이 분석 및 치료 옵션 제시"""
        
        # Somatic mutation calling
        mutations = self.call_somatic_mutations(vcf_file, bam_file)
        
        # Driver mutation identification
        driver_mutations = []
        for mutation in mutations:
            if self.is_driver_mutation(mutation):
                driver_mutations.append({
                    'gene': mutation['gene'],
                    'variant': mutation['variant'],
                    'vaf': mutation['vaf'],  # Variant Allele Frequency
                    'cosmic_id': mutation['cosmic_id'],
                    'clinical_significance': mutation['significance']
                })
        
        # Tumor Mutational Burden (TMB) 계산
        tmb = len(mutations) / 30  # mutations per Mb
        
        # Microsatellite Instability (MSI) 검사
        msi_status = self.check_msi_status(bam_file)
        
        # 치료 옵션 추천
        treatment_options = self.recommend_treatments(
            driver_mutations, tmb, msi_status
        )
        
        return {
            'driver_mutations': driver_mutations,
            'tmb': tmb,
            'msi_status': msi_status,
            'treatment_options': treatment_options
        }
    
    def liquid_biopsy_analysis(self, cfDNA_data):
        """Circulating tumor DNA (ctDNA) 분석"""
        
        results = {
            'ctDNA_fraction': None,
            'mutations_detected': [],
            'copy_number_alterations': [],
            'monitoring_recommendation': None
        }
        
        # ctDNA fraction 계산
        results['ctDNA_fraction'] = self.calculate_ctDNA_fraction(cfDNA_data)
        
        # Ultra-deep sequencing으로 low-frequency mutations 검출
        for region in cfDNA_data['target_regions']:
            mutations = self.detect_low_freq_mutations(
                region, 
                min_vaf=0.001  # 0.1% detection limit
            )
            results['mutations_detected'].extend(mutations)
        
        # Copy number alterations
        results['copy_number_alterations'] = self.detect_cna_from_cfDNA(cfDNA_data)
        
        # Monitoring recommendation
        if results['ctDNA_fraction'] > 0.01:
            results['monitoring_recommendation'] = 'High risk - monthly monitoring'
        elif results['ctDNA_fraction'] > 0.001:
            results['monitoring_recommendation'] = 'Medium risk - quarterly monitoring'
        else:
            results['monitoring_recommendation'] = 'Low risk - biannual monitoring'
        
        return results
    
    def pharmacogenomics_analysis(self, patient_genotype):
        """약물유전체 분석으로 개인 맞춤 처방"""
        
        recommendations = []
        
        # Key pharmacogenes 검사
        pharmacogenes = ['CYP2D6', 'CYP2C19', 'CYP2C9', 'VKORC1', 
                        'TPMT', 'NUDT15', 'DPYD', 'HLA-B']
        
        for gene in pharmacogenes:
            genotype = patient_genotype.get(gene)
            if genotype:
                # 약물 대사 표현형 예측
                phenotype = self.predict_metabolizer_status(gene, genotype)
                
                # 영향받는 약물 목록
                affected_drugs = self.drug_db[gene]
                
                for drug in affected_drugs:
                    recommendation = {
                        'drug': drug['name'],
                        'gene': gene,
                        'genotype': genotype,
                        'phenotype': phenotype,
                        'dosing_guideline': self.get_dosing_guideline(
                            drug, phenotype
                        ),
                        'alternative_drugs': drug.get('alternatives', [])
                    }
                    recommendations.append(recommendation)
        
        return recommendations
    
    def survival_analysis(self, genomic_data, clinical_data):
        """생존 분석 및 예후 예측"""
        
        # Cox proportional hazards model
        cph = CoxPHFitter()
        
        # Genomic and clinical features 결합
        combined_data = pd.concat([genomic_data, clinical_data], axis=1)
        
        # Model fitting
        cph.fit(combined_data, 
               duration_col='survival_time', 
               event_col='event_occurred')
        
        # Hazard ratios 계산
        hazard_ratios = cph.summary
        
        # Risk score 계산
        risk_scores = cph.predict_partial_hazard(combined_data)
        
        # Risk groups 분류
        risk_groups = pd.qcut(risk_scores, q=3, 
                              labels=['Low', 'Medium', 'High'])
        
        # Kaplan-Meier curves for each risk group
        kmf = KaplanMeierFitter()
        survival_curves = {}
        
        for group in ['Low', 'Medium', 'High']:
            group_data = combined_data[risk_groups == group]
            kmf.fit(group_data['survival_time'], 
                   group_data['event_occurred'])
            survival_curves[group] = kmf.survival_function_
        
        return {
            'hazard_ratios': hazard_ratios,
            'risk_scores': risk_scores,
            'risk_groups': risk_groups,
            'survival_curves': survival_curves
        }`

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          1. Cancer Genomics
        </h2>
        <p className="mb-4">
          암 유전체학은 종양의 유전적 변화를 분석하여 정밀 치료를 가능하게 합니다.
        </p>
        
        <div className="bg-emerald-50 dark:bg-emerald-900/20 rounded-lg p-6 mb-6">
          <h3 className="font-bold mb-3">종양 유전체 분석 항목</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2">변이 분석</h4>
              <ul className="space-y-1 text-sm">
                <li>• Somatic mutations</li>
                <li>• Copy number alterations</li>
                <li>• Structural variants</li>
                <li>• Gene fusions</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2">바이오마커</h4>
              <ul className="space-y-1 text-sm">
                <li>• TMB (Tumor Mutational Burden)</li>
                <li>• MSI (Microsatellite Instability)</li>
                <li>• HRD (Homologous Recombination Deficiency)</li>
                <li>• PD-L1 expression</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          2. 임상 유전체 분석 파이프라인
        </h2>
        
        <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-mono text-gray-600 dark:text-gray-400">clinical_genomics.py</span>
            <button
              onClick={() => copyCode(clinicalCode, 'clinical')}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-800 rounded"
            >
              {copiedCode === 'clinical' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
          <pre className="text-sm overflow-x-auto">
            <code>{clinicalCode}</code>
          </pre>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          3. Liquid Biopsy
        </h2>
        <p className="mb-4">
          혈액에서 순환 종양 DNA (ctDNA)를 분석하여 비침습적으로 암을 진단하고 모니터링합니다.
        </p>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-emerald-200 dark:border-emerald-800">
          <h3 className="font-bold mb-3">Liquid Biopsy 응용</h3>
          <ul className="space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">✓</span>
              <span><strong>조기 진단:</strong> 암 스크리닝 및 조기 발견</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">✓</span>
              <span><strong>치료 반응:</strong> 실시간 치료 효과 모니터링</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">✓</span>
              <span><strong>내성 감지:</strong> 약물 내성 변이 조기 발견</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-500 mt-1">✓</span>
              <span><strong>재발 모니터링:</strong> MRD (Minimal Residual Disease) 추적</span>
            </li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4 text-emerald-700 dark:text-emerald-400">
          4. Pharmacogenomics
        </h2>
        
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
          <h3 className="font-bold mb-3">주요 약물유전자</h3>
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left p-2">유전자</th>
                <th className="text-left p-2">영향 약물</th>
                <th className="text-left p-2">임상 영향</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="p-2">CYP2D6</td>
                <td className="p-2">Codeine, Tamoxifen</td>
                <td className="p-2">진통/항암 효과</td>
              </tr>
              <tr>
                <td className="p-2">VKORC1</td>
                <td className="p-2">Warfarin</td>
                <td className="p-2">항응고제 용량</td>
              </tr>
              <tr>
                <td className="p-2">TPMT</td>
                <td className="p-2">6-MP, Azathioprine</td>
                <td className="p-2">골수 독성</td>
              </tr>
              <tr>
                <td className="p-2">HLA-B*5701</td>
                <td className="p-2">Abacavir</td>
                <td className="p-2">과민반응</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}