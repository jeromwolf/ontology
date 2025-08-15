'use client';

export default function Chapter13() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-6">Chapter 13: 실전 프로젝트 - 의료 온톨로지</h1>
        
        <div className="bg-indigo-50 dark:bg-indigo-900/20 rounded-xl p-6 mb-8">
          <p className="text-lg">
            실제 의료 분야에서 사용되는 온톨로지를 구축해봅시다. 
            질병, 증상, 치료법 간의 복잡한 관계를 모델링하고, 
            의료 의사결정 지원 시스템의 기반을 만들어봅니다.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">프로젝트 개요</h2>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <h3 className="font-semibold mb-3">목표</h3>
          <ul className="space-y-2 text-gray-700 dark:text-gray-300">
            <li>• 일반적인 질병과 증상의 관계 모델링</li>
            <li>• 약물과 치료법 정보 통합</li>
            <li>• 환자 기록과 진단 이력 관리</li>
            <li>• 의료진을 위한 의사결정 지원</li>
            <li>• 약물 상호작용 경고 시스템</li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">온톨로지 설계</h2>
        
        <div className="bg-gray-50 dark:bg-gray-900 rounded-xl p-6 font-mono text-sm">
          <div className="space-y-4">
            <div>
              <div className="text-gray-500"># 핵심 클래스</div>
              <div className="text-blue-600">:Disease</div>
              <div className="text-blue-600">:Symptom</div>
              <div className="text-blue-600">:Treatment</div>
              <div className="text-blue-600">:Medication</div>
              <div className="text-blue-600">:Patient</div>
              <div className="text-blue-600">:MedicalProfessional</div>
            </div>
            
            <div>
              <div className="text-gray-500"># 질병 계층</div>
              <div className="text-blue-600">:InfectiousDisease</div>
              <div className="ml-4">rdfs:subClassOf :Disease .</div>
              <div className="text-blue-600">:ChronicDisease</div>
              <div className="ml-4">rdfs:subClassOf :Disease .</div>
              <div className="text-blue-600">:GeneticDisorder</div>
              <div className="ml-4">rdfs:subClassOf :Disease .</div>
            </div>
            
            <div>
              <div className="text-gray-500"># 관계 정의</div>
              <div className="text-green-600">:hasSymptom</div>
              <div className="ml-4">rdfs:domain :Disease ;</div>
              <div className="ml-4">rdfs:range :Symptom .</div>
              <div className="text-green-600">:treatedBy</div>
              <div className="ml-4">rdfs:domain :Disease ;</div>
              <div className="ml-4">rdfs:range :Treatment .</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">실제 데이터 예시</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">질병 인스턴스</h3>
            <div className="font-mono text-sm">
              :COVID19 rdf:type :InfectiousDisease ;<br/>
              <span className="ml-4">:hasSymptom :Fever ,</span><br/>
              <span className="ml-4">:Cough ,</span><br/>
              <span className="ml-4">:Fatigue ;</span><br/>
              <span className="ml-4">:transmissionRoute "airborne" .</span>
            </div>
          </div>
          
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-6">
            <h3 className="font-semibold mb-3">치료법 인스턴스</h3>
            <div className="font-mono text-sm">
              :Vaccination rdf:type :PreventiveTreatment ;<br/>
              <span className="ml-4">:prevents :COVID19 ;</span><br/>
              <span className="ml-4">:efficacy "95%" ;</span><br/>
              <span className="ml-4">:recommendedFor :AdultPatient .</span>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">추론 규칙</h2>
        
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6">
          <h3 className="font-semibold mb-3">의료 추론 예시</h3>
          <div className="font-mono text-sm space-y-4">
            <div>
              <div className="text-gray-500"># 고위험군 환자 정의</div>
              <div>:HighRiskPatient owl:equivalentClass [</div>
              <div className="ml-4">owl:unionOf (</div>
              <div className="ml-8">[rdf:type :Patient ; :age &gt;= 65]</div>
              <div className="ml-8">[rdf:type :Patient ; :hasCondition :ChronicDisease]</div>
              <div className="ml-8">[rdf:type :Patient ; :immunocompromised true]</div>
              <div className="ml-4">)</div>
              <div>] .</div>
            </div>
            
            <div>
              <div className="text-gray-500"># 약물 상호작용 경고</div>
              <div>:DrugInteraction a owl:Class ;</div>
              <div className="ml-4">owl:equivalentClass [</div>
              <div className="ml-8">owl:onProperty :interactsWith ;</div>
              <div className="ml-8">owl:minCardinality 1</div>
              <div className="ml-4">] .</div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">SPARQL 쿼리 예시</h2>
        
        <div className="space-y-4">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <h3 className="font-semibold mb-3">1. 특정 증상에 대한 가능한 질병 조회</h3>
            <div className="font-mono text-sm">
              SELECT ?disease ?diseaseName WHERE {`{`}<br/>
              <span className="ml-2">?disease :hasSymptom :Fever .</span><br/>
              <span className="ml-2">?disease :hasSymptom :Cough .</span><br/>
              <span className="ml-2">?disease rdfs:label ?diseaseName .</span><br/>
              {`}`} ORDER BY ?diseaseName
            </div>
          </div>
          
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
            <h3 className="font-semibold mb-3">2. 약물 상호작용 확인</h3>
            <div className="font-mono text-sm">
              ASK WHERE {`{`}<br/>
              <span className="ml-2">:Medication1 :interactsWith :Medication2 .</span><br/>
              <span className="ml-2">?patient :takes :Medication1 .</span><br/>
              <span className="ml-2">?patient :takes :Medication2 .</span><br/>
              {`}`}
            </div>
          </div>
        </div>
      </section>

      <section className="bg-yellow-50 dark:bg-yellow-900/20 rounded-xl p-6">
        <h2 className="text-xl font-bold mb-3 flex items-center gap-2">
          <span className="text-2xl">⚕️</span>
          의료 온톨로지 표준
        </h2>
        <div className="grid md:grid-cols-3 gap-4">
          <div>
            <h3 className="font-semibold text-blue-600 dark:text-blue-400">SNOMED CT</h3>
            <p className="text-sm">포괄적인 임상 용어 체계</p>
          </div>
          <div>
            <h3 className="font-semibold text-green-600 dark:text-green-400">ICD-11</h3>
            <p className="text-sm">질병 분류 국제 표준</p>
          </div>
          <div>
            <h3 className="font-semibold text-purple-600 dark:text-purple-400">HL7 FHIR</h3>
            <p className="text-sm">의료 정보 교환 표준</p>
          </div>
        </div>
      </section>
    </div>
  )
}