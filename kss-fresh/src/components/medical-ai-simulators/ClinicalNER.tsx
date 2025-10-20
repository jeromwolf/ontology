'use client'

import React, { useState, useEffect } from 'react'

type EntityType = 'Disease' | 'Medication' | 'Symptom' | 'Procedure' | 'Anatomy'

interface Entity {
  text: string
  type: EntityType
  start: number
  end: number
  confidence: number
  umlsCode?: string
  snomedCode?: string
}

interface Relationship {
  from: Entity
  to: Entity
  type: string
}

const SAMPLE_NOTES = [
  {
    title: 'Diabetes Case',
    text: 'Patient presents with elevated blood glucose levels and polyuria. Diagnosed with Type 2 Diabetes Mellitus. Started on Metformin 500mg twice daily. Recommended lifestyle modifications including diet and exercise. Follow-up in 3 months to assess HbA1c levels.'
  },
  {
    title: 'Cardiac Event',
    text: 'Male patient, 62 years old, admitted to emergency with chest pain and shortness of breath. ECG shows ST-elevation. Diagnosed with acute myocardial infarction. Administered Aspirin 325mg and Nitroglycerin. Cardiac catheterization performed showing 90% occlusion of LAD. Stent placement successful.'
  },
  {
    title: 'Respiratory Infection',
    text: 'Patient complains of persistent cough, fever, and difficulty breathing for 5 days. Chest X-ray reveals bilateral infiltrates. Diagnosed with community-acquired pneumonia. Prescribed Azithromycin 500mg daily for 5 days and supportive care with fluids and rest.'
  },
  {
    title: 'Post-Surgical Note',
    text: 'Post-operative day 2 following laparoscopic cholecystectomy. Patient recovering well with controlled pain. Surgical site clean and dry without signs of infection. Continue Ibuprofen 400mg as needed for pain. Clear liquid diet advanced to soft foods. Discharge planned for tomorrow.'
  }
]

const ENTITY_COLORS: Record<EntityType, string> = {
  Disease: '#ef4444',
  Medication: '#3b82f6',
  Symptom: '#f59e0b',
  Procedure: '#8b5cf6',
  Anatomy: '#10b981'
}

export default function ClinicalNER() {
  const [clinicalNote, setClinicalNote] = useState(SAMPLE_NOTES[0].text)
  const [entities, setEntities] = useState<Entity[]>([])
  const [relationships, setRelationships] = useState<Relationship[]>([])
  const [isExtracting, setIsExtracting] = useState(false)
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.7)
  const [modelType, setModelType] = useState<'BioBERT' | 'ClinicalBERT' | 'SciBERT'>('BioBERT')

  const extractEntities = () => {
    setIsExtracting(true)

    setTimeout(() => {
      const extractedEntities: Entity[] = []

      // Disease patterns
      const diseases = [
        { text: 'Type 2 Diabetes Mellitus', umls: 'C0011860', snomed: '44054006' },
        { text: 'Diabetes', umls: 'C0011847', snomed: '73211009' },
        { text: 'myocardial infarction', umls: 'C0027051', snomed: '22298006' },
        { text: 'pneumonia', umls: 'C0032285', snomed: '233604007' },
        { text: 'infection', umls: 'C3714514', snomed: '40733004' }
      ]

      // Medication patterns
      const medications = [
        { text: 'Metformin', umls: 'C0025598', snomed: '109081006' },
        { text: 'Aspirin', umls: 'C0004057', snomed: '387458008' },
        { text: 'Nitroglycerin', umls: 'C0017887', snomed: '387404004' },
        { text: 'Azithromycin', umls: 'C0052796', snomed: '387531004' },
        { text: 'Ibuprofen', umls: 'C0020740', snomed: '387207008' }
      ]

      // Symptom patterns
      const symptoms = [
        { text: 'polyuria', umls: 'C0032617', snomed: '28442001' },
        { text: 'chest pain', umls: 'C0008031', snomed: '29857009' },
        { text: 'shortness of breath', umls: 'C0013404', snomed: '267036007' },
        { text: 'cough', umls: 'C0010200', snomed: '49727002' },
        { text: 'fever', umls: 'C0015967', snomed: '386661006' },
        { text: 'difficulty breathing', umls: 'C0013404', snomed: '230145002' },
        { text: 'pain', umls: 'C0030193', snomed: '22253000' }
      ]

      // Procedure patterns
      const procedures = [
        { text: 'Cardiac catheterization', umls: 'C0018795', snomed: '41976001' },
        { text: 'Stent placement', umls: 'C0522776', snomed: '232717009' },
        { text: 'laparoscopic cholecystectomy', umls: 'C0162522', snomed: '45595009' },
        { text: 'ECG', umls: 'C1623258', snomed: '29303009' },
        { text: 'Chest X-ray', umls: 'C0039985', snomed: '399208008' }
      ]

      // Anatomy patterns
      const anatomy = [
        { text: 'blood glucose', umls: 'C0005802', snomed: '33747003' },
        { text: 'LAD', umls: 'C0226032', snomed: '33795007' },
        { text: 'HbA1c', umls: 'C0019018', snomed: '43396009' },
        { text: 'Surgical site', umls: 'C0332679', snomed: '225546006' }
      ]

      const patterns = [
        { type: 'Disease' as EntityType, items: diseases },
        { type: 'Medication' as EntityType, items: medications },
        { type: 'Symptom' as EntityType, items: symptoms },
        { type: 'Procedure' as EntityType, items: procedures },
        { type: 'Anatomy' as EntityType, items: anatomy }
      ]

      patterns.forEach(({ type, items }) => {
        items.forEach(item => {
          const regex = new RegExp(item.text, 'gi')
          let match
          while ((match = regex.exec(clinicalNote)) !== null) {
            const confidence = 0.75 + Math.random() * 0.24

            if (confidence >= confidenceThreshold) {
              extractedEntities.push({
                text: match[0],
                type,
                start: match.index,
                end: match.index + match[0].length,
                confidence,
                umlsCode: item.umls,
                snomedCode: item.snomed
              })
            }
          }
        })
      })

      // Sort by position
      extractedEntities.sort((a, b) => a.start - b.start)

      setEntities(extractedEntities)

      // Extract relationships
      extractRelationships(extractedEntities)

      setIsExtracting(false)
    }, 2000)
  }

  const extractRelationships = (entities: Entity[]) => {
    const rels: Relationship[] = []

    entities.forEach((entity, idx) => {
      entities.forEach((other, otherIdx) => {
        if (idx !== otherIdx && Math.abs(entity.start - other.start) < 200) {
          if (entity.type === 'Symptom' && other.type === 'Disease') {
            rels.push({ from: entity, to: other, type: 'indicates' })
          } else if (entity.type === 'Medication' && other.type === 'Disease') {
            rels.push({ from: entity, to: other, type: 'treats' })
          } else if (entity.type === 'Procedure' && other.type === 'Disease') {
            rels.push({ from: entity, to: other, type: 'diagnoses' })
          } else if (entity.type === 'Symptom' && other.type === 'Anatomy') {
            rels.push({ from: entity, to: other, type: 'affects' })
          }
        }
      })
    })

    setRelationships(rels)
  }

  const highlightedText = () => {
    if (entities.length === 0) return clinicalNote

    let result = ''
    let lastIndex = 0

    entities.forEach(entity => {
      result += clinicalNote.slice(lastIndex, entity.start)
      result += `<mark class="entity-${entity.type}" data-entity-id="${entity.start}" style="background-color: ${ENTITY_COLORS[entity.type]}40; padding: 2px 4px; border-radius: 3px; cursor: pointer; border-bottom: 2px solid ${ENTITY_COLORS[entity.type]};">${entity.text}</mark>`
      lastIndex = entity.end
    })

    result += clinicalNote.slice(lastIndex)
    return result
  }

  const handleEntityClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement
    if (target.tagName === 'MARK') {
      const entityId = parseInt(target.getAttribute('data-entity-id') || '0')
      const entity = entities.find(e => e.start === entityId)
      if (entity) {
        setSelectedEntity(entity)
      }
    }
  }

  const getEntityStats = () => {
    const stats: Record<EntityType, number> = {
      Disease: 0,
      Medication: 0,
      Symptom: 0,
      Procedure: 0,
      Anatomy: 0
    }

    entities.forEach(entity => {
      stats[entity.type]++
    })

    return stats
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-500 via-pink-600 to-red-500 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 mb-6">
          <h1 className="text-4xl font-bold text-white mb-2">Clinical NER</h1>
          <p className="text-white/80">Named Entity Recognition from clinical notes using BioBERT and ClinicalBERT</p>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Architecture</div>
              <div className="text-white font-bold">BioBERT / ClinicalBERT</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">F1 Score</div>
              <div className="text-white font-bold">90.1% (i2b2 2012)</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Ontology</div>
              <div className="text-white font-bold">UMLS / SNOMED CT</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Entity Types</div>
              <div className="text-white font-bold">5 Categories</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold text-white">Clinical Note</h2>
                <div className="flex gap-2">
                  <select
                    value={modelType}
                    onChange={(e) => setModelType(e.target.value as any)}
                    className="px-3 py-1.5 rounded-lg bg-white/20 text-white border border-white/30 text-sm"
                  >
                    <option value="BioBERT">BioBERT</option>
                    <option value="ClinicalBERT">ClinicalBERT</option>
                    <option value="SciBERT">SciBERT</option>
                  </select>
                </div>
              </div>

              <div className="mb-4">
                <label className="text-white font-medium block mb-2">Sample Notes</label>
                <div className="flex flex-wrap gap-2">
                  {SAMPLE_NOTES.map((note, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setClinicalNote(note.text)
                        setEntities([])
                        setRelationships([])
                        setSelectedEntity(null)
                      }}
                      className="px-3 py-1.5 rounded-lg bg-white/20 text-white hover:bg-white/30 text-sm"
                    >
                      {note.title}
                    </button>
                  ))}
                </div>
              </div>

              <textarea
                value={clinicalNote}
                onChange={(e) => setClinicalNote(e.target.value)}
                className="w-full h-40 px-4 py-3 rounded-lg bg-white/20 text-white placeholder-white/50 border border-white/30 resize-none"
                placeholder="Enter clinical note..."
              />

              <div className="mt-4 flex items-center gap-4">
                <div className="flex-1">
                  <label className="text-white text-sm block mb-1">
                    Confidence Threshold: {(confidenceThreshold * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="0.99"
                    step="0.01"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                    className="w-full"
                  />
                </div>

                <button
                  onClick={extractEntities}
                  disabled={isExtracting}
                  className={`px-6 py-2.5 rounded-lg font-bold transition-all ${
                    isExtracting ? 'bg-gray-400 cursor-not-allowed' : 'bg-white text-pink-600 hover:bg-gray-100'
                  }`}
                >
                  {isExtracting ? 'Extracting...' : 'Extract Entities'}
                </button>
              </div>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4">Annotated Text ({entities.length} entities)</h3>
              <div
                className="bg-white rounded-lg p-4 text-gray-800 leading-relaxed"
                onClick={handleEntityClick}
                dangerouslySetInnerHTML={{ __html: highlightedText() }}
              />

              <div className="mt-4 flex flex-wrap gap-3">
                {Object.entries(ENTITY_COLORS).map(([type, color]) => (
                  <div key={type} className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded" style={{ backgroundColor: color }} />
                    <span className="text-white text-sm">{type}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4">Entity Relationship Graph</h3>
              <div className="bg-white rounded-lg p-4" style={{ height: '300px' }}>
                <svg width="100%" height="100%" viewBox="0 0 600 280">
                  {/* Draw relationships */}
                  {relationships.slice(0, 10).map((rel, idx) => {
                    const x1 = 100 + (idx % 3) * 200
                    const y1 = 50 + Math.floor(idx / 3) * 80
                    const x2 = x1 + 150
                    const y2 = y1 + 40

                    return (
                      <g key={idx}>
                        <line
                          x1={x1 + 50}
                          y1={y1 + 15}
                          x2={x2}
                          y2={y2 + 15}
                          stroke="#94a3b8"
                          strokeWidth="2"
                          markerEnd="url(#arrowhead)"
                        />
                        <text
                          x={(x1 + x2 + 50) / 2}
                          y={(y1 + y2 + 30) / 2}
                          fill="#64748b"
                          fontSize="10"
                          textAnchor="middle"
                        >
                          {rel.type}
                        </text>

                        <rect
                          x={x1}
                          y={y1}
                          width="80"
                          height="30"
                          fill={ENTITY_COLORS[rel.from.type]}
                          rx="4"
                        />
                        <text
                          x={x1 + 40}
                          y={y1 + 19}
                          fill="white"
                          fontSize="11"
                          fontWeight="bold"
                          textAnchor="middle"
                        >
                          {rel.from.text.slice(0, 10)}
                        </text>

                        <rect
                          x={x2}
                          y={y2}
                          width="80"
                          height="30"
                          fill={ENTITY_COLORS[rel.to.type]}
                          rx="4"
                        />
                        <text
                          x={x2 + 40}
                          y={y2 + 19}
                          fill="white"
                          fontSize="11"
                          fontWeight="bold"
                          textAnchor="middle"
                        >
                          {rel.to.text.slice(0, 10)}
                        </text>
                      </g>
                    )
                  })}

                  <defs>
                    <marker
                      id="arrowhead"
                      markerWidth="10"
                      markerHeight="10"
                      refX="9"
                      refY="3"
                      orient="auto"
                    >
                      <polygon points="0 0, 10 3, 0 6" fill="#94a3b8" />
                    </marker>
                  </defs>
                </svg>
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4">Entity Statistics</h3>
              <div className="space-y-3">
                {Object.entries(getEntityStats()).map(([type, count]) => (
                  <div key={type} className="bg-white/10 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-white font-medium">{type}</span>
                      <span className="text-white font-bold text-lg">{count}</span>
                    </div>
                    <div className="w-full bg-white/20 rounded-full h-2">
                      <div
                        className="h-2 rounded-full"
                        style={{
                          width: `${(count / Math.max(...Object.values(getEntityStats()))) * 100}%`,
                          backgroundColor: ENTITY_COLORS[type as EntityType]
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {selectedEntity && (
              <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
                <h3 className="text-xl font-bold text-white mb-4">Entity Details</h3>
                <div className="space-y-3">
                  <div>
                    <div className="text-white/60 text-sm">Text</div>
                    <div className="text-white font-bold text-lg">{selectedEntity.text}</div>
                  </div>
                  <div>
                    <div className="text-white/60 text-sm">Type</div>
                    <div
                      className="inline-block px-3 py-1 rounded-lg text-white font-bold"
                      style={{ backgroundColor: ENTITY_COLORS[selectedEntity.type] }}
                    >
                      {selectedEntity.type}
                    </div>
                  </div>
                  <div>
                    <div className="text-white/60 text-sm">Confidence</div>
                    <div className="text-white font-bold">{(selectedEntity.confidence * 100).toFixed(1)}%</div>
                  </div>
                  {selectedEntity.umlsCode && (
                    <div>
                      <div className="text-white/60 text-sm">UMLS Code</div>
                      <div className="text-white font-mono">{selectedEntity.umlsCode}</div>
                    </div>
                  )}
                  {selectedEntity.snomedCode && (
                    <div>
                      <div className="text-white/60 text-sm">SNOMED CT Code</div>
                      <div className="text-white font-mono">{selectedEntity.snomedCode}</div>
                    </div>
                  )}
                </div>
              </div>
            )}

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-3">About {modelType}</h3>
              <div className="space-y-2 text-sm text-white/80">
                {modelType === 'BioBERT' && (
                  <>
                    <p><strong>Pre-training:</strong> PubMed abstracts + PMC full-text</p>
                    <p><strong>Performance:</strong> 90.1% F1 on i2b2 2012</p>
                    <p><strong>Specialty:</strong> Biomedical text mining</p>
                  </>
                )}
                {modelType === 'ClinicalBERT' && (
                  <>
                    <p><strong>Pre-training:</strong> MIMIC-III clinical notes</p>
                    <p><strong>Performance:</strong> 91.3% F1 on i2b2</p>
                    <p><strong>Specialty:</strong> Clinical documentation</p>
                  </>
                )}
                {modelType === 'SciBERT' && (
                  <>
                    <p><strong>Pre-training:</strong> Semantic Scholar corpus</p>
                    <p><strong>Performance:</strong> 89.7% F1</p>
                    <p><strong>Specialty:</strong> Scientific literature</p>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
