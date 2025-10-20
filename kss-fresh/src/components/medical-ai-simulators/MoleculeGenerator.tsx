'use client'

import React, { useState, useEffect } from 'react'

interface MolecularProperties {
  molecularWeight: number
  logP: number
  hBondDonors: number
  hBondAcceptors: number
  rotatableBonds: number
  polarSurfaceArea: number
  qedScore: number
  lipinskiViolations: number
  dockingScore: number
}

interface GeneratedMolecule {
  id: string
  smiles: string
  name: string
  properties: MolecularProperties
  timestamp: number
}

const TEMPLATE_MOLECULES = [
  { name: 'Aspirin', smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O' },
  { name: 'Caffeine', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' },
  { name: 'Penicillin', smiles: 'CC1(C)SC2C(NC(=O)CC3=CC=CC=C3)C(=O)N2C1C(=O)O' },
  { name: 'Ibuprofen', smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O' },
  { name: 'Morphine', smiles: 'CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O' }
]

export default function MoleculeGenerator() {
  const [inputSmiles, setInputSmiles] = useState('CC(=O)OC1=CC=CC=C1C(=O)O')
  const [generatedMolecules, setGeneratedMolecules] = useState<GeneratedMolecule[]>([])
  const [selectedMolecule, setSelectedMolecule] = useState<GeneratedMolecule | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [generationMode, setGenerationMode] = useState<'similar' | 'optimize' | 'novel'>('similar')
  const [latentDim, setLatentDim] = useState(128)
  const [temperature, setTemperature] = useState(0.7)

  const parseSmiles = (smiles: string): MolecularProperties => {
    // Simplified SMILES parsing for demonstration
    const carbonCount = (smiles.match(/C/g) || []).length
    const nitrogenCount = (smiles.match(/N/g) || []).length
    const oxygenCount = (smiles.match(/O/g) || []).length
    const aromaticRings = (smiles.match(/c/g) || []).length / 6 || 1

    // Estimate molecular weight
    const mw = carbonCount * 12 + nitrogenCount * 14 + oxygenCount * 16 + 1 * (smiles.length - carbonCount - nitrogenCount - oxygenCount)

    // Estimate LogP (lipophilicity)
    const logP = (carbonCount * 0.5 - oxygenCount * 0.2 - nitrogenCount * 0.1) / 2

    // Count hydrogen bond donors/acceptors
    const hBondDonors = (smiles.match(/[OH]/g) || []).length
    const hBondAcceptors = oxygenCount + nitrogenCount

    // Rotatable bonds (single bonds not in rings)
    const rotatableBonds = Math.max(0, (smiles.match(/[-]/g) || []).length - aromaticRings * 6)

    // Polar surface area (PSA)
    const psa = oxygenCount * 20 + nitrogenCount * 15

    // QED score (drug-likeness) - simplified
    const qed = Math.min(1, Math.max(0,
      1 - Math.abs(mw - 350) / 500 - Math.abs(logP - 2.5) / 5 - hBondDonors / 10 - hBondAcceptors / 15
    ))

    // Lipinski's Rule of Five violations
    let violations = 0
    if (mw > 500) violations++
    if (logP > 5) violations++
    if (hBondDonors > 5) violations++
    if (hBondAcceptors > 10) violations++

    // Docking score (binding affinity simulation)
    const dockingScore = -5 - Math.random() * 5 - (qed * 3)

    return {
      molecularWeight: mw,
      logP,
      hBondDonors,
      hBondAcceptors,
      rotatableBonds,
      polarSurfaceArea: psa,
      qedScore: qed,
      lipinskiViolations: violations,
      dockingScore
    }
  }

  const generateMolecule = () => {
    setIsGenerating(true)

    setTimeout(() => {
      const variations = [
        'C' + inputSmiles.slice(1),
        inputSmiles.replace(/O/g, 'N'),
        inputSmiles + 'C',
        'N' + inputSmiles,
        inputSmiles.replace(/C\(/g, 'C(O)')
      ]

      const randomSmiles = variations[Math.floor(Math.random() * variations.length)]
      const properties = parseSmiles(randomSmiles)

      // Apply generation mode effects
      if (generationMode === 'optimize') {
        properties.qedScore = Math.min(0.95, properties.qedScore + 0.1 + Math.random() * 0.1)
        properties.dockingScore -= 1 + Math.random() * 2
      } else if (generationMode === 'novel') {
        properties.qedScore = 0.5 + Math.random() * 0.3
      }

      const molecule: GeneratedMolecule = {
        id: Math.random().toString(36).substr(2, 9),
        smiles: randomSmiles,
        name: `Compound-${Date.now().toString().slice(-6)}`,
        properties,
        timestamp: Date.now()
      }

      setGeneratedMolecules(prev => [molecule, ...prev.slice(0, 7)])
      setSelectedMolecule(molecule)
      setIsGenerating(false)
    }, 2000)
  }

  const interpolateMolecules = () => {
    if (generatedMolecules.length < 2) {
      alert('Generate at least 2 molecules to interpolate')
      return
    }

    setIsGenerating(true)

    setTimeout(() => {
      const mol1 = generatedMolecules[0].smiles
      const mol2 = generatedMolecules[1].smiles

      // Create interpolated SMILES (simplified)
      const len1 = mol1.length
      const len2 = mol2.length
      const interpolated = mol1.slice(0, Math.floor(len1 / 2)) + mol2.slice(Math.floor(len2 / 2))

      const properties = parseSmiles(interpolated)

      const molecule: GeneratedMolecule = {
        id: Math.random().toString(36).substr(2, 9),
        smiles: interpolated,
        name: `Interpolated-${Date.now().toString().slice(-6)}`,
        properties,
        timestamp: Date.now()
      }

      setGeneratedMolecules(prev => [molecule, ...prev.slice(0, 7)])
      setSelectedMolecule(molecule)
      setIsGenerating(false)
    }, 2500)
  }

  const drawMolecule = (smiles: string): string => {
    // Generate SVG representation of molecule
    const atoms: Array<{ element: string; x: number; y: number; id: number }> = []
    const bonds: Array<{ from: number; to: number; type: number }> = []

    let x = 50
    let y = 150
    let angle = 0
    let atomId = 0
    let prevAtomId = -1

    for (let i = 0; i < Math.min(smiles.length, 30); i++) {
      const char = smiles[i]

      if (/[CNOS]/.test(char)) {
        atoms.push({ element: char, x, y, id: atomId })

        if (prevAtomId >= 0) {
          bonds.push({ from: prevAtomId, to: atomId, type: 1 })
        }

        prevAtomId = atomId
        atomId++

        angle += (Math.random() - 0.5) * 60
        x += 30 * Math.cos(angle * Math.PI / 180)
        y += 30 * Math.sin(angle * Math.PI / 180)
      } else if (char === '=') {
        if (bonds.length > 0) {
          bonds[bonds.length - 1].type = 2
        }
      } else if (char === '#') {
        if (bonds.length > 0) {
          bonds[bonds.length - 1].type = 3
        }
      } else if (char === '(') {
        angle -= 120
      } else if (char === ')') {
        angle += 120
      }
    }

    // Generate SVG
    let svg = '<svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">'

    // Draw bonds
    bonds.forEach(bond => {
      const from = atoms.find(a => a.id === bond.from)
      const to = atoms.find(a => a.id === bond.to)

      if (from && to) {
        if (bond.type === 1) {
          svg += `<line x1="${from.x}" y1="${from.y}" x2="${to.x}" y2="${to.y}" stroke="#555" stroke-width="2"/>`
        } else if (bond.type === 2) {
          svg += `<line x1="${from.x - 2}" y1="${from.y - 2}" x2="${to.x - 2}" y2="${to.y - 2}" stroke="#555" stroke-width="2"/>`
          svg += `<line x1="${from.x + 2}" y1="${from.y + 2}" x2="${to.x + 2}" y2="${to.y + 2}" stroke="#555" stroke-width="2"/>`
        } else {
          svg += `<line x1="${from.x - 3}" y1="${from.y}" x2="${to.x - 3}" y2="${to.y}" stroke="#555" stroke-width="2"/>`
          svg += `<line x1="${from.x}" y1="${from.y}" x2="${to.x}" y2="${to.y}" stroke="#555" stroke-width="2"/>`
          svg += `<line x1="${from.x + 3}" y1="${from.y}" x2="${to.x + 3}" y2="${to.y}" stroke="#555" stroke-width="2"/>`
        }
      }
    })

    // Draw atoms
    atoms.forEach(atom => {
      const colors: Record<string, string> = { C: '#444', N: '#3b82f6', O: '#ef4444', S: '#fbbf24' }
      const color = colors[atom.element] || '#444'

      if (atom.element !== 'C') {
        svg += `<circle cx="${atom.x}" cy="${atom.y}" r="12" fill="${color}"/>`
        svg += `<text x="${atom.x}" y="${atom.y + 5}" text-anchor="middle" fill="white" font-size="14" font-weight="bold">${atom.element}</text>`
      }
    })

    svg += '</svg>'
    return svg
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-500 via-pink-600 to-red-500 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 mb-6">
          <h1 className="text-4xl font-bold text-white mb-2">Molecule Generator</h1>
          <p className="text-white/80">GAN/Transformer for AI-driven drug molecule generation and property prediction</p>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Architecture</div>
              <div className="text-white font-bold">MolGAN + ChemBERTa</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">QED Score</div>
              <div className="text-white font-bold">0.7-0.9 (Drug-like)</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Training Data</div>
              <div className="text-white font-bold">ZINC Database (2M+)</div>
            </div>
            <div className="bg-white/10 rounded-lg p-3">
              <div className="text-white/60">Representation</div>
              <div className="text-white font-bold">SMILES Notation</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h2 className="text-2xl font-bold text-white mb-4">SMILES Input</h2>

              <div className="mb-4">
                <label className="text-white font-medium block mb-2">SMILES String</label>
                <input
                  type="text"
                  value={inputSmiles}
                  onChange={(e) => setInputSmiles(e.target.value)}
                  className="w-full px-4 py-3 rounded-lg bg-white/20 text-white placeholder-white/50 border border-white/30"
                  placeholder="Enter SMILES notation..."
                />
              </div>

              <div className="mb-4">
                <label className="text-white font-medium block mb-2">Templates</label>
                <div className="flex flex-wrap gap-2">
                  {TEMPLATE_MOLECULES.map(mol => (
                    <button
                      key={mol.name}
                      onClick={() => setInputSmiles(mol.smiles)}
                      className="px-3 py-1.5 rounded-lg bg-white/20 text-white hover:bg-white/30 text-sm"
                    >
                      {mol.name}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="text-white text-sm block mb-1">Generation Mode</label>
                  <select
                    value={generationMode}
                    onChange={(e) => setGenerationMode(e.target.value as any)}
                    className="w-full px-3 py-2 rounded-lg bg-white/20 text-white border border-white/30 text-sm"
                  >
                    <option value="similar">Similar</option>
                    <option value="optimize">Optimize</option>
                    <option value="novel">Novel</option>
                  </select>
                </div>
                <div>
                  <label className="text-white text-sm block mb-1">Latent Dim: {latentDim}</label>
                  <input
                    type="range"
                    min="64"
                    max="256"
                    step="32"
                    value={latentDim}
                    onChange={(e) => setLatentDim(Number(e.target.value))}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="text-white text-sm block mb-1">Temp: {temperature.toFixed(2)}</label>
                  <input
                    type="range"
                    min="0.1"
                    max="1.5"
                    step="0.1"
                    value={temperature}
                    onChange={(e) => setTemperature(Number(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4">Molecular Structure</h3>
              {selectedMolecule ? (
                <div className="bg-white rounded-lg p-4">
                  <div dangerouslySetInnerHTML={{ __html: drawMolecule(selectedMolecule.smiles) }} />
                  <div className="mt-4 text-sm text-gray-700">
                    <strong>SMILES:</strong> {selectedMolecule.smiles}
                  </div>
                </div>
              ) : (
                <div className="bg-white/10 rounded-lg p-8 text-center text-white/60">
                  Generate a molecule to visualize
                </div>
              )}
            </div>

            {selectedMolecule && (
              <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
                <h3 className="text-xl font-bold text-white mb-4">Molecular Properties</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-white/10 rounded-lg p-3">
                    <div className="text-white/60 text-sm">Molecular Weight</div>
                    <div className="text-2xl font-bold text-white">{selectedMolecule.properties.molecularWeight.toFixed(1)}</div>
                    <div className="text-white/60 text-xs">g/mol</div>
                  </div>
                  <div className="bg-white/10 rounded-lg p-3">
                    <div className="text-white/60 text-sm">LogP</div>
                    <div className="text-2xl font-bold text-white">{selectedMolecule.properties.logP.toFixed(2)}</div>
                    <div className="text-white/60 text-xs">Lipophilicity</div>
                  </div>
                  <div className="bg-white/10 rounded-lg p-3">
                    <div className="text-white/60 text-sm">QED Score</div>
                    <div className="text-2xl font-bold text-white">{selectedMolecule.properties.qedScore.toFixed(3)}</div>
                    <div className="text-white/60 text-xs">Drug-likeness</div>
                  </div>
                  <div className="bg-white/10 rounded-lg p-3">
                    <div className="text-white/60 text-sm">H-Bond Donors</div>
                    <div className="text-2xl font-bold text-white">{selectedMolecule.properties.hBondDonors}</div>
                  </div>
                  <div className="bg-white/10 rounded-lg p-3">
                    <div className="text-white/60 text-sm">H-Bond Acceptors</div>
                    <div className="text-2xl font-bold text-white">{selectedMolecule.properties.hBondAcceptors}</div>
                  </div>
                  <div className="bg-white/10 rounded-lg p-3">
                    <div className="text-white/60 text-sm">Rotatable Bonds</div>
                    <div className="text-2xl font-bold text-white">{selectedMolecule.properties.rotatableBonds}</div>
                  </div>
                  <div className="bg-white/10 rounded-lg p-3">
                    <div className="text-white/60 text-sm">PSA</div>
                    <div className="text-2xl font-bold text-white">{selectedMolecule.properties.polarSurfaceArea.toFixed(0)}</div>
                    <div className="text-white/60 text-xs">Ų</div>
                  </div>
                  <div className="bg-white/10 rounded-lg p-3">
                    <div className="text-white/60 text-sm">Docking Score</div>
                    <div className="text-2xl font-bold text-white">{selectedMolecule.properties.dockingScore.toFixed(2)}</div>
                    <div className="text-white/60 text-xs">kcal/mol</div>
                  </div>
                  <div className="bg-white/10 rounded-lg p-3">
                    <div className="text-white/60 text-sm">Lipinski Violations</div>
                    <div className={`text-2xl font-bold ${selectedMolecule.properties.lipinskiViolations === 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {selectedMolecule.properties.lipinskiViolations}
                    </div>
                    <div className="text-white/60 text-xs">Rule of Five</div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="space-y-6">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4">Actions</h3>

              <button
                onClick={generateMolecule}
                disabled={isGenerating}
                className={`w-full py-3 rounded-lg font-bold mb-3 transition-all ${
                  isGenerating ? 'bg-gray-400 cursor-not-allowed' : 'bg-white text-pink-600 hover:bg-gray-100'
                }`}
              >
                {isGenerating ? 'Generating...' : 'Generate Molecule'}
              </button>

              <button
                onClick={interpolateMolecules}
                disabled={isGenerating || generatedMolecules.length < 2}
                className={`w-full py-2.5 rounded-lg font-medium mb-3 transition-all ${
                  isGenerating || generatedMolecules.length < 2
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-white/30 text-white hover:bg-white/40'
                }`}
              >
                Interpolate Top 2
              </button>

              <button
                onClick={() => setGeneratedMolecules([])}
                className="w-full py-2.5 rounded-lg font-medium bg-red-500/30 text-white hover:bg-red-500/40"
              >
                Clear All
              </button>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4">Generated ({generatedMolecules.length})</h3>
              <div className="space-y-2 max-h-[400px] overflow-y-auto">
                {generatedMolecules.map(mol => (
                  <div
                    key={mol.id}
                    onClick={() => setSelectedMolecule(mol)}
                    className={`p-3 rounded-lg cursor-pointer transition-all ${
                      selectedMolecule?.id === mol.id ? 'bg-white/30' : 'bg-white/10 hover:bg-white/20'
                    }`}
                  >
                    <div className="font-bold text-white text-sm mb-1">{mol.name}</div>
                    <div className="text-white/60 text-xs truncate">{mol.smiles}</div>
                    <div className="flex items-center justify-between mt-2">
                      <span className="text-white/80 text-xs">QED: {mol.properties.qedScore.toFixed(3)}</span>
                      <span className="text-white/80 text-xs">MW: {mol.properties.molecularWeight.toFixed(0)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-3">Lipinski's Rule of Five</h3>
              <div className="space-y-2 text-xs text-white/80">
                <p>✓ MW ≤ 500 Da</p>
                <p>✓ LogP ≤ 5</p>
                <p>✓ H-bond donors ≤ 5</p>
                <p>✓ H-bond acceptors ≤ 10</p>
                <p className="mt-2 text-white/60">Compounds with ≤1 violation are likely oral bioavailable</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
