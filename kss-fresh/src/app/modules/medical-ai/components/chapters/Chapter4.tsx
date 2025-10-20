import React from 'react';
import { Microscope, Brain, Zap, Code, TrendingUp, Shield, Activity, Database } from 'lucide-react';
import References from '../References';

export default function Chapter4() {
  return (
    <div className="space-y-8">
      {/* 헤더 */}
      <div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          AI 신약 개발
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300">
          분자 설계부터 임상시험까지, AI가 바꾸는 신약 개발의 미래
        </p>
      </div>

      {/* 신약 개발 파이프라인 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Microscope className="w-7 h-7 text-purple-600" />
          AI 신약 개발 파이프라인
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          {/* 표적 발굴 */}
          <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 p-6 rounded-lg border-2 border-blue-300">
            <Database className="w-12 h-12 text-blue-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-blue-900 dark:text-blue-300">
              1. 표적 발굴 (Target Discovery)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              질병 관련 단백질, 유전자, 경로 식별
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">AI 기법:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• Graph Neural Network (단백질 상호작용)</li>
                <li>• NLP (의학 논문 텍스트 마이닝)</li>
                <li>• Knowledge Graph (질병-유전자 연관)</li>
                <li>• Deep Learning (오믹스 데이터 분석)</li>
              </ul>
            </div>
            <div className="bg-blue-900/10 dark:bg-blue-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-blue-900 dark:text-blue-300 mb-1">성공 사례:</p>
              <p className="text-gray-700 dark:text-gray-300">
                BenevolentAI: ALS 치료 표적 발굴 (2년 → 6개월)
              </p>
            </div>
          </div>

          {/* 분자 설계 */}
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 p-6 rounded-lg border-2 border-purple-300">
            <Brain className="w-12 h-12 text-purple-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-purple-900 dark:text-purple-300">
              2. 분자 설계 (Molecule Design)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              신규 화합물 구조 생성 및 최적화
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">생성 모델:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• GAN (Generative Adversarial Networks)</li>
                <li>• VAE (Variational Autoencoder)</li>
                <li>• Reinforcement Learning (보상 기반)</li>
                <li>• Diffusion Models (2024 최신)</li>
              </ul>
            </div>
            <div className="bg-purple-900/10 dark:bg-purple-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-purple-900 dark:text-purple-300 mb-1">혁신:</p>
              <p className="text-gray-700 dark:text-gray-300">
                Insilico Medicine: 특발성 폐섬유증 치료제 18개월 만에 임상 진입
              </p>
            </div>
          </div>

          {/* 단백질 구조 예측 */}
          <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 p-6 rounded-lg border-2 border-green-300">
            <Zap className="w-12 h-12 text-green-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-green-900 dark:text-green-300">
              3. 단백질 구조 예측 (Protein Folding)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              3D 구조 예측으로 약물-단백질 결합 이해
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">핵심 모델:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• AlphaFold 2/3 (DeepMind) - 원자 수준 정확도</li>
                <li>• ESMFold (Meta AI) - 60배 빠른 예측</li>
                <li>• RoseTTAFold (UW) - 복합체 구조</li>
                <li>• OmegaFold - 단일 서열 예측</li>
              </ul>
            </div>
            <div className="bg-green-900/10 dark:bg-green-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-green-900 dark:text-green-300 mb-1">획기적 성과:</p>
              <p className="text-gray-700 dark:text-gray-300">
                AlphaFold 3: 단백질-DNA-RNA 복합체 예측 정확도 90%+
              </p>
            </div>
          </div>

          {/* 임상시험 설계 */}
          <div className="bg-gradient-to-br from-pink-50 to-pink-100 dark:from-pink-900/20 dark:to-pink-800/20 p-6 rounded-lg border-2 border-pink-300">
            <Activity className="w-12 h-12 text-pink-600 mb-4" />
            <h3 className="text-xl font-bold mb-3 text-pink-900 dark:text-pink-300">
              4. 임상시험 최적화 (Clinical Trials)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
              환자 선정, 프로토콜 설계, 성공률 예측
            </p>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg mb-4">
              <p className="text-sm font-semibold mb-2">AI 활용:</p>
              <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
                <li>• 환자 모집 자동화 (EHR 기반)</li>
                <li>• Adverse Event 조기 탐지</li>
                <li>• 바이오마커 식별 (반응 예측)</li>
                <li>• Digital Twin (가상 임상시험)</li>
              </ul>
            </div>
            <div className="bg-pink-900/10 dark:bg-pink-900/30 p-3 rounded text-xs">
              <p className="font-semibold text-pink-900 dark:text-pink-300 mb-1">효율화:</p>
              <p className="text-gray-700 dark:text-gray-300">
                Deep 6 AI: 환자 모집 시간 50% 단축, 비용 30% 절감
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 실전 코드 - 분자 생성 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <Code className="w-7 h-7 text-indigo-600" />
          실전 코드: VAE 기반 신약 후보 물질 생성
        </h2>

        <div className="space-y-6">
          {/* SMILES 기반 VAE */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-blue-900 dark:text-blue-300">
              1. SMILES VAE - 분자 생성 모델 (RDKit + PyTorch)
            </h3>
            <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-gray-100">
                <code>{`import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import numpy as np

# SMILES를 원-핫 인코딩
class SMILESTokenizer:
    def __init__(self):
        self.vocab = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
                      '(', ')', '=', '#', '@', '[', ']', '+', '-', '1', '2', '3', '4', '5', '6']
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

    def encode(self, smiles, max_len=120):
        encoded = np.zeros((max_len, len(self.vocab)))
        for i, char in enumerate(smiles[:max_len]):
            if char in self.char_to_idx:
                encoded[i, self.char_to_idx[char]] = 1
        return encoded

    def decode(self, one_hot):
        smiles = ''
        for probs in one_hot:
            idx = np.argmax(probs)
            smiles += self.idx_to_char.get(idx, '')
        return smiles

# VAE 모델 정의
class MoleculeVAE(nn.Module):
    def __init__(self, vocab_size=24, max_len=120, latent_dim=256):
        super(MoleculeVAE, self).__init__()
        self.max_len = max_len

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(max_len * vocab_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, max_len * vocab_size)
        )
        self.vocab_size = vocab_size

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        # Encode
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)
        x_recon = x_recon.view(batch_size, self.max_len, self.vocab_size)

        return x_recon, mu, logvar

# 손실 함수
def vae_loss(x_recon, x, mu, logvar):
    # Reconstruction loss (Cross-Entropy)
    recon_loss = nn.functional.cross_entropy(
        x_recon.view(-1, x_recon.size(-1)),
        x.argmax(dim=-1).view(-1),
        reduction='sum'
    )

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss

# 신약 후보 물질 생성
def generate_novel_molecules(model, tokenizer, n_molecules=10):
    model.eval()
    latent_dim = 256
    molecules = []

    with torch.no_grad():
        for _ in range(n_molecules):
            # 랜덤 잠재 벡터 샘플링
            z = torch.randn(1, latent_dim)
            x_gen = model.decoder(z)
            x_gen = torch.softmax(x_gen, dim=-1).numpy()[0]

            smiles = tokenizer.decode(x_gen)

            # RDKit로 유효성 검증
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # 약물 유사성 평가 (Lipinski's Rule of Five)
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)

                if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
                    molecules.append({
                        'SMILES': smiles,
                        'MolWeight': round(mw, 2),
                        'LogP': round(logp, 2),
                        'HBD': hbd,
                        'HBA': hba,
                        'Lipinski': 'PASS'
                    })

    return molecules

# 사용 예시
tokenizer = SMILESTokenizer()
model = MoleculeVAE()

# 학습된 모델 로드
checkpoint = torch.load('molecule_vae_epoch50.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# 신약 후보 10개 생성
novel_drugs = generate_novel_molecules(model, tokenizer, n_molecules=50)
print(f"\\n생성된 유효 분자: {len(novel_drugs)}개\\n")

for i, drug in enumerate(novel_drugs[:5], 1):
    print(f"{i}. {drug['SMILES']}")
    print(f"   분자량: {drug['MolWeight']}, LogP: {drug['LogP']}")
    print(f"   Lipinski: {drug['Lipinski']}\\n")`}</code>
              </pre>
            </div>
          </div>

          {/* 약물-단백질 결합 예측 */}
          <div>
            <h3 className="font-bold text-lg mb-3 text-green-900 dark:text-green-300">
              2. 약물-단백질 결합 친화도 예측 (DeepDTA)
            </h3>
            <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-gray-100">
                <code>{`import tensorflow as tf
from tensorflow import keras

# 약물-단백질 결합 예측 CNN 모델
def build_deepdta_model():
    # 약물 입력 (SMILES)
    drug_input = keras.Input(shape=(100, 64), name='drug')  # (seq_len, embedding_dim)
    drug_conv1 = keras.layers.Conv1D(32, 4, activation='relu')(drug_input)
    drug_conv2 = keras.layers.Conv1D(64, 6, activation='relu')(drug_conv1)
    drug_conv3 = keras.layers.Conv1D(96, 8, activation='relu')(drug_conv2)
    drug_pool = keras.layers.GlobalMaxPooling1D()(drug_conv3)

    # 단백질 입력 (아미노산 서열)
    protein_input = keras.Input(shape=(1000, 64), name='protein')
    protein_conv1 = keras.layers.Conv1D(32, 4, activation='relu')(protein_input)
    protein_conv2 = keras.layers.Conv1D(64, 8, activation='relu')(protein_conv1)
    protein_conv3 = keras.layers.Conv1D(96, 12, activation='relu')(protein_conv2)
    protein_pool = keras.layers.GlobalMaxPooling1D()(protein_conv3)

    # 결합
    concat = keras.layers.Concatenate()([drug_pool, protein_pool])
    dense1 = keras.layers.Dense(1024, activation='relu')(concat)
    dropout1 = keras.layers.Dropout(0.3)(dense1)
    dense2 = keras.layers.Dense(512, activation='relu')(dropout1)
    dropout2 = keras.layers.Dropout(0.3)(dense2)
    output = keras.layers.Dense(1, name='affinity')(dropout2)  # pKd 값

    model = keras.Model(inputs=[drug_input, protein_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

# 약물-표적 결합 예측
def predict_binding_affinity(drug_smiles, protein_sequence, model):
    """
    drug_smiles: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'  # Ibuprofen
    protein_sequence: 'MVHLTPEEKS...'  # Target protein
    Returns: pKd (결합 친화도, 높을수록 강함)
    """
    # 전처리
    drug_encoded = encode_smiles(drug_smiles)  # (100, 64)
    protein_encoded = encode_protein(protein_sequence)  # (1000, 64)

    # 예측
    pkd = model.predict([drug_encoded, protein_encoded])[0, 0]

    # 해석
    if pkd >= 9:
        affinity = 'Very High (< 1 nM)'
    elif pkd >= 7:
        affinity = 'High (1-100 nM)'
    elif pkd >= 5:
        affinity = 'Medium (100 nM - 10 μM)'
    else:
        affinity = 'Low (> 10 μM)'

    return {
        'pKd': round(pkd, 2),
        'affinity': affinity,
        'Kd_nM': round(10 ** (9 - pkd), 2)
    }

# 사용 예시
model = build_deepdta_model()
model.load_weights('deepdta_davis_dataset.h5')

result = predict_binding_affinity(
    drug_smiles='CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # Ibuprofen
    protein_sequence='MVHLTPEEKSAVTALWGKVN...',  # COX-2
    model=model
)

print(f"결합 친화도 (pKd): {result['pKd']}")
print(f"해석: {result['affinity']}")
print(f"Kd 값: {result['Kd_nM']} nM")`}</code>
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* 2024-2025 최신 동향 */}
      <section className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold mb-6 text-gray-900 dark:text-white flex items-center gap-2">
          <TrendingUp className="w-7 h-7 text-orange-600" />
          2024-2025 AI 신약 개발 혁신 동향
        </h2>

        <div className="space-y-4">
          <div className="border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-blue-900 dark:text-blue-300">
              1. AlphaFold 3 - 단백질 복합체 구조 예측
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              단백질-DNA-RNA-리간드 복합체를 원자 수준으로 예측 (2024.05)
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>정확도:</strong> 단백질-리간드 결합 예측 90%+ (실험 데이터 대비)</li>
              <li>• <strong>속도:</strong> 복합체 구조 예측 수 분 이내 (기존 수개월)</li>
              <li>• <strong>활용:</strong> 약물 설계, 항체 개발, CRISPR 타겟팅</li>
            </ul>
          </div>

          <div className="border-l-4 border-green-500 bg-green-50 dark:bg-green-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-green-900 dark:text-green-300">
              2. Diffusion Models for Molecules
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              이미지 생성 AI 기법을 분자 설계에 적용 (2024 ICML)
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>DiffSBDD:</strong> 단백질 구조 기반 약물 설계, GAN 대비 품질 30% 향상</li>
              <li>• <strong>GeoDiff:</strong> 3D 분자 구조 생성, Lipinski Rule 통과율 85%</li>
              <li>• <strong>Pocket2Drug:</strong> 단백질 결합 포켓에 맞춤형 분자 생성</li>
            </ul>
          </div>

          <div className="border-l-4 border-purple-500 bg-purple-50 dark:bg-purple-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-purple-900 dark:text-purple-300">
              3. Foundation Models for Drug Discovery
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              대규모 화학 데이터로 사전 학습된 범용 AI 모델
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>MolFormer (IBM):</strong> 11억 분자 학습, Zero-shot 약물 특성 예측</li>
              <li>• <strong>ChemBERTa-2:</strong> 7700만 분자 SMILES 학습, 독성 예측 SOTA</li>
              <li>• <strong>Uni-Mol (DP Technology):</strong> 2D+3D 통합 학습, 결합 친화도 예측 최고</li>
            </ul>
          </div>

          <div className="border-l-4 border-pink-500 bg-pink-50 dark:bg-pink-900/20 p-4 rounded-r-lg">
            <h3 className="font-bold text-lg mb-2 text-pink-900 dark:text-pink-300">
              4. AI로 첫 FDA 승인 신약 등장 (2024)
            </h3>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-3">
              Insilico Medicine의 특발성 폐섬유증 치료제 임상 2상 성공
            </p>
            <ul className="text-sm space-y-1 text-gray-600 dark:text-gray-400">
              <li>• <strong>개발 기간:</strong> 18개월 (기존 평균 5-7년)</li>
              <li>• <strong>비용:</strong> $260만 (기존 평균 $3억)</li>
              <li>• <strong>AI 역할:</strong> 표적 발굴, 분자 설계, 전임상 독성 예측 자동화</li>
            </ul>
          </div>
        </div>
      </section>

      {/* AI 신약 개발 통계 */}
      <section className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-xl p-6 shadow-lg text-white">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <Shield className="w-7 h-7" />
          AI 신약 개발 시장 & 성과 (2024)
        </h2>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">$7.7B</p>
            <p className="text-sm opacity-90">2030 AI 신약 개발 시장 규모</p>
            <p className="text-xs mt-2 opacity-75">출처: MarketsandMarkets</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">40%</p>
            <p className="text-sm opacity-90">AI로 개발 시간 단축 (10년 → 6년)</p>
            <p className="text-xs mt-2 opacity-75">출처: Deloitte 2024</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">18</p>
            <p className="text-sm opacity-90">AI 설계 신약 임상시험 진행 중</p>
            <p className="text-xs mt-2 opacity-75">출처: Nature Biotechnology</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4">
            <p className="text-4xl font-bold mb-2">90%</p>
            <p className="text-sm opacity-90">AlphaFold 3 단백질 구조 예측 정확도</p>
            <p className="text-xs mt-2 opacity-75">출처: DeepMind 2024</p>
          </div>
        </div>
      </section>

      {/* References */}
      <References
        sections={[
          {
            title: '📚 핵심 데이터베이스 & 도구',
            icon: 'docs' as const,
            color: 'border-blue-500',
            items: [
              {
                title: 'ChEMBL (EMBL-EBI)',
                url: 'https://www.ebi.ac.uk/chembl/',
                description: '230만+ 화합물, 1,900만+ 생물학적 활성 데이터'
              },
              {
                title: 'PubChem (NIH)',
                url: 'https://pubchem.ncbi.nlm.nih.gov/',
                description: '1.1억+ 화합물 구조, 물성, 생물학적 활성'
              },
              {
                title: 'Protein Data Bank (PDB)',
                url: 'https://www.rcsb.org/',
                description: '20만+ 단백질 3D 구조, X-ray/Cryo-EM 데이터'
              },
              {
                title: 'ZINC Database',
                url: 'https://zinc.docking.org/',
                description: '7억+ 구매 가능 화합물 (가상 스크리닝용)'
              },
            ]
          },
          {
            title: '🔬 최신 연구 논문 (2023-2024)',
            icon: 'research' as const,
            color: 'border-pink-500',
            items: [
              {
                title: 'AlphaFold 3 (DeepMind, Science 2024)',
                url: 'https://www.science.org/doi/10.1126/science.adl2528',
                description: '단백질-DNA-RNA-리간드 복합체 구조 예측 90%+ 정확도'
              },
              {
                title: 'Diffusion Models for Molecule Generation (ICML 2024)',
                url: 'https://arxiv.org/abs/2403.18114',
                description: 'DiffSBDD: 단백질 기반 약물 설계, GAN 대비 30% 향상'
              },
              {
                title: 'MolFormer Foundation Model (Nature Machine Intelligence 2024)',
                url: 'https://www.nature.com/articles/s42256-024-00812-5',
                description: '11억 분자 학습, Zero-shot 약물 특성 예측'
              },
              {
                title: 'AI Drug Discovery Clinical Trial (Nature Biotechnology 2024)',
                url: 'https://www.nature.com/articles/s41587-024-02156-7',
                description: 'Insilico Medicine: 18개월 만에 임상 2상 진입'
              },
            ]
          },
          {
            title: '🛠️ 실전 프레임워크 & 라이브러리',
            icon: 'tools' as const,
            color: 'border-green-500',
            items: [
              {
                title: 'RDKit',
                url: 'https://www.rdkit.org/',
                description: '화학정보학 표준 라이브러리 (SMILES, 분자 디스크립터)'
              },
              {
                title: 'DeepChem',
                url: 'https://deepchem.io/',
                description: '신약 개발 전용 딥러닝 프레임워크 (PyTorch/TF)'
              },
              {
                title: 'AlphaFold (ColabFold)',
                url: 'https://github.com/sokrypton/ColabFold',
                description: '단백질 구조 예측 무료 도구 (Google Colab)'
              },
              {
                title: 'OpenMM',
                url: 'https://openmm.org/',
                description: '분자 동역학 시뮬레이션 (약물-단백질 결합)'
              },
              {
                title: 'PyTorch Geometric',
                url: 'https://pytorch-geometric.readthedocs.io/',
                description: 'Graph Neural Network (분자 그래프 학습)'
              },
            ]
          },
          {
            title: '📖 주요 AI 신약 개발 기업',
            icon: 'docs' as const,
            color: 'border-purple-500',
            items: [
              {
                title: 'Insilico Medicine',
                url: 'https://insilico.com/',
                description: '18개월 만에 AI 신약 임상 진입, Pharma.AI 플랫폼'
              },
              {
                title: 'Recursion Pharmaceuticals',
                url: 'https://www.recursion.com/',
                description: '세포 이미지 기반 AI, 100+ 파이프라인'
              },
              {
                title: 'Exscientia',
                url: 'https://www.exscientia.ai/',
                description: '첫 AI 설계 약물 임상시험 (2020), 브리스톨마이어스와 협력'
              },
              {
                title: 'Atomwise',
                url: 'https://www.atomwise.com/',
                description: 'AtomNet: 가상 스크리닝 AI, 1조+ 분자 탐색'
              },
            ]
          },
        ]}
      />

      {/* 요약 */}
      <section className="bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800 rounded-xl p-6">
        <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          🎯 핵심 요약
        </h2>
        <ul className="space-y-2 text-gray-700 dark:text-gray-300">
          <li className="flex items-start gap-2">
            <span className="text-purple-600 font-bold">•</span>
            <span>AI 신약 개발 4단계: <strong>표적 발굴, 분자 설계, 단백질 구조 예측, 임상시험 최적화</strong></span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-600 font-bold">•</span>
            <span>핵심 기술: <strong>VAE/GAN (분자 생성), AlphaFold (구조 예측), DeepDTA (결합 예측)</strong></span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-600 font-bold">•</span>
            <span><strong>2024 트렌드</strong>: AlphaFold 3 (90% 정확도), Diffusion Models, Foundation Models</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-600 font-bold">•</span>
            <span>Insilico Medicine: 18개월 만에 임상 진입 (기존 5-7년), 비용 <strong>99% 절감</strong></span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-600 font-bold">•</span>
            <span>필수 도구: <strong>RDKit (화학), DeepChem (딥러닝), AlphaFold (구조 예측)</strong></span>
          </li>
        </ul>
      </section>
    </div>
  );
}
