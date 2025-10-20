'use client';

import { useState, useRef, useEffect } from 'react';
import {
  Layers, Eye, Type, AudioLines, Merge, Settings,
  Play, Download, Trash2, Plus, ArrowRight, Cpu
} from 'lucide-react';

interface ArchitectureComponent {
  id: string;
  type: 'encoder' | 'fusion' | 'decoder';
  modality?: 'vision' | 'text' | 'audio';
  position: { x: number; y: number };
  config: {
    layers?: number;
    hiddenSize?: number;
    dropout?: number;
  };
}

interface Connection {
  from: string;
  to: string;
}

export default function MultimodalArchitect() {
  const [components, setComponents] = useState<ArchitectureComponent[]>([
    {
      id: 'vision-enc-1',
      type: 'encoder',
      modality: 'vision',
      position: { x: 50, y: 100 },
      config: { layers: 12, hiddenSize: 768, dropout: 0.1 }
    },
    {
      id: 'text-enc-1',
      type: 'encoder',
      modality: 'text',
      position: { x: 50, y: 250 },
      config: { layers: 12, hiddenSize: 768, dropout: 0.1 }
    },
    {
      id: 'fusion-1',
      type: 'fusion',
      position: { x: 350, y: 175 },
      config: { layers: 6, hiddenSize: 1024, dropout: 0.1 }
    }
  ]);
  const [connections, setConnections] = useState<Connection[]>([
    { from: 'vision-enc-1', to: 'fusion-1' },
    { from: 'text-enc-1', to: 'fusion-1' }
  ]);
  const [selectedComponent, setSelectedComponent] = useState<string | null>(null);
  const [fusionStrategy, setFusionStrategy] = useState<'early' | 'late' | 'hybrid'>('late');
  const [dragging, setDragging] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const componentTemplates = {
    vision: {
      icon: Eye,
      color: 'from-blue-500 to-cyan-600',
      borderColor: 'border-blue-500',
      bgColor: 'bg-blue-50 dark:bg-blue-900/30'
    },
    text: {
      icon: Type,
      color: 'from-green-500 to-emerald-600',
      borderColor: 'border-green-500',
      bgColor: 'bg-green-50 dark:bg-green-900/30'
    },
    audio: {
      icon: AudioLines,
      color: 'from-orange-500 to-red-600',
      borderColor: 'border-orange-500',
      bgColor: 'bg-orange-50 dark:bg-orange-900/30'
    },
    fusion: {
      icon: Merge,
      color: 'from-purple-500 to-pink-600',
      borderColor: 'border-purple-500',
      bgColor: 'bg-purple-50 dark:bg-purple-900/30'
    }
  };

  // Draw connections on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Clear canvas
    ctx.clearRect(0, 0, rect.width, rect.height);

    // Draw connections
    connections.forEach(conn => {
      const fromComp = components.find(c => c.id === conn.from);
      const toComp = components.find(c => c.id === conn.to);

      if (fromComp && toComp) {
        ctx.strokeStyle = '#8b5cf6';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(fromComp.position.x + 120, fromComp.position.y + 50);
        ctx.lineTo(toComp.position.x, toComp.position.y + 50);
        ctx.stroke();

        // Draw arrow head
        const angle = Math.atan2(
          toComp.position.y - fromComp.position.y,
          toComp.position.x - fromComp.position.x
        );
        const arrowX = toComp.position.x - 10;
        const arrowY = toComp.position.y + 50;
        ctx.beginPath();
        ctx.moveTo(arrowX, arrowY);
        ctx.lineTo(arrowX - 10 * Math.cos(angle - Math.PI / 6), arrowY - 10 * Math.sin(angle - Math.PI / 6));
        ctx.moveTo(arrowX, arrowY);
        ctx.lineTo(arrowX - 10 * Math.cos(angle + Math.PI / 6), arrowY - 10 * Math.sin(angle + Math.PI / 6));
        ctx.stroke();
      }
    });
  }, [components, connections]);

  const addComponent = (type: 'encoder' | 'fusion' | 'decoder', modality?: 'vision' | 'text' | 'audio') => {
    const newComponent: ArchitectureComponent = {
      id: `${type}-${Date.now()}`,
      type,
      modality,
      position: { x: 50, y: components.length * 80 + 50 },
      config: { layers: 12, hiddenSize: 768, dropout: 0.1 }
    };
    setComponents([...components, newComponent]);
  };

  const deleteComponent = (id: string) => {
    setComponents(components.filter(c => c.id !== id));
    setConnections(connections.filter(c => c.from !== id && c.to !== id));
    if (selectedComponent === id) setSelectedComponent(null);
  };

  const handleMouseDown = (id: string, e: React.MouseEvent) => {
    const component = components.find(c => c.id === id);
    if (!component) return;

    setDragging(id);
    setDragOffset({
      x: e.clientX - component.position.x,
      y: e.clientY - component.position.y
    });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!dragging) return;

    const newComponents = components.map(c => {
      if (c.id === dragging) {
        return {
          ...c,
          position: {
            x: Math.max(0, Math.min(600, e.clientX - dragOffset.x)),
            y: Math.max(0, Math.min(400, e.clientY - dragOffset.y))
          }
        };
      }
      return c;
    });
    setComponents(newComponents);
  };

  const handleMouseUp = () => {
    setDragging(null);
  };

  const generateCode = () => {
    return `import torch
import torch.nn as nn
from transformers import ViTModel, BertModel

class MultimodalArchitecture(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoders
${components.filter(c => c.type === 'encoder').map(c => {
  if (c.modality === 'vision') {
    return `        self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')`;
  } else if (c.modality === 'text') {
    return `        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')`;
  } else if (c.modality === 'audio') {
    return `        self.audio_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')`;
  }
  return '';
}).filter(Boolean).join('\n')}

        # Fusion Module (${fusionStrategy} fusion)
        self.fusion = nn.Sequential(
            nn.Linear(${components.filter(c => c.type === 'encoder').length * 768}, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512)
        )

    def forward(self, vision_input=None, text_input=None, audio_input=None):
        embeddings = []

${components.filter(c => c.type === 'encoder').map(c => {
  if (c.modality === 'vision') {
    return `        if vision_input is not None:\n            vision_emb = self.vision_encoder(vision_input).pooler_output\n            embeddings.append(vision_emb)`;
  } else if (c.modality === 'text') {
    return `        if text_input is not None:\n            text_emb = self.text_encoder(**text_input).pooler_output\n            embeddings.append(text_emb)`;
  } else if (c.modality === 'audio') {
    return `        if audio_input is not None:\n            audio_emb = self.audio_encoder(audio_input).last_hidden_state.mean(dim=1)\n            embeddings.append(audio_emb)`;
  }
  return '';
}).filter(Boolean).join('\n\n')}

        # Concatenate and fuse
        combined = torch.cat(embeddings, dim=-1)
        fused = self.fusion(combined)

        return fused

# Initialize model
model = MultimodalArchitecture()
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")`;
  };

  const selectedComp = components.find(c => c.id === selectedComponent);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-500 to-purple-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <Layers size={32} />
          <h2 className="text-2xl font-bold">멀티모달 아키텍처 빌더</h2>
        </div>
        <p className="text-violet-100">
          드래그 앤 드롭으로 멀티모달 AI 모델 아키텍처를 시각적으로 설계하세요
        </p>
      </div>

      {/* Component Palette */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Plus className="text-purple-600" />
          컴포넌트 팔레트
        </h3>
        <div className="space-y-3">
          <div>
            <h4 className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2">인코더 (Encoders)</h4>
            <div className="flex gap-2 flex-wrap">
              <button
                onClick={() => addComponent('encoder', 'vision')}
                className="flex items-center gap-2 px-4 py-2 bg-blue-50 dark:bg-blue-900/30 border-2 border-blue-500 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-900/50 transition-colors"
              >
                <Eye size={18} />
                <span>Vision Encoder</span>
              </button>
              <button
                onClick={() => addComponent('encoder', 'text')}
                className="flex items-center gap-2 px-4 py-2 bg-green-50 dark:bg-green-900/30 border-2 border-green-500 rounded-lg hover:bg-green-100 dark:hover:bg-green-900/50 transition-colors"
              >
                <Type size={18} />
                <span>Text Encoder</span>
              </button>
              <button
                onClick={() => addComponent('encoder', 'audio')}
                className="flex items-center gap-2 px-4 py-2 bg-orange-50 dark:bg-orange-900/30 border-2 border-orange-500 rounded-lg hover:bg-orange-100 dark:hover:bg-orange-900/50 transition-colors"
              >
                <AudioLines size={18} />
                <span>Audio Encoder</span>
              </button>
            </div>
          </div>
          <div>
            <h4 className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-2">융합 모듈 (Fusion)</h4>
            <div className="flex gap-2">
              <button
                onClick={() => addComponent('fusion')}
                className="flex items-center gap-2 px-4 py-2 bg-purple-50 dark:bg-purple-900/30 border-2 border-purple-500 rounded-lg hover:bg-purple-100 dark:hover:bg-purple-900/50 transition-colors"
              >
                <Merge size={18} />
                <span>Fusion Module</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Architecture Canvas */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold flex items-center gap-2">
            <Cpu className="text-purple-600" />
            아키텍처 캔버스
          </h3>
          <div className="flex gap-2">
            <select
              value={fusionStrategy}
              onChange={(e) => setFusionStrategy(e.target.value as any)}
              className="px-3 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-sm"
            >
              <option value="early">Early Fusion</option>
              <option value="late">Late Fusion</option>
              <option value="hybrid">Hybrid Fusion</option>
            </select>
            <button
              onClick={() => {
                setComponents([]);
                setConnections([]);
              }}
              className="flex items-center gap-2 px-3 py-2 bg-red-50 dark:bg-red-900/30 text-red-600 rounded-lg hover:bg-red-100 dark:hover:bg-red-900/50 transition-colors"
            >
              <Trash2 size={18} />
              초기화
            </button>
          </div>
        </div>

        <div
          className="relative bg-gray-50 dark:bg-gray-900 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-700 overflow-hidden"
          style={{ height: 500 }}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          {/* Canvas for connections */}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 pointer-events-none"
            style={{ width: '100%', height: '100%' }}
          />

          {/* Components */}
          {components.map(comp => {
            const template = componentTemplates[comp.modality || comp.type];
            const Icon = template.icon;

            return (
              <div
                key={comp.id}
                className={`absolute cursor-move ${template.bgColor} border-2 ${template.borderColor} rounded-lg p-3 shadow-lg transition-all ${
                  selectedComponent === comp.id ? 'ring-4 ring-yellow-400' : ''
                }`}
                style={{
                  left: comp.position.x,
                  top: comp.position.y,
                  width: 120
                }}
                onMouseDown={(e) => handleMouseDown(comp.id, e)}
                onClick={() => setSelectedComponent(comp.id)}
              >
                <div className="flex items-center gap-2 mb-2">
                  <Icon size={20} />
                  <span className="text-xs font-bold">
                    {comp.modality ? `${comp.modality.toUpperCase()}` : comp.type.toUpperCase()}
                  </span>
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                  <div>Layers: {comp.config.layers}</div>
                  <div>Size: {comp.config.hiddenSize}</div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteComponent(comp.id);
                  }}
                  className="absolute top-1 right-1 text-red-500 hover:text-red-700"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            );
          })}

          {components.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-400">
              위의 컴포넌트를 추가하여 아키텍처를 설계하세요
            </div>
          )}
        </div>
      </div>

      {/* Configuration Panel */}
      {selectedComp && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
            <Settings className="text-purple-600" />
            컴포넌트 설정
          </h3>
          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-semibold mb-2">레이어 수</label>
              <input
                type="number"
                value={selectedComp.config.layers}
                onChange={(e) => {
                  const newComponents = components.map(c =>
                    c.id === selectedComp.id
                      ? { ...c, config: { ...c.config, layers: parseInt(e.target.value) } }
                      : c
                  );
                  setComponents(newComponents);
                }}
                className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg"
                min={1}
                max={24}
              />
            </div>
            <div>
              <label className="block text-sm font-semibold mb-2">Hidden Size</label>
              <input
                type="number"
                value={selectedComp.config.hiddenSize}
                onChange={(e) => {
                  const newComponents = components.map(c =>
                    c.id === selectedComp.id
                      ? { ...c, config: { ...c.config, hiddenSize: parseInt(e.target.value) } }
                      : c
                  );
                  setComponents(newComponents);
                }}
                className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg"
                step={64}
                min={64}
                max={2048}
              />
            </div>
            <div>
              <label className="block text-sm font-semibold mb-2">Dropout</label>
              <input
                type="number"
                value={selectedComp.config.dropout}
                onChange={(e) => {
                  const newComponents = components.map(c =>
                    c.id === selectedComp.id
                      ? { ...c, config: { ...c.config, dropout: parseFloat(e.target.value) } }
                      : c
                  );
                  setComponents(newComponents);
                }}
                className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg"
                step={0.05}
                min={0}
                max={0.5}
              />
            </div>
          </div>
        </div>
      )}

      {/* Generated Code */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold">🔧 생성된 PyTorch 코드</h3>
          <button
            onClick={() => {
              navigator.clipboard.writeText(generateCode());
              alert('코드가 클립보드에 복사되었습니다!');
            }}
            className="flex items-center gap-2 px-3 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors"
          >
            <Download size={18} />
            복사
          </button>
        </div>
        <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto max-h-96 overflow-y-auto">
          {generateCode()}
        </pre>
      </div>

      {/* Model Stats */}
      <div className="grid md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg p-4 text-white">
          <div className="text-3xl font-bold">{components.filter(c => c.type === 'encoder').length}</div>
          <div className="text-sm text-blue-100">Encoders</div>
        </div>
        <div className="bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg p-4 text-white">
          <div className="text-3xl font-bold">{components.filter(c => c.type === 'fusion').length}</div>
          <div className="text-sm text-purple-100">Fusion Modules</div>
        </div>
        <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-lg p-4 text-white">
          <div className="text-3xl font-bold">{connections.length}</div>
          <div className="text-sm text-green-100">Connections</div>
        </div>
        <div className="bg-gradient-to-br from-orange-500 to-orange-600 rounded-lg p-4 text-white">
          <div className="text-3xl font-bold">
            {(components.reduce((sum, c) => sum + (c.config.layers || 0), 0)).toLocaleString()}
          </div>
          <div className="text-sm text-orange-100">Total Layers</div>
        </div>
      </div>
    </div>
  );
}
