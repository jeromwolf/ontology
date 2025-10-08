import LithographySimulator from '../../components/simulators/LithographySimulator'

export const metadata = {
  title: '리소그래피 공정 시뮬레이터 - KSS',
  description: '반도체 회로 패턴을 형성하는 핵심 공정을 단계별로 학습하세요'
}

export default function LithographyPage() {
  return <LithographySimulator />
}
