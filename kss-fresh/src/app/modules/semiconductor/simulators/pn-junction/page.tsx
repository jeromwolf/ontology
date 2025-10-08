import PNJunctionSimulator from '../../components/simulators/PNJunctionSimulator'

export const metadata = {
  title: 'PN 접합 시뮬레이터 - KSS',
  description: '순방향/역방향 바이어스를 적용하여 PN 접합의 동작 원리를 학습하세요'
}

export default function PNJunctionPage() {
  return <PNJunctionSimulator />
}
