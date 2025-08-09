// Sample RDF Triple data for demonstration purposes
export const sampleTriples = {
  basic: {
    name: '기본 온톨로지',
    description: '사람과 조직에 대한 간단한 온톨로지',
    data: [
      { subject: '사람', predicate: 'rdf:type', object: 'owl:Class', type: 'resource' as const },
      { subject: '조직', predicate: 'rdf:type', object: 'owl:Class', type: 'resource' as const },
      { subject: '직원', predicate: 'rdfs:subClassOf', object: '사람', type: 'resource' as const },
      { subject: '회사', predicate: 'rdfs:subClassOf', object: '조직', type: 'resource' as const },
      { subject: '근무하다', predicate: 'rdf:type', object: 'owl:ObjectProperty', type: 'resource' as const },
      { subject: '근무하다', predicate: 'rdfs:domain', object: '직원', type: 'resource' as const },
      { subject: '근무하다', predicate: 'rdfs:range', object: '회사', type: 'resource' as const },
      { subject: '김철수', predicate: 'rdf:type', object: '직원', type: 'resource' as const },
      { subject: '김철수', predicate: '근무하다', object: '삼성전자', type: 'resource' as const },
      { subject: '삼성전자', predicate: 'rdf:type', object: '회사', type: 'resource' as const },
    ]
  },
  foaf: {
    name: '친구 관계 (FOAF)',
    description: '소셜 네트워크와 사람 관계를 표현하는 온톨로지',
    data: [
      { subject: '김영희', predicate: 'rdf:type', object: 'foaf:Person', type: 'resource' as const },
      { subject: '김영희', predicate: '이름', object: '김영희', type: 'literal' as const },
      { subject: '김영희', predicate: '나이', object: '28', type: 'literal' as const },
      { subject: '김영희', predicate: '친구이다', object: '이철수', type: 'resource' as const },
      { subject: '이철수', predicate: 'rdf:type', object: 'foaf:Person', type: 'resource' as const },
      { subject: '이철수', predicate: '이름', object: '이철수', type: 'literal' as const },
      { subject: '이철수', predicate: '친구이다', object: '박민수', type: 'resource' as const },
      { subject: '박민수', predicate: 'rdf:type', object: 'foaf:Person', type: 'resource' as const },
      { subject: '박민수', predicate: '이름', object: '박민수', type: 'literal' as const },
      { subject: '박민수', predicate: '홈페이지', object: 'http://minsu.blog.com', type: 'literal' as const },
    ]
  },
  academic: {
    name: '학술 도메인',
    description: '대학, 교수, 학생, 강좌 관계를 표현하는 온톨로지',
    data: [
      { subject: '대학', predicate: 'rdf:type', object: 'owl:Class', type: 'resource' as const },
      { subject: '교수', predicate: 'rdf:type', object: 'owl:Class', type: 'resource' as const },
      { subject: '학생', predicate: 'rdf:type', object: 'owl:Class', type: 'resource' as const },
      { subject: '강의', predicate: 'rdf:type', object: 'owl:Class', type: 'resource' as const },
      { subject: '가르치다', predicate: 'rdf:type', object: 'owl:ObjectProperty', type: 'resource' as const },
      { subject: '수강하다', predicate: 'rdf:type', object: 'owl:ObjectProperty', type: 'resource' as const },
      { subject: '김교수', predicate: 'rdf:type', object: '교수', type: 'resource' as const },
      { subject: '김교수', predicate: '가르치다', object: '온톨로지개론', type: 'resource' as const },
      { subject: '온톨로지개론', predicate: 'rdf:type', object: '강의', type: 'resource' as const },
      { subject: '온톨로지개론', predicate: '학점', object: '3', type: 'literal' as const },
      { subject: '이학생', predicate: 'rdf:type', object: '학생', type: 'resource' as const },
      { subject: '이학생', predicate: '수강하다', object: '온톨로지개론', type: 'resource' as const },
    ]
  },
  ecommerce: {
    name: '전자상거래',
    description: '상품, 고객, 주문 관계를 표현하는 온톨로지',
    data: [
      { subject: '상품', predicate: 'rdf:type', object: 'owl:Class', type: 'resource' as const },
      { subject: '고객', predicate: 'rdf:type', object: 'owl:Class', type: 'resource' as const },
      { subject: '주문', predicate: 'rdf:type', object: 'owl:Class', type: 'resource' as const },
      { subject: '가격', predicate: 'rdf:type', object: 'owl:DatatypeProperty', type: 'resource' as const },
      { subject: '주문자', predicate: 'rdf:type', object: 'owl:ObjectProperty', type: 'resource' as const },
      { subject: '포함하다', predicate: 'rdf:type', object: 'owl:ObjectProperty', type: 'resource' as const },
      { subject: '갤럭시북프로', predicate: 'rdf:type', object: '상품', type: 'resource' as const },
      { subject: '갤럭시북프로', predicate: '가격', object: '1200000', type: 'literal' as const },
      { subject: '갤럭시북프로', predicate: '상품명', object: '삼성 갤럭시북 프로', type: 'literal' as const },
      { subject: '홍길동고객', predicate: 'rdf:type', object: '고객', type: 'resource' as const },
      { subject: '주문번호5678', predicate: 'rdf:type', object: '주문', type: 'resource' as const },
      { subject: '주문번호5678', predicate: '주문자', object: '홍길동고객', type: 'resource' as const },
      { subject: '주문번호5678', predicate: '포함하다', object: '갤럭시북프로', type: 'resource' as const },
    ]
  },
  semantic: {
    name: '동물 분류 체계',
    description: '동물 계층 구조와 속성을 보여주는 예제',
    data: [
      { subject: '동물', predicate: 'rdf:type', object: 'owl:Class', type: 'resource' as const },
      { subject: '포유류', predicate: 'rdf:type', object: 'owl:Class', type: 'resource' as const },
      { subject: '포유류', predicate: 'rdfs:subClassOf', object: '동물', type: 'resource' as const },
      { subject: '개', predicate: 'rdf:type', object: 'owl:Class', type: 'resource' as const },
      { subject: '개', predicate: 'rdfs:subClassOf', object: '포유류', type: 'resource' as const },
      { subject: '고양이', predicate: 'rdf:type', object: 'owl:Class', type: 'resource' as const },
      { subject: '고양이', predicate: 'rdfs:subClassOf', object: '포유류', type: 'resource' as const },
      { subject: '애완동물', predicate: 'rdf:type', object: 'owl:ObjectProperty', type: 'resource' as const },
      { subject: '애완동물', predicate: 'rdfs:domain', object: '사람', type: 'resource' as const },
      { subject: '애완동물', predicate: 'rdfs:range', object: '동물', type: 'resource' as const },
      { subject: '나이', predicate: 'rdf:type', object: 'owl:DatatypeProperty', type: 'resource' as const },
      { subject: '바둑이', predicate: 'rdf:type', object: '개', type: 'resource' as const },
      { subject: '바둑이', predicate: '나이', object: '5', type: 'literal' as const },
    ]
  }
};

export type SampleTripleKey = keyof typeof sampleTriples;