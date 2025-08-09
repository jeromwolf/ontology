export interface Concept {
  id: string
  label: string
  description?: string
  type?: 'class' | 'property' | 'individual'
}

export interface Triple {
  subject: Concept
  predicate: string
  object: Concept
  id?: string
}

export interface OntologyGraph {
  concepts: Concept[]
  triples: Triple[]
  namespaces?: Record<string, string>
}

export interface SPARQLQuery {
  query: string
  type: 'SELECT' | 'CONSTRUCT' | 'ASK' | 'DESCRIBE'
}

export interface SPARQLResult {
  head: {
    vars: string[]
  }
  results: {
    bindings: Array<Record<string, { type: string; value: string }>>
  }
}