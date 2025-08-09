# Cognosphere Database Schema Design

## 1. PostgreSQL (Relational Data)

### Users & Authentication
```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    email_verified BOOLEAN DEFAULT FALSE,
    status VARCHAR(50) DEFAULT 'active'
);

-- User profiles
CREATE TABLE user_profiles (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    display_name VARCHAR(200),
    bio TEXT,
    avatar_url VARCHAR(500),
    learning_style VARCHAR(50),
    preferences JSONB DEFAULT '{}',
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Learning progress
CREATE TABLE learning_progress (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    module_id VARCHAR(100) NOT NULL,
    chapter_id VARCHAR(100) NOT NULL,
    completion_percentage DECIMAL(5,2) DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    time_spent_minutes INTEGER DEFAULT 0,
    quiz_scores JSONB DEFAULT '[]',
    simulation_results JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, module_id, chapter_id)
);

-- Achievements
CREATE TABLE achievements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    icon_url VARCHAR(500),
    points INTEGER DEFAULT 0,
    criteria JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User achievements
CREATE TABLE user_achievements (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    achievement_id UUID NOT NULL REFERENCES achievements(id) ON DELETE CASCADE,
    earned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, achievement_id)
);

-- Sessions
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT
);
```

## 2. Neo4j (Graph Data - Ontology & Knowledge)

### Node Types
```cypher
// Concept nodes
CREATE (c:Concept {
    id: 'concept_uuid',
    name: 'Ontology',
    definition: 'A formal representation of knowledge',
    category: 'core',
    difficulty_level: 1,
    created_at: datetime(),
    metadata: {}
})

// Module nodes
CREATE (m:Module {
    id: 'module_uuid',
    title: 'Introduction to Ontology',
    description: 'Basic concepts and principles',
    order: 1,
    estimated_hours: 2.5,
    prerequisites: []
})

// Chapter nodes
CREATE (ch:Chapter {
    id: 'chapter_uuid',
    title: 'What is Ontology?',
    content_id: 'mongodb_content_id',
    order: 1,
    type: 'theory' // theory, practice, simulation
})

// Learning Path nodes
CREATE (lp:LearningPath {
    id: 'path_uuid',
    name: 'Semantic Web Fundamentals',
    description: 'Complete path to understanding semantic web',
    total_modules: 5,
    estimated_weeks: 8
})

// Relationship types
CREATE (c1:Concept)-[:PREREQUISITE_OF {weight: 0.8}]->(c2:Concept)
CREATE (c:Concept)-[:BELONGS_TO]->(m:Module)
CREATE (m:Module)-[:PART_OF {order: 1}]->(lp:LearningPath)
CREATE (ch:Chapter)-[:COVERS]->(c:Concept)
CREATE (u:User {id: 'user_uuid'})-[:COMPLETED {score: 95, date: datetime()}]->(ch:Chapter)
CREATE (u:User)-[:CURRENTLY_LEARNING {started: datetime()}]->(m:Module)
```

## 3. MongoDB (Document Store - Content & Simulations)

### Collections

#### content
```javascript
{
  _id: ObjectId("..."),
  chapter_id: "chapter_uuid",
  type: "theory", // theory, exercise, simulation
  title: "Introduction to Ontology",
  content: {
    markdown: "# Introduction to Ontology\n...",
    html: "<h1>Introduction to Ontology</h1>...",
    components: [
      {
        type: "text",
        content: "..."
      },
      {
        type: "interactive_diagram",
        config: {
          diagram_type: "ontology_tree",
          data: {...}
        }
      },
      {
        type: "quiz",
        questions: [...]
      }
    ]
  },
  media: [
    {
      type: "image",
      url: "/assets/images/ontology-diagram.png",
      alt: "Ontology structure diagram"
    }
  ],
  metadata: {
    author: "system",
    created_at: ISODate("2024-01-20"),
    updated_at: ISODate("2024-01-20"),
    version: 1,
    tags: ["ontology", "basics", "introduction"]
  }
}
```

#### simulations
```javascript
{
  _id: ObjectId("..."),
  simulation_id: "sim_uuid",
  name: "RDF Triple Builder",
  description: "Interactive RDF triple creation and visualization",
  type: "interactive",
  category: "rdf",
  difficulty: "beginner",
  config: {
    initial_state: {
      subjects: [],
      predicates: [],
      objects: []
    },
    rules: {
      max_triples: 50,
      validation: "strict",
      allow_blank_nodes: true
    },
    ui_config: {
      layout: "graph",
      theme: "light",
      controls: ["add", "edit", "delete", "validate", "export"]
    }
  },
  instructions: {
    steps: [...],
    hints: [...],
    solution: {...}
  },
  scoring: {
    criteria: [...],
    max_points: 100
  },
  created_at: ISODate("2024-01-20"),
  updated_at: ISODate("2024-01-20")
}
```

#### user_simulation_states
```javascript
{
  _id: ObjectId("..."),
  user_id: "user_uuid",
  simulation_id: "sim_uuid",
  session_id: "session_uuid",
  state: {
    current_step: 3,
    data: {
      triples: [...],
      variables: {...}
    },
    score: 75,
    time_elapsed_seconds: 450
  },
  actions: [
    {
      timestamp: ISODate("2024-01-20T10:30:00Z"),
      action_type: "add_triple",
      data: {...}
    }
  ],
  completed: false,
  created_at: ISODate("2024-01-20"),
  updated_at: ISODate("2024-01-20")
}
```

## 4. Redis (Cache & Sessions)

### Key Patterns

```redis
# User sessions
session:{session_id} -> {user_id, expires_at, data}
user:sessions:{user_id} -> SET of session_ids

# Learning progress cache
progress:{user_id}:{module_id} -> {completion%, last_accessed}

# Concept relationships cache (from Neo4j)
concepts:prerequisites:{concept_id} -> SET of prerequisite_ids
concepts:graph:{depth} -> Serialized graph data

# Real-time collaboration
collab:room:{room_id} -> {participants, state}
collab:user:{user_id}:rooms -> SET of room_ids

# Rate limiting
rate:api:{user_id}:{endpoint} -> request count
rate:simulation:{user_id} -> simulation count

# Temporary data
temp:quiz:{session_id} -> {questions, answers, start_time}
temp:export:{job_id} -> {status, progress, result_url}
```

## Database Integration Strategy

1. **PostgreSQL**: Primary source of truth for user data, authentication, and structured progress tracking
2. **Neo4j**: Knowledge graph for concepts, relationships, and learning paths
3. **MongoDB**: Flexible content storage and simulation state management
4. **Redis**: High-performance caching and real-time features

## Data Synchronization

- Use event-driven architecture with message queues
- Implement CQRS pattern for read/write separation
- Cache frequently accessed Neo4j queries in Redis
- Use MongoDB change streams for real-time updates