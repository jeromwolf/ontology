# RDF Editor ↔ 3D Graph Data Transfer Implementation

## Overview
This implementation allows seamless data transfer between the RDF Editor and 3D Graph Visualization using localStorage as the transfer mechanism.

## Implementation Details

### 1. Data Flow: RDF Editor → 3D Graph

#### Modified Files:
- `/src/components/rdf-editor/TripleVisualization.tsx`
  - Added `handleView3D` function that saves triples to localStorage before navigation
  - Data is saved with timestamp and source identifier

#### Data Structure:
```javascript
{
  triples: [
    {
      subject: string,
      predicate: string,
      object: string,
      type: 'resource' | 'literal'
    }
  ],
  timestamp: ISO 8601 string,
  source: 'rdf-editor'
}
```

#### localStorage Key: `rdf-editor-triples`

### 2. Data Flow: 3D Graph → RDF Editor

#### Modified Files:
- `/src/app/3d-graph/page.tsx`
  - Added state tracking for data source and changes
  - Implemented `handleSaveToRDFEditor` function
  - Added UI buttons for saving changes back to RDF Editor

- `/src/components/rdf-editor/RDFTripleEditor.tsx`
  - Added `useEffect` to check for data from 3D Graph on mount
  - Shows notification when data is imported
  - Asks user whether to replace or merge triples

#### Data Structure:
```javascript
{
  triples: [
    {
      subject: string,
      predicate: string,
      object: string,
      type: 'resource' | 'literal'
    }
  ],
  timestamp: ISO 8601 string,
  source: '3d-graph'
}
```

#### localStorage Key: `3d-graph-triples`

## Features Implemented

### 1. Automatic Data Loading
- 3D Graph automatically loads triples from RDF Editor when opened via "3D 보기" button
- RDF Editor automatically detects and loads changes from 3D Graph
- Data is cleared after loading (one-time use)
- Time validation ensures only recent data (< 5 minutes) is loaded

### 2. Visual Feedback
- 3D Graph shows green notification when triples are loaded from RDF Editor
- RDF Editor shows notification when importing from 3D Graph
- Button states change based on whether there are unsaved changes

### 3. User Controls
- "RDF 에디터로 돌아가기" - Returns without saving changes
- "변경사항 저장하고 돌아가기" - Saves changes and returns to RDF Editor
- Merge/Replace dialog when importing to RDF Editor

## Usage Instructions

### For Users:
1. Create or edit triples in RDF Editor
2. Click "3D 보기" button to visualize in 3D
3. Make changes in 3D Graph if needed
4. Click "변경사항 저장하고 돌아가기" to save changes back to RDF Editor

### For Developers:
- Data format is consistent between both components
- Triple interface: `{ subject, predicate, object, type? }`
- localStorage is used for temporary data transfer
- Data expires after 5 minutes to prevent stale data issues

## Testing
Use `test-rdf-3d-transfer.html` to test the data transfer mechanism:
- Simulate data transfer in both directions
- Check localStorage contents
- Verify data format compatibility

## Future Enhancements
1. Add support for bulk operations
2. Implement conflict resolution for concurrent edits
3. Add export/import history tracking
4. Support for metadata preservation (e.g., triple IDs, timestamps)