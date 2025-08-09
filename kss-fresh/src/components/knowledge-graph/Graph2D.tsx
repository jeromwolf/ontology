'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Triple, LayoutType, Node, Edge } from './types';

interface Graph2DProps {
  triples: Triple[];
  layout: LayoutType;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  onNodeSelect: (nodeId: string | null) => void;
  onEdgeSelect: (edgeId: string | null) => void;
  nodeSize: number;
  linkDistance: number;
  showLabels: boolean;
  currentTool?: 'select' | 'move' | 'add-node' | 'add-edge' | 'delete';
  pendingNode?: { label: string; type: 'resource' | 'literal' | 'class' } | null;
  pendingEdge?: { source: string; predicate: string } | null;
  onPlaceNode?: (position: { x: number; y: number }) => void;
  onAddEdgeSelect?: (nodeId: string) => void;
}

export const Graph2D: React.FC<Graph2DProps> = ({
  triples,
  layout,
  selectedNodeId,
  selectedEdgeId,
  onNodeSelect,
  onEdgeSelect,
  nodeSize,
  linkDistance,
  showLabels,
  currentTool = 'select',
  pendingNode,
  pendingEdge,
  onPlaceNode,
  onAddEdgeSelect
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [draggedNodeId, setDraggedNodeId] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [lastPanPosition, setLastPanPosition] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const animationRef = useRef<number>();

  // Convert triples to nodes and edges
  useEffect(() => {
    console.log('Graph2D - Received triples:', triples);
    const nodeMap = new Map<string, Node>();
    const edgeList: Edge[] = [];

    // Preserve existing node positions
    const existingPositions = new Map<string, { x: number; y: number }>();
    nodes.forEach(node => {
      existingPositions.set(node.id, { ...node.position! });
    });

    triples.forEach((triple, idx) => {
      // Add subject node
      if (!nodeMap.has(triple.subject)) {
        nodeMap.set(triple.subject, {
          id: triple.subject,
          label: triple.subject,
          type: 'resource',
          position: existingPositions.get(triple.subject) || (triple as any).position || { x: 0, y: 0 },
          color: '#4ade80'
        });
      }

      // Add object node
      if (!nodeMap.has(triple.object)) {
        nodeMap.set(triple.object, {
          id: triple.object,
          label: triple.object,
          type: triple.type || 'resource',
          position: existingPositions.get(triple.object) || (triple as any).position || { x: 0, y: 0 },
          color: triple.type === 'literal' ? '#fb923c' : '#4ade80'
        });
      }

      // Add edge
      edgeList.push({
        id: `${triple.subject}-${triple.predicate}-${triple.object}`,
        source: triple.subject,
        target: triple.object,
        label: triple.predicate
      });
    });

    // Apply layout only to new nodes or if layout changed
    const nodeArray = Array.from(nodeMap.values());
    const hasNewNodes = nodeArray.some(node => !existingPositions.has(node.id));
    
    if (hasNewNodes || nodes.length === 0) {
      applyLayout(nodeArray, edgeList, layout);
    }

    console.log('Graph2D - Created nodes:', nodeArray);
    setNodes(nodeArray);
    setEdges(edgeList);
  }, [triples]); // Remove layout dependency to prevent re-layout on drag

  // Separate effect for layout changes
  useEffect(() => {
    if (nodes.length > 0) {
      const nodesCopy = nodes.map(node => ({ ...node }));
      applyLayout(nodesCopy, edges, layout);
      setNodes(nodesCopy);
    }
  }, [layout]);

  // Apply different layouts
  const applyLayout = (nodes: Node[], edges: Edge[], layoutType: LayoutType) => {
    const width = 1200;
    const height = 800;
    const centerX = width / 2;
    const centerY = height / 2;

    switch (layoutType) {
      case 'circular':
        const radius = Math.min(width, height) * 0.4;
        nodes.forEach((node, idx) => {
          const angle = (idx / nodes.length) * 2 * Math.PI;
          node.position = {
            x: centerX + radius * Math.cos(angle),
            y: centerY + radius * Math.sin(angle)
          };
        });
        break;

      case 'grid':
        const cols = Math.ceil(Math.sqrt(nodes.length));
        const cellWidth = width / (cols + 1);
        const cellHeight = height / (cols + 1);
        nodes.forEach((node, idx) => {
          const col = idx % cols;
          const row = Math.floor(idx / cols);
          node.position = {
            x: cellWidth * (col + 1),
            y: cellHeight * (row + 1)
          };
        });
        break;

      case 'hierarchical':
        // Simple hierarchical layout
        const levels = new Map<string, number>();
        const visited = new Set<string>();
        
        // Find root nodes (nodes with no incoming edges)
        const incomingEdges = new Map<string, number>();
        nodes.forEach(node => incomingEdges.set(node.id, 0));
        edges.forEach(edge => {
          incomingEdges.set(edge.target, (incomingEdges.get(edge.target) || 0) + 1);
        });
        
        const roots = nodes.filter(node => incomingEdges.get(node.id) === 0);
        
        // Assign levels
        const assignLevel = (nodeId: string, level: number) => {
          if (visited.has(nodeId)) return;
          visited.add(nodeId);
          levels.set(nodeId, level);
          
          edges.filter(e => e.source === nodeId).forEach(edge => {
            assignLevel(edge.target, level + 1);
          });
        };
        
        roots.forEach(root => assignLevel(root.id, 0));
        
        // Position nodes by level
        const levelNodes = new Map<number, string[]>();
        nodes.forEach(node => {
          const level = levels.get(node.id) || 0;
          if (!levelNodes.has(level)) levelNodes.set(level, []);
          levelNodes.get(level)!.push(node.id);
        });
        
        levelNodes.forEach((nodeIds, level) => {
          const levelHeight = height / (levelNodes.size + 1);
          const levelWidth = width / (nodeIds.length + 1);
          nodeIds.forEach((nodeId, idx) => {
            const node = nodes.find(n => n.id === nodeId);
            if (node) {
              node.position = {
                x: levelWidth * (idx + 1),
                y: levelHeight * (level + 1)
              };
            }
          });
        });
        break;

      case 'force-directed':
      default:
        // Initialize random positions with more spacing
        nodes.forEach((node, idx) => {
          const angle = (idx / nodes.length) * 2 * Math.PI;
          const radius = Math.min(width, height) * 0.3;
          node.position = {
            x: centerX + radius * Math.cos(angle) + (Math.random() - 0.5) * 50,
            y: centerY + radius * Math.sin(angle) + (Math.random() - 0.5) * 50
          };
        });

        // Improved force simulation
        for (let i = 0; i < 200; i++) {
          // Repulsion between all nodes
          nodes.forEach((node1, idx1) => {
            let fx = 0, fy = 0;
            
            nodes.forEach((node2, idx2) => {
              if (idx1 !== idx2) {
                const dx = node1.position!.x - node2.position!.x;
                const dy = node1.position!.y - node2.position!.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < 150) { // Only repel if close enough
                  const force = 3000 / (distance * distance + 1);
                  fx += (dx / (distance + 0.1)) * force;
                  fy += (dy / (distance + 0.1)) * force;
                }
              }
            });
            
            // Apply forces with damping
            node1.position!.x += fx * 0.01;
            node1.position!.y += fy * 0.01;
            
            // Keep nodes within bounds
            node1.position!.x = Math.max(50, Math.min(width - 50, node1.position!.x));
            node1.position!.y = Math.max(50, Math.min(height - 50, node1.position!.y));
          });

          // Attraction along edges
          edges.forEach(edge => {
            const source = nodes.find(n => n.id === edge.source);
            const target = nodes.find(n => n.id === edge.target);
            if (source && target) {
              const dx = target.position!.x - source.position!.x;
              const dy = target.position!.y - source.position!.y;
              const distance = Math.sqrt(dx * dx + dy * dy);
              const idealDistance = 120; // Ideal distance between connected nodes
              
              if (distance > idealDistance) {
                const force = (distance - idealDistance) * 0.001;
                source.position!.x += dx / distance * force;
                source.position!.y += dy / distance * force;
                target.position!.x -= dx / distance * force;
                target.position!.y -= dy / distance * force;
              }
            }
          });
        }
        break;
    }
  };

  // Drawing function
  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Save context and apply transformations
    ctx.save();
    ctx.translate(panOffset.x, panOffset.y);
    ctx.scale(zoom, zoom);


    // Draw edges
    edges.forEach(edge => {
      const source = nodes.find(n => n.id === edge.source);
      const target = nodes.find(n => n.id === edge.target);
      if (!source || !target) return;

      ctx.strokeStyle = selectedEdgeId === edge.id ? '#60a5fa' : '#4b5563';
      ctx.lineWidth = selectedEdgeId === edge.id ? 2 : 1;
      ctx.beginPath();
      ctx.moveTo(source.position!.x, source.position!.y);
      ctx.lineTo(target.position!.x, target.position!.y);
      ctx.stroke();

      // Draw edge label
      if (showLabels) {
        const midX = (source.position!.x + target.position!.x) / 2;
        const midY = (source.position!.y + target.position!.y) / 2;
        ctx.fillStyle = '#9ca3af';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(edge.label, midX, midY - 5);
      }
    });

    // Draw nodes
    nodes.forEach(node => {
      const isSelected = selectedNodeId === node.id;
      const isHovered = hoveredNodeId === node.id;
      const isDragged = draggedNodeId === node.id;
      const radius = 15 * nodeSize * (isSelected || isHovered ? 1.3 : 1);

      // Node circle
      ctx.fillStyle = isSelected ? '#60a5fa' : node.color || '#4ade80';
      ctx.strokeStyle = isHovered || isDragged ? '#ffffff' : 'transparent';
      ctx.lineWidth = isHovered || isDragged ? 3 : 2;
      ctx.beginPath();
      ctx.arc(node.position!.x, node.position!.y, radius, 0, 2 * Math.PI);
      ctx.fill();
      if (isHovered || isDragged) ctx.stroke();

      // Debug: Show node position
      if (isHovered || isSelected) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.font = '10px monospace';
        ctx.fillText(`(${Math.round(node.position!.x)}, ${Math.round(node.position!.y)})`, 
                     node.position!.x + radius + 5, 
                     node.position!.y - radius);
      }

      // Node label
      if (showLabels) {
        ctx.fillStyle = '#e5e7eb';
        ctx.font = 'bold 14px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        
        // Add background for better readability
        const text = node.label;
        const metrics = ctx.measureText(text);
        const textWidth = metrics.width;
        const textHeight = 20;
        
        ctx.fillStyle = 'rgba(17, 24, 39, 0.8)';
        ctx.fillRect(
          node.position!.x - textWidth / 2 - 4,
          node.position!.y + radius + 10,
          textWidth + 8,
          textHeight
        );
        
        ctx.fillStyle = '#e5e7eb';
        ctx.fillText(text, node.position!.x, node.position!.y + radius + 14);
      }
    });

    // Restore context
    ctx.restore();

    // Draw pending elements (not affected by zoom/pan)
    ctx.save();
    
    // Draw pending node preview at mouse position
    if (currentTool === 'add-node' && pendingNode) {
      ctx.strokeStyle = '#60a5fa';
      ctx.fillStyle = 'rgba(96, 165, 250, 0.2)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(mousePosition.x, mousePosition.y, 15, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      
      // Draw label
      ctx.fillStyle = '#e5e7eb';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(pendingNode.label, mousePosition.x, mousePosition.y - 20);
    }
    
    // Draw pending edge preview
    if (currentTool === 'add-edge' && pendingEdge && pendingEdge.source) {
      ctx.save();
      ctx.translate(panOffset.x, panOffset.y);
      ctx.scale(zoom, zoom);
      
      const sourceNode = nodes.find(n => n.id === pendingEdge.source);
      if (sourceNode) {
        ctx.strokeStyle = '#60a5fa';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(sourceNode.position!.x, sourceNode.position!.y);
        // Convert mouse position to graph coordinates
        const graphX = (mousePosition.x - panOffset.x) / zoom;
        const graphY = (mousePosition.y - panOffset.y) / zoom;
        ctx.lineTo(graphX, graphY);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Draw predicate label
        const midX = (sourceNode.position!.x + graphX) / 2;
        const midY = (sourceNode.position!.y + graphY) / 2;
        ctx.fillStyle = '#60a5fa';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(pendingEdge.predicate, midX, midY - 5);
      }
      
      ctx.restore();
    }
    
    ctx.restore();
  };

  // Draw function as a callback to avoid recreating it
  const drawCanvas = useCallback(() => {
    draw();
  }, [nodes, edges, selectedNodeId, selectedEdgeId, hoveredNodeId, showLabels, panOffset, draggedNodeId, zoom, currentTool, pendingNode, pendingEdge, mousePosition]);

  // Animation loop
  useEffect(() => {
    const animate = () => {
      drawCanvas();
      animationRef.current = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [drawCanvas]);

  // Handle mouse events
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    // Calculate scale factors to handle CSS scaling
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (((e.clientX - rect.left) * scaleX) - panOffset.x) / zoom;
    const y = (((e.clientY - rect.top) * scaleY) - panOffset.y) / zoom;

    // Check if clicking on a node
    const clickedNode = nodes.find(node => {
      const dx = x - node.position!.x;
      const dy = y - node.position!.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      return distance <= 20 * nodeSize;
    });

    if (clickedNode) {
      console.log('Node clicked:', clickedNode.id, 'at', clickedNode.position);
      setIsDragging(true);
      setDraggedNodeId(clickedNode.id);
      setDragOffset({
        x: x - clickedNode.position!.x,
        y: y - clickedNode.position!.y
      });
      onNodeSelect(clickedNode.id);
    } else {
      console.log('Canvas clicked at:', x, y);
      // Start panning
      setIsPanning(true);
      setLastPanPosition({
        x: e.clientX,
        y: e.clientY
      });
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    // Update mouse position for preview
    setMousePosition({
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    });
    
    const x = (((e.clientX - rect.left) * scaleX) - panOffset.x) / zoom;
    const y = (((e.clientY - rect.top) * scaleY) - panOffset.y) / zoom;

    if (isDragging && draggedNodeId) {
      // Update the position of the dragged node
      const newX = x - dragOffset.x;
      const newY = y - dragOffset.y;
      console.log('Dragging node:', draggedNodeId, 'to', newX, newY);
      
      setNodes(prevNodes => prevNodes.map(node => {
        if (node.id === draggedNodeId) {
          return {
            ...node,
            position: {
              x: newX,
              y: newY
            }
          };
        }
        return node;
      }));
    } else if (isPanning) {
      // Update pan offset
      const dx = e.clientX - lastPanPosition.x;
      const dy = e.clientY - lastPanPosition.y;
      setPanOffset(prev => ({
        x: prev.x + dx,
        y: prev.y + dy
      }));
      setLastPanPosition({
        x: e.clientX,
        y: e.clientY
      });
    } else {
      // Check if hovering over a node
      const hoveredNode = nodes.find(node => {
        const dx = x - node.position!.x;
        const dy = y - node.position!.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        return distance <= 20 * nodeSize;
      });

      setHoveredNodeId(hoveredNode?.id || null);
      
      // Update cursor based on current tool
      if (currentTool === 'add-node' || currentTool === 'add-edge') {
        canvas.style.cursor = 'crosshair';
      } else if (currentTool === 'delete') {
        canvas.style.cursor = hoveredNode ? 'not-allowed' : 'default';
      } else {
        canvas.style.cursor = hoveredNode ? 'pointer' : 'grab';
      }
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setDraggedNodeId(null);
    setIsPanning(false);
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    // Get mouse position relative to canvas
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;
    
    // Calculate zoom
    const delta = e.deltaY < 0 ? 1.1 : 0.9;
    const newZoom = Math.min(Math.max(zoom * delta, 0.1), 5); // Limit zoom between 0.1 and 5
    
    // Adjust pan offset to zoom towards mouse position
    const zoomRatio = newZoom / zoom;
    setPanOffset(prev => ({
      x: mouseX - (mouseX - prev.x) * zoomRatio,
      y: mouseY - (mouseY - prev.y) * zoomRatio
    }));
    
    setZoom(newZoom);
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Don't handle click if we were dragging or panning
    if (isDragging || isPanning) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (((e.clientX - rect.left) * scaleX) - panOffset.x) / zoom;
    const y = (((e.clientY - rect.top) * scaleY) - panOffset.y) / zoom;

    // Check if clicking on a node
    const clickedNode = nodes.find(node => {
      const dx = x - node.position!.x;
      const dy = y - node.position!.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      return distance <= 20 * nodeSize;
    });

    // Handle different tool modes
    if (currentTool === 'add-node' && pendingNode && onPlaceNode) {
      // Place new node at clicked position
      onPlaceNode({ x, y });
    } else if (currentTool === 'add-edge' && pendingEdge && onAddEdgeSelect) {
      // Select node for edge connection
      if (clickedNode) {
        onAddEdgeSelect(clickedNode.id);
      }
    } else if (currentTool === 'delete' && clickedNode) {
      // Delete mode will be handled by parent component
      onNodeSelect(clickedNode.id);
    } else {
      // Default select mode
      if (clickedNode) {
        onNodeSelect(clickedNode.id);
      } else {
        onNodeSelect(null);
        onEdgeSelect(null);
      }
    }
  };

  return (
    <canvas
      ref={canvasRef}
      width={1200}
      height={800}
      className="w-full h-full"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onClick={handleClick}
      onWheel={handleWheel}
      style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
    />
  );
};