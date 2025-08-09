// 온톨로지 그래프 시각화 모듈

// D3.js를 활용한 온톨로지 네트워크 그래프
class OntologyGraph {
    constructor(container, data, options = {}) {
        this.container = container;
        this.data = data;
        this.options = {
            width: options.width || 800,
            height: options.height || 600,
            nodeRadius: options.nodeRadius || 20,
            linkDistance: options.linkDistance || 150,
            chargeStrength: options.chargeStrength || -300,
            ...options
        };
        
        this.init();
    }
    
    init() {
        // SVG 생성
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', this.options.width)
            .attr('height', this.options.height)
            .attr('viewBox', `0 0 ${this.options.width} ${this.options.height}`);
        
        // 화살표 마커 정의
        this.svg.append('defs').selectAll('marker')
            .data(['end'])
            .enter().append('marker')
            .attr('id', 'arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 25)
            .attr('refY', 0)
            .attr('markerWidth', 8)
            .attr('markerHeight', 8)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#666');
        
        // 그룹 생성
        this.g = this.svg.append('g');
        
        // 줌 기능
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                this.g.attr('transform', event.transform);
            });
        
        this.svg.call(zoom);
        
        this.render();
    }
    
    render() {
        // 시뮬레이션 설정
        this.simulation = d3.forceSimulation(this.data.nodes)
            .force('link', d3.forceLink(this.data.links)
                .id(d => d.id)
                .distance(this.options.linkDistance))
            .force('charge', d3.forceManyBody()
                .strength(this.options.chargeStrength))
            .force('center', d3.forceCenter(
                this.options.width / 2, 
                this.options.height / 2
            ))
            .force('collision', d3.forceCollide()
                .radius(this.options.nodeRadius + 5));
        
        // 링크 그리기
        this.link = this.g.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(this.data.links)
            .enter().append('line')
            .attr('stroke', '#999')
            .attr('stroke-opacity', 0.6)
            .attr('stroke-width', d => Math.sqrt(d.value || 1))
            .attr('marker-end', 'url(#arrow)');
        
        // 노드 그룹
        this.node = this.g.append('g')
            .attr('class', 'nodes')
            .selectAll('g')
            .data(this.data.nodes)
            .enter().append('g')
            .call(this.drag());
        
        // 노드 원
        this.node.append('circle')
            .attr('r', d => d.radius || this.options.nodeRadius)
            .attr('fill', d => this.getNodeColor(d.type))
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .style('cursor', 'pointer')
            .on('mouseover', (event, d) => this.handleNodeHover(event, d, true))
            .on('mouseout', (event, d) => this.handleNodeHover(event, d, false))
            .on('click', (event, d) => this.handleNodeClick(event, d));
        
        // 노드 라벨
        this.node.append('text')
            .text(d => d.label || d.id)
            .attr('x', 0)
            .attr('y', 0)
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em')
            .style('font-size', '12px')
            .style('pointer-events', 'none')
            .style('user-select', 'none');
        
        // 시뮬레이션 업데이트
        this.simulation.on('tick', () => this.ticked());
    }
    
    ticked() {
        this.link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        this.node
            .attr('transform', d => `translate(${d.x},${d.y})`);
    }
    
    drag() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    }
    
    getNodeColor(type) {
        const colorMap = {
            'class': '#6366f1',
            'instance': '#8b5cf6',
            'property': '#ec4899',
            'literal': '#10b981',
            'concept': '#f59e0b',
            'default': '#64748b'
        };
        return colorMap[type] || colorMap.default;
    }
    
    handleNodeHover(event, d, isHover) {
        const node = d3.select(event.target);
        
        if (isHover) {
            // 노드 강조
            node.transition()
                .duration(200)
                .attr('r', (d.radius || this.options.nodeRadius) * 1.2);
            
            // 연결된 노드와 링크 강조
            const connectedNodes = new Set();
            this.data.links.forEach(link => {
                if (link.source.id === d.id) connectedNodes.add(link.target.id);
                if (link.target.id === d.id) connectedNodes.add(link.source.id);
            });
            
            this.node.style('opacity', n => 
                n.id === d.id || connectedNodes.has(n.id) ? 1 : 0.3
            );
            
            this.link.style('opacity', l => 
                l.source.id === d.id || l.target.id === d.id ? 1 : 0.1
            );
            
            // 툴팁 표시
            this.showTooltip(event, d);
        } else {
            // 원래 상태로 복원
            node.transition()
                .duration(200)
                .attr('r', d.radius || this.options.nodeRadius);
            
            this.node.style('opacity', 1);
            this.link.style('opacity', 0.6);
            
            this.hideTooltip();
        }
    }
    
    handleNodeClick(event, d) {
        // 노드 클릭 이벤트
        if (this.options.onNodeClick) {
            this.options.onNodeClick(d);
        }
    }
    
    showTooltip(event, d) {
        const tooltip = d3.select('body').append('div')
            .attr('class', 'graph-tooltip')
            .style('opacity', 0);
        
        tooltip.transition()
            .duration(200)
            .style('opacity', .9);
        
        tooltip.html(`
            <strong>${d.label || d.id}</strong><br/>
            타입: ${d.type}<br/>
            ${d.description || ''}
        `)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px');
    }
    
    hideTooltip() {
        d3.selectAll('.graph-tooltip').remove();
    }
    
    // 데이터 업데이트
    updateData(newData) {
        this.data = newData;
        this.svg.selectAll('*').remove();
        this.init();
    }
    
    // 노드 추가
    addNode(node) {
        this.data.nodes.push(node);
        this.render();
    }
    
    // 링크 추가
    addLink(link) {
        this.data.links.push(link);
        this.render();
    }
    
    // 노드 검색 및 하이라이트
    searchNode(query) {
        const matches = this.data.nodes.filter(n => 
            n.label.toLowerCase().includes(query.toLowerCase())
        );
        
        this.node.style('opacity', n => 
            matches.includes(n) ? 1 : 0.3
        );
        
        return matches;
    }
    
    // 그래프 리셋
    reset() {
        this.svg.transition().duration(750).call(
            this.zoom.transform,
            d3.zoomIdentity
        );
        this.node.style('opacity', 1);
        this.link.style('opacity', 0.6);
    }
}

// 계층 구조 트리 그래프
class HierarchyTree {
    constructor(container, data, options = {}) {
        this.container = container;
        this.data = data;
        this.options = {
            width: options.width || 800,
            height: options.height || 600,
            nodeSize: options.nodeSize || [100, 200],
            duration: options.duration || 750,
            ...options
        };
        
        this.init();
    }
    
    init() {
        this.margin = {top: 20, right: 90, bottom: 30, left: 90};
        this.width = this.options.width - this.margin.left - this.margin.right;
        this.height = this.options.height - this.margin.top - this.margin.bottom;
        
        this.svg = d3.select(this.container)
            .append('svg')
            .attr('width', this.options.width)
            .attr('height', this.options.height);
        
        this.g = this.svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);
        
        this.tree = d3.tree()
            .size([this.height, this.width]);
        
        this.root = d3.hierarchy(this.data, d => d.children);
        this.root.x0 = this.height / 2;
        this.root.y0 = 0;
        
        this.update(this.root);
    }
    
    update(source) {
        const treeData = this.tree(this.root);
        const nodes = treeData.descendants();
        const links = treeData.descendants().slice(1);
        
        // 노드 위치 조정
        nodes.forEach(d => {
            d.y = d.depth * 180;
        });
        
        // 노드 업데이트
        const node = this.g.selectAll('g.node')
            .data(nodes, d => d.id || (d.id = ++this.nodeId));
        
        const nodeEnter = node.enter().append('g')
            .attr('class', 'node')
            .attr('transform', d => `translate(${source.y0},${source.x0})`)
            .on('click', (event, d) => this.click(d));
        
        nodeEnter.append('circle')
            .attr('r', 1e-6)
            .style('fill', d => d._children ? 'lightsteelblue' : '#fff');
        
        nodeEnter.append('text')
            .attr('dy', '.35em')
            .attr('x', d => d.children || d._children ? -13 : 13)
            .attr('text-anchor', d => d.children || d._children ? 'end' : 'start')
            .text(d => d.data.name);
        
        const nodeUpdate = nodeEnter.merge(node);
        
        nodeUpdate.transition()
            .duration(this.options.duration)
            .attr('transform', d => `translate(${d.y},${d.x})`);
        
        nodeUpdate.select('circle')
            .attr('r', 10)
            .style('fill', d => d._children ? 'lightsteelblue' : '#fff');
        
        const nodeExit = node.exit().transition()
            .duration(this.options.duration)
            .attr('transform', d => `translate(${source.y},${source.x})`)
            .remove();
        
        nodeExit.select('circle')
            .attr('r', 1e-6);
        
        nodeExit.select('text')
            .style('fill-opacity', 1e-6);
        
        // 링크 업데이트
        const link = this.g.selectAll('path.link')
            .data(links, d => d.id);
        
        const linkEnter = link.enter().insert('path', 'g')
            .attr('class', 'link')
            .attr('d', d => {
                const o = {x: source.x0, y: source.y0};
                return this.diagonal(o, o);
            });
        
        const linkUpdate = linkEnter.merge(link);
        
        linkUpdate.transition()
            .duration(this.options.duration)
            .attr('d', d => this.diagonal(d, d.parent));
        
        const linkExit = link.exit().transition()
            .duration(this.options.duration)
            .attr('d', d => {
                const o = {x: source.x, y: source.y};
                return this.diagonal(o, o);
            })
            .remove();
        
        // 현재 위치 저장
        nodes.forEach(d => {
            d.x0 = d.x;
            d.y0 = d.y;
        });
    }
    
    diagonal(s, d) {
        const path = `M ${s.y} ${s.x}
                C ${(s.y + d.y) / 2} ${s.x},
                  ${(s.y + d.y) / 2} ${d.x},
                  ${d.y} ${d.x}`;
        return path;
    }
    
    click(d) {
        if (d.children) {
            d._children = d.children;
            d.children = null;
        } else {
            d.children = d._children;
            d._children = null;
        }
        this.update(d);
    }
}

// 차트 유틸리티
class ChartUtils {
    static createPieChart(container, data, options = {}) {
        const width = options.width || 400;
        const height = options.height || 400;
        const radius = Math.min(width, height) / 2;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', `translate(${width / 2},${height / 2})`);
        
        const color = d3.scaleOrdinal()
            .domain(data.map(d => d.label))
            .range(d3.schemeSet2);
        
        const pie = d3.pie()
            .value(d => d.value);
        
        const arc = d3.arc()
            .innerRadius(options.innerRadius || 0)
            .outerRadius(radius);
        
        const arcs = svg.selectAll('arc')
            .data(pie(data))
            .enter()
            .append('g');
        
        arcs.append('path')
            .attr('d', arc)
            .attr('fill', d => color(d.data.label))
            .style('stroke', 'white')
            .style('stroke-width', 2)
            .on('mouseover', function(event, d) {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('transform', 'scale(1.05)');
            })
            .on('mouseout', function(event, d) {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('transform', 'scale(1)');
            });
        
        // 라벨 추가
        arcs.append('text')
            .attr('transform', d => `translate(${arc.centroid(d)})`)
            .attr('text-anchor', 'middle')
            .text(d => `${d.data.label}: ${d.data.value}`);
        
        return svg;
    }
    
    static createBarChart(container, data, options = {}) {
        const margin = {top: 20, right: 20, bottom: 70, left: 40};
        const width = (options.width || 600) - margin.left - margin.right;
        const height = (options.height || 400) - margin.top - margin.bottom;
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        const x = d3.scaleBand()
            .range([0, width])
            .padding(0.1)
            .domain(data.map(d => d.label));
        
        const y = d3.scaleLinear()
            .range([height, 0])
            .domain([0, d3.max(data, d => d.value)]);
        
        // 바 그리기
        svg.selectAll('.bar')
            .data(data)
            .enter().append('rect')
            .attr('class', 'bar')
            .attr('x', d => x(d.label))
            .attr('width', x.bandwidth())
            .attr('y', height)
            .attr('height', 0)
            .attr('fill', '#6366f1')
            .transition()
            .duration(800)
            .attr('y', d => y(d.value))
            .attr('height', d => height - y(d.value));
        
        // X축
        svg.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x))
            .selectAll('text')
            .attr('transform', 'rotate(-45)')
            .style('text-anchor', 'end');
        
        // Y축
        svg.append('g')
            .call(d3.axisLeft(y));
        
        return svg;
    }
}

// 전역으로 노출
window.OntologyGraphs = {
    OntologyGraph,
    HierarchyTree,
    ChartUtils,
    
    // 헬퍼 메서드들
    createNetworkGraph: function(container, data, options = {}) {
        return new OntologyGraph(container, data, options);
    },
    
    createInteractiveGraph: function(container, data, options = {}) {
        return new OntologyGraph(container, data, {
            ...options,
            onNodeClick: function(d) {
                console.log('Node clicked:', d);
            }
        });
    },
    
    createHierarchyTree: function(container, data, options = {}) {
        return new HierarchyTree(container, data, options);
    },
    
    createPieChart: function(container, data, options = {}) {
        return ChartUtils.createPieChart(container, data, options);
    },
    
    createBarChart: function(container, data, options = {}) {
        return ChartUtils.createBarChart(container, data, options);
    }
};