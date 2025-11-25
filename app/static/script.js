document.addEventListener('DOMContentLoaded', function() {
    const graph = document.querySelector('.nodeGraph');
    const edgeList = document.querySelector('.edgeList');
    let edges = new Set();
    const reset = document.querySelector('.resetButton');

    function createGraph() {
        graph.innerHTML = '';

        // Create nodes with data attributes for row/col
        for (let i = 0; i < 12; i++) {
            for (let j = 0; j < 12; j++) {
                const node = document.createElement('div');
                node.className = 'node';
                node.dataset.row = i;
                node.dataset.col = j;
                node.addEventListener('click', handleNodeClick);
                graph.appendChild(node);
            }
        }
    }

    function handleNodeClick(e) {
        const node = e.target;
        const row = parseInt(node.dataset.row);
        const col = parseInt(node.dataset.col);

        // Find diagonal counterpart
        const diagSelector = `.node[data-row="${col}"][data-col="${row}"]`;
        const diagNode = graph.querySelector(diagSelector);

        const isDiagonal = row === col;
        const isSelected = node.classList.contains('selected');

        if (isSelected) {
            node.classList.remove('selected');
            if (diagNode && !isDiagonal) diagNode.classList.remove('selected');
            // Remove edge from set
            if (isDiagonal) {
                edges.delete(`[${row},${col}]`);
            } else {
                edges.delete(`[${row},${col}]\n[${col},${row}]`);
            }
        } else {
            node.classList.add('selected');
            if (diagNode && !isDiagonal) diagNode.classList.add('selected');
            // Add edge to set
            if (isDiagonal) {
                edges.add(`[${row},${col}]`);
            } else {
                edges.add(`[${row},${col}]\n[${col},${row}]`);
            }
        }
        updateEdgeList();
    }

    function updateEdgeList() {
        edgeList.value = Array.from(edges).join('\n');
    }

    function resetGraph() {
        edges.clear();
        updateEdgeList();
        const selectedNodes = graph.querySelectorAll('.node.selected');
        selectedNodes.forEach(node => node.classList.remove('selected'));
    }

    reset.addEventListener('click', resetGraph);

    createGraph();
});