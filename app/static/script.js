document.addEventListener('DOMContentLoaded', function() {
    const graph = document.querySelector('.nodeGraph');
    const edgeList = document.querySelector('.edgeList');
    let edges = new Set();
    const reset = document.querySelector('.resetButton');
    const predict = document.querySelector('.predictButton');

    function createGraph() {
        graph.innerHTML = '';

        // Create nodes with data attributes for row/col
        for (let i = 0; i < 11; i++) {
            for (let j = 0; j < 11; j++) {
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

    function sendPredictionData() {
        const data = {
            edges: Array.from(edges),
            rho: document.querySelector('.rhoValue').value,
            compliance_max: document.querySelector('.complianceMax').value,
            compliance_min: document.querySelector('.complianceMin').value
        }
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        })
        .then(response => response.json())
        .then(data => {
            // Display the prediction results
            let resultsDiv = document.querySelector('.results');
            resultsDiv.innerHTML = `
                <table border="1" style="border-collapse: collapse; margin-top: 1em;">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Average Young's Modulus</th>
                            <th>Average Shear Modulus</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>GCN</td>
                            <td>${data.gcn_youngs_prediction}</td>
                            <td>${data.gcn_shear_prediction}</td>
                        </tr>
                        <tr>
                            <td>GAT</td>
                            <td>${data.gat_youngs_prediction}</td>
                            <td>${data.gat_shear_prediction}</td>
                        </tr>
                    </tbody>
                </table>
            `;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }

    reset.addEventListener('click', resetGraph);
    predict.addEventListener('click', sendPredictionData);

    createGraph();
});