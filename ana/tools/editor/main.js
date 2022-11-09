let cy;

function getModuleNodes() {
    return cy.nodes('[id != "_dummy-node"][type = "module"]');
}

function getStatusNodes() {
    return cy.nodes('[id != "_dummy-node"][type = "status"]');
}


function toJSON() {
    const data = {};
    const modules = getModuleNodes();
    for (const module of modules) {
        const moduleConnection = {
            condition: [],
            add: [],
            delete: [],
            threshold: 100
        };
        data[module.data('name')] = moduleConnection;
        for (const edge of module.connectedEdges('[id != "_dummy-edge"]')) {
            if (edge.source() !== module || edge.target() === module) {
                continue;
            }
            moduleConnection[edge.data('type')].push(edge.target().data('name'))
        }
    }
    return JSON.stringify(data, null, ' '.repeat(4));
}

function addModuleNode(position) {
    const id = `module-${getModuleNodes().length}`;
    const name = id;
    cy.add({
        group: 'nodes',
        data: { id, name, type: 'module' },
        position,
    });
    return id;
}

function addStatusNode(position) {
    const id = `status-${getStatusNodes().length}`;
    const name = id;
    cy.add({
        group: 'nodes',
        data: { id, name, type: 'status' },
        position,
    });
    return id;
}

function addConditionLink(source, target) {
    addLink(source, target, 'condition');
}

function addAddLink(source, target) {
    addLink(source, target, 'add');
}

function addDeleteLink(source, target) {
    addLink(source, target, 'delete');
}

function toLinkId(source, target, type) {
    return `${type}-${source}-${target}`;
}

function addLink(source, target, type) {
    const id = toLinkId(source, target, type);
    cy.add({
        group: 'edges',
        data: { id, source, target, type },
    });
}

function linkExists(source, target, type) {
    const id = toLinkId(source, target, type);
    return cy.edges(`#${id}`).length > 0;
}

window.onload = function() {
    cy = cytoscape({
        container: document.getElementById('cy'),
        elements: [
        ],
        style: [
            {
                selector: 'node',
                style: {
                    'background-color': 'gray',
                    'color': 'black',
                    'label': 'data(name)',
                    'border-width': 0.5,
                    'z-index': 100,
                }
            },
            {
                selector: 'node:selected',
                style: {
                    'background-color': 'orange',
                    'color': 'orange',
                }
            },
            {
                selector: 'node[type = "module"]',
                style: {
                    'shape': 'ellipse',
                }
            },
            {
                selector: 'node[type = "status"]',
                style: {
                    'shape': 'round-rectangle',
                }
            },

            {
                selector: 'edge',
                style: {
                    'width': 3,
                    'line-color': 'gray',
                    'source-arrow-color': 'gray',
                    'target-arrow-color': 'gray',
                    'source-arrow-shape': 'none',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'z-index': 100,
                }
            },
            {
                selector: 'edge[type = "condition"]',
                style: {
                    'line-style': 'dashed',
                }
            },
            {
                selector: 'edge[type != "condition"]',
                style: {
                    'line-style': 'solid',
                }
            },
            {
                selector: 'edge[type = "add"]',
                style: {
                    'line-color': '#00f',
                    'source-arrow-color': '#00f',
                    'target-arrow-color': '#00f',
                }
            },
            {
                selector: 'edge[type = "delete"]',
                style: {
                    'line-color': '#f00',
                    'source-arrow-color': '#f00',
                    'target-arrow-color': '#f00',
                }
            },

            {
                selector: '#_dummy-node',
                style: {
                    'label': '',
                    'background-color': 'rgba(255, 255, 255, 0.9)',
                    'border-color': 'rgba(200, 200, 200, 0.9)',
                    'border-width': 1,
                    'z-index': 0,
                }
            },
            {
                selector: '#_dummy-edge',
                style: {
                    'source-arrow-shape': 'none',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                }
            },
            {
                selector: '#_dummy-edge[type = "condition"]',
                style: {
                    'line-color': 'rgba(200, 200, 200, 0.9)',
                    'source-arrow-color': 'rgba(200, 200, 200, 0.9)',
                    'target-arrow-color': 'rgba(200, 200, 200, 0.9)',
                }
            },
            {
                selector: '#_dummy-edge[type = "add"]',
                style: {
                    'line-color': 'rgba(200, 200, 255, 0.5)',
                    'source-arrow-color': 'rgba(200, 200, 255, 0.5)',
                    'target-arrow-color': 'rgba(200, 200, 255, 0.5)',
                }
            },
            {
                selector: '#_dummy-edge[type = "delete"]',
                style: {
                    'line-color': 'rgba(255, 200, 200, 0.5)',
                    'source-arrow-color': 'rgba(255, 200, 200, 0.5)',
                    'target-arrow-color': 'rgba(255, 200, 200, 0.5)',
                }
            }
        ],
        autounselectify: true,
    });

    let mode = 'module';

    function isLinkMode() {
        return ['condition', 'add', 'delete'].includes(mode);
    }

    function isNodeMode() {
        return ['module', 'status'].includes(mode);
    }

    function isRemoveMode() {
        return mode === 'remove';
    }

    let dummyNode = null;
    let dummyEdge = null;
    const select = document.getElementById('mode-select');
    function setMode() {
        mode = select.value;
        if (dummyEdge) {
            cy.remove(dummyEdge);
            dummyEdge = null;
        }
        if (dummyNode) {
            cy.remove(dummyNode);
            dummyNode = null;
        }
        if (isLinkMode()) {
            dummyNode = cy.add({
                group: 'nodes',
                data: { id: '_dummy-node', 'type': 'module' },
            });
        }
    }
    select.addEventListener('change', setMode);
    setMode();

    cy.on('click', e => {
        if (isLinkMode() && e.target.id() !== dummyNode.id()) {
            if (e.target.data('type') === 'module') {
                dummyNode.data('type', 'status');
                if (dummyEdge) {
                    cy.remove(dummyEdge);
                }
                dummyEdge = cy.add({
                    group: 'edges',
                    data: { id: '_dummy-edge', type: mode, source: e.target.id(), target: dummyNode.id() },
                });
            } else if (e.target.data('type') === 'status') {
                const sourceId = dummyEdge.data('source');
                const targetId = e.target.id();
                if (linkExists(sourceId, targetId, mode)) {
                    return;
                }
                dummyNode.data('type', 'module');
                if (dummyEdge) {
                    cy.remove(dummyEdge);
                }
                if (mode === 'condition') {
                    addConditionLink(sourceId, targetId)
                } else if (mode === 'add') {
                    addAddLink(sourceId, targetId)
                } else if (mode === 'delete') {
                    addDeleteLink(sourceId, targetId)
                }
            }
        }
        if (isNodeMode()) {
            if (mode === 'module') {
                addModuleNode(e.position);
            } else if (mode === 'status') {
                addStatusNode(e.position);
            }
        }
        if (isRemoveMode()) {
            cy.remove(e.target);
        }
    });

    cy.on('mousemove', e => {
        if (dummyNode) {
            dummyNode.position(e.position);
        }
    });

    cy.on('cxttap', e => {
        if (dummyNode && e.target.id() === dummyNode.id()) {
            return;
        }
        if (e.target === cy) {
            return;
        }
        if (!e.target.isNode()) {
            return;
        }
        const newName = prompt("change name", e.target.data('name'));
        if (newName) {
            e.target.data('name', newName);
        }
    });

    document.getElementById('download-button').addEventListener('click', () => {
        const jsonStr = toJSON();
        const blob = new Blob([jsonStr], {'type': 'application/json'});
        const anchor = document.createElement('a');
        anchor.href = URL.createObjectURL(blob);
        anchor.download = 'ana.json';
        anchor.click();
    });

    document.getElementById('upload-button').addEventListener('change', e => {
        const file = e.target.files[0];
        const reader = new FileReader();
        reader.readAsText(file, 'utf-8');
        reader.addEventListener('load', event => {
            const modules = new Map();
            const statuses = new Map();
            const networkData = JSON.parse(event.target.result);
            for (const moduleName in networkData) {
                const nodeId = addModuleNode({x: 0, y: 0})
                cy.getElementById(nodeId).data('name', moduleName);
                modules.set(moduleName, nodeId);
            }
            function setLink(moduleName, statusName, type) {
                const source = modules.get(moduleName)
                let target;
                if (statuses.has(statusName)) {
                    target = statuses.get(statusName);
                } else {
                    target = addStatusNode({x: 0, y: 0});
                    statuses.set(statusName, target);
                    cy.getElementById(target).data('name', statusName);
                }
                addLink(source, target, type);
            }
            for (const [moduleName, connection] of Object.entries(networkData)) {
                for (const statusName of connection.condition) {
                    setLink(moduleName, statusName, 'condition');
                }
                for (const statusName of connection.add) {
                    setLink(moduleName, statusName, 'add');
                }
                for (const statusName of connection.delete) {
                    setLink(moduleName, statusName, 'delete');
                }
            }
            cy.layout({ name: 'cose', animate: false }).run();
        });
    });

    const filterCheckboxElem = document.getElementById('filter-checkbox');
    const filterTextElem = document.getElementById('filter-text');

    function filterNodes() {
        getModuleNodes().map(node => node.style({ display: 'element' }));
        getStatusNodes().map(node => node.style({ display: 'element' }));
        if (filterTextElem.value.trim() === '') {
            return;
        }
        function matchingFilter(re) {
            return node => node.data('name').match(re) === null;
        }
        function visibleNeighborhoodFilter(node) {
            for (const node2 of node.neighborhood()) {
                if (node2.isNode() && node2.visible()) {
                    return true;
                }
            }
            return false;
        }
        try {
            filterTextElem.classList.remove('is-invalid');
            const filterFunc = matchingFilter(new RegExp(filterTextElem.value.trim()));
            const hiddenModules = getModuleNodes().filter(filterFunc).map(node => node.style({ display: 'none' }));
            const hiddenStatuses = getStatusNodes().filter(filterFunc).map(node => node.style({ display: 'none' }));
            if (filterCheckboxElem.checked) {
                const modules = hiddenModules.filter(visibleNeighborhoodFilter);
                const statuses = hiddenStatuses.filter(visibleNeighborhoodFilter);
                modules.map(node => node.style({ display: 'element' }));
                statuses.map(node => node.style({ display: 'element' }));
            }
        } catch (_) {
            filterTextElem.classList.add('is-invalid');
        }
    }

    filterCheckboxElem.addEventListener('change', () => {
        filterNodes();
    });

    filterTextElem.addEventListener('input', () => {
        filterNodes();
    });
};
