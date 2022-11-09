let cy;
let needsLayout = false;

eel.expose(cyStartBatch)
function cyStartBatch() {
    cy.startBatch();
}

eel.expose(cyEndBatch)
function cyEndBatch() {
    cy.endBatch();
}

eel.expose(cyLayout)
function cyLayout() {
    if (needsLayout) {
        needsLayout = false;
        cy.layout({ name: 'cose', animate: false }).run();
    }
}

eel.expose(setModules);
function setModules(modules) {
    setNodes(modules, 'module');
}

eel.expose(setData);
function setData(data) {
    setNodes(data, 'data');
}

eel.expose(setGoals);
function setGoals(goals) {
    setNodes(goals, 'goal');
}

eel.expose(setProtectedGoals);
function setProtectedGoals(protectedGoals) {
    setNodes(protectedGoals, 'protected-goal');
}

eel.expose(setConditionLinks);
function setConditionLinks(links) {
    setLinks(links, 'condition');
}

eel.expose(setAddLinks);
function setAddLinks(links) {
    setLinks(links, 'add');
}

eel.expose(setDeleteLinks);
function setDeleteLinks(links) {
    setLinks(links, 'delete');
}

function setNodes(nodes, type) {
    const nodeIds = nodes.map(node => node.id)
    for (const element of cy.nodes(`[type = "${type}"]`)) {
        if (!nodeIds.includes(element.data('id'))) {
            cy.remove(element);
        }
    }

    for (const { id, active, activationLevel } of nodes) {
        const element = cy.getElementById(id);
        const h = 120 - (120 * Math.tanh(activationLevel / 100));
        const activationLevelColor = `hsl(${h}, 100%, 50%)`;
        if (element.isNode()) {
            element.data({ active, activationLevelColor });
        } else {
            needsLayout = true;
            cy.add({
                group: 'nodes',
                data : { id, type, active, activationLevelColor },
            });
        }
    }
}

function setLinks(links, type) {
    function toLinkId([source, target]) {
        return `${source}-${target}`
    }
    const linkIds = links.map(toLinkId);
    for (const element of cy.edges(`[type = "${type}"]`)) {
        if (!linkIds.includes(element.data('id'))) {
            cy.remove(element);
        }
    }

    for (const [ source, target ] of links) {
        const id = toLinkId([ source, target ]);
        const element = cy.getElementById(id);
        if (!element.isEdge()) {
            needsLayout = true;
            cy.add({
                group: 'edges',
                data: { id, source, target, type },
            });
        }
    }
}

window.onload = function() {
    cy = cytoscape({
        container: document.getElementById('cy'),
        elements: [],
        style: [
            {
                selector: 'node',
                style: {
                    'label': 'data(id)',
                    'border-width': 0.5,
                }
            },
            {
                selector: 'node[type = "module"]',
                style: {
                    'background-color': 'data(activationLevelColor)',
                    'shape': 'ellipse',
                    'color': 'black',
                }
            },
            {
                selector: 'node[type != "module"][?active]',
                style: {
                    'background-color': 'red',
                    'color': 'red',
                }
            },
            {
                selector: 'node[type != "module"][!active]',
                style: {
                    'background-color': 'orange',
                    'color': 'orange',
                }
            },
            {
                selector: 'node[type = "data"]',
                style: {
                    'shape': 'round-rectangle',
                }
            },
            {
                selector: 'node[type = "goal"]',
                style: {
                    'shape': 'star',
                }
            },
            {
                selector: 'node[type = "protected-goal"]',
                style: {
                    'shape': 'diamond',
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
        ],
    });

    eel.ready()
};
