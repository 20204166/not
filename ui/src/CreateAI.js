
import React, { useCallback, useRef, useState } from "react";
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
  ReactFlowProvider,
} from "reactflow";
import "reactflow/dist/style.css";

const nodeLibrary = [
  { label: "🟢 Input", type: "input", code: "return { value: 1 }", category: "IO" },
  { label: "🔴 Output", type: "output", code: "return final_output", category: "IO" },
  { label: "🔁 LSTM", type: "default", code: "h_t = tanh(c_t + x_t)", category: "Layers" },
  { label: "🧠 Dense", type: "default", code: "y = Wx + b", category: "Layers" },
  { label: "⚡ ReLU", type: "default", code: "return max(0, x)", category: "Activations" },
];

const Sidebar = ({ onDragStart }) => (
  <div style={{ padding: 10, width: 200, backgroundColor: "#1e1e2f", color: "white" }}>
    <h3>📚 Node Library</h3>
    {nodeLibrary.map((node, i) => (
      <div
        key={i}
        draggable
        onDragStart={(e) => onDragStart(e, node)}
        style={{ padding: "6px", margin: "4px 0", border: "1px solid #ccc", cursor: "grab" }}
      >
        {node.label}
      </div>
    ))}
  </div>
);

const FlowCanvas = () => {
  const reactFlowWrapper = useRef(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const { project } = useReactFlow();
  const lastNodeRef = useRef(null);

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const onDragStart = (event, node) => {
    event.dataTransfer.setData("application/reactflow", JSON.stringify(node));
    event.dataTransfer.effectAllowed = "move";
  };

  const onDrop = (event) => {
    event.preventDefault();
    const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
    const nodeData = JSON.parse(event.dataTransfer.getData("application/reactflow"));

    const position = project({
      x: event.clientX - reactFlowBounds.left,
      y: event.clientY - reactFlowBounds.top,
    });

    const newNode = {
      id: `${+new Date()}`,
      type: nodeData.type,
      position,
      data: { label: nodeData.label, code: nodeData.code },
    };

    setNodes((nds) => [...nds, newNode]);

    if (lastNodeRef.current) {
      setEdges((eds) =>
        addEdge({ id: `e${lastNodeRef.current}-${newNode.id}`, source: lastNodeRef.current, target: newNode.id }, eds)
      );
    }

    lastNodeRef.current = newNode.id;
  };

  const onDragOver = (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  };

  const simulateGraph = async () => {
    const graph = {
      nodes: nodes.map((node) => ({
        id: node.id,
        type: node.type,
        code: node.data.code,
        params: {},
        inputs: edges.filter((e) => e.target === node.id).map((e) => e.source),
        outputs: edges.filter((e) => e.source === node.id).map((e) => e.target),
      })),
      edges: edges.map((e) => ({ from: e.source, to: e.target })),
    };

    const res = await fetch("/ai/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: "demo", dataset: "demo", graph }),
    });

    console.log(await res.json());
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      <div style={{ backgroundColor: "#1f2937", color: "#fff", padding: "10px 20px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <strong>AI Builder</strong>
        </div>
        <div>
          <button style={{ marginRight: 8 }}>⚙️ Settings</button>
          <button style={{ marginRight: 8 }}>💾 Save</button>
          <button style={{ marginRight: 8 }}>📤 Export</button>
          <button onClick={simulateGraph}>▶ Simulate</button>
        </div>
      </div>

      <div style={{ display: "flex", flex: 1, backgroundColor: "#0f172a" }}>
        <Sidebar onDragStart={onDragStart} />
        <div
          ref={reactFlowWrapper}
          className="reactflow-wrapper"
          style={{ flexGrow: 1, height: "100%" }}
          onDrop={onDrop}
          onDragOver={onDragOver}
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            fitView
          >
            <MiniMap />
            <Controls />
            <Background />
          </ReactFlow>
        </div>
      </div>
    </div>
  );
};

const CreateAI = () => (
  <ReactFlowProvider>
    <FlowCanvas />
  </ReactFlowProvider>
);

export default CreateAI;
