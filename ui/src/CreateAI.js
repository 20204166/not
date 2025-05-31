import React, { useCallback } from "react";
import ReactFlow, {
  addEdge,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
} from "reactflow";
import "reactflow/dist/style.css";

const initialNodes = [
  {
    id: "1",
    type: "input",
    data: { label: "🟢 Input Node", code: "return { value: 1.0 }" },
    position: { x: 250, y: 5 },
  },
  {
    id: "2",
    data: { label: "🔵 LSTM Cell", code: "h_t = tanh(c_t + x_t)" },
    position: { x: 100, y: 100 },
  },
  {
    id: "3",
    type: "output",
    data: { label: "🔴 Output Node", code: "final_output = h_t" },
    position: { x: 400, y: 200 },
  },
];

const initialEdges = [
  { id: "e1-2", source: "1", target: "2" },
  { id: "e2-3", source: "2", target: "3" },
];

const CreateAI = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const simulateGraph = async () => {
    const graph = {
      nodes: nodes.map((node) => ({
        id: node.id,
        type: node.data.label.replace(/[^a-zA-Z0-9]/g, "_"),
        params: {},
        inputs: edges.filter((e) => e.target === node.id).map((e) => e.source),
        outputs: edges.filter((e) => e.source === node.id).map((e) => e.target),
      })),
      edges: edges.map((e) => ({ from: e.source, to: e.target })),
    };

    const response = await fetch("/ai/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: "demo-model", dataset: "demo-dataset", graph }),
    });

    const result = await response.json();
    console.log("Simulation result:", result);
  };

  return (
    <div style={{ width: "100vw", height: "100vh", backgroundColor: "#1e1e2f" }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        fitView
        style={{ background: "#0f172a", color: "#fff" }}
      >
        <MiniMap nodeColor={() => "#10b981"} maskColor="#1f2937" />
        <Controls />
        <Background color="#334155" gap={16} />
      </ReactFlow>
      <button
        onClick={simulateGraph}
        style={{
          position: "absolute",
          top: 10,
          left: 10,
          zIndex: 10,
          padding: "10px 16px",
          background: "#4f46e5",
          color: "white",
          border: "none",
          borderRadius: "8px",
          cursor: "pointer",
          boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
        }}
      >
        ▶ Simulate
      </button>
    </div>
  );
};

export default CreateAI;