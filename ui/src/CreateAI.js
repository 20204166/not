import React, { useCallback, useEffect, useState } from "react";
import ReactFlow, {
  addEdge,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
} from "reactflow";
import "reactflow/dist/style.css";

const CreateAI = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [nodeLibrary, setNodeLibrary] = useState([]);

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

    const response = await fetch("/api/notes/ai/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_id: "demo-model",
        dataset: "demo-dataset",
        graph,
      }),
    });

    const result = await response.json();
    console.log("Simulation result:", result);
  };

  const addNode = (libraryNode) => {
    const id = `${nodes.length + 1}`;
    const newNode = {
      id,
      type: libraryNode.type || "default",
      data: { label: libraryNode.label, code: libraryNode.code },
      position: {
        x: Math.random() * 200 + 100,
        y: Math.random() * 200 + 100,
      },
    };
    setNodes((nds) => [...nds, newNode]);
  };

  useEffect(() => {
    // ✅ Load node library from backend
    fetch("/api/notes/nodes/library")
      .then((res) => res.json())
      .then((data) => setNodeLibrary(data))
      .catch((err) => console.error("Failed to load node library:", err));
  }, []);

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <div style={{ width: "20%", backgroundColor: "#111827", color: "#fff", padding: "1rem" }}>
        <h2>📚 Node Library</h2>
        {nodeLibrary.map((node, index) => (
          <button
            key={index}
            onClick={() => addNode(node)}
            style={{
              display: "block",
              width: "100%",
              margin: "0.5rem 0",
              background: "#1f2937",
              border: "1px solid #374151",
              padding: "0.5rem",
              color: "#fff",
              textAlign: "left",
              cursor: "pointer",
              borderRadius: "4px",
            }}
          >
            {node.label}
          </button>
        ))}
      </div>
      <div style={{ flexGrow: 1, backgroundColor: "#1e1e2f", position: "relative" }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          fitView
          style={{ background: "#0f172a", height: "100%" }}
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
            left: "22%",
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
    </div>
  );
};

export default CreateAI;
