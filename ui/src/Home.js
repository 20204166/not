import React from 'react';
import { Link } from 'react-router-dom';

function Home() {
  return (
    <div style={{ textAlign: 'center', padding: '4rem' }}>
      <h1>Welcome to AI Platform</h1>
      <p>Pick a tool to begin:</p>
      <div style={{ marginTop: '2rem' }}>
        <Link to="/summarization">
          <button style={{ padding: '1rem 2rem', marginRight: '2rem' }}>
            📝 Summarization
          </button>
        </Link>
        <Link to="/create">
          <button style={{ padding: '1rem 2rem' }}>
            🧠 Create AI
          </button>
        </Link>
      </div>
    </div>
  );
}

export default Home;