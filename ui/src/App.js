import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Home from './Home';
import Summarizer from './Summarizer';
import CreateAI from './CreateAI';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/summarization" element={<Summarizer />} />
        <Route path="/create" element={<CreateAI />} />
      </Routes>
    </Router>
  );
}

export default App;