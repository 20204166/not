import React, { useState } from 'react';
import './App.css';

function App() {
  const [textInput, setTextInput] = useState('');
  const [audioFile, setAudioFile]   = useState(null);
  const [summary, setSummary]       = useState('');
  const [evaluation, setEvaluation] = useState(null);
  const [loading, setLoading]       = useState(false);
  const [evalLoading, setEvalLoading] = useState(false);

  const handleTextChange = e => setTextInput(e.target.value);
  const handleFileChange = e => setAudioFile(e.target.files[0] || null);

  const handleSummarize = async () => {
    if (!textInput && !audioFile) {
      alert("Enter text or select an audio file.");
      return;
    }
    setLoading(true);
    setSummary('');
    setEvaluation(null);

    try {
      let res;
      if (audioFile) {
        const form = new FormData();
        form.append('audio_file', audioFile);
        res = await fetch('/notes/process', { method: 'POST', body: form });
      } else {
        res = await fetch('/notes/process', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text_input: textInput })
        });
      }
      const data = await res.json();
      setSummary(data.summary || JSON.stringify(data));
    } catch (err) {
      console.error(err);
      alert("Error summarizing—check console for details.");
    } finally {
      setLoading(false);
    }
  };

  const handleEvaluate = async () => {
    if (!summary) {
      alert("Please generate a summary first.");
      return;
    }
    setEvalLoading(true);
    try {
      const res = await fetch('/notes/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text_input: textInput, summary })
      });
      const data = await res.json();
      setEvaluation(data);
    } catch (err) {
      console.error(err);
      alert("Error evaluating—check console for details.");
    } finally {
      setEvalLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>AI Note-Taking App</h1>

      <textarea
        placeholder="Type your notes here…"
        value={textInput}
        onChange={handleTextChange}
      />

      <div>
        <label>
          Or upload audio:&nbsp;
          <input type="file" accept="audio/*" onChange={handleFileChange} />
        </label>
      </div>

      <button onClick={handleSummarize} disabled={loading}>
        {loading ? 'Working…' : 'Summarize'}
      </button>

      {summary && (
        <div className="output">
          <h2>Summary</h2>
          <p>{summary}</p>
          <button onClick={handleEvaluate} disabled={evalLoading}>
            {evalLoading ? 'Evaluating…' : 'Evaluate Summary'}
          </button>
        </div>
      )}

      {evaluation && (
        <div className="output">
          <h2>Evaluation Results</h2>
          <ul>
            {Object.entries(evaluation).map(([k, v]) => (
              <li key={k}><strong>{k}:</strong> {String(v)}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
