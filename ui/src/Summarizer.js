import React, { useState, useRef } from 'react';
import './App.css';

function Summarizer() {
  const [textInput, setTextInput]     = useState('');
  const [summary, setSummary]         = useState('');
  const [evaluation, setEvaluation]   = useState(null);
  const [loading, setLoading]         = useState(false);
  const [evalLoading, setEvalLoading] = useState(false);

  const [recording, setRecording]   = useState(false);
  const mediaRecorderRef             = useRef(null);
  const audioChunksRef               = useRef([]);

  const handleTextChange = e => setTextInput(e.target.value);

  const handleFileChange = e => {
    const file = e.target.files[0];
    if (file) sendAudioBlob(file);
  };

  const handleSummarize = async () => {
    if (!textInput.trim()) {
      alert('Please type some text to summarize.');
      return;
    }
    setLoading(true);
    setSummary('');
    setEvaluation(null);

    try {
      const res = await fetch('/notes/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text_input: textInput.trim() })
      });
      const data = await res.json();
      setSummary(data.summary || JSON.stringify(data));
    } catch (err) {
      console.error(err);
      alert('Text summarization failed—check the console.');
    } finally {
      setLoading(false);
    }
  };

  const handleEvaluate = async () => {
    if (!summary) {
      alert('Nothing to evaluate. Please summarize first.');
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
      alert('Evaluation failed—check the console.');
    } finally {
      setEvalLoading(false);
    }
  };

  const startRecording = async () => {
    audioChunksRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      mediaRecorderRef.current = mr;

      mr.ondataavailable = e => audioChunksRef.current.push(e.data);
      mr.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        sendAudioBlob(blob);
      };

      mr.start();
      setRecording(true);
    } catch (err) {
      console.error(err);
      alert('Could not start recording. Check microphone permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const sendAudioBlob = async blob => {
    setLoading(true);
    setSummary('');
    setEvaluation(null);

    try {
      const formData = new FormData();
      formData.append('audio_file', blob, 'recording.webm');

      const res = await fetch('/notes/process', { method: 'POST', body: formData });
      const data = await res.json();
      setSummary(data.summary || JSON.stringify(data));
    } catch (err) {
      console.error(err);
      alert('Audio summarization failed—check the console.');
    } finally {
      setLoading(false);
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
      <button onClick={handleSummarize} disabled={loading}>
        {loading ? 'Summarizing…' : 'Summarize Text'}
      </button>
      <hr />
      <div>
        <button onClick={startRecording} disabled={recording || loading}>
          {recording ? 'Recording…' : 'Start Recording'}
        </button>
        <button onClick={stopRecording} disabled={!recording}>
          Stop Recording
        </button>
      </div>
      <p>— or —</p>
      <label>
        Upload audio file:&nbsp;
        <input type="file" accept="audio/*" onChange={handleFileChange} disabled={loading} />
      </label>
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
              <li key={k}>
                <strong>{k}:</strong> {String(v)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default Summarizer;
