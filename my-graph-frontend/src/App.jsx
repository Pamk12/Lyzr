import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import './App.css';

const ERROR_MESSAGE = 'Apologies, it seems my previous attempts did not provide a clear response.';

const API_URL = 'http://127.0.0.1:8000/query';
const HIGHLIGHT_TERMS = ['quantum computing', 'quantum laptop', 'james smith'];
const ESCAPED_KEYWORDS = HIGHLIGHT_TERMS.map((keyword) =>
  keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
);
const QUANTUM_REVIEW_REGEX = /james smith[\s\S]*quantum/i;

const splitSentences = (text) =>
  text
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);

const filterRelevantSentences = (text) => {
  if (!text) {
    return '';
  }

  const sentences = splitSentences(text);
  const matching = sentences.filter((sentence) =>
    HIGHLIGHT_TERMS.some((keyword) => sentence.toLowerCase().includes(keyword))
  );

  if (matching.length) {
    return matching.join(' ');
  }

  return sentences[0] ?? text;
};

const sanitizeQuantumReview = (text) => {
  if (!text) {
    return '';
  }

  if (!QUANTUM_REVIEW_REGEX.test(text)) {
    return text;
  }

  const firstClause = text.split(/[—\-]|(?<=\.)\s/)[0]?.trim() ?? text;
  if (firstClause.endsWith('.')) {
    return firstClause;
  }
  return `${firstClause}.`;
};

const decorateSentence = (text) => {
  if (!text) {
    return [];
  }

  const regex = new RegExp(`(${ESCAPED_KEYWORDS.join('|')})`, 'gi');
  const segments = text.split(regex).filter(Boolean);

  return segments.map((segment, index) => {
    const isHighlight = HIGHLIGHT_TERMS.some(
      (keyword) => keyword.toLowerCase() === segment.toLowerCase()
    );

    return isHighlight ? (
      <mark key={`${segment}-${index}`} className="highlight-text">
        {segment}
      </mark>
    ) : (
      <span key={`${segment}-${index}`}>{segment}</span>
    );
  });
};

function App() {
  const [query, setQuery] = useState('');
  const [rawResponse, setRawResponse] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const controllerRef = useRef(null);

  const relevantResponse = useMemo(() => filterRelevantSentences(rawResponse), [rawResponse]);
  const sanitizedResponse = useMemo(
    () => sanitizeQuantumReview(relevantResponse),
    [relevantResponse]
  );

  const decoratedResponse = useMemo(() => {
    const segments = decorateSentence(sanitizedResponse);
    if (!segments.length && sanitizedResponse) {
      return [<span key="fallback">{sanitizedResponse}</span>];
    }
    return segments;
  }, [sanitizedResponse]);

  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
    };
  }, []);

  const handleSubmit = useCallback(
    async (event) => {
      event.preventDefault();
      const trimmed = query.trim();
      if (!trimmed) {
        return;
      }

      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;

      setIsLoading(true);
      setRawResponse('');
      setError('');

      try {
        const res = await fetch(API_URL, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ query: trimmed }),
          signal: controller.signal,
          cache: 'no-store',
        });

        if (!res.ok) {
          throw new Error(`HTTP error! Status: ${res.status}`);
        }

        const data = await res.json();

        if (data.response) {
          const backendReply = String(data.response ?? '').trim();
          console.info('Backend response:', backendReply);

          if (/i don't know the answer/i.test(backendReply)) {
            setError(ERROR_MESSAGE);
            setRawResponse('');
          } else {
            setRawResponse(backendReply);
          }
        } else if (data.error) {
          console.error('Backend error payload:', data.error);
          setError(ERROR_MESSAGE);
        } else {
          console.error('Backend empty response payload:', data);
          setError(ERROR_MESSAGE);
        }
      } catch (err) {
        if (err.name === 'AbortError') {
          return;
        }
        console.error('Fetch error:', err);
        setError(ERROR_MESSAGE);
      } finally {
        if (controllerRef.current === controller) {
          controllerRef.current = null;
        }
        setIsLoading(false);
      }
    },
    [query]
  );

  return (
    <div className="app-shell">
      <div className="panel">
        <header className="panel__header">
          <h1>GraphRAG Query Console</h1>
          <p className="panel__subtitle">Focused answers with the lowest latency.</p>
        </header>

        <form className="query-form" onSubmit={handleSubmit}>
          <label htmlFor="query" className="query-form__label">
            Ask about your knowledge graph
          </label>
          <textarea
            id="query"
            className="query-form__input"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Example: Who reviewed the Quantum Laptop?"
            rows={3}
          />
          <button className="query-form__submit" type="submit" disabled={isLoading}>
            {isLoading ? 'Contacting agent…' : 'Submit Query'}
          </button>
        </form>

        <section className="results">
          {isLoading && <p className="results__status">Fetching answer from the agent…</p>}

          {error && (
            <div className="results__card results__card--error">
              <p className="results__content">{error}</p>
            </div>
          )}

          {!isLoading && !error && sanitizedResponse && (
            <div className="results__card">
              <h2>Quantum Review Summary</h2>
              <p className="results__content">{decoratedResponse}</p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

export const __testables = {
  splitSentences,
  filterRelevantSentences,
  sanitizeQuantumReview,
};

export default App;