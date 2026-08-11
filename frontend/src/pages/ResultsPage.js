import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import '../styles/pages/ResultsPage.css';
import api from '../api';

const RISK_COLORS = {
  low: '#10b981',
  medium: '#f59e0b',
  high: '#ef4444',
  critical: '#7f1d1d',
};

const RISK_BG = {
  low: '#ecfdf5',
  medium: '#fffbeb',
  high: '#fef2f2',
  critical: '#450a0a',
};

const RISK_ICONS = {
  low: '✅',
  medium: '⚠️',
  high: '🚨',
  critical: '⛔',
};

const STAGE_DESCRIPTIONS = {
  stage_0: { label: 'Stage 0', sub: 'Normal Liver', color: '#10b981' },
  stage_1: { label: 'Stage 1', sub: 'Mild Fibrosis', color: '#84cc16' },
  stage_2: { label: 'Stage 2', sub: 'Moderate Fibrosis', color: '#f59e0b' },
  stage_3: { label: 'Stage 3', sub: 'Severe Fibrosis', color: '#ef4444' },
  stage_4: { label: 'Stage 4', sub: 'Cirrhosis', color: '#7f1d1d' },
};

const ResultsPage = ({ prediction: initialPrediction }) => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [prediction, setPrediction] = useState(initialPrediction);
  const [loading, setLoading] = useState(!initialPrediction);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!initialPrediction && id) {
      fetchResults();
    }
  }, [id, initialPrediction]); // eslint-disable-line

  const fetchResults = async () => {
    try {
      setLoading(true);
      const response = await api.get(`/diagnosis/results/${id}`);
      setPrediction(response.data);
    } catch (err) {
      setError('Failed to load diagnosis results. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPDF = async () => {
    setDownloading(true);
    try {
      const response = await api.get(`/reports/${id}`, { responseType: 'blob' });

      // Check if the response is actually an error JSON (can happen with blob responseType)
      if (response.data.type === 'application/json') {
        const text = await response.data.text();
        const err = JSON.parse(text);
        throw new Error(err.detail || 'Report generation failed');
      }

      const url = window.URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }));
      const link = document.createElement('a');
      link.href = url;
      link.download = `liver_diagnosis_${id}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      const msg = err?.response?.data?.detail || err?.message || 'Could not download the report.';
      alert(`Download failed: ${msg}\n\nPlease try again in a moment.`);
    } finally {
      setDownloading(false);
    }
  };

  const handleNewDiagnosis = () => navigate('/diagnosis');
  const handleViewHistory = () => navigate('/results');

  if (loading) {
    return (
      <div className="results-loading">
        <div className="loading-spinner"></div>
        <p>Loading diagnosis results…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="results-error">
        <div className="error-card">
          <div className="error-icon">⚠️</div>
          <h2>Results Unavailable</h2>
          <p>{error}</p>
          <button className="btn-primary" onClick={handleNewDiagnosis}>Run New Diagnosis</button>
        </div>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="results-error">
        <div className="error-card">
          <div className="error-icon">📋</div>
          <h2>No Data Found</h2>
          <p>No prediction data is available for this diagnosis.</p>
          <button className="btn-primary" onClick={handleNewDiagnosis}>Run New Diagnosis</button>
        </div>
      </div>
    );
  }

  const pred = prediction.prediction || prediction;
  const explanations = prediction.explanations;
  const probabilities = pred.probabilities || {};
  const stage = pred.stage || 'stage_0';
  const riskLevel = (pred.risk_level || 'low').toLowerCase();
  const stageInfo = STAGE_DESCRIPTIONS[stage] || STAGE_DESCRIPTIONS.stage_0;
  const riskColor = RISK_COLORS[riskLevel] || '#64748b';
  const riskBg = RISK_BG[riskLevel] || '#f8fafc';
  const confidence = pred.confidence || 0;
  const recommendations = pred.recommendations || [];

  const cirrhosisRisk = Object.entries(probabilities)
    .filter(([s]) => ['stage_2', 'stage_3', 'stage_4'].includes(s))
    .reduce((sum, [, v]) => sum + v, 0) * 100;

  return (
    <div className="results-page">
      <div className="results-header">
        <div>
          <h1>Diagnosis Results</h1>
          <p className="results-subtitle">Diagnosis #{id} • AI-Assisted Analysis</p>
        </div>
        <div className="header-actions">
          <button className="btn-outline" onClick={handleViewHistory}>📋 My History</button>
          <button className="btn-primary" onClick={handleNewDiagnosis}>＋ New Diagnosis</button>
        </div>
      </div>

      <div className="results-container">
        {/* ── Risk Banner ── */}
        <div className="risk-banner" style={{ background: riskBg, borderColor: riskColor }}>
          <div className="risk-icon-lg">{RISK_ICONS[riskLevel]}</div>
          <div className="risk-info">
            <div className="risk-title" style={{ color: riskColor }}>
              {riskLevel.toUpperCase()} RISK
            </div>
            <div className="risk-stage">{stageInfo.label} — {stageInfo.sub}</div>
          </div>
          <div className="risk-confidence">
            <div className="conf-value">{(confidence * 100).toFixed(1)}%</div>
            <div className="conf-label">AI Confidence</div>
          </div>
        </div>

        {/* ── Metrics Row ── */}
        <div className="metrics-row">
          <div className="metric-card">
            <div className="metric-icon">📊</div>
            <div>
              <div className="metric-value" style={{ color: riskColor }}>
                {cirrhosisRisk.toFixed(1)}%
              </div>
              <div className="metric-label">Cirrhosis Risk</div>
            </div>
            <div className="metric-bar-wrap">
              <div className="metric-bar-track">
                <div
                  className="metric-bar-fill"
                  style={{ width: `${cirrhosisRisk}%`, background: riskColor }}
                ></div>
              </div>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-icon">🔬</div>
            <div>
              <div className="metric-value">{(confidence * 100).toFixed(1)}%</div>
              <div className="metric-label">Model Confidence</div>
            </div>
            <div className="metric-bar-wrap">
              <div className="metric-bar-track">
                <div
                  className="metric-bar-fill confidence-fill"
                  style={{ width: `${confidence * 100}%` }}
                ></div>
              </div>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-icon">🎯</div>
            <div>
              <div className="metric-value" style={{ color: stageInfo.color }}>
                {stageInfo.label}
              </div>
              <div className="metric-label">{stageInfo.sub}</div>
            </div>
          </div>
        </div>

        {/* ── Stage Probabilities ── */}
        {Object.keys(probabilities).length > 0 && (
          <div className="card stage-probs-card">
            <h3>Stage Probability Breakdown</h3>
            <p className="card-subtitle">Probability distribution across all five liver health stages</p>
            <div className="stage-bars">
              {Object.entries(probabilities).map(([s, prob]) => {
                const info = STAGE_DESCRIPTIONS[s] || { label: s, sub: '', color: '#94a3b8' };
                const pct = (prob * 100).toFixed(1);
                return (
                  <div key={s} className={`stage-bar-row ${s === stage ? 'active-stage' : ''}`}>
                    <div className="stage-bar-label">
                      <span className="stage-bar-name">{info.label}</span>
                      <span className="stage-bar-sub">{info.sub}</span>
                    </div>
                    <div className="stage-bar-track">
                      <div
                        className="stage-bar-fill"
                        style={{ width: `${pct}%`, background: info.color }}
                      ></div>
                    </div>
                    <div className="stage-bar-pct">{pct}%</div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── Recommendations ── */}
        {recommendations.length > 0 && (
          <div className="card recommendations-card">
            <h3>💊 Medical Recommendations</h3>
            <p className="card-subtitle">AI-generated guidance based on the assessment findings</p>
            <div className="rec-list">
              {recommendations.map((rec, i) => (
                <div key={i} className="rec-item">
                  <div className="rec-num">{i + 1}</div>
                  <p>{rec}</p>
                </div>
              ))}
            </div>
            <div className="disclaimer">
              ⚕️ This AI output is for screening support only and does not replace a qualified
              medical professional's diagnosis.
            </div>
          </div>
        )}

        {/* ── XAI Explanations ── */}
        {explanations && (
          <div className="card xai-card">
            <h3>🧠 AI Explainability Analysis</h3>
            <p className="card-subtitle">Transparency into how the AI reached this conclusion</p>

            {explanations.natural_language_explanation && (
              <div className="xai-section">
                <h4>Plain-Language Explanation</h4>
                <p className="xai-text">{explanations.natural_language_explanation}</p>
              </div>
            )}

            {explanations.confidence_interpretation && (
              <div className="xai-section">
                <h4>Confidence Assessment</h4>
                <p className="xai-text">{explanations.confidence_interpretation}</p>
              </div>
            )}

            {explanations.key_factors && explanations.key_factors.length > 0 && (
              <div className="xai-section">
                <h4>Key Contributing Factors</h4>
                <div className="factor-list">
                  {explanations.key_factors.map((factor, idx) => (
                    <div key={idx} className="factor-tag">{factor}</div>
                  ))}
                </div>
              </div>
            )}

            {explanations.shap_explanation?.feature_importance && (
              <div className="xai-section">
                <h4>Feature Importance (SHAP)</h4>
                <div className="feature-bars">
                  {Object.entries(explanations.shap_explanation.feature_importance)
                    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                    .slice(0, 8)
                    .map(([feat, val]) => {
                      const pct = Math.min(Math.abs(val) * 100, 100);
                      return (
                        <div key={feat} className="feature-bar-row">
                          <span className="feature-name">{feat.replace(/_/g, ' ')}</span>
                          <div className="feature-track">
                            <div
                              className="feature-fill"
                              style={{
                                width: `${pct}%`,
                                background: val >= 0 ? '#ef4444' : '#10b981',
                              }}
                            ></div>
                          </div>
                          <span className="feature-val">{val >= 0 ? '+' : ''}{val.toFixed(3)}</span>
                        </div>
                      );
                    })}
                </div>
                <p className="shap-legend">
                  <span style={{ color: '#ef4444' }}>■ Red = increases risk</span>
                  &nbsp;&nbsp;
                  <span style={{ color: '#10b981' }}>■ Green = decreases risk</span>
                </p>
              </div>
            )}
          </div>
        )}

        {/* ── Actions ── */}
        <div className="results-actions">
          <button
            className="btn-primary"
            onClick={handleDownloadPDF}
            disabled={downloading}
          >
            {downloading ? '⏳ Generating…' : '⬇️ Download PDF Report'}
          </button>
          <button className="btn-outline" onClick={handleNewDiagnosis}>
            🔍 New Diagnosis
          </button>
          <button className="btn-outline" onClick={handleViewHistory}>
            📋 View History
          </button>
        </div>
      </div>
    </div>
  );
};

export default ResultsPage;
