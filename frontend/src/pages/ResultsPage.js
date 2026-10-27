import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ExternalLink, MapPin, Navigation, PhoneCall, Stethoscope } from 'lucide-react';
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
  const [consultationOpen, setConsultationOpen] = useState(false);
  const [locationStatus, setLocationStatus] = useState('idle');
  const [locationError, setLocationError] = useState('');
  const [location, setLocation] = useState(null);
  const [providers, setProviders] = useState([]);
  const [providersLoading, setProvidersLoading] = useState(false);

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

  const findNearbyDoctors = () => {
    setConsultationOpen(true); setLocationStatus('loading'); setLocationError('');
    if (!navigator.geolocation) { setLocationStatus('manual'); setLocationError('Location is not available in this browser.'); return; }
    navigator.geolocation.getCurrentPosition(({ coords }) => {
      setLocation({ latitude: coords.latitude, longitude: coords.longitude });
      fetchNearbyProviders(coords.latitude, coords.longitude);
      const query = encodeURIComponent('hepatologist gastroenterologist liver specialist');
      window.open(`https://www.google.com/maps/search/${query}/@${coords.latitude},${coords.longitude},13z`, '_blank', 'noopener,noreferrer');
      setLocationStatus('found');
    }, () => { setLocationStatus('manual'); setLocationError('We could not access your location. Search by city or postcode instead.'); }, { enableHighAccuracy: false, timeout: 8000 });
  };

  const fetchNearbyProviders = async (latitude, longitude) => {
    setProvidersLoading(true);
    try {
      const query = `[out:json][timeout:15];(nwr[amenity~"hospital|clinic|doctors"](around:10000,${latitude},${longitude}););out center tags;`;
      const response = await fetch(`https://overpass-api.de/api/interpreter?data=${encodeURIComponent(query)}`);
      if (!response.ok) throw new Error('Provider search failed');
      const data = await response.json();
      const mapped = (data.elements || []).map(provider => {
        const tags = provider.tags || {}; const center = provider.center || provider;
        const address = [tags['addr:housenumber'], tags['addr:street'], tags['addr:city'], tags['addr:postcode']].filter(Boolean).join(', ');
        const name = tags.name || 'Healthcare provider';
        return { id: provider.id, name, type: tags.amenity === 'hospital' ? 'Hospital' : tags.amenity === 'clinic' ? 'Clinic' : 'Doctors', address: address || null, phone: tags.phone || tags['contact:phone'] || null, mapUrl: `https://www.google.com/maps/search/?api=1&query=${center.lat},${center.lon}`, searchUrl: `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${name} ${address}`)}` };
      }).filter(provider => provider.name !== 'Healthcare provider' || provider.address).slice(0, 8);
      setProviders(mapped);
      if (!mapped.length) setLocationError('No listed providers were found within 10 km. Try searching by city or postcode.');
    } catch (err) { setLocationError('Nearby provider details could not be loaded. You can still search by city or postcode.'); }
    finally { setProvidersLoading(false); }
  };

  const openManualSearch = (event) => { event.preventDefault(); const area = new FormData(event.currentTarget).get('location'); if (!area?.trim()) return; window.open(`https://www.google.com/maps/search/${encodeURIComponent(`hepatologist gastroenterologist liver specialist near ${area.trim()}`)}`, '_blank', 'noopener,noreferrer'); setLocationStatus('found'); };

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
  const needsConsultation = ['medium', 'high', 'critical'].includes(riskLevel) || ['stage_2', 'stage_3', 'stage_4'].includes(stage);

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

        {needsConsultation && <div className="consultation-card consultation-priority" role="region" aria-label="Nearby doctor consultation">
          <div className="consultation-icon"><Stethoscope size={25} /></div>
          <div className="consultation-copy">
            <span className="consultation-kicker">{needsConsultation ? 'Recommended next step' : 'Care directory'}</span>
            <h2>{needsConsultation ? 'Speak with a liver specialist nearby' : 'Find a liver specialist nearby'}</h2>
            <p>{needsConsultation ? 'Your result suggests follow-up with a hepatologist or gastroenterologist. Find local care and bring this report to your appointment.' : 'Search for a hepatologist or gastroenterologist near you. This screening result does not replace a medical consultation.'}</p>
            {consultationOpen && locationStatus === 'manual' && <form className="location-form" onSubmit={openManualSearch}><input name="location" type="text" placeholder="City or postcode" aria-label="City or postcode" /><button type="submit" className="location-submit"><ExternalLink size={16} /> Search area</button></form>}
            {consultationOpen && locationError && <small className="location-error">{locationError}</small>}
            {consultationOpen && locationStatus === 'found' && <small className="location-success"><PhoneCall size={14} /> Search opened in a new tab. Call the clinic before visiting.</small>}
            {consultationOpen && providersLoading && <div className="provider-loading">Searching nearby hospitals and clinics...</div>}
            {providers.length > 0 && <div className="provider-list">{providers.map(provider => <article className="provider-item" key={`${provider.id}-${provider.name}`}><div><strong>{provider.name}</strong><span>{provider.type}</span></div><p><MapPin size={14} /> {provider.address || 'Address not listed'}</p><p><PhoneCall size={14} /> {provider.phone || 'Phone not listed'}</p>{(!provider.address || !provider.phone) && <a className="provider-contact-link" href={provider.searchUrl} target="_blank" rel="noreferrer">Find missing contact details <ExternalLink size={13} /></a>}<a href={provider.mapUrl} target="_blank" rel="noreferrer">Open map <ExternalLink size={13} /></a></article>)}</div>}
          {!consultationOpen && <button className="consultation-button" onClick={findNearbyDoctors}><Navigation size={17} /> Find nearby doctors</button>}
          </div></div>}

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
