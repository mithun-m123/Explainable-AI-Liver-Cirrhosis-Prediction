import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/pages/HistoryPage.css';
import api from '../api';

const RISK_COLORS = {
  low: '#10b981',
  medium: '#f59e0b',
  high: '#ef4444',
  critical: '#7f1d1d',
};

const RISK_ICONS = { low: '✅', medium: '⚠️', high: '🚨', critical: '⛔' };

const STAGE_LABELS = {
  stage_0: 'Normal',
  stage_1: 'Stage 1 – Mild Fibrosis',
  stage_2: 'Stage 2 – Moderate Fibrosis',
  stage_3: 'Stage 3 – Severe Fibrosis',
  stage_4: 'Stage 4 – Cirrhosis',
};

const PER_PAGE = 8;

const HistoryPage = () => {
  const navigate = useNavigate();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(false);

  useEffect(() => {
    fetchHistory();
  }, [page]); // eslint-disable-line

  const fetchHistory = async () => {
    setLoading(true);
    setError('');
    try {
      const skip = (page - 1) * PER_PAGE;
      const res = await api.get('/history', { params: { skip, limit: PER_PAGE } });
      const data = res.data || [];
      setHistory(data);
      setHasMore(data.length === PER_PAGE);
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to load diagnosis history.');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (d) => {
    if (!d) return '—';
    return new Date(d).toLocaleDateString('en-US', {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  };

  const getStageName = (record) => {
    const stage = record.prediction_result?.stage;
    return STAGE_LABELS[stage] || record.prediction_result?.stage_description || '—';
  };

  return (
    <div className="history-page">
      <div className="history-header">
        <div>
          <h1>📊 My Results</h1>
          <p>All your previous diagnosis assessments</p>
        </div>
        <button className="btn-primary" onClick={() => navigate('/diagnosis')}>
          ＋ New Diagnosis
        </button>
      </div>

      <div className="history-container">
        {loading ? (
          <div className="history-loading">
            <div className="spinner"></div>
            <p>Loading your results…</p>
          </div>
        ) : error ? (
          <div className="history-empty">
            <div className="empty-icon">⚠️</div>
            <h2>Could Not Load Results</h2>
            <p>{error}</p>
            <button className="btn-retry" onClick={fetchHistory}>Retry</button>
          </div>
        ) : history.length === 0 ? (
          <div className="history-empty">
            <div className="empty-icon">🏥</div>
            <h2>No Results Yet</h2>
            <p>Complete your first diagnosis to see results here.</p>
            <button className="btn-primary" onClick={() => navigate('/diagnosis')}>
              Start Diagnosis
            </button>
          </div>
        ) : (
          <>
            {/* Summary bar */}
            <div className="summary-bar">
              <span>{history.length} result{history.length !== 1 ? 's' : ''} on this page</span>
            </div>

            <div className="history-table-wrap">
              <table className="history-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Date</th>
                    <th>Stage</th>
                    <th>Risk Level</th>
                    <th>Confidence</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((record) => {
                    const risk = (record.risk_level || 'low').toLowerCase();
                    const riskColor = RISK_COLORS[risk] || '#64748b';
                    const confidence = record.confidence_score != null
                      ? `${(record.confidence_score * 100).toFixed(1)}%`
                      : '—';

                    return (
                      <tr key={record.id} className="history-row">
                        <td className="col-id">
                          <span className="id-badge">#{record.id}</span>
                        </td>
                        <td className="col-date">{formatDate(record.created_at)}</td>
                        <td className="col-stage">{getStageName(record)}</td>
                        <td className="col-risk">
                          <span
                            className="risk-tag"
                            style={{
                              background: riskColor + '18',
                              color: riskColor,
                              border: `1px solid ${riskColor}44`,
                            }}
                          >
                            {RISK_ICONS[risk]} {risk.toUpperCase()}
                          </span>
                        </td>
                        <td className="col-conf">
                          <div className="conf-wrap">
                            <div className="conf-bar-track">
                              <div
                                className="conf-bar-fill"
                                style={{
                                  width: record.confidence_score
                                    ? `${record.confidence_score * 100}%`
                                    : '0%',
                                  background: riskColor,
                                }}
                              ></div>
                            </div>
                            <span>{confidence}</span>
                          </div>
                        </td>
                        <td className="col-actions">
                          <button
                            className="btn-view-result"
                            onClick={() => navigate(`/results/${record.id}`)}
                          >
                            View Details
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div className="pagination">
              <button
                className="page-btn"
                disabled={page === 1}
                onClick={() => setPage(p => p - 1)}
              >
                ← Previous
              </button>
              <span className="page-info">Page {page}</span>
              <button
                className="page-btn"
                disabled={!hasMore}
                onClick={() => setPage(p => p + 1)}
              >
                Next →
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default HistoryPage;
