import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/pages/ReportsPage.css';
import api from '../api';

const RISK_COLORS = {
  low: '#10b981',
  medium: '#f59e0b',
  high: '#ef4444',
  critical: '#7f1d1d',
};

const RISK_ICONS = {
  low: '✅',
  medium: '⚠️',
  high: '🚨',
  critical: '⛔',
};

const ReportsPage = () => {
  const navigate = useNavigate();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [downloading, setDownloading] = useState(null);
  const [page, setPage] = useState(1);
  const PER_PAGE = 10;

  useEffect(() => {
    fetchHistory();
  }, [page]); // eslint-disable-line

  const fetchHistory = async () => {
    setLoading(true);
    setError('');
    try {
      const skip = (page - 1) * PER_PAGE;
      const res = await api.get('/history', { params: { skip, limit: PER_PAGE } });
      setHistory(res.data || []);
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to load diagnosis history.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (diagnosisId) => {
    setDownloading(diagnosisId);
    try {
      const response = await api.get(`/reports/${diagnosisId}`, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }));
      const link = document.createElement('a');
      link.href = url;
      link.download = `liver_diagnosis_${diagnosisId}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch {
      alert('Could not download the report. Please try again.');
    } finally {
      setDownloading(null);
    }
  };

  const handleViewResults = (diagnosisId) => {
    navigate(`/results/${diagnosisId}`);
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return '—';
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-US', {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  };

  return (
    <div className="reports-page">
      <div className="reports-header">
        <div>
          <h1>📄 Diagnosis Reports</h1>
          <p>View, download, and review all your previous diagnoses</p>
        </div>
        <button className="btn-new" onClick={() => navigate('/diagnosis')}>
          ＋ New Diagnosis
        </button>
      </div>

      <div className="reports-container">
        {loading ? (
          <div className="reports-loading">
            <div className="spinner"></div>
            <p>Loading your reports…</p>
          </div>
        ) : error ? (
          <div className="reports-error">
            <div className="error-icon">⚠️</div>
            <p>{error}</p>
            <button className="btn-retry" onClick={fetchHistory}>Retry</button>
          </div>
        ) : history.length === 0 ? (
          <div className="reports-empty">
            <div className="empty-icon">🏥</div>
            <h2>No Reports Yet</h2>
            <p>Run your first diagnosis to generate a report.</p>
            <button className="btn-new" onClick={() => navigate('/diagnosis')}>
              Start Diagnosis
            </button>
          </div>
        ) : (
          <>
            <div className="reports-grid">
              {history.map((record) => {
                const riskLevel = (record.risk_level || 'low').toLowerCase();
                const riskColor = RISK_COLORS[riskLevel] || '#64748b';
                const stageName = record.prediction_result?.stage_description
                  || record.prediction_result?.stage
                  || '—';
                const confidence = record.confidence_score
                  ? `${(record.confidence_score * 100).toFixed(1)}%`
                  : '—';

                return (
                  <div key={record.id} className="report-card">
                    <div className="report-card-header" style={{ borderLeftColor: riskColor }}>
                      <div className="report-id">
                        <span className="id-label">Diagnosis</span>
                        <span className="id-value">#{record.id}</span>
                      </div>
                      <div
                        className="risk-pill"
                        style={{ background: riskColor + '22', color: riskColor, border: `1px solid ${riskColor}` }}
                      >
                        {RISK_ICONS[riskLevel]} {riskLevel.toUpperCase()}
                      </div>
                    </div>

                    <div className="report-card-body">
                      <div className="report-detail">
                        <span className="detail-label">Stage</span>
                        <span className="detail-value">{stageName}</span>
                      </div>
                      <div className="report-detail">
                        <span className="detail-label">Confidence</span>
                        <span className="detail-value">{confidence}</span>
                      </div>
                      <div className="report-detail">
                        <span className="detail-label">Date</span>
                        <span className="detail-value">{formatDate(record.created_at)}</span>
                      </div>
                    </div>

                    <div className="report-card-actions">
                      <button
                        className="btn-view"
                        onClick={() => handleViewResults(record.id)}
                      >
                        👁 View Results
                      </button>
                      <button
                        className="btn-download"
                        onClick={() => handleDownload(record.id)}
                        disabled={downloading === record.id}
                      >
                        {downloading === record.id ? '⏳ Downloading…' : '⬇️ Download PDF'}
                      </button>
                    </div>
                  </div>
                );
              })}
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
                disabled={history.length < PER_PAGE}
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

export default ReportsPage;
