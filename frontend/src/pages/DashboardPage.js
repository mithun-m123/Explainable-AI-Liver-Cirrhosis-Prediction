import React, { useEffect, useState } from 'react';
import { 
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend,
  AreaChart, Area, XAxis, YAxis, CartesianGrid, BarChart, Bar
} from 'recharts';
import { analyticsAPI } from '../api';
import '../styles/pages/DashboardPage.css';

const DashboardPage = () => {
  const [stats, setStats] = useState(null);
  const [trends, setTrends] = useState([]);
  const [distribution, setDistribution] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        // Fetch stats, trends, and risk distribution in parallel
        const [statsData, trendsData, distData] = await Promise.all([
          analyticsAPI.getUserStats(),
          analyticsAPI.getDiagnosisTrends(null, 30),
          analyticsAPI.getRiskDistribution()
        ]);

        setStats(statsData);
        setTrends(trendsData);
        setDistribution(distData);
      } catch (err) {
        console.error('Failed to load dashboard data', err);
        setError('Error loading analytics. Make sure the backend server is running and populated.');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  if (loading) {
    return (
      <div className="dashboard-page loading-container">
        <div className="spinner"></div>
        <p>Loading medical analytics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-page error-container">
        <div className="error-card">
          <h2>⚠️ Analytics Unavailable</h2>
          <p>{error}</p>
          <button onClick={() => window.location.reload()} className="btn-retry">Retry Loading</button>
        </div>
      </div>
    );
  }

  const { total_diagnoses = 0, average_confidence = 0, risk_counts = {} } = stats || {};
  const criticalAlerts = (risk_counts.critical || 0) + (risk_counts.high || 0);

  return (
    <div className="dashboard-page">
      <div className="dashboard-header-section">
        <h1>📈 Clinical Dashboard & Analytics</h1>
        <p className="subtitle">Real-time surveillance and predictions of Liver Cirrhosis cases</p>
      </div>

      <div className="dashboard-container">
        {/* KPI Grid */}
        <div className="kpi-grid">
          <div className="kpi-card total-diagnoses">
            <div className="kpi-icon">📋</div>
            <div className="kpi-info">
              <h3>Total Diagnoses</h3>
              <div className="kpi-value">{total_diagnoses}</div>
              <p>AI Screening Assessments</p>
            </div>
          </div>

          <div className="kpi-card confidence">
            <div className="kpi-icon">🔬</div>
            <div className="kpi-info">
              <h3>Avg. AI Confidence</h3>
              <div className="kpi-value">{(average_confidence * 100).toFixed(1)}%</div>
              <p>Model Prediction Certitude</p>
            </div>
          </div>

          <div className="kpi-card alerts">
            <div className="kpi-icon">🚨</div>
            <div className="kpi-info">
              <h3>Critical Alerts</h3>
              <div className="kpi-value">{criticalAlerts}</div>
              <p>High & Critical Risk Patients</p>
            </div>
          </div>
        </div>

        {/* Charts Section */}
        <div className="charts-grid">
          {/* Trend Chart */}
          <div className="chart-card trends-card">
            <h3>📈 Screening Activity & Risk Level Trends</h3>
            <p className="chart-description">Diagnosis count and average risk score progression (last 30 days)</p>
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={trends} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorRisk" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#F59E0B" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis yAxisId="left" label={{ value: 'Diagnoses', angle: -90, position: 'insideLeft' }} />
                  <YAxis yAxisId="right" orientation="right" domain={[0, 4]} label={{ value: 'Avg Risk Score', angle: 90, position: 'insideRight' }} />
                  <Tooltip />
                  <Legend />
                  <Area yAxisId="left" type="monotone" dataKey="count" name="Diagnoses Done" stroke="#3B82F6" fillOpacity={1} fill="url(#colorCount)" />
                  <Area yAxisId="right" type="monotone" dataKey="average_risk_score" name="Avg Risk (1-4 Scale)" stroke="#F59E0B" fillOpacity={1} fill="url(#colorRisk)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Distribution Chart */}
          <div className="chart-card distribution-card">
            <h3>📊 Patient Risk Class Distribution</h3>
            <p className="chart-description">Percentage breakdown of patients across liver health risk brackets</p>
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={distribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={90}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {distribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Confidence Distribution */}
          <div className="chart-card bar-card">
            <h3>🔬 Confidence Score Distribution by Date</h3>
            <p className="chart-description">Average AI classification confidence index progression</p>
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={trends} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis domain={[0, 1]} tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} />
                  <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
                  <Bar dataKey="average_confidence" name="Model Confidence" fill="#10B981" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
