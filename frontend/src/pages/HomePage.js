import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/pages/HomePage.css';
import api from '../api';

const HomePage = ({ onLogin }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    fullName: '',
    role: 'patient'
  });
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      if (isLogin) {
        const response = await api.post('/auth/login', {
          email: formData.email,
          password: formData.password
        });
        localStorage.setItem('auth_token', response.data.access_token);
        
        onLogin({
          ...response.data.user,
          token: response.data.access_token
        });
        
        navigate('/diagnosis');
      } else {
        // Client-side validation to avoid server 422 errors
        if (!formData.fullName || formData.fullName.trim().length === 0) {
          setError('Full name is required for registration');
          setLoading(false);
          return;
        }
        if (!formData.password || formData.password.length < 8) {
          setError('Password must be at least 8 characters');
          setLoading(false);
          return;
        }
        await api.post('/auth/register', {
          email: formData.email,
          password: formData.password,
          full_name: formData.fullName,
          role: formData.role
        });
        
        setIsLogin(true);
        setError('');
        setFormData({
          email: formData.email,
          password: formData.password,
          fullName: '',
          role: 'patient'
        });
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="home-page">
      <div className="hero-section">
        <div className="hero-content">
          <h1>🏥 Liver Cirrhosis Detection System</h1>
          <p>AI-Powered Non-invasive Detection using Machine Learning & Deep Learning</p>
          <p className="subtitle">Early detection for better outcomes</p>
        </div>
      </div>

      <div className="content-container">
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🔬</div>
            <h3>Accurate Diagnosis</h3>
            <p>Hybrid ML+DL models for precise cirrhosis detection and staging</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Explainable AI</h3>
            <p>SHAP, LIME, and Grad-CAM explanations for transparent predictions</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🩺</div>
            <h3>Medical Insights</h3>
            <p>Comprehensive analysis of symptoms and medical imaging</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">💬</div>
            <h3>AI Chatbot</h3>
            <p>Medical chatbot for personalized guidance and recommendations</p>
          </div>
        </div>

        <div className="auth-section">
          <div className="auth-container">
            <div className="auth-box">
              <h2>{isLogin ? 'Login' : 'Register'}</h2>
              
              {error && <div className="error-message">{error}</div>}
              
              <form onSubmit={handleSubmit}>
                {!isLogin && (
                  <div className="form-group">
                    <label>Full Name</label>
                    <input
                      type="text"
                      name="fullName"
                      value={formData.fullName}
                      onChange={handleChange}
                      placeholder="Enter your full name"
                      required={!isLogin}
                    />
                  </div>
                )}

                <div className="form-group">
                  <label>Email</label>
                  <input
                    type="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    placeholder="Enter your email"
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Password</label>
                  <input
                    type="password"
                    name="password"
                    value={formData.password}
                    onChange={handleChange}
                    placeholder="Enter your password"
                    required
                  />
                </div>

                {!isLogin && (
                  <div className="form-group">
                    <label>User Role</label>
                    <select
                      name="role"
                      value={formData.role}
                      onChange={handleChange}
                    >
                      <option value="patient">Patient</option>
                      <option value="doctor">Doctor</option>
                    </select>
                  </div>
                )}

                <button 
                  type="submit" 
                  className="btn-submit"
                  disabled={loading}
                >
                  {loading ? 'Processing...' : isLogin ? 'Login' : 'Register'}
                </button>
              </form>

              <div className="auth-toggle">
                <p>
                  {isLogin ? "Don't have an account? " : "Already have an account? "}
                  <button
                    type="button"
                    onClick={() => setIsLogin(!isLogin)}
                    className="toggle-btn"
                  >
                    {isLogin ? 'Register' : 'Login'}
                  </button>
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
