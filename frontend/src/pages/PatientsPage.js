import React, { useEffect, useState } from 'react';
import '../styles/pages/PatientsPage.css';
import { patientsAPI } from '../api';

const EMPTY_FORM = { full_name: '', email: '', password: 'liver1234' };

const PatientsPage = () => {
  const [patients,    setPatients]    = useState([]);
  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState('');
  const [search,      setSearch]      = useState('');
  const [showModal,   setShowModal]   = useState(false);
  const [form,        setForm]        = useState(EMPTY_FORM);
  const [formError,   setFormError]   = useState('');
  const [formLoading, setFormLoading] = useState(false);
  const [addedUid,    setAddedUid]    = useState(null);   // show after success

  useEffect(() => { fetchPatients(); }, []);

  const fetchPatients = async () => {
    setLoading(true); setError('');
    try {
      const res = await patientsAPI.list();
      setPatients(res.patients || []);
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to load patients');
    } finally { setLoading(false); }
  };

  const handleDeactivate = async (patientId) => {
    if (!window.confirm('Deactivate this patient account?')) return;
    try {
      const res = await fetch(`http://localhost:8000/patients/${patientId}/deactivate`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${localStorage.getItem('auth_token')}` },
      });
      if (!res.ok) throw new Error('Failed');
      setPatients(prev => prev.map(p =>
        p.id === patientId ? { ...p, is_active: false } : p
      ));
    } catch {
      setError('Failed to deactivate patient');
    }
  };

  const handleFormChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const handleAddPatient = async (e) => {
    e.preventDefault();
    setFormError('');
    if (!form.full_name.trim()) { setFormError('Patient name is required'); return; }
    if (!form.email.trim())     { setFormError('Email is required'); return; }
    if (form.password.length < 6) { setFormError('Password must be at least 6 characters'); return; }

    setFormLoading(true);
    try {
      const res = await patientsAPI.add(form);
      setAddedUid(res.patient.patient_uid);
      setPatients(prev => [res.patient, ...prev]);
      setForm(EMPTY_FORM);
    } catch (err) {
      setFormError(err.response?.data?.detail || 'Failed to add patient');
    } finally { setFormLoading(false); }
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setForm(EMPTY_FORM);
    setFormError('');
    setAddedUid(null);
  };

  const filtered = patients.filter(p =>
    p.full_name.toLowerCase().includes(search.toLowerCase()) ||
    p.email.toLowerCase().includes(search.toLowerCase()) ||
    (p.patient_uid || '').toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="patients-page">
      {/* Header */}
      <div className="patients-header">
        <div>
          <h1>👥 Patients</h1>
          <p>{patients.length} registered patient{patients.length !== 1 ? 's' : ''}</p>
        </div>
        <button className="btn-add-patient" onClick={() => setShowModal(true)}>
          ＋ Add Patient
        </button>
      </div>

      {/* Search */}
      <div className="patients-search">
        <input
          type="text"
          placeholder="🔍  Search by name, email or Patient ID…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="search-input"
        />
      </div>

      {/* Table */}
      <div className="patients-container">
        {loading ? (
          <div className="patients-loading">
            <div className="spinner"></div>
            <p>Loading patients…</p>
          </div>
        ) : error ? (
          <div className="error-message">{error}</div>
        ) : filtered.length === 0 ? (
          <div className="patients-empty">
            {search ? `No patients matching "${search}"` : 'No patients registered yet.'}
          </div>
        ) : (
          <table className="patients-table">
            <thead>
              <tr>
                <th>Patient ID</th>
                <th>Full Name</th>
                <th>Email</th>
                <th>Status</th>
                <th>Joined</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(p => (
                <tr key={p.id} className={!p.is_active ? 'row-inactive' : ''}>
                  <td>
                    <span className="uid-badge">{p.patient_uid || `PAT-${String(p.id).padStart(5,'0')}`}</span>
                  </td>
                  <td className="col-name">{p.full_name}</td>
                  <td className="col-email">{p.email}</td>
                  <td>
                    <span className={`status-pill ${p.is_active ? 'active' : 'inactive'}`}>
                      {p.is_active ? '● Active' : '● Inactive'}
                    </span>
                  </td>
                  <td className="col-date">
                    {new Date(p.created_at).toLocaleDateString('en-US', {
                      year: 'numeric', month: 'short', day: 'numeric'
                    })}
                  </td>
                  <td>
                    {p.is_active ? (
                      <button className="btn-deactivate" onClick={() => handleDeactivate(p.id)}>
                        Deactivate
                      </button>
                    ) : (
                      <span className="muted">Inactive</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Add Patient Modal */}
      {showModal && (
        <div className="modal-overlay" onClick={handleCloseModal}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>

            {addedUid ? (
              /* ── Success screen ── */
              <div className="modal-success">
                <div className="success-icon">✅</div>
                <h2>Patient Added!</h2>
                <div className="uid-display">
                  <span className="uid-label">Unique Patient ID</span>
                  <span className="uid-value">{addedUid}</span>
                </div>
                <p className="uid-note">
                  Share this ID with the patient. They can use it to reference their records.
                </p>
                <button className="btn-primary" onClick={handleCloseModal}>Done</button>
              </div>
            ) : (
              /* ── Form ── */
              <>
                <div className="modal-header">
                  <h2>➕ Add New Patient</h2>
                  <button className="modal-close" onClick={handleCloseModal}>✕</button>
                </div>

                <form onSubmit={handleAddPatient} className="modal-form">
                  {formError && <div className="error-message">⚠ {formError}</div>}

                  <div className="form-group">
                    <label>Patient Full Name <span className="required">*</span></label>
                    <input
                      type="text"
                      name="full_name"
                      value={form.full_name}
                      onChange={handleFormChange}
                      placeholder="e.g. Ravi Kumar"
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label>Email Address <span className="required">*</span></label>
                    <input
                      type="email"
                      name="email"
                      value={form.email}
                      onChange={handleFormChange}
                      placeholder="patient@example.com"
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label>Temporary Password</label>
                    <input
                      type="text"
                      name="password"
                      value={form.password}
                      onChange={handleFormChange}
                      placeholder="Minimum 6 characters"
                    />
                    <span className="field-hint">Patient can change this after first login</span>
                  </div>

                  <div className="modal-note">
                    <span>🪪</span>
                    <span>A unique Patient ID (e.g. <strong>PAT-00006</strong>) will be automatically assigned.</span>
                  </div>

                  <div className="modal-actions">
                    <button type="button" className="btn-outline" onClick={handleCloseModal}>
                      Cancel
                    </button>
                    <button type="submit" className="btn-primary" disabled={formLoading}>
                      {formLoading ? 'Adding…' : 'Add Patient'}
                    </button>
                  </div>
                </form>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PatientsPage;
