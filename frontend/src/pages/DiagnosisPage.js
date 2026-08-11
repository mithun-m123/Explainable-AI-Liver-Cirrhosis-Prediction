import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/pages/DiagnosisPage.css';
import api from '../api';

/* ─── Test modes ───────────────────────────────────────────────────── */
const TEST_MODES = [
  {
    id: 'symptoms_only',
    icon: '📋',
    title: 'Symptoms Only',
    subtitle: 'Early Screening Tool',
    accuracy: 'Basic',
    accuracyColor: '#f59e0b',
    description: 'Quick screening based on visible symptoms and clinical history. No lab work needed.',
    tabs: ['symptoms'],
    badge: 'Fast · No labs required',
    badgeColor: '#f59e0b',
  },
  {
    id: 'symptoms_labs',
    icon: '🧪',
    title: 'Symptoms + Lab Values',
    subtitle: 'Clinical Assessment',
    accuracy: 'High',
    accuracyColor: '#10b981',
    description: 'Combines symptom data with blood test results for a comprehensive and accurate prediction.',
    tabs: ['symptoms', 'labs'],
    badge: 'Recommended · High accuracy',
    badgeColor: '#10b981',
  },
  {
    id: 'full',
    icon: '🏥',
    title: 'Full Assessment',
    subtitle: 'Complete Diagnostic',
    accuracy: 'Highest',
    accuracyColor: '#667eea',
    description: 'Most thorough evaluation — symptoms, lab values, and liver imaging combined.',
    tabs: ['symptoms', 'labs', 'imaging'],
    badge: 'Maximum accuracy',
    badgeColor: '#667eea',
  },
];

/* ─── Dataset-aligned field definitions ────────────────────────────── */
/*
 * Source: Mayo Clinic PBC Trial (cirrhosis.csv)
 * - Ascites, Hepatomegaly, Spiders: Y / N  → boolean checkboxes
 * - Edema: N / S (diuretic-controlled) / Y (overt)  → 3-option selector
 * - Sex: F=0, M=1
 * - Platelets stored in ×10³/µL in dataset (62–563), NOT raw count
 * - Alk_Phos actual range 289–13862 U/L
 * - Cholesterol actual range 120–1775 mg/dL
 */

// Defaults = dataset medians from cirrhosis.csv
const DEFAULT_SYMPTOMS = {
  age:                  '',
  sex:                  0,        // 0=Female (89% of dataset), 1=Male
  fatigue_level:        5,
  alcohol_consumption:  0,
  weight_loss_kg:       0,
  // Symptom checklist
  abdominal_swelling:   false,
  appetite_loss:        false,
  jaundice:             false,
  fever:                false,
  // Clinical exam — Y/N in dataset
  ascites:              false,
  hepatomegaly:         false,
  spiders:              false,
  // Edema: 0=None(N), 1=Diuretic-controlled(S), 2=Overt(Y)
  edema:                0,
  // Lab values — dataset medians
  bilirubin:            1.4,
  cholesterol:          310,
  albumin:              3.5,
  copper:               73,
  alk_phos:             1259,
  ast:                  115,
  triglycerides:        108,
  platelets:            251,   // ×10³/µL — dataset stores e.g. 251 meaning 251,000
  prothrombin:          10.6,
};

// Lab fields — min/max = actual patient range in Mayo Clinic PBC dataset
// "normal" = healthy adult reference range (for context only)
const LAB_FIELDS = [
  { name: 'bilirubin',     label: 'Bilirubin',            unit: 'mg/dL',   min: 0.1,  max: 28,    step: 0.1,  normal: '0.2–1.2'  },
  { name: 'albumin',       label: 'Albumin',              unit: 'g/dL',     min: 1.0,  max: 5.0,    step: 0.1,  normal: '3.5–5.0'  },
  { name: 'cholesterol',   label: 'Cholesterol',          unit: 'mg/dL',   min: 50,   max: 1775,  step: 1,    normal: '<200'     },
  { name: 'ast',           label: 'AST (SGOT)',           unit: 'U/L',     min: 5,    max: 460,   step: 1,    normal: '10–40'    },
  { name: 'alk_phos',      label: 'Alkaline Phosphatase', unit: 'U/L',      min: 35,   max: 14000, step: 1,    normal: '44–147'   },
  { name: 'copper',        label: 'Copper',               unit: 'μg/dL',   min: 4,    max: 590,   step: 1,    normal: '70–140'   },
  { name: 'triglycerides', label: 'Triglycerides',        unit: 'mg/dL',   min: 20,   max: 600,   step: 1,    normal: '<150'     },
  { name: 'platelets',     label: 'Platelets',            unit: '×10³/μL', min: 50,   max: 600,   step: 1,    normal: '150–400'  },
  { name: 'prothrombin',   label: 'Prothrombin Time',     unit: 'sec',     min: 8.0,  max: 18.0,  step: 0.1,  normal: '11–13.5'  },
];

const OCR_EXTRACTABLE = new Set(LAB_FIELDS.map(f => f.name));

/* ─── Component ─────────────────────────────────────────────────────── */
const DiagnosisPage = ({ onPrediction }) => {
  const navigate = useNavigate();

  const [selectedMode,     setSelectedMode]     = useState(null);
  const [activeTab,        setActiveTab]        = useState('symptoms');
  const [symptoms,         setSymptoms]         = useState(DEFAULT_SYMPTOMS);
  const [imageFile,        setImageFile]        = useState(null);
  const [imagePreview,     setImagePreview]     = useState(null);
  const [loading,          setLoading]          = useState(false);
  const [error,            setError]            = useState('');
  const [labMode,          setLabMode]          = useState('manual');
  const [labReportFile,    setLabReportFile]    = useState(null);
  const [labReportPreview, setLabReportPreview] = useState(null);
  const [ocrLoading,       setOcrLoading]       = useState(false);
  const [ocrResult,        setOcrResult]        = useState(null);
  const labInputRef = useRef(null);

  const currentMode = TEST_MODES.find(m => m.id === selectedMode);
  const tabs = currentMode?.tabs || [];

  /* ── Handlers ────────────────────────────────────────────────── */
  const handleSelectMode = (id) => { setSelectedMode(id); setActiveTab('symptoms'); setError(''); };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setSymptoms(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked
        : (type === 'number' || type === 'range') ? (parseFloat(value) || 0)
        : value,
    }));
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImageFile(file);
    const r = new FileReader();
    r.onload = ev => setImagePreview(ev.target.result);
    r.readAsDataURL(file);
  };

  const handleLabReportChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setLabReportFile(file);
    setOcrResult(null);
    const r = new FileReader();
    r.onload = ev => setLabReportPreview(ev.target.result);
    r.readAsDataURL(file);
  };

  const handleRemoveLabReport = () => {
    setLabReportFile(null); setLabReportPreview(null); setOcrResult(null);
    if (labInputRef.current) labInputRef.current.value = '';
  };

  const handleExtractLabValues = async () => {
    if (!labReportFile) return;
    setOcrLoading(true); setOcrResult(null);
    try {
      const form = new FormData();
      form.append('image', labReportFile);
      const res = await api.post('/labs/extract-from-image', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const data = res.data;
      if (!data._ocr_available) {
        setOcrResult({ status: 'unavailable', message: 'OCR not available on this server. Please fill manually.' });
        setLabMode('manual'); return;
      }
      if (data._error) { setOcrResult({ status: 'error', message: data._error }); return; }
      const extracted = {}, missing = [];
      LAB_FIELDS.forEach(({ name }) => {
        if (data[name] !== undefined && OCR_EXTRACTABLE.has(name)) extracted[name] = data[name];
        else missing.push(name);
      });
      if (!Object.keys(extracted).length) {
        setOcrResult({ status: 'empty', message: 'No values found. Try a clearer image or fill manually.' });
        return;
      }
      setSymptoms(prev => ({ ...prev, ...extracted }));
      setOcrResult({ status: 'success', extracted, missing });
    } catch (err) {
      setOcrResult({ status: 'error', message: err.response?.data?.detail || 'Failed to process image.' });
    } finally { setOcrLoading(false); }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!symptoms.age || isNaN(symptoms.age) || symptoms.age < 1) {
      setError('Please enter a valid age.'); return;
    }
    setLoading(true); setError('');
    try {
      const form = new FormData();
      form.append('symptoms', JSON.stringify(symptoms));
      if (imageFile && tabs.includes('imaging')) form.append('image', imageFile);
      const response = await api.post('/predict', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      onPrediction(response.data);
      navigate(`/results/${response.data.diagnosis_id}`);
    } catch (err) {
      setError(err.response?.data?.detail || 'Diagnosis failed. Please try again.');
    } finally { setLoading(false); }
  };

  /* ── Mode selector ───────────────────────────────────────────── */
  if (!selectedMode) {
    return (
      <div className="diagnosis-page">
        <div className="mode-selector-header">
          <h1>🩺 Start a New Diagnosis</h1>
          <p>Choose the assessment type based on available information</p>
        </div>
        <div className="mode-cards">
          {TEST_MODES.map(mode => (
            <div key={mode.id} className="mode-card" onClick={() => handleSelectMode(mode.id)}>
              <div className="mode-card-top">
                <div className="mode-card-icon">{mode.icon}</div>
                <div className="mode-accuracy-badge"
                  style={{ background: mode.accuracyColor + '20', color: mode.accuracyColor, border: `1px solid ${mode.accuracyColor}` }}>
                  {mode.accuracy} Accuracy
                </div>
              </div>
              <div className="mode-card-title">{mode.title}</div>
              <div className="mode-card-subtitle">{mode.subtitle}</div>
              <p className="mode-card-desc">{mode.description}</p>
              <div className="mode-card-steps">
                {mode.tabs.map(tab => (
                  <span key={tab} className="mode-step">
                    {tab === 'symptoms' && '📋 Symptoms'}
                    {tab === 'labs'    && '🧪 Lab Values'}
                    {tab === 'imaging' && '🖼️ Imaging'}
                  </span>
                ))}
              </div>
              <div className="mode-card-footer"
                style={{ background: mode.badgeColor + '15', color: mode.badgeColor }}>
                {mode.badge}
              </div>
              <button className="mode-card-btn"
                style={{ background: `linear-gradient(135deg, ${mode.badgeColor}, ${mode.badgeColor}cc)` }}>
                Start This Assessment →
              </button>
            </div>
          ))}
        </div>
      </div>
    );
  }

  /* ── Diagnosis form ──────────────────────────────────────────── */
  return (
    <div className="diagnosis-page">
      <div className="diagnosis-header">
        <div className="dh-left">
          <button className="btn-change-mode" onClick={() => setSelectedMode(null)}>← Change Mode</button>
          <div>
            <h1>{currentMode.icon} {currentMode.title}</h1>
            <p>{currentMode.subtitle} — {currentMode.description.split('.')[0]}</p>
          </div>
        </div>
        <div className="mode-accuracy-pill"
          style={{ background: currentMode.accuracyColor + '20', color: currentMode.accuracyColor, border: `1px solid ${currentMode.accuracyColor}` }}>
          {currentMode.accuracy} Accuracy Mode
        </div>
      </div>

      <div className="diagnosis-container">
        {/* Tabs */}
        <div className="tabs">
          {tabs.map(tabId => {
            const META = {
              symptoms: { icon: '📋', label: 'Symptoms & History' },
              labs:     { icon: '🧪', label: 'Lab Reports' },
              imaging:  { icon: '🖼️', label: 'Medical Imaging' },
            };
            const isDone = tabs.indexOf(tabId) < tabs.indexOf(activeTab);
            return (
              <button key={tabId} type="button"
                className={`tab ${activeTab === tabId ? 'active' : ''} ${isDone ? 'done' : ''}`}
                onClick={() => setActiveTab(tabId)}>
                <span className="tab-icon">{isDone ? '✓' : META[tabId].icon}</span>
                <span className="tab-label">{META[tabId].label}</span>
              </button>
            );
          })}
        </div>

        <form onSubmit={handleSubmit} className="diagnosis-form">
          {error && <div className="error-message">⚠ {error}</div>}

          {/* ══════════ SYMPTOMS TAB ══════════════════════════════════ */}
          {activeTab === 'symptoms' && (
            <div className="form-section">
              <div className="section-header">
                <h3>Patient Demographics & Symptoms</h3>
                <p className="section-hint">Based on the Mayo Clinic PBC trial dataset (cirrhosis.csv)</p>
              </div>

              {/* Demographics */}
              <div className="form-row">
                <div className="form-group">
                  <label>Age <span className="required">*</span></label>
                  <input type="number" name="age" value={symptoms.age} onChange={handleChange}
                    placeholder="e.g. 50" required min="1" max="120" />
                </div>
                <div className="form-group">
                  <label>Biological Sex <span className="hint-text">(89% female in dataset)</span></label>
                  <select name="sex" value={symptoms.sex} onChange={handleChange}>
                    <option value={0}>Female</option>
                    <option value={1}>Male</option>
                  </select>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>Fatigue Level <span className="hint-text">(1 = none, 10 = severe)</span></label>
                  <div className="slider-row">
                    <input type="range" name="fatigue_level" value={symptoms.fatigue_level}
                      onChange={handleChange} min="1" max="10" step="1" />
                    <span className="range-badge">{symptoms.fatigue_level}/10</span>
                  </div>
                </div>
                <div className="form-group">
                  <label>Alcohol Consumption <span className="hint-text">(units/week)</span></label>
                  <input type="number" name="alcohol_consumption" value={symptoms.alcohol_consumption}
                    onChange={handleChange} min="0" max="100" step="0.5" placeholder="0" />
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>Recent Weight Loss <span className="hint-text">(kg, last 3 months)</span></label>
                  <input type="number" name="weight_loss_kg" value={symptoms.weight_loss_kg}
                    onChange={handleChange} min="0" max="30" step="0.1" placeholder="0" />
                </div>
              </div>

              {/* Symptom checklist */}
              <div className="form-subsection">
                <h4>Symptom Checklist</h4>
                <div className="checkbox-grid">
                  {[
                    { name: 'abdominal_swelling', label: 'Abdominal Swelling',   icon: '🫁' },
                    { name: 'appetite_loss',       label: 'Loss of Appetite',     icon: '🍽️' },
                    { name: 'jaundice',            label: 'Jaundice (yellowing)', icon: '🟡' },
                    { name: 'fever',               label: 'Persistent Fever',     icon: '🌡️' },
                  ].map(({ name, label, icon }) => (
                    <label key={name} className={`checkbox-card ${symptoms[name] ? 'checked' : ''}`}>
                      <input type="checkbox" name={name} checked={symptoms[name]} onChange={handleChange} />
                      <span className="checkbox-icon">{icon}</span>
                      <span>{label}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Clinical findings — Y/N as per dataset */}
              <div className="form-subsection">
                <h4>Clinical Examination Findings
                  <span className="hint-text"> (as recorded in the PBC trial — present or absent)</span>
                </h4>
                <div className="checkbox-grid">
                  {[
                    { name: 'ascites',      label: 'Ascites (abdominal fluid)',  icon: '💧' },
                    { name: 'hepatomegaly', label: 'Hepatomegaly (enlarged liver)', icon: '🫀' },
                    { name: 'spiders',      label: 'Spider Angiomas',            icon: '🕸️' },
                  ].map(({ name, label, icon }) => (
                    <label key={name} className={`checkbox-card ${symptoms[name] ? 'checked' : ''}`}>
                      <input type="checkbox" name={name} checked={symptoms[name]} onChange={handleChange} />
                      <span className="checkbox-icon">{icon}</span>
                      <span>{label}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Edema — 3-level as per dataset (N / S / Y) */}
              <div className="form-subsection">
                <h4>Peripheral Edema
                  <span className="hint-text"> (N = none, S = diuretic-controlled, Y = overt)</span>
                </h4>
                <div className="edema-selector">
                  {[
                    { value: 0, label: 'None (N)',              desc: 'No peripheral edema' },
                    { value: 1, label: 'Controlled (S)',        desc: 'Resolved with diuretics' },
                    { value: 2, label: 'Overt (Y)',             desc: 'Present without diuretics' },
                  ].map(opt => (
                    <button key={opt.value} type="button"
                      className={`edema-btn ${symptoms.edema === opt.value ? 'active' : ''}`}
                      onClick={() => setSymptoms(prev => ({ ...prev, edema: opt.value }))}>
                      <span className="edema-label">{opt.label}</span>
                      <span className="edema-desc">{opt.desc}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="tab-nav-btns">
                {tabs.indexOf('symptoms') < tabs.length - 1 && (
                  <button type="button" className="btn-next"
                    onClick={() => setActiveTab(tabs[tabs.indexOf('symptoms') + 1])}>
                    Next: {tabs[1] === 'labs' ? 'Lab Reports' : 'Medical Imaging'} →
                  </button>
                )}
              </div>
            </div>
          )}

          {/* ══════════ LABS TAB ══════════════════════════════════════ */}
          {activeTab === 'labs' && (
            <div className="form-section">
              <div className="section-header">
                <h3>Laboratory Results</h3>
                <p className="section-hint">
                  Ranges match the Mayo Clinic PBC trial dataset. Upload a lab report image for auto-fill.
                </p>
              </div>

              {/* Platelets note */}
              <div className="platelets-note">
                <span className="info-icon">💡</span>
                <span>
                  <strong>Platelets:</strong> Enter in ×10³/µL — e.g. enter <strong>251</strong> for 251,000/µL.
                  PBC patients often show elevated Alk. Phosphatase and Cholesterol — values above the healthy
                  reference range are expected and normal for this disease.
                </span>
              </div>

              {/* Upload / Manual toggle */}
              <div className="lab-mode-toggle">
                <button type="button" className={`mode-btn ${labMode === 'upload' ? 'active' : ''}`}
                  onClick={() => setLabMode('upload')}>📷 Upload Lab Report</button>
                <button type="button" className={`mode-btn ${labMode === 'manual' ? 'active' : ''}`}
                  onClick={() => setLabMode('manual')}>✏️ Fill Manually</button>
              </div>

              {/* Upload mode */}
              {labMode === 'upload' && (
                <div className="lab-upload-section">
                  <div className="lab-upload-info">
                    <span className="info-icon">ℹ️</span>
                    <p>Upload a clear photo or scan of your liver function / blood test report.
                      Values are auto-extracted. You can review and correct them before submitting.</p>
                  </div>
                  {!labReportPreview ? (
                    <div className="lab-drop-zone"
                      onClick={() => labInputRef.current?.click()}
                      onDragOver={e => e.preventDefault()}
                      onDrop={e => {
                        e.preventDefault();
                        const f = e.dataTransfer.files[0];
                        if (f) {
                          setLabReportFile(f); setOcrResult(null);
                          const r = new FileReader();
                          r.onload = ev => setLabReportPreview(ev.target.result);
                          r.readAsDataURL(f);
                        }
                      }}>
                      <input ref={labInputRef} type="file" accept="image/*"
                        style={{ display: 'none' }} onChange={handleLabReportChange} />
                      <div className="drop-icon">📋</div>
                      <p><strong>Click to upload</strong> or drag &amp; drop</p>
                      <p className="drop-hint">JPG, PNG, BMP — max 15 MB</p>
                    </div>
                  ) : (
                    <div className="lab-preview-area">
                      <div className="lab-preview-img-wrap">
                        <img src={labReportPreview} alt="Lab report" className="lab-report-img" />
                      </div>
                      <div className="lab-preview-actions">
                        <span className="file-name-chip">📎 {labReportFile?.name}</span>
                        <button type="button" className="btn-remove-lab" onClick={handleRemoveLabReport}>✕ Remove</button>
                      </div>
                    </div>
                  )}
                  {labReportFile && !ocrLoading && (
                    <button type="button" className="btn-extract" onClick={handleExtractLabValues}>
                      🔍 Extract Lab Values from Image
                    </button>
                  )}
                  {ocrLoading && (
                    <div className="ocr-loading">
                      <div className="ocr-spinner"></div>
                      <span>Reading lab report…</span>
                    </div>
                  )}
                  {ocrResult && (
                    <div className={`ocr-result ocr-${ocrResult.status}`}>
                      {ocrResult.status === 'success' ? (
                        <>
                          <div className="ocr-result-title">✅ {Object.keys(ocrResult.extracted).length} values extracted</div>
                          <div className="ocr-chips">
                            {Object.entries(ocrResult.extracted).map(([k, v]) => {
                              const f = LAB_FIELDS.find(x => x.name === k);
                              return <span key={k} className="ocr-chip extracted">{f?.label || k}: <strong>{v}</strong> {f?.unit}</span>;
                            })}
                          </div>
                          {ocrResult.missing?.length > 0 && (
                            <p className="ocr-missing">⚠️ Could not read: {ocrResult.missing.map(m => LAB_FIELDS.find(x => x.name === m)?.label || m).join(', ')} — fill manually.</p>
                          )}
                        </>
                      ) : (
                        <div className="ocr-result-title">⚠️ {ocrResult.message}</div>
                      )}
                    </div>
                  )}
                  {(ocrResult?.status === 'success' || ocrResult?.status === 'empty' || ocrResult?.status === 'error') && (
                    <div className="ocr-review-label">📝 Review &amp; correct extracted values below:</div>
                  )}
                </div>
              )}

              {/* Lab fields grid */}
              <div className={`lab-fields-section ${labMode === 'upload' && !ocrResult ? 'lab-fields-hidden' : ''}`}>
                <div className="lab-grid">
                  {LAB_FIELDS.map(({ name, label, unit, min, max, step, normal }) => {
                    const wasExtracted = ocrResult?.status === 'success' && ocrResult.extracted?.[name] !== undefined;
                    return (
                      <div key={name} className={`form-group lab-group ${wasExtracted ? 'ocr-filled' : ''}`}>
                        <label>
                          {label} <span className="unit-badge">{unit}</span>
                          {wasExtracted && <span className="ocr-badge">OCR</span>}
                        </label>
                        <input type="number" name={name} value={symptoms[name]}
                          onChange={handleChange} min={min} max={max} step={step}
                          placeholder={`e.g. ${DEFAULT_SYMPTOMS[name]}`} />
                        <span className="normal-range">Healthy ref: {normal} {unit}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="tab-nav-btns">
                <button type="button" className="btn-back" onClick={() => setActiveTab('symptoms')}>← Back</button>
                {tabs.includes('imaging') && (
                  <button type="button" className="btn-next" onClick={() => setActiveTab('imaging')}>
                    Next: Medical Imaging →
                  </button>
                )}
              </div>
            </div>
          )}

          {/* ══════════ IMAGING TAB ═══════════════════════════════════ */}
          {activeTab === 'imaging' && (
            <div className="form-section">
              <div className="section-header">
                <h3>Medical Imaging <span className="optional-badge">Optional</span></h3>
                <p className="section-hint">
                  Upload a liver ultrasound, CT, or MRI scan. Trained on the 7272660 dataset
                  (Normal / Benign / Malignant, 735 liver ultrasound images).
                </p>
              </div>
              <div className={`image-upload-zone ${imagePreview ? 'has-image' : ''}`}>
                {!imagePreview ? (
                  <>
                    <input type="file" accept="image/*" onChange={handleImageChange} id="image-input" />
                    <label htmlFor="image-input" className="upload-label">
                      <div className="upload-icon">📁</div>
                      <div className="upload-text"><strong>Click to upload</strong> or drag and drop</div>
                      <div className="upload-hint">JPG, PNG, BMP — up to 10 MB</div>
                    </label>
                  </>
                ) : (
                  <div className="image-preview">
                    <img src={imagePreview} alt="Scan" />
                    <div className="image-info">
                      <span>📎 {imageFile?.name}</span>
                      <button type="button" className="btn-remove-img"
                        onClick={() => { setImageFile(null); setImagePreview(null); }}>✕ Remove</button>
                    </div>
                  </div>
                )}
              </div>
              <div className="imaging-note">
                <div className="note-icon">ℹ️</div>
                <p>Image classifier trained on 735 liver ultrasound images — Normal / Benign / Malignant.
                  Prediction still works without an image; imaging adds 25% weight to the final score.</p>
              </div>
              <div className="tab-nav-btns">
                <button type="button" className="btn-back" onClick={() => setActiveTab('labs')}>← Back</button>
              </div>
            </div>
          )}

          {/* Submit */}
          <div className="form-actions">
            <div className="submit-info">
              Mode: <strong>{currentMode?.title}</strong>
              <span className="submit-mode-badge"
                style={{ background: currentMode?.badgeColor + '20', color: currentMode?.badgeColor }}>
                {currentMode?.accuracy} Accuracy
              </span>
            </div>
            <button type="submit"
              className={`btn-submit btn-primary ${loading ? 'loading' : ''}`}
              disabled={loading}>
              {loading ? <><span className="spinner-sm"></span> Analyzing…</> : '🔍 Run Diagnosis'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default DiagnosisPage;
