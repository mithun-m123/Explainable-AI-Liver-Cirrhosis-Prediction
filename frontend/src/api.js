import axios from 'axios';

const API_BASE_URL = (process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
  headers: { Accept: 'application/json' },
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  // Do not set Content-Type globally: the browser/Axios must set multipart boundaries for FormData.
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const url = String(error.config?.url || '');
    const isAuthRequest = url.startsWith('/auth/login') || url.startsWith('/auth/register');
    if (error.response?.status === 401 && !isAuthRequest) {
      localStorage.removeItem('auth_token');
      localStorage.removeItem('userData');
      window.dispatchEvent(new Event('liverai:auth-expired'));
    }
    return Promise.reject(error);
  }
);

export function getErrorMessage(error) {
  const detail = error?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg || item.message || JSON.stringify(item)).join('; ');
  if (error?.response?.data?.message) return error.response.data.message;
  if (error?.code === 'ECONNABORTED') return 'The request timed out. Check that the backend is running and try again.';
  if (error?.request && !error?.response) return `Cannot reach the backend at ${API_BASE_URL}. Start FastAPI and verify REACT_APP_API_URL.`;
  return error?.message || 'An unexpected error occurred.';
}

export const authAPI = {
  async login(credentials) { return (await api.post('/auth/login', credentials)).data; },
  async register(user) { return (await api.post('/auth/register', user)).data; },
  async me() { return (await api.get('/auth/me')).data; },
};

export const diagnosisAPI = {
  async predict({ symptoms, assessmentMode, imageFile }) {
    const form = new FormData();
    form.append('symptoms', JSON.stringify(symptoms));
    form.append('assessment_mode', assessmentMode);
    if (imageFile) form.append('image', imageFile);
    return (await api.post('/predict', form)).data;
  },
  async extractLabs(imageFile) {
    const form = new FormData();
    form.append('image', imageFile);
    return (await api.post('/labs/extract-from-image', form)).data;
  },
  async history({ skip = 0, limit = 10 } = {}) {
    return (await api.get('/history', { params: { skip, limit } })).data;
  },
  async result(id) { return (await api.get(`/diagnosis/results/${encodeURIComponent(id)}`)).data; },
};

export const reportsAPI = {
  async download(id, filename = `liver_diagnosis_${id}.pdf`) {
    const response = await api.get(`/reports/${encodeURIComponent(id)}`, { responseType: 'blob' });
    const blobUrl = window.URL.createObjectURL(response.data);
    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => window.URL.revokeObjectURL(blobUrl), 1000);
  },
};

export const analyticsAPI = {
  async userStats() { return (await api.get('/analytics/user-stats')).data; },
  async trends(days = 30) { return (await api.get('/analytics/trends', { params: { days } })).data; },
  async riskDistribution() { return (await api.get('/analytics/risk-distribution')).data; },
};

export const patientsAPI = {
  async list() { return (await api.get('/patients')).data; },
  async add(data) { return (await api.post('/patients/add', data)).data; },
  async deactivate(id) { return (await api.post(`/patients/${encodeURIComponent(id)}/deactivate`)).data; },
};

export const chatAPI = {
  async start() { return (await api.post('/chat/start', { message: 'start', session_type: 'medical_qa', context: {} })).data; },
  async send(sessionId, message) { return (await api.post('/chat/message', { session_id: String(sessionId), message })).data; },
  async history(sessionId) { return (await api.get(`/chat/history/${encodeURIComponent(sessionId)}`)).data; },
};

export default api;