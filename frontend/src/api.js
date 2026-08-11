import axios from 'axios';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

// Create axios instance with default config
const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000, // 30 seconds timeout
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor for auth
api.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('auth_token');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// Response interceptor for error handling
api.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401) {
            // Token expired or invalid
            localStorage.removeItem('auth_token');
            localStorage.removeItem('userData');
            window.location.href = '/';
        }
        return Promise.reject(error);
    }
);

// Auth API
export const authAPI = {
    register: async (userData) => {
        const response = await api.post('/auth/register', userData);
        return response.data;
    },

    login: async (credentials) => {
        const response = await api.post('/auth/login', credentials);
        const { access_token } = response.data;
        if (access_token) {
            localStorage.setItem('auth_token', access_token);
        }
        return response.data;
    },

    logout: () => {
        localStorage.removeItem('auth_token');
    },

    getCurrentUser: async () => {
        const response = await api.get('/auth/me');
        return response.data;
    },

    updateProfile: async (userData) => {
        const response = await api.put('/auth/profile', userData);
        return response.data;
    },
};

// Diagnosis API
export const diagnosisAPI = {
    performDiagnosis: async (diagnosisData) => {
        const formData = new FormData();

        // Add user ID
        formData.append("user_id", diagnosisData.userId);

        // Add symptoms as JSON string
        formData.append("symptoms", JSON.stringify(diagnosisData.symptoms));

        // Add image if provided
        if (diagnosisData.imageFile) {
            formData.append("image", diagnosisData.imageFile);
        }

        const response = await api.post('/predict', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    getDiagnosisHistory: async (page = 1, limit = 10) => {
        const skip = (page - 1) * limit;
        const response = await api.get('/history', {
            params: { skip, limit }
        });
        return response.data;
    },

    getDiagnosisById: async (diagnosisId) => {
        const response = await api.get(`/diagnoses/${diagnosisId}`);
        return response.data;
    },

    deleteDiagnosis: async (diagnosisId) => {
        const response = await api.delete(`/diagnoses/${diagnosisId}`);
        return response.data;
    },
};

// Chatbot API
export const chatbotAPI = {
    startChatSession: async (userId) => {
        const response = await api.post('/chat/start', {
            message: 'start',
            session_type: 'medical_qa',
            context: { user_id: userId }
        });
        return response.data;
    },

    sendMessage: async (sessionId, message) => {
        const response = await api.post('/chat/message', {
            session_id: sessionId,
            message: message
        });
        return response.data;
    },

    getChatHistory: async (sessionId) => {
        const response = await api.get(`/chat/history/${sessionId}`);
        return response.data;
    },

    getChatSessions: async (userId) => {
        const response = await api.get('/chat/sessions', {
            params: { user_id: userId }
        });
        return response.data;
    },
};

// Patients API (doctor tools)
export const patientsAPI = {
    list: async () => {
        const response = await api.get('/patients');
        return response.data;
    },
    add: async (patientData) => {
        const response = await api.post('/patients/add', patientData);
        return response.data;
    },
};

// Deactivate (remove active) a patient
patientsAPI.deactivate = async (patientId) => {
    const response = await api.post(`/patients/${patientId}/deactivate`);
    return response.data;
};

// Reports API
export const reportsAPI = {
    generateReport: async (diagnosisId) => {
        const response = await api.post(`/reports/generate/${diagnosisId}`, {}, {
            responseType: 'blob' // For file download
        });
        return response.data;
    },

    downloadReport: async (diagnosisId, filename = 'liver_report.pdf') => {
        const response = await api.get(`/reports/download/${diagnosisId}`, {
            responseType: 'blob'
        });

        // Create download link
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', filename);
        document.body.appendChild(link);
        link.click();
        link.remove();
    },

    generateSummaryReport: async (userId, startDate, endDate) => {
        const response = await api.post('/reports/summary', {
            user_id: userId,
            start_date: startDate,
            end_date: endDate
        }, {
            responseType: 'blob'
        });
        return response.data;
    },
};

// Analytics API
export const analyticsAPI = {
    getUserStats: async (userId) => {
        const response = await api.get('/analytics/user-stats', {
            params: { user_id: userId }
        });
        return response.data;
    },

    getDiagnosisTrends: async (userId, days = 30) => {
        const response = await api.get('/analytics/trends', {
            params: { user_id: userId, days }
        });
        return response.data;
    },

    getRiskDistribution: async () => {
        const response = await api.get('/analytics/risk-distribution');
        return response.data;
    },
};

// Health Tips API
export const healthTipsAPI = {
    getTips: async (category = null) => {
        const response = await api.get('/health-tips', {
            params: { category }
        });
        return response.data;
    },

    getTipById: async (tipId) => {
        const response = await api.get(`/health-tips/${tipId}`);
        return response.data;
    },

    markTipRead: async (tipId) => {
        const response = await api.post(`/health-tips/${tipId}/read`);
        return response.data;
    },
};

// Utility functions
export const utils = {
    handleApiError: (error) => {
        if (error.response) {
            // Server responded with error status
            const { status, data } = error.response;
            switch (status) {
                case 400:
                    return data.detail || 'Bad request. Please check your input.';
                case 401:
                    return 'Authentication required. Please log in.';
                case 403:
                    return 'Access denied. You do not have permission.';
                case 404:
                    return 'Resource not found.';
                case 422:
                    return 'Validation error. Please check your input data.';
                case 500:
                    return 'Server error. Please try again later.';
                default:
                    return data.detail || 'An unexpected error occurred.';
            }
        } else if (error.request) {
            // Network error
            return 'Network error. Please check your connection.';
        } else {
            // Other error
            return error.message || 'An unexpected error occurred.';
        }
    },

    formatApiResponse: (response) => {
        return {
            success: true,
            data: response.data,
            message: response.data.message || 'Success'
        };
    },

    isAuthenticated: () => {
        return !!localStorage.getItem('auth_token');
    },

    getAuthToken: () => {
        return localStorage.getItem('auth_token');
    }
};

// Legacy function for backward compatibility
export const performDiagnosis = async (userId, symptoms, imageFile) => {
    return diagnosisAPI.performDiagnosis({
        userId,
        symptoms,
        imageFile
    });
};

export default api;