import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import HomePage from './pages/HomePage';
import DiagnosisPage from './pages/DiagnosisPage';
import ResultsPage from './pages/ResultsPage';
import HistoryPage from './pages/HistoryPage';
import ReportsPage from './pages/ReportsPage';
import DashboardPage from './pages/DashboardPage';
import ChatbotPage from './pages/ChatbotPage';
import PatientsPage from './pages/PatientsPage';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [userAuthenticated, setUserAuthenticated] = useState(false);
  const [userRole, setUserRole] = useState('patient');
  const [userData, setUserData] = useState(null);
  const [lastPrediction, setLastPrediction] = useState(null);

  const toggleSidebar = () => setSidebarOpen(prev => !prev);

  useEffect(() => {
    const savedUser = localStorage.getItem('userData');
    if (savedUser) {
      try {
        const user = JSON.parse(savedUser);
        setUserData(user);
        setUserAuthenticated(true);
        setUserRole(user.role || 'patient');
      } catch {
        localStorage.removeItem('userData');
      }
    }
  }, []);

  const handleLogin = (user) => {
    setUserData(user);
    setUserAuthenticated(true);
    setUserRole(user.role || 'patient');
    localStorage.setItem('userData', JSON.stringify(user));
  };

  const handleLogout = () => {
    setUserAuthenticated(false);
    setUserData(null);
    setUserRole('patient');
    localStorage.removeItem('userData');
    localStorage.removeItem('auth_token');
  };

  const sidebarClass = sidebarOpen && userAuthenticated ? 'sidebar-open' : 'sidebar-closed';

  return (
    <Router>
      <div className="App">
        <Navbar
          toggleSidebar={toggleSidebar}
          isAuthenticated={userAuthenticated}
          userRole={userRole}
          userData={userData}
          onLogout={handleLogout}
        />
        <div className="app-container">
          {userAuthenticated && (
            <Sidebar isOpen={sidebarOpen} userRole={userRole} />
          )}
          <main className={`main-content ${sidebarClass}`}>
            <Routes>
              {/* Public */}
              <Route path="/" element={<HomePage onLogin={handleLogin} />} />
              <Route path="/chatbot" element={<ChatbotPage />} />

              {/* Patient routes */}
              <Route
                path="/diagnosis"
                element={userAuthenticated
                  ? <DiagnosisPage onPrediction={setLastPrediction} />
                  : <Navigate to="/" replace />}
              />
              <Route
                path="/results"
                element={userAuthenticated
                  ? <HistoryPage />
                  : <Navigate to="/" replace />}
              />
              <Route
                path="/results/:id"
                element={userAuthenticated
                  ? <ResultsPage prediction={lastPrediction} />
                  : <Navigate to="/" replace />}
              />
              <Route
                path="/reports"
                element={userAuthenticated
                  ? <ReportsPage />
                  : <Navigate to="/" replace />}
              />

              {/* Doctor/Admin routes */}
              <Route
                path="/dashboard"
                element={userAuthenticated && ['doctor', 'admin'].includes(userRole)
                  ? <DashboardPage />
                  : <Navigate to="/diagnosis" replace />}
              />
              <Route
                path="/patients"
                element={userAuthenticated && ['doctor', 'admin'].includes(userRole)
                  ? <PatientsPage />
                  : <Navigate to="/diagnosis" replace />}
              />

              {/* Fallback */}
              <Route path="*" element={<Navigate to={userAuthenticated ? '/diagnosis' : '/'} replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;
