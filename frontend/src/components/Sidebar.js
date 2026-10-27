import React from 'react';
import { NavLink } from 'react-router-dom';
import '../styles/Sidebar.css';

const Sidebar = ({ isOpen, userRole }) => {
  const isPrivileged = ['doctor', 'admin'].includes(userRole);

  return (
    <aside className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
      <div className="sidebar-content">
        <div className="sidebar-section">
          <h3>Diagnosis</h3>
          <NavLink to="/diagnosis" className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}>
            🔍 New Diagnosis
          </NavLink>
          <NavLink to="/results" className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}>
            📊 My Results
          </NavLink>
          <NavLink to="/reports" className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}>
            📄 Download Reports
          </NavLink>
        </div>

        {isPrivileged && (
          <div className="sidebar-section">
            <h3>{userRole === 'admin' ? 'Admin' : 'Doctor'} Tools</h3>
            <NavLink to="/dashboard" className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}>
              📈 Dashboard
            </NavLink>
            <NavLink to="/patients" className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}>
              👥 Patients
            </NavLink>
          </div>
        )}

        <div className="sidebar-section">
          <h3>Support</h3>
          <NavLink to="/chatbot" className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}>
            💬 AI Chatbot
          </NavLink>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
