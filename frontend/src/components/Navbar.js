import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Activity, LogOut, Menu, MessageCircle } from 'lucide-react';
import '../styles/Navbar.css';

const Navbar = ({ toggleSidebar, isAuthenticated, userRole, userData, onLogout }) => {
  const navigate = useNavigate();

  const handleLogout = () => {
    onLogout();
    navigate('/');
  };

  const roleLabel = {
    patient: '🧑 Patient',
    doctor: '🩺 Doctor',
    admin: '⚙️ Admin',
  };

  return (
    <nav className="navbar">
      <div className="navbar-left">
        {isAuthenticated && (
          <button className="sidebar-toggle" onClick={toggleSidebar} aria-label="Toggle sidebar">
            <Menu size={20} />
          </button>
        )}
        <Link to={isAuthenticated ? '/diagnosis' : '/'} className="navbar-brand">
          <span className="brand-icon"><Activity size={21} /></span>
          <span className="brand-text">LiverAI</span>
        </Link>
      </div>

      <div className="navbar-center">
        <span className="navbar-title">AI-Powered Liver Cirrhosis Detection</span>
      </div>

      <div className="navbar-right">
        {isAuthenticated ? (
          <>
            <Link to="/chatbot" className="nav-link"><MessageCircle size={17} /> Chatbot</Link>
            {userData && (
              <div className="user-chip">
                <span className="user-name">{userData.full_name || userData.email}</span>
                <span className="user-role">{roleLabel[userRole] || userRole}</span>
              </div>
            )}
            <button className="nav-logout" onClick={handleLogout}><LogOut size={16} /> Logout</button>
          </>
        ) : (
          <Link to="/" className="nav-link">Login</Link>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
