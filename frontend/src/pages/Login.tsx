import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Bubble } from '../components/ui/Bubble';
import { User, Lock } from 'lucide-react';
import './Login.css';

export function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [errorMsg, setErrorMsg] = useState('');
  const navigate = useNavigate();

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    // Hardcoded authentication
    if (username === 'admin' && password === 'admin123') {
      setErrorMsg('');
      navigate('/app/dashboard');
    } else {
      setErrorMsg('Invalid username or password. Use admin / admin123');
    }
  };

  return (
    <div className="login-page">
      <div className="login-background"></div>
      
      <Bubble className="login-bubble" variant="default">
        {errorMsg && (
          <Bubble variant="error" className="login-error-bubble" style={{ padding: '1rem', marginBottom: '1.5rem' }}>
            <p style={{ margin: 0, color: '#fca5a5', fontSize: '0.875rem', textAlign: 'center' }}>
              {errorMsg}
            </p>
          </Bubble>
        )}
        
        <div className="login-header">
          <div className="logo-placeholder">P</div>
          <h2>PROSH</h2>
          <p>Predictive Resource Optimization System</p>
        </div>

        <form onSubmit={handleLogin} className="login-form">
          <div className="input-group">
            <div className="input-icon">
              <User size={18} />
            </div>
            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>

          <div className="input-group">
            <div className="input-icon">
              <Lock size={18} />
            </div>
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <button type="submit" className="login-button">
            Sign In
          </button>
        </form>
      </Bubble>
    </div>
  );
}
