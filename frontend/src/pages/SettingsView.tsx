import React, { useState } from 'react';
import { Bell, Moon, Sun, Lock, Shield, Database } from 'lucide-react';
import './Settings.css';

export function SettingsView() {
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [alertsEnabled, setAlertsEnabled] = useState(true);
  const [autoRebalance, setAutoRebalance] = useState(false);

  return (
    <div className="settings-page">
      <div className="page-header">
        <div className="header-title">
          <h1>System Settings</h1>
          <p>Configure preferences and manage application settings.</p>
        </div>
      </div>

      <div className="settings-grid">
        <div className="settings-section glass-card">
          <div className="settings-section-header">
            <h3>Preferences</h3>
          </div>
          
          <div className="settings-list">
            <div className="setting-item">
              <div className="setting-info">
                <div className="setting-icon">
                  {isDarkMode ? <Moon size={20} /> : <Sun size={20} />}
                </div>
                <div>
                  <h4>Appearance</h4>
                  <p>Toggle dark or light mode</p>
                </div>
              </div>
              <label className="switch">
                <input type="checkbox" checked={isDarkMode} onChange={() => setIsDarkMode(!isDarkMode)} />
                <span className="slider round"></span>
              </label>
            </div>

            <div className="setting-item">
              <div className="setting-info">
                <div className="setting-icon"><Bell size={20} /></div>
                <div>
                  <h4>Push Notifications</h4>
                  <p>Receive alerts for capacity limits</p>
                </div>
              </div>
              <label className="switch">
                <input type="checkbox" checked={alertsEnabled} onChange={() => setAlertsEnabled(!alertsEnabled)} />
                <span className="slider round"></span>
              </label>
            </div>
          </div>
        </div>

        <div className="settings-section glass-card">
          <div className="settings-section-header">
            <h3>Engine Configuration</h3>
          </div>
          
          <div className="settings-list">
            <div className="setting-item">
              <div className="setting-info">
                <div className="setting-icon"><Database size={20} /></div>
                <div>
                  <h4>Auto-Rebalance</h4>
                  <p>Allow system to auto-approve safe optimizations</p>
                </div>
              </div>
              <label className="switch">
                <input type="checkbox" checked={autoRebalance} onChange={() => setAutoRebalance(!autoRebalance)} />
                <span className="slider round"></span>
              </label>
            </div>
            
            <div className="setting-item">
              <div className="setting-info">
                <div className="setting-icon"><Shield size={20} /></div>
                <div>
                  <h4>Strict Constraints</h4>
                  <p>Enforce hard regulatory constraints only</p>
                </div>
              </div>
              <label className="switch">
                <input type="checkbox" checked={true} readOnly />
                <span className="slider round"></span>
              </label>
            </div>
          </div>
        </div>
      </div>
      
      <div className="settings-footer">
        <button className="btn-secondary">
          <Lock size={16} style={{marginRight: '8px'}} /> Change Password
        </button>
      </div>
    </div>
  );
}
