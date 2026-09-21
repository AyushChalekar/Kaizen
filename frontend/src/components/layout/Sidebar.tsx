import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, GitMerge, Activity, Calendar, Users, Settings, FileSearch } from 'lucide-react';
import './Sidebar.css';

export function Sidebar() {
  const navItems = [
    { name: 'Overview', path: '/app/dashboard', icon: LayoutDashboard },
    { name: 'Simulation', path: '/app/simulation', icon: GitMerge },
    { name: 'Optimization', path: '/app/optimization', icon: Activity },
    { name: 'Explainability (XAI)', path: '/app/explainability', icon: FileSearch },
    { name: 'Forecast', path: '/app/forecast', icon: Calendar },
    { name: 'Queue & Staff', path: '/app/queue', icon: Users },
    { name: 'Settings', path: '/app/settings', icon: Settings },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="logo-placeholder-small">P</div>
        <h2>PROSH</h2>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.name}
              to={item.path}
              className={({ isActive }) =>
                `nav-item ${isActive ? 'active' : ''}`
              }
            >
              <Icon size={20} className="nav-icon" />
              <span>{item.name}</span>
            </NavLink>
          );
        })}
      </nav>
      
      <div className="sidebar-footer">
        <div className="user-profile">
          <div className="avatar">A</div>
          <div className="user-info">
            <span className="user-name">Admin User</span>
            <span className="user-role">System Admin</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
