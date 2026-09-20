import React, { useState } from 'react';
import { ShieldAlert, Users, BedDouble, AlertTriangle, ChevronDown, ChevronRight, Activity } from 'lucide-react';
import './Explainability.css';

const reasoningLogs = [
  {
    id: 1,
    time: '14:30',
    type: 'staffing',
    title: 'Shifted 2 Ward Nurses to ICU',
    icon: Users,
    rationale: 'ICU occupancy is predicted to reach 95% (critical threshold) at 15:00 based on current ED severe admissions.',
    activeConstraints: [
      'ICU Nurse-to-Patient Ratio (1:1 minimum)',
      'Ward Minimum Safe Staffing Level (Maintained at 1:5)'
    ]
  },
  {
    id: 2,
    time: '14:32',
    type: 'capacity',
    title: 'Converted 10 General Beds to HDU',
    icon: BedDouble,
    rationale: 'High influx of post-op patients expected over the next 4 hours from scheduled surgeries, requiring higher monitoring.',
    activeConstraints: [
      'Max Flex Bed Capacity (15 beds max)',
      'HDU Equipment Availability (Monitors confirmed)'
    ]
  },
  {
    id: 3,
    time: '14:35',
    type: 'alert',
    title: 'Deferred 1 Elective Admission',
    icon: ShieldAlert,
    rationale: 'ED queue time exceeded 45 minutes. System prioritizing acute incoming trauma patients over non-urgent electives.',
    activeConstraints: [
      'Hard Constraint: ED Wait Time < 60m',
      'Elective Deferral Policy (Allowed up to 24h)'
    ]
  }
];

export function ExplainabilityView() {
  const [expandedId, setExpandedId] = useState<number | null>(1);

  const toggleExpand = (id: number) => {
    setExpandedId(expandedId === id ? null : id);
  };

  return (
    <div className="xai-page">
      <div className="page-header">
        <div className="header-title">
          <h1>Explainability Log</h1>
          <p>Transparent AI reasoning for every optimization decision.</p>
        </div>
      </div>

      <div className="xai-timeline">
        {reasoningLogs.map((log) => {
          const Icon = log.icon;
          const isExpanded = expandedId === log.id;

          return (
            <div key={log.id} className={`xai-card glass-card ${isExpanded ? 'expanded' : ''}`}>
              <div className="xai-card-header" onClick={() => toggleExpand(log.id)}>
                <div className="xai-card-title-group">
                  <div className={`xai-icon ${log.type}`}>
                    <Icon size={20} />
                  </div>
                  <div>
                    <span className="xai-time">{log.time}</span>
                    <h3 className="xai-title">{log.title}</h3>
                  </div>
                </div>
                <button className="expand-btn">
                  {isExpanded ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
                </button>
              </div>
              
              {isExpanded && (
                <div className="xai-card-body">
                  <div className="xai-section">
                    <h4 className="xai-section-title">
                      <Activity size={16} /> Optimizer Rationale
                    </h4>
                    <p className="xai-text">{log.rationale}</p>
                  </div>
                  
                  <div className="xai-section">
                    <h4 className="xai-section-title">
                      <AlertTriangle size={16} /> Active Constraints
                    </h4>
                    <ul className="xai-constraints-list">
                      {log.activeConstraints.map((constraint, index) => (
                        <li key={index}>
                          <span className="constraint-dot"></span>
                          {constraint}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
