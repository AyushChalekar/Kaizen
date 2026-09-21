import React from 'react';
import { Clock, UserCircle, Activity } from 'lucide-react';
import './QueueStaff.css';

const activePatients = [
  { id: 'PT-8932', name: 'James Wilson', acuity: 'Urgent', complaint: 'Chest Pain', unit: 'Triage -> ED', waitTime: '12m' },
  { id: 'PT-8933', name: 'Sarah Connor', acuity: 'Acute', complaint: 'Fracture', unit: 'ED -> Ward', waitTime: '45m' },
  { id: 'PT-8934', name: 'Mike Ross', acuity: 'Non-Urgent', complaint: 'Laceration', unit: 'Triage', waitTime: '1h 15m' },
  { id: 'PT-8935', name: 'Elena Gilbert', acuity: 'Urgent', complaint: 'Shortness of Breath', unit: 'Triage -> ICU', waitTime: '8m' },
  { id: 'PT-8936', name: 'Bruce Wayne', acuity: 'Acute', complaint: 'Concussion', unit: 'ED -> HDU', waitTime: '32m' },
];

const staffRoster = [
  { dept: 'Emergency Dept', nurses: 16, doctors: 6, overtime: 2 },
  { dept: 'General Ward', nurses: 20, doctors: 8, overtime: 0 },
  { dept: 'ICU / HDU', nurses: 12, doctors: 4, overtime: 4 },
];

export function QueueStaffView() {
  return (
    <div className="queue-page">
      <div className="page-header">
        <div className="header-title">
          <h1>Queue & Staff Roster</h1>
          <p>Real-time patient waitlist and active staff assignment.</p>
        </div>
      </div>

      <div className="queue-grid">
        {/* Patient Queue Table */}
        <div className="queue-section glass-card">
          <div className="section-header">
            <h3>Active Patient Queue</h3>
            <span className="badge neutral">24 Waiting</span>
          </div>
          
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Patient ID</th>
                  <th>Acuity</th>
                  <th>Chief Complaint</th>
                  <th>Current Unit</th>
                  <th>Wait Time</th>
                </tr>
              </thead>
              <tbody>
                {activePatients.map((pt) => (
                  <tr key={pt.id}>
                    <td className="font-medium text-white">{pt.id}</td>
                    <td>
                      <span className={`acuity-badge ${pt.acuity.toLowerCase()}`}>
                        {pt.acuity}
                      </span>
                    </td>
                    <td>{pt.complaint}</td>
                    <td>{pt.unit}</td>
                    <td className="wait-time">
                      <Clock size={14} /> {pt.waitTime}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Staff Roster Sidebar */}
        <div className="roster-section">
          <div className="roster-card glass-card">
            <h3>Current Shift Roster</h3>
            <p className="roster-desc">Day Shift (08:00 - 16:00)</p>
            
            <div className="roster-list">
              {staffRoster.map((dept, idx) => (
                <div key={idx} className="roster-item">
                  <div className="roster-item-header">
                    <h4>{dept.dept}</h4>
                    {dept.overtime > 0 && (
                      <span className="overtime-badge" title="Staff on overtime">
                        {dept.overtime} OT
                      </span>
                    )}
                  </div>
                  <div className="roster-stats">
                    <div className="roster-stat">
                      <UserCircle size={16} className="text-blue" />
                      <span>{dept.nurses} Nurses</span>
                    </div>
                    <div className="roster-stat">
                      <Activity size={16} className="text-teal" />
                      <span>{dept.doctors} Doctors</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            <button className="btn-secondary roster-action-btn">
              Request Additional Cover
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
