import React, { useState } from 'react';
import { Play, Pause, FastForward, Rewind, Activity, Ambulance, Stethoscope, Bed, HeartPulse, LogOut } from 'lucide-react';
import './Simulation.css';

export function SimulationView() {
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);

  const togglePlay = () => setIsPlaying(!isPlaying);
  
  const handleSpeed = (newSpeed: number) => {
    setSpeed(newSpeed);
  };

  return (
    <div className="simulation-page">
      <div className="page-header">
        <div className="header-title">
          <h1>Digital Twin Simulation</h1>
          <p>Live animated view of patient flow and state transitions.</p>
        </div>
        
        {/* Controls */}
        <div className="sim-controls glass-card">
          <button className="control-btn" onClick={() => handleSpeed(0.5)} title="Slow">
            <Rewind size={18} />
          </button>
          
          <button className={`control-btn play-pause ${isPlaying ? 'playing' : ''}`} onClick={togglePlay}>
            {isPlaying ? <Pause size={24} /> : <Play size={24} />}
          </button>
          
          <button className="control-btn" onClick={() => handleSpeed(2)} title="Fast">
            <FastForward size={18} />
          </button>
          
          <div className="speed-indicator">{speed}x Speed</div>
        </div>
      </div>

      <div className="sim-canvas glass-card">
        <div className={`pipeline-container ${isPlaying ? 'animating' : 'paused'}`} style={{ '--sim-speed': `${1 / speed}s` } as React.CSSProperties}>
          
          {/* Nodes */}
          <div className="node node-arrival">
            <div className="node-icon bg-blue"><Ambulance size={24} /></div>
            <span>Arrivals</span>
            <div className="node-stats">12 / hr</div>
          </div>

          <div className="node node-triage">
            <div className="node-icon bg-teal"><Stethoscope size={24} /></div>
            <span>Triage</span>
            <div className="node-stats">Wait: 15m</div>
          </div>

          <div className="node node-ward">
            <div className="node-icon bg-amber"><Bed size={24} /></div>
            <span>General Ward</span>
            <div className="node-stats">198 Beds</div>
          </div>

          <div className="node node-icu">
            <div className="node-icon bg-red"><HeartPulse size={24} /></div>
            <span>ICU</span>
            <div className="node-stats">19 Beds</div>
          </div>

          <div className="node node-discharge">
            <div className="node-icon bg-green"><LogOut size={24} /></div>
            <span>Discharge</span>
            <div className="node-stats">8 / hr</div>
          </div>

          {/* Connectors & Animated Particles */}
          <div className="path path-arrival-triage">
            <div className="particle"></div>
            <div className="particle delay-1"></div>
          </div>
          
          <div className="path path-triage-ward">
            <div className="particle"></div>
          </div>
          
          <div className="path path-triage-icu">
            <div className="particle delay-2 alert"></div>
          </div>

          <div className="path path-ward-discharge">
            <div className="particle delay-1"></div>
          </div>

          <div className="path path-icu-ward">
            <div className="particle delay-2"></div>
          </div>

        </div>
      </div>
      
      <div className="sim-details">
        <div className="glass-card detail-panel">
          <h3>Simulation Parameters</h3>
          <ul className="sim-params">
            <li><strong>Engine Mode:</strong> Discrete Event Simulation (SimPy)</li>
            <li><strong>Time Horizon:</strong> Next 24 Hours</li>
            <li><strong>Random Seed:</strong> 42 (Deterministic Mode)</li>
            <li><strong>Current Status:</strong> {isPlaying ? <span className="status-live">Running</span> : <span className="status-paused">Paused</span>}</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
