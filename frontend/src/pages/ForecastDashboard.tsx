// frontend/src/pages/ForecastDashboard.tsx
/**
 * Real-time operational command center and predictive intelligence dashboard.
 * Consumes telemetry and machine learning inferences from useForecastData,
 * enforcing biological and temporal invariants across five clinical UI zones.
 */

import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  useForecastData,
  type SimulationDataResponse,
  type PatientStayContract,
  type HourlyCensusContract,
  type ShiftType,
  type CareUnitType,
} from "../api/useForecastData";

// ---------------------------------------------------------------------------
// Constants & Baselines
// ---------------------------------------------------------------------------
const MIN_PULSE_PRESSURE = 15.0;

const HISTORICAL_TARGETS = {
  average_los: 24.0,
  ed_los: 4.2,
  ward_los: 48.0,
  icu_los: 72.0,
};

const SHIFT_BASELINES: Record<ShiftType, { nurses: number; doctors: number }> = {
  Day: { nurses: 35, doctors: 10 },
  Evening: { nurses: 30, doctors: 8 },
  Night: { nurses: 20, doctors: 5 },
};

// ---------------------------------------------------------------------------
// Helper Utilities
// ---------------------------------------------------------------------------
function getUtilizationColor(pct: number): { bg: string; text: string; label: string } {
  if (pct >= 90.0) {
    return { bg: "#ef4444", text: "#450a0a", label: "Critical" };
  }
  if (pct >= 80.0) {
    return { bg: "#d97706", text: "#451a03", label: "Warning" };
  }
  return { bg: "#4ade80", text: "#052e16", label: "Optimal" };
}

function getEsiBadgeStyle(acuity: number): { bg: string; text: string } {
  switch (acuity) {
    case 1:
      return { bg: "#ef4444", text: "#ffffff" }; // Resuscitation (Red)
    case 2:
      return { bg: "#ea580c", text: "#ffffff" }; // Emergent (Orange)
    case 3:
      return { bg: "#ca8a04", text: "#ffffff" }; // Urgent (Yellow)
    case 4:
      return { bg: "#2563eb", text: "#ffffff" }; // Less Urgent (Blue)
    case 5:
    default:
      return { bg: "#4ade80", text: "#ffffff" }; // Non-Urgent (Green)
  }
}

// ---------------------------------------------------------------------------
// 5. Asynchronous Data States: Skeletons & Latency Placeholders
// ---------------------------------------------------------------------------
interface SkeletonBlockProps {
  height?: string;
  width?: string;
  className?: string;
  style?: React.CSSProperties;
}

const SkeletonBlock: React.FC<SkeletonBlockProps> = ({
  height = "1.5rem",
  width = "100%",
  className = "",
  style,
}) => (
  <div
    className={`animate-pulse ${className}`}
    style={{
      height,
      width,
      backgroundColor: "#334155",
      borderRadius: "0.375rem",
      ...style,
    }}
  />
);

const DashboardSkeleton: React.FC = () => (
  <div style={{ padding: "1.5rem", maxWidth: "1600px", margin: "0 auto", color: "#f1f5f9" }}>
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "1.5rem" }}>
      <SkeletonBlock height="2.5rem" width="300px" />
      <SkeletonBlock height="2.5rem" width="220px" />
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem", marginBottom: "1.5rem" }}>
      <SkeletonBlock height="7rem" />
      <SkeletonBlock height="7rem" />
      <SkeletonBlock height="7rem" />
      <SkeletonBlock height="7rem" />
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem", marginBottom: "1.5rem" }}>
      <SkeletonBlock height="18rem" />
      <SkeletonBlock height="18rem" />
    </div>
    <SkeletonBlock height="20rem" style={{ marginBottom: "1.5rem" }} />
    <SkeletonBlock height="24rem" />
  </div>
);

// ---------------------------------------------------------------------------
// 1. Executive Prescriptive KPI Strip
// ---------------------------------------------------------------------------
interface KpiStripProps {
  metrics: SimulationDataResponse["metrics"];
  hourlyCensus: HourlyCensusContract[];
  stays: PatientStayContract[];
}

const ExecutiveKpiStrip: React.FC<KpiStripProps> = ({ metrics, hourlyCensus, stays }) => {
  // 1. Rolling 4-Hour Prescriptive Arrival Gauge
  const latestCensus = hourlyCensus[0];
  const projected4h = latestCensus?.incoming_arrivals_next_4h ?? 0;
  const currentArrivals = latestCensus?.arrivals_count ?? 0;
  const projectedHourlyAverage = projected4h / 4.0;
  const isUpwardTrend = projectedHourlyAverage >= currentArrivals;

  // 2. Departmental Length of Stay & Delta
  const edStays = stays.filter((s) => s.disposition === "ED_Discharge");
  const wardStays = stays.filter((s) => s.disposition === "Ward");

  const avgEdLos = edStays.length > 0
    ? edStays.reduce((acc, curr) => acc + curr.los_hours, 0) / edStays.length
    : HISTORICAL_TARGETS.ed_los;

  const avgWardLos = wardStays.length > 0
    ? wardStays.reduce((acc, curr) => acc + curr.los_hours, 0) / wardStays.length
    : HISTORICAL_TARGETS.ward_los;

  const overallLosDelta = metrics.average_los_hours - HISTORICAL_TARGETS.average_los;

  // 3. Peak Queue Bottleneck & Time-To-Peak
  let peakQueueCount = metrics.peak_queue_length;
  let peakQueueTime = "Current Shift";
  if (hourlyCensus.length > 0) {
    const peakRecord = hourlyCensus.reduce((prev, curr) =>
      curr.patients_in_queue > prev.patients_in_queue ? curr : prev
    );
    peakQueueCount = Math.max(peakQueueCount, peakRecord.patients_in_queue);
    peakQueueTime = `${peakRecord.hour.toString().padStart(2, "0")}:00 hrs`;
  }

  // 4. Active Escalations Badge (ICU transfer flagged or requires mechanical ventilation)
  const activeEscalationsCount = stays.filter(
    (s) => s.icu_transfer_flag === 1 || s.requires_ventilation === 1
  ).length;

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
        gap: "1rem",
        marginBottom: "1.5rem",
      }}
    >
      {/* 4h Prescriptive Arrivals */}
      <div
        style={{
          backgroundColor: "#1e293b",
          borderRadius: "0.5rem",
          padding: "1rem 1.25rem",
          border: "1px solid #334155",
          boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
        }}
      >
        <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "#94a3b8", textTransform: "uppercase" }}>
          Prescriptive Influx (4-Hour)
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem", marginTop: "0.25rem" }}>
          <span style={{ fontSize: "1.75rem", fontWeight: 700, color: "#f8fafc" }}>
            +{Math.round(projected4h)}
          </span>
          <span
            style={{
              fontSize: "0.875rem",
              fontWeight: 600,
              color: isUpwardTrend ? "#ef4444" : "#4ade80",
            }}
          >
            {isUpwardTrend ? "[^] High Influx" : "[v] Stable Flow"}
          </span>
        </div>
        <div style={{ fontSize: "0.75rem", color: "#94a3b8", marginTop: "0.25rem" }}>
          Next 4h projected arrivals ({projectedHourlyAverage.toFixed(1)}/hr)
        </div>
      </div>

      {/* Average Length of Stay */}
      <div
        style={{
          backgroundColor: "#1e293b",
          borderRadius: "0.5rem",
          padding: "1rem 1.25rem",
          border: "1px solid #334155",
          boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
        }}
      >
        <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "#94a3b8", textTransform: "uppercase" }}>
          Projected Mean LoS
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem", marginTop: "0.25rem" }}>
          <span style={{ fontSize: "1.75rem", fontWeight: 700, color: "#f8fafc" }}>
            {metrics.average_los_hours.toFixed(1)}h
          </span>
          <span
            style={{
              fontSize: "0.75rem",
              fontWeight: 600,
              padding: "0.15rem 0.4rem",
              borderRadius: "0.25rem",
              backgroundColor: overallLosDelta > 0 ? "#450a0a" : "#052e16",
              color: overallLosDelta > 0 ? "#fca5a5" : "#86efac",
            }}
          >
            {overallLosDelta >= 0 ? `+${overallLosDelta.toFixed(1)}h` : `${overallLosDelta.toFixed(1)}h`} vs Target
          </span>
        </div>
        <div style={{ fontSize: "0.75rem", color: "#94a3b8", marginTop: "0.25rem" }}>
          ED: {avgEdLos.toFixed(1)}h | Med-Surg: {avgWardLos.toFixed(1)}h
        </div>
      </div>

      {/* Peak Queue Bottleneck */}
      <div
        style={{
          backgroundColor: "#1e293b",
          borderRadius: "0.5rem",
          padding: "1rem 1.25rem",
          border: "1px solid #334155",
          boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
        }}
      >
        <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "#94a3b8", textTransform: "uppercase" }}>
          Bottleneck Peak Queue
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem", marginTop: "0.25rem" }}>
          <span style={{ fontSize: "1.75rem", fontWeight: 700, color: "#f8fafc" }}>
            {peakQueueCount}
          </span>
          <span style={{ fontSize: "0.875rem", fontWeight: 500, color: "#cbd5e1" }}>
            patients waiting
          </span>
        </div>
        <div style={{ fontSize: "0.75rem", color: "#94a3b8", marginTop: "0.25rem" }}>
          Anticipated peak at {peakQueueTime}
        </div>
      </div>

      {/* Active Escalations Alert Badge */}
      <div
        style={{
          backgroundColor: activeEscalationsCount > 0 ? "#450a0a" : "#1e293b",
          borderRadius: "0.5rem",
          padding: "1rem 1.25rem",
          border: activeEscalationsCount > 0 ? "1px solid #7f1d1d" : "1px solid #334155",
          boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
        }}
      >
        <div style={{ fontSize: "0.75rem", fontWeight: 600, color: activeEscalationsCount > 0 ? "#991b1b" : "#94a3b8", textTransform: "uppercase" }}>
          Escalation Watchlist
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: "0.5rem", marginTop: "0.25rem" }}>
          <span
            style={{
              fontSize: "1.75rem",
              fontWeight: 700,
              color: activeEscalationsCount > 0 ? "#ef4444" : "#f8fafc",
            }}
          >
            {activeEscalationsCount}
          </span>
          <span
            style={{
              fontSize: "0.75rem",
              fontWeight: 600,
              padding: "0.15rem 0.4rem",
              borderRadius: "0.25rem",
              backgroundColor: activeEscalationsCount > 0 ? "#ef4444" : "#334155",
              color: activeEscalationsCount > 0 ? "#ffffff" : "#cbd5e1",
            }}
          >
            {activeEscalationsCount > 0 ? "CRITICAL ALERT" : "NORMAL"}
          </span>
        </div>
        <div style={{ fontSize: "0.75rem", color: activeEscalationsCount > 0 ? "#fca5a5" : "#94a3b8", marginTop: "0.25rem" }}>
          Predicted ICU transfer or invasive ventilation
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// 2. Resource Utilization & Bottleneck Analytics
// ---------------------------------------------------------------------------
interface ResourceAnalyticsProps {
  metrics: SimulationDataResponse["metrics"];
  hourlyCensus: HourlyCensusContract[];
}

const ResourceUtilizationSection: React.FC<ResourceAnalyticsProps> = ({ metrics, hourlyCensus }) => {
  const units: Array<{ name: string; pct: number; code: CareUnitType }> = [
    { name: "Emergency Department (ED)", pct: metrics.ed_utilization_pct, code: "ED_Only" },
    { name: "Med-Surg General Ward", pct: metrics.ward_utilization_pct, code: "Ward" },
    { name: "Intensive Care Unit (ICU)", pct: metrics.icu_utilization_pct, code: "ICU" },
  ];

  // Derive maximum arrivals or discharges for dual-axis chart normalization
  const maxVolume = useMemo(() => {
    if (hourlyCensus.length === 0) return 20;
    const maxVal = Math.max(
      ...hourlyCensus.map((h) => Math.max(h.arrivals_count, h.discharges_count, h.patients_in_queue))
    );
    return Math.max(maxVal, 10);
  }, [hourlyCensus]);

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1.6fr",
        gap: "1.5rem",
        marginBottom: "1.5rem",
      }}
    >
      {/* Bed Utilization Capacity Gauges */}
      <div
        style={{
          backgroundColor: "#1e293b",
          borderRadius: "0.5rem",
          padding: "1.25rem",
          border: "1px solid #334155",
        }}
      >
        <h3 style={{ fontSize: "1rem", fontWeight: 700, margin: "0 0 1rem 0", color: "#f8fafc" }}>
          Department Bed Utilization Gauges
        </h3>
        <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          {units.map((unit) => {
            const status = getUtilizationColor(unit.pct);
            return (
              <div key={unit.code}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.35rem" }}>
                  <span style={{ fontSize: "0.875rem", fontWeight: 600, color: "#e2e8f0" }}>
                    {unit.name}
                  </span>
                  <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                    <span
                      style={{
                        fontSize: "0.7rem",
                        fontWeight: 600,
                        padding: "0.1rem 0.4rem",
                        borderRadius: "0.25rem",
                        backgroundColor: status.bg,
                        color: status.text,
                      }}
                    >
                      {status.label}
                    </span>
                    <span style={{ fontSize: "0.875rem", fontWeight: 700, color: "#f8fafc" }}>
                      {unit.pct.toFixed(1)}%
                    </span>
                  </div>
                </div>
                <div
                  style={{
                    height: "0.75rem",
                    width: "100%",
                    backgroundColor: "#334155",
                    borderRadius: "0.375rem",
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${Math.min(100, Math.max(0, unit.pct))}%`,
                      backgroundColor: status.bg,
                      transition: "width 0.4s ease-in-out",
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
        <div style={{ marginTop: "1.25rem", fontSize: "0.75rem", color: "#94a3b8" }}>
          Thresholds: Optimal (&lt;80%) | Warning (80-90%) | Critical Alert (&gt;90%)
        </div>
      </div>

      {/* Hourly Patient Flow Dual-Axis Chart Area */}
      <div
        style={{
          backgroundColor: "#1e293b",
          borderRadius: "0.5rem",
          padding: "1.25rem",
          border: "1px solid #334155",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
          <h3 style={{ fontSize: "1rem", fontWeight: 700, margin: 0, color: "#f8fafc" }}>
            24h Projected Patient Influx vs. Discharges
          </h3>
          <div style={{ display: "flex", gap: "1rem", fontSize: "0.75rem" }}>
            <span style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
              <span style={{ width: "10px", height: "10px", backgroundColor: "#2563eb", borderRadius: "2px" }} />
              Arrivals
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
              <span style={{ width: "10px", height: "10px", backgroundColor: "#4ade80", borderRadius: "2px" }} />
              Discharges
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
              <span style={{ width: "10px", height: "10px", backgroundColor: "#ea580c", borderRadius: "2px" }} />
              Waiting Queue
            </span>
          </div>
        </div>

        {/* Visual Dual-Axis Timeline representation */}
        <div
          style={{
            height: "170px",
            display: "flex",
            alignItems: "flex-end",
            gap: "4px",
            paddingTop: "1rem",
            borderBottom: "1px solid #cbd5e1",
            position: "relative",
          }}
        >
          {hourlyCensus.slice(0, 24).map((h, idx) => {
            const arrHeight = (h.arrivals_count / maxVolume) * 130;
            const disHeight = (h.discharges_count / maxVolume) * 130;
            const qHeight = (h.patients_in_queue / maxVolume) * 130;
            const isShiftChange = h.hour === 7 || h.hour === 15 || h.hour === 23;

            return (
              <div
                key={idx}
                style={{
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  height: "100%",
                  justifyContent: "flex-end",
                  position: "relative",
                }}
                title={`Hour ${h.hour}:00 | Arr: ${h.arrivals_count} | Dis: ${h.discharges_count} | Queue: ${h.patients_in_queue}`}
              >
                {isShiftChange && (
                  <div
                    style={{
                      position: "absolute",
                      top: 0,
                      bottom: 0,
                      width: "1px",
                      backgroundColor: "#94a3b8",
                      borderStyle: "dashed",
                    }}
                  />
                )}
                <div style={{ display: "flex", alignItems: "flex-end", gap: "1px", width: "100%" }}>
                  <div style={{ width: "33%", height: `${arrHeight}px`, backgroundColor: "#2563eb" }} />
                  <div style={{ width: "33%", height: `${disHeight}px`, backgroundColor: "#4ade80" }} />
                  <div style={{ width: "33%", height: `${qHeight}px`, backgroundColor: "#ea580c" }} />
                </div>
              </div>
            );
          })}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: "0.5rem", fontSize: "0.7rem", color: "#94a3b8" }}>
          <span>00:00 (Night)</span>
          <span>07:00 (Day Shift Shiftover)</span>
          <span>15:00 (Evening Shiftover)</span>
          <span>23:00 (Night Shiftover)</span>
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// 3. Hourly Census & Shift-Staffing Matrix
// ---------------------------------------------------------------------------
interface ShiftStaffingProps {
  hourlyCensus: HourlyCensusContract[];
}

const ShiftStaffingMatrix: React.FC<ShiftStaffingProps> = ({ hourlyCensus }) => {
  const shiftGroups = useMemo(() => {
    const shifts: Record<ShiftType, HourlyCensusContract[]> = {
      Day: [],
      Evening: [],
      Night: [],
    };

    hourlyCensus.forEach((record) => {
      if (shifts[record.shift_id]) {
        shifts[record.shift_id].push(record);
      }
    });

    return shifts;
  }, [hourlyCensus]);

  const summary = (["Day", "Evening", "Night"] as ShiftType[]).map((shiftKey) => {
    const records = shiftGroups[shiftKey];
    const baseline = SHIFT_BASELINES[shiftKey];

    if (records.length === 0) {
      return {
        shift: shiftKey,
        meanCensus: 0,
        activeNurses: baseline.nurses,
        activeDoctors: baseline.doctors,
        targetNurses: baseline.nurses,
        targetDoctors: baseline.doctors,
        nurseGap: 0,
        safeRatioViolated: false,
      };
    }

    const totalCensus = records.reduce(
      (sum, r) => sum + r.ed_occupancy + r.ward_occupancy + r.icu_occupancy,
      0
    );
    const meanCensus = Math.round(totalCensus / records.length);

    const avgNurses = Math.round(
      records.reduce((sum, r) => sum + r.active_nurses, 0) / records.length
    );
    const avgDoctors = Math.round(
      records.reduce((sum, r) => sum + r.active_doctors, 0) / records.length
    );

    // Dynamic staffing target adjustment based on occupancy volume
    const targetNurses = baseline.nurses;
    const targetDoctors = baseline.doctors;
    const nurseGap = avgNurses - targetNurses;

    // Safe ratio check: if total hospital census to active nurses exceeds 7:1
    const safeRatioViolated = meanCensus / Math.max(1, avgNurses) > 6.5;

    return {
      shift: shiftKey,
      meanCensus,
      activeNurses: avgNurses,
      activeDoctors: avgDoctors,
      targetNurses,
      targetDoctors,
      nurseGap,
      safeRatioViolated,
    };
  });

  return (
    <div
      style={{
        backgroundColor: "#1e293b",
        borderRadius: "0.5rem",
        padding: "1.25rem",
        border: "1px solid #334155",
        marginBottom: "1.5rem",
      }}
    >
      <h3 style={{ fontSize: "1rem", fontWeight: 700, margin: "0 0 1rem 0", color: "#f8fafc" }}>
        Shift-Staffing Alignment Matrix & Ratios
      </h3>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.875rem", textAlign: "left" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #334155", color: "#cbd5e1" }}>
              <th style={{ padding: "0.6rem" }}>Shift Window</th>
              <th style={{ padding: "0.6rem" }}>Hours (Active)</th>
              <th style={{ padding: "0.6rem" }}>Mean Census</th>
              <th style={{ padding: "0.6rem" }}>Target RN / Active</th>
              <th style={{ padding: "0.6rem" }}>Target MD / Active</th>
              <th style={{ padding: "0.6rem" }}>Staffing Delta / Status</th>
              <th style={{ padding: "0.6rem" }}>Ratio Safety Feedback</th>
            </tr>
          </thead>
          <tbody>
            {summary.map((row) => {
              const hoursWindow =
                row.shift === "Day"
                  ? "07:00 - 15:00"
                  : row.shift === "Evening"
                  ? "15:00 - 23:00"
                  : "23:00 - 07:00";

              return (
                <tr key={row.shift} style={{ borderBottom: "1px solid #0f172a" }}>
                  <td style={{ padding: "0.6rem", fontWeight: 600, color: "#f1f5f9" }}>{row.shift}</td>
                  <td style={{ padding: "0.6rem", color: "#94a3b8" }}>{hoursWindow}</td>
                  <td style={{ padding: "0.6rem", fontWeight: 600 }}>{row.meanCensus} pts</td>
                  <td style={{ padding: "0.6rem" }}>
                    {row.targetNurses} / <strong>{row.activeNurses} RNs</strong>
                  </td>
                  <td style={{ padding: "0.6rem" }}>
                    {row.targetDoctors} / <strong>{row.activeDoctors} MDs</strong>
                  </td>
                  <td style={{ padding: "0.6rem" }}>
                    <span
                      style={{
                        padding: "0.2rem 0.5rem",
                        borderRadius: "0.25rem",
                        fontSize: "0.75rem",
                        fontWeight: 600,
                        backgroundColor:
                          row.nurseGap < 0 ? "#450a0a" : row.nurseGap > 0 ? "#052e16" : "#0f172a",
                        color:
                          row.nurseGap < 0 ? "#fca5a5" : row.nurseGap > 0 ? "#86efac" : "#cbd5e1",
                      }}
                    >
                      {row.nurseGap > 0
                        ? `+${row.nurseGap} RN (Overstaffed)`
                        : row.nurseGap < 0
                        ? `${row.nurseGap} RNs (Understaffed Alert)`
                        : "Balanced"}
                    </span>
                  </td>
                  <td style={{ padding: "0.6rem" }}>
                    {row.safeRatioViolated ? (
                      <span style={{ color: "#ef4444", fontWeight: 700, fontSize: "0.75rem" }}>
                        [!] Ratio Breach Warning (&gt;6.5:1)
                      </span>
                    ) : (
                      <span style={{ color: "#4ade80", fontWeight: 600, fontSize: "0.75rem" }}>
                        Safe Workforce Baselines Maintained
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// 4. Patient Flow Trajectories & Clinical Escalation Radar
// ---------------------------------------------------------------------------
interface ClinicalRadarProps {
  stays: PatientStayContract[];
}

const ClinicalEscalationRadar: React.FC<ClinicalRadarProps> = ({ stays }) => {
  // Disposition split calculation
  const dispositionBreakdown = useMemo(() => {
    if (stays.length === 0) {
      return { edDischarge: 0, wardAdmit: 0, icuAdmit: 0 };
    }
    const edDischarge = (stays.filter((s) => s.disposition === "ED_Discharge").length / stays.length) * 100;
    const wardAdmit = (stays.filter((s) => s.disposition === "Ward").length / stays.length) * 100;
    const icuAdmit = (stays.filter((s) => s.disposition === "ICU").length / stays.length) * 100;

    return {
      edDischarge: Math.round(edDischarge),
      wardAdmit: Math.round(wardAdmit),
      icuAdmit: Math.round(icuAdmit),
    };
  }, [stays]);

  // Prioritize high acuity, ICU transfer flags, or ventilation requirement
  const prioritizedWatchlist = useMemo(() => {
    return [...stays]
      .sort((a, b) => {
        // High risk sort: ventilation flag -> ICU transfer flag -> Acuity (lower is more acute)
        if (b.requires_ventilation !== a.requires_ventilation) {
          return b.requires_ventilation - a.requires_ventilation;
        }
        if (b.icu_transfer_flag !== a.icu_transfer_flag) {
          return b.icu_transfer_flag - a.icu_transfer_flag;
        }
        return a.triage_acuity - b.triage_acuity;
      })
      .slice(0, 8);
  }, [stays]);

  return (
    <div
      style={{
        backgroundColor: "#1e293b",
        borderRadius: "0.5rem",
        padding: "1.25rem",
        border: "1px solid #334155",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
        <h3 style={{ fontSize: "1rem", fontWeight: 700, margin: 0, color: "#f8fafc" }}>
          Patient Flow Trajectories & Clinical Escalation Radar
        </h3>
        {/* Trajectory percentage split indicator */}
        <div style={{ display: "flex", gap: "1rem", fontSize: "0.8rem", fontWeight: 600 }}>
          <span style={{ color: "#4ade80" }}>ED Discharge: {dispositionBreakdown.edDischarge}%</span>
          <span style={{ color: "#2563eb" }}>Med-Surg Ward: {dispositionBreakdown.wardAdmit}%</span>
          <span style={{ color: "#ef4444" }}>Direct ICU: {dispositionBreakdown.icuAdmit}%</span>
        </div>
      </div>

      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.875rem", textAlign: "left" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #334155", color: "#cbd5e1" }}>
              <th style={{ padding: "0.6rem" }}>Stay / Patient</th>
              <th style={{ padding: "0.6rem" }}>Acuity (ESI)</th>
              <th style={{ padding: "0.6rem" }}>Comorbidity & Demographics</th>
              <th style={{ padding: "0.6rem" }}>Complaint / Path</th>
              <th style={{ padding: "0.6rem" }}>Escalation Flags</th>
              <th style={{ padding: "0.6rem" }}>Physiological Vitals Trajectory</th>
              <th style={{ padding: "0.6rem" }}>Invariant Verification</th>
            </tr>
          </thead>
          <tbody>
            {prioritizedWatchlist.map((stay) => {
              const esiStyle = getEsiBadgeStyle(stay.triage_acuity);
              const pulsePressure = stay.sbp - stay.dbp;
              const isPulsePressureValid = pulsePressure >= MIN_PULSE_PRESSURE;

              return (
                <tr
                  key={stay.stay_id}
                  style={{
                    borderBottom: "1px solid #0f172a",
                    backgroundColor: stay.icu_transfer_flag === 1 || stay.requires_ventilation === 1 ? "#1e293b" : "#1e293b",
                  }}
                >
                  {/* Identifiers */}
                  <td style={{ padding: "0.6rem", fontWeight: 600, color: "#f1f5f9" }}>
                    <div>#{stay.stay_id}</div>
                    <div style={{ fontSize: "0.75rem", color: "#94a3b8" }}>PT-{stay.patient_id}</div>
                  </td>

                  {/* Triage Acuity */}
                  <td style={{ padding: "0.6rem" }}>
                    <span
                      style={{
                        padding: "0.2rem 0.6rem",
                        borderRadius: "0.25rem",
                        fontSize: "0.75rem",
                        fontWeight: 700,
                        backgroundColor: esiStyle.bg,
                        color: esiStyle.text,
                      }}
                    >
                      ESI Level {stay.triage_acuity}
                    </span>
                  </td>

                  {/* Comorbidity & Demographics */}
                  <td style={{ padding: "0.6rem", color: "#e2e8f0" }}>
                    <div>{stay.age}yo {stay.gender}</div>
                    <div style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Charlson CCI: {stay.charlson_index}/10</div>
                  </td>

                  {/* Complaint & Disposition */}
                  <td style={{ padding: "0.6rem" }}>
                    <div style={{ fontWeight: 500 }}>{stay.chief_complaint}</div>
                    <div style={{ fontSize: "0.75rem", color: "#2563eb", fontWeight: 600 }}>
                      -&gt; {stay.disposition} ({stay.initial_care_unit})
                    </div>
                  </td>

                  {/* Escalation Risk Flags */}
                  <td style={{ padding: "0.6rem" }}>
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                      {stay.icu_transfer_flag === 1 && (
                        <span
                          style={{
                            backgroundColor: "#450a0a",
                            color: "#fca5a5",
                            fontSize: "0.7rem",
                            fontWeight: 700,
                            padding: "0.15rem 0.4rem",
                            borderRadius: "0.25rem",
                            border: "1px solid #7f1d1d",
                          }}
                        >
                          Ward -&gt; ICU Transfer Triggered
                        </span>
                      )}
                      {stay.requires_ventilation === 1 && (
                        <span
                          style={{
                            backgroundColor: "#172554",
                            color: "#1d4ed8",
                            fontSize: "0.7rem",
                            fontWeight: 700,
                            padding: "0.15rem 0.4rem",
                            borderRadius: "0.25rem",
                            border: "1px solid #1e3a8a",
                          }}
                        >
                          Mechanical Ventilation
                        </span>
                      )}
                      {stay.icu_transfer_flag === 0 && stay.requires_ventilation === 0 && (
                        <span style={{ fontSize: "0.75rem", color: "#94a3b8" }}>Stable Path</span>
                      )}
                    </div>
                  </td>

                  {/* Vitals Panel */}
                  <td style={{ padding: "0.6rem", fontSize: "0.75rem", color: "#e2e8f0" }}>
                    <div>HR: {stay.heart_rate} bpm | SpO2: {stay.o2_sat}%</div>
                    <div>BP: {stay.sbp}/{stay.dbp} mmHg | RR: {stay.resp_rate}</div>
                  </td>

                  {/* Invariant Verification */}
                  <td style={{ padding: "0.6rem" }}>
                    {isPulsePressureValid ? (
                      <span
                        style={{
                          backgroundColor: "#052e16",
                          color: "#166534",
                          fontSize: "0.7rem",
                          fontWeight: 700,
                          padding: "0.15rem 0.4rem",
                          borderRadius: "0.25rem",
                          border: "1px solid #14532d",
                          display: "inline-block",
                        }}
                      >
                        [PP Valid: {pulsePressure.toFixed(1)} mmHg]
                      </span>
                    ) : (
                      <span
                        style={{
                          backgroundColor: "#450a0a",
                          color: "#991b1b",
                          fontSize: "0.7rem",
                          fontWeight: 700,
                          padding: "0.15rem 0.4rem",
                          borderRadius: "0.25rem",
                          border: "1px solid #f87171",
                          display: "inline-block",
                        }}
                      >
                        Sensor Invariant Failure: Plausibility Check Failed
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Main Dashboard Page Component
// ---------------------------------------------------------------------------
export const ForecastDashboard: React.FC = () => {
  const { data, isLoading, error, fetchSimulationData, runSimulation, resetSimulation } =
    useForecastData({ autoFetch: true });

  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);

  // Background polling heartbeat setup (every 20 seconds)
  useEffect(() => {
    const interval = setInterval(async () => {
      setIsRefreshing(true);
      await fetchSimulationData(true);
      setIsRefreshing(false);
    }, 20000);

    return () => clearInterval(interval);
  }, [fetchSimulationData]);

  const handleManualRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await fetchSimulationData(true);
    setIsRefreshing(false);
  }, [fetchSimulationData]);

  const handleTriggerRun = useCallback(async () => {
    setIsRefreshing(true);
    await runSimulation({
      duration_hours: 24,
      ed_capacity: 50,
      ward_capacity: 150,
      icu_capacity: 30,
      seed: 42,
    });
    setIsRefreshing(false);
  }, [runSimulation]);

  // Loading skeleton on initial cold load
  if (isLoading && !data) {
    return <DashboardSkeleton />;
  }

  // Error Banner State
  if (error && !data) {
    return (
      <div style={{ padding: "2rem", maxWidth: "1200px", margin: "0 auto", textAlign: "center" }}>
        <div
          style={{
            backgroundColor: "#450a0a",
            border: "1px solid #f87171",
            color: "#991b1b",
            padding: "1.5rem",
            borderRadius: "0.5rem",
          }}
        >
          <h2 style={{ fontSize: "1.25rem", fontWeight: 700, margin: "0 0 0.5rem 0" }}>
            Operational Telemetry Stream Disconnected
          </h2>
          <p style={{ margin: "0 0 1rem 0" }}>{error}</p>
          <button
            onClick={handleManualRefresh}
            style={{
              backgroundColor: "#ef4444",
              color: "#ffffff",
              border: "none",
              padding: "0.5rem 1.25rem",
              borderRadius: "0.375rem",
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            Retry Telemetry Ingestion
          </button>
        </div>
      </div>
    );
  }

  if (!data) {
    return null;
  }

  return (
    <div
      style={{
        backgroundColor: "#0f172a",
        minHeight: "100vh",
        padding: "1.5rem",
        color: "#f8fafc",
        fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      }}
    >
      <div style={{ maxWidth: "1600px", margin: "0 auto" }}>
        {/* Header with background pulse indicator and controls */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "1.5rem",
            flexWrap: "wrap",
            gap: "1rem",
          }}
        >
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <h1 style={{ fontSize: "1.5rem", fontWeight: 800, margin: 0, color: "#f8fafc" }}>
                Hospital Digital Twin - Forecast & Telemetry Command Center
              </h1>
              {/* Background Revalidation Pulse */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.35rem",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  padding: "0.2rem 0.5rem",
                  borderRadius: "9999px",
                  backgroundColor: isRefreshing ? "#78350f" : "#052e16",
                  color: isRefreshing ? "#b45309" : "#166534",
                }}
              >
                <span
                  style={{
                    width: "8px",
                    height: "8px",
                    borderRadius: "50%",
                    backgroundColor: isRefreshing ? "#d97706" : "#22c55e",
                  }}
                />
                {isRefreshing ? "Live Sync Active..." : "Telemetry Synchronized"}
              </div>
            </div>
            <div style={{ fontSize: "0.8rem", color: "#94a3b8", marginTop: "0.25rem" }}>
              Snapshot generated: {data.generated_at} | SimClock Anchor: 2026-09-23 00:00:00
            </div>
          </div>

          <div style={{ display: "flex", gap: "0.75rem" }}>
            <button
              onClick={handleManualRefresh}
              disabled={isRefreshing}
              style={{
                backgroundColor: "#1e293b",
                border: "1px solid #cbd5e1",
                padding: "0.5rem 1rem",
                borderRadius: "0.375rem",
                fontSize: "0.875rem",
                fontWeight: 600,
                color: "#e2e8f0",
                cursor: isRefreshing ? "not-allowed" : "pointer",
              }}
            >
              Refresh Data
            </button>
            <button
              onClick={handleTriggerRun}
              disabled={isRefreshing}
              style={{
                backgroundColor: "#2563eb",
                border: "none",
                padding: "0.5rem 1rem",
                borderRadius: "0.375rem",
                fontSize: "0.875rem",
                fontWeight: 600,
                color: "#ffffff",
                cursor: isRefreshing ? "not-allowed" : "pointer",
              }}
            >
              Execute ML Forecast Run
            </button>
            <button
              onClick={resetSimulation}
              disabled={isRefreshing}
              style={{
                backgroundColor: "#1e293b",
                border: "1px solid #cbd5e1",
                padding: "0.5rem 1rem",
                borderRadius: "0.375rem",
                fontSize: "0.875rem",
                fontWeight: 600,
                color: "#ef4444",
                cursor: isRefreshing ? "not-allowed" : "pointer",
              }}
            >
              Reset Baseline
            </button>
          </div>
        </div>

        {/* 1. Executive Prescriptive KPI Strip */}
        <ExecutiveKpiStrip
          metrics={data.metrics}
          hourlyCensus={data.hourly_census}
          stays={data.stays}
        />

        {/* 2. Resource Utilization & Bottleneck Analytics */}
        <ResourceUtilizationSection
          metrics={data.metrics}
          hourlyCensus={data.hourly_census}
        />

        {/* 3. Hourly Census & Shift-Staffing Matrix */}
        <ShiftStaffingMatrix hourlyCensus={data.hourly_census} />

        {/* 4. Patient Flow Trajectories & Clinical Escalation Radar */}
        <ClinicalEscalationRadar stays={data.stays} />
      </div>
    </div>
  );
};

export default ForecastDashboard;