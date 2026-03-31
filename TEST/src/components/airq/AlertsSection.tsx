
import { useEffect, useRef } from "react";
import { AlertTriangle, CheckCircle, Info, TrendingUp, TrendingDown, Wind } from "lucide-react";

const alerts = [
  {
    id: 1,
    type: "warning",
    icon: AlertTriangle,
    city: "Delhi",
    time: "2 min ago",
    title: "AQI predicted to spike by 45 points",
    detail: "Model detects sustained low wind speeds (<4 km/h) combined with rising traffic hours. AQI expected to cross 220 by 9:00 PM.",
    color: "#FB923C",
  },
  {
    id: 2,
    type: "success",
    icon: CheckCircle,
    city: "Mumbai",
    time: "11 min ago",
    title: "Coastal winds expected to clear pollution",
    detail: "Sea-breeze pattern developing offshore. AQI forecast to drop from 85 to 55 over the next 6 hours.",
    color: "#34D399",
  },
  {
    id: 3,
    type: "info",
    icon: TrendingUp,
    city: "Kolkata",
    time: "28 min ago",
    title: "Moderate deterioration trend detected",
    detail: "AQI has been climbing for 3 consecutive hours. Current: 148. Predicted: 172 in 4h. Cause: vehicle congestion + calm meteorology.",
    color: "#FACC15",
  },
  {
    id: 4,
    type: "info",
    icon: Info,
    city: "Hyderabad",
    time: "45 min ago",
    title: "Stable AQI conditions forecast",
    detail: "No significant changes expected in the next 24 hours. Moderate wind activity keeping pollutant dispersion at healthy levels.",
    color: "#60A5FA",
  },
  {
    id: 5,
    type: "warning",
    icon: Wind,
    city: "Delhi",
    time: "1 hr ago",
    title: "Stubble burning impact detected in north districts",
    detail: "Satellite data and lag-24 feature showing elevated PM2.5 precursor signals. Peak AQI likely between 11 PM–2 AM.",
    color: "#F87171",
  },
  {
    id: 6,
    type: "success",
    icon: TrendingDown,
    city: "Bangalore",
    time: "2 hrs ago",
    title: "Rain event cleared overnight pollution",
    detail: "Rainfall of 8mm recorded early morning. AQI has improved from 95 to 48 — now in 'Good' category.",
    color: "#34D399",
  },
];

export default function AlertsSection() {
  const sectionRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = sectionRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => entry.isIntersecting && el.classList.add("visible"),
      { threshold: 0.05 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <section id="alerts" className="alerts-section">
      <div ref={sectionRef} className="section-fade alerts-container">
        {/* Header */}
        <div className="alerts-header">
          <p className="alerts-eyebrow">Live Feed</p>
          <h2 className="alerts-title">
            Alerts &amp; <span>Notifications</span>
          </h2>
          <p className="alerts-desc">
            Real-time model-generated alerts for air quality events across monitored cities.
          </p>
        </div>

        {/* Stats bar */}
        <div className="alerts-stats">
          {[
            { label: "Active Warnings", value: "2", color: "#FB923C" },
            { label: "Clearance Events", value: "2", color: "#34D399" },
            { label: "Info Updates", value: "2", color: "#60A5FA" },
          ].map((s) => (
            <div key={s.label} className="depth-card alerts-stat-card">
              <p className="alerts-stat-val" style={{ color: s.color }}>{s.value}</p>
              <p className="alerts-stat-label">{s.label}</p>
            </div>
          ))}
        </div>

        {/* Timeline */}
        <div className="alerts-timeline">
          {/* vertical line */}
          <div className="alerts-timeline-line" />

          <div className="alerts-list">
            {alerts.map((alert) => {
              const Icon = alert.icon;
              return (
                <div
                  key={alert.id}
                  className="depth-card alerts-card"
                  style={{ borderColor: `${alert.color}33` }}
                >
                  {/* Timeline dot */}
                  <div
                    className="alerts-dot-outer"
                    style={{ backgroundColor: `${alert.color}1A`, borderColor: `${alert.color}33` }}
                  >
                    <span className="alerts-dot-inner" style={{ backgroundColor: alert.color }} />
                  </div>

                  <div className="alerts-content">
                    <div
                      className="alerts-mobile-icon"
                      style={{ backgroundColor: `${alert.color}1A`, borderColor: `${alert.color}33` }}
                    >
                      <Icon className="w-4 h-4" style={{ color: alert.color }} />
                    </div>
                    <div className="alerts-info">
                      <div className="alerts-meta">
                        <div className="alerts-city-tag">
                          <Icon className="alerts-desktop-icon" style={{ color: alert.color }} />
                          <span style={{ color: alert.color }}>
                            {alert.city} Alert
                          </span>
                        </div>
                        <span className="alerts-time">{alert.time}</span>
                      </div>
                      <p className="alerts-item-title">{alert.title}</p>
                      <p className="alerts-item-desc">{alert.detail}</p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
