
import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";

const aqiScale = [
  {
    range: "0–50",
    label: "Good",
    color: "#10B981",

    description: "Air quality is considered satisfactory, and air pollution poses little or no risk.",
    advice: [
      "Ideal for all outdoor activities.",
      "Open windows to enjoy fresh air.",
      "No special precautions needed.",
      "Good time for exercise and sports.",
    ],
  },
  {
    range: "51–100",
    label: "Satisfactory",
    color: "#84CC16",

    description: "Air quality is acceptable; however, some pollutants may pose a moderate concern for a small number of people.",
    advice: [
      "Unusually sensitive people should consider reducing prolonged outdoor exertion.",
      "Most people can remain outdoors safely.",
      "Watch for symptoms if you have respiratory issues.",
    ],
  },
  {
    range: "101–200",
    label: "Moderate",
    color: "#FBBF24",

    description: "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
    advice: [
      "Sensitive groups (elderly, children, asthma patients) should limit outdoor activity.",
      "Consider wearing a surgical mask outdoors.",
      "Keep windows closed during peak hours (7–10 AM, 5–8 PM).",
      "Stay hydrated.",
    ],
  },
  {
    range: "201–300",
    label: "Poor",
    color: "#F97316",

    description: "Health warnings of emergency conditions. The entire population is more likely to be affected.",
    advice: [
      "Avoid prolonged outdoor physical exertion.",
      "Wear an N95 mask if going out.",
      "Keep indoor air purifiers running.",
      "Sensitive groups should stay indoors.",
      "Avoid morning and evening rush-hour exposure.",
    ],
  },
  {
    range: "301–400",
    label: "Very Poor",
    color: "#EF4444",

    description: "Health alert: everyone may experience serious health effects.",
    advice: [
      "Everyone should avoid outdoor activities.",
      "N95 masks mandatory if you must go out.",
      "Seal windows and doors with weather stripping.",
      "Run air purifiers on highest setting.",
      "Consult a doctor if experiencing breathing difficulties.",
    ],
  },
  {
    range: "400+",
    label: "Severe",
    color: "#7C3AED",

    description: "Hazardous: emergency health conditions. Everyone is at risk.",
    advice: [
      "Avoid ALL outdoor physical activity.",
      "Wear an N95 mask. Do not use surgical masks.",
      "Keep indoor air purifiers running 24/7.",
      "Seek medical attention if experiencing any respiratory symptoms.",
      "Children, elderly, and patients must stay indoors.",
      "Consider temporary relocation if possible.",
    ],
  },
];

export default function HealthSection() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

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
    <section id="health" className="health-section">
      <div ref={sectionRef} className="section-fade health-container">
        {/* Header */}
        <div className="health-header">
          <p className="health-eyebrow">Health Advisory</p>
          <h2 className="health-title">
            AQI Scale &amp; <span>Recommendations</span>
          </h2>
          <p className="health-desc">Click any category to expand personalized health advice.</p>
        </div>

        {/* Scale table */}
        <div className="depth-card health-table-card">
          <div className="health-table-wrap">
            <table className="health-table">
              <thead>
                <tr>
                  <th>AQI Range</th>
                  <th>Category</th>
                  <th className="health-th-md">Color</th>
                  <th className="health-th-lg">Description</th>
                </tr>
              </thead>
              <tbody>
                {aqiScale.map((row) => (
                  <tr key={row.label}>
                    <td className="health-td-range" style={{ color: row.color }}>{row.range}</td>
                    <td>
                      <span className="health-td-cat" style={{ color: row.color }}>
                        <span className="health-td-dot" style={{ backgroundColor: row.color }} />
                        {row.label}
                      </span>
                    </td>
                    <td className="health-td-md">
                      <span className="health-color-box" style={{ backgroundColor: row.color }} />
                    </td>
                    <td className="health-td-lg"><div className="health-td-desc">{row.description}</div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Expandable cards */}
        <h3 className="health-subtitle">Dynamic Recommendations</h3>
        <div className="health-grid">
          {aqiScale.map((item) => (
            <div
              key={item.label}
              className={`depth-card health-accordion-card ${expanded === item.label ? "expanded" : ""}`}
              style={{ borderColor: `${item.color}33` }}
            >
              <button
                className="health-accordion-btn"
                onClick={() => setExpanded(expanded === item.label ? null : item.label)}
              >
                <div className="health-accordion-title">
                  <span className="health-accordion-dot" style={{ backgroundColor: item.color }} />
                  <span className="health-accordion-label" style={{ color: item.color }}>
                    {item.label}
                    <span className="health-accordion-range">({item.range})</span>
                  </span>
                </div>
                <ChevronDown className={`health-accordion-icon ${expanded === item.label ? "rotated" : ""}`} />
              </button>

              {expanded === item.label && (
                <div className="health-accordion-content">
                  <p className="health-accordion-desc">{item.description}</p>
                  <ul className="health-accordion-list">
                    {item.advice.map((tip) => (
                      <li key={tip} className="health-accordion-li">
                        <span className="health-accordion-list-dot" style={{ backgroundColor: item.color }} />
                        {tip}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
