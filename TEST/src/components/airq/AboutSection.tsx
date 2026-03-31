
import { useEffect, useRef } from "react";
import { AlertCircle, Lightbulb, Github, Linkedin, Code2, Cpu, FileCode, Database, Layers } from "lucide-react";

const techStack = [
  { icon: Code2, name: "React", desc: "Frontend UI framework", color: "#22D3EE" },
  { icon: Layers, name: "Tailwind CSS", desc: "Utility-first styling", color: "#38BDF8" },
  { icon: FileCode, name: "Python", desc: "Backend & model training", color: "#FACC15" },
  { icon: Database, name: "Pandas", desc: "Data manipulation & EDA", color: "#4ADE80" },
  { icon: Cpu, name: "Scikit-Learn", desc: "ML model training", color: "#FB923C" },
  { icon: Database, name: "NumPy", desc: "Numerical computing", color: "#60A5FA" },
];

export default function AboutSection() {
  const aboutRef = useRef<HTMLDivElement>(null);
  const projectRef = useRef<HTMLDivElement>(null);
  const creditsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const elements = [aboutRef.current, projectRef.current, creditsRef.current];
    const obs = new IntersectionObserver(
      (entries) =>
        entries.forEach((e) => e.isIntersecting && e.target.classList.add("visible")),
      { threshold: 0.05 }
    );
    elements.forEach((el) => el && obs.observe(el));
    return () => obs.disconnect();
  }, []);

  return (
    <>
      {/* About */}
      <section id="about" className="about-section">
        <div ref={aboutRef} className="section-fade about-container">
          {/* Liquid Glass Header */}
          <div className="about-header-card">
            <div className="about-header-glow-1"></div>
            <div className="about-header-glow-2"></div>

            <div className="about-header-content">
              <p className="about-eyebrow">About AirQ</p>
              <h2 className="about-title">
                Understanding the <span>Challenge</span>
              </h2>
            </div>
          </div>

          <div className="about-grid">
            {/* The Problem */}
            <div className="depth-card about-card">
              <div className="about-card-title-wrap">
                <div className="about-card-icon-box" style={{ backgroundColor: "rgba(248, 113, 113, 0.1)", borderColor: "rgba(248, 113, 113, 0.2)" }}>
                  <AlertCircle className="w-5 h-5" style={{ color: "#F87171" }} />
                </div>
                <h3 className="about-card-title">The Problem</h3>
              </div>
              <div className="about-card-body">
                <p>
                  Air pollution in Indian cities is one of the most severe public health crises of our time.
                  India is home to 9 of the world&apos;s 10 most polluted cities, with Delhi routinely recording
                  AQI levels above 400 — classified as &quot;Severe&quot; on the national index.
                </p>
                <p>
                  The danger lies not just in the pollution itself, but in its <span>unpredictability</span>.
                  Sudden spikes driven by vehicle emissions, industrial output, crop burning, and adverse
                  meteorological conditions catch residents off-guard, leading to preventable hospitalizations
                  and long-term respiratory damage.
                </p>
                <p>
                  Without advance warning, individuals, hospitals, and city planners cannot take protective
                  measures before conditions deteriorate. The gap between data collection and actionable
                  forecasting has historically been too wide.
                </p>
              </div>
            </div>

            {/* The Solution */}
            <div className="depth-card about-card">
              <div className="about-card-title-wrap">
                <div className="about-card-icon-box" style={{ backgroundColor: "rgba(52, 211, 153, 0.1)", borderColor: "rgba(52, 211, 153, 0.2)" }}>
                  <Lightbulb className="w-5 h-5" style={{ color: "#34D399" }} />
                </div>
                <h3 className="about-card-title">The Solution</h3>
              </div>
              <div className="about-card-body">
                <p>
                  AirQ addresses this challenge with a machine learning-powered forecasting system trained on
                  <span> historical lag data and weather variables</span>.
                  By treating past AQI values (1-hour and 24-hour lags) as the most predictive features,
                  the model captures the momentum of pollution events before they peak.
                </p>
                <p>
                  The core algorithm — a <span className="highlight">Random Forest Regressor</span> —
                  was chosen for its robustness to non-linear relationships, resistance to overfitting on tabular
                  data, and interpretability through feature importance scores.
                </p>
                <p>
                  The result is a system that can predict AQI up to 24 hours ahead with strong accuracy,
                  giving residents, city officials, and health workers the lead time needed to act — before
                  the smog arrives.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Project Details */}
      <section id="project" className="about-section-alt">
        <div ref={projectRef} className="section-fade about-container">
          {/* Liquid Glass Header */}
          <div className="about-header-card">
            <div className="about-header-glow-1"></div>
            <div className="about-header-glow-2"></div>

            <div className="about-header-content">
              <p className="about-eyebrow">Technical</p>
              <h2 className="about-title">
                Project <span>Details</span>
              </h2>
            </div>
          </div>

          {/* Tech stack */}
          <div>
            <h3 className="about-tech-subtitle">Tech Stack</h3>
            <div className="about-tech-grid">
              {techStack.map((tech) => {
                const Icon = tech.icon;
                return (
                  <div
                    key={tech.name}
                    className="depth-card about-tech-card"
                    style={{ backgroundColor: `${tech.color}1A`, borderColor: `${tech.color}33` }}
                  >
                    <Icon className="w-6 h-6 mx-auto mb-2" style={{ color: tech.color }} />
                    <p className="about-tech-name" style={{ color: tech.color }}>{tech.name}</p>
                    <p className="about-tech-desc">{tech.desc}</p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Model explanation */}
          <div className="depth-card about-model-card">
            <h3 className="about-model-title">
              <Cpu className="w-5 h-5" style={{ color: "#34D399" }} />
              Random Forest Regressor — How It Works
            </h3>
            <div className="about-model-grid">
              <div>
                <h4 className="about-model-sub">Architecture</h4>
                <p style={{ marginBottom: "0.75rem" }}>
                  A Random Forest is an ensemble of decision trees, each trained on a random bootstrap sample
                  of the training data. For regression tasks like AQI prediction, the final output is the
                  mean prediction across all trees — typically 100–500 estimators.
                </p>
                <p>
                  Each tree independently learns non-linear splits on features like{" "}
                  <span style={{ color: "#6EE7B7", fontFamily: "monospace", fontSize: "0.75rem" }}>AQI_lag24</span>,{" "}
                  <span style={{ color: "#6EE7B7", fontFamily: "monospace", fontSize: "0.75rem" }}>temperature</span>, and{" "}
                  <span style={{ color: "#6EE7B7", fontFamily: "monospace", fontSize: "0.75rem" }}>wind_speed</span>.
                  By averaging, the ensemble cancels out individual tree variance.
                </p>
              </div>
              <div>
                <h4 className="about-model-sub">Key Features Used</h4>
                <ul className="about-model-list">
                  {[
                    { name: "AQI_lag24", pct: "41%", desc: "AQI value 24 hours prior" },
                    { name: "AQI_lag1", pct: "28%", desc: "AQI value 1 hour prior" },
                    { name: "Temperature", pct: "12%", desc: "Affects pollutant dispersion" },
                    { name: "Wind Speed", pct: "9%", desc: "Ventilation coefficient" },
                    { name: "Hour", pct: "5%", desc: "Diurnal traffic patterns" },
                    { name: "Precipitation", pct: "3%", desc: "Wet deposition effect" },
                  ].map((f) => (
                    <li key={f.name} className="about-model-li">
                      <span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                        <span style={{ width: "0.375rem", height: "0.375rem", backgroundColor: "#34D399", borderRadius: "50%" }} />
                        <span style={{ color: "#6EE7B7", fontFamily: "monospace", fontSize: "0.75rem" }}>{f.name}</span>
                        <span style={{ color: "var(--color-gray-500)", fontSize: "0.75rem" }}>{f.desc}</span>
                      </span>
                      <span style={{ color: "#34D399", fontSize: "0.75rem", fontWeight: "700" }}>{f.pct}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Developer Credits */}
      <section id="credits" className="about-section">
        <div ref={creditsRef} className="section-fade about-credits-container">
          {/* Liquid Glass Header */}
          <div className="about-header-card">
            <div className="about-header-glow-1" style={{ top: "50%", left: "0", transform: "translate(-50%, -50%)" }}></div>
            <div className="about-header-glow-2" style={{ top: "50%", right: "0", transform: "translate(50%, -50%)" }}></div>

            <div className="about-header-content">
              <p className="about-eyebrow">Credits</p>
              <h2 className="about-title">
                Developer <span>Credits</span>
              </h2>
            </div>
          </div>

          <div className="depth-card about-card">
            {/* Avatar */}
            <div className="about-avatar">
              <span>P</span>
            </div>

            <h3 className="about-dev-name">Pratham</h3>
            <p className="about-dev-role">Full-Stack Developer & ML Engineer</p>
            <p className="about-dev-edu">BCA Student · Christ University, Bangalore</p>
            <p className="about-dev-desc" style={{ marginBottom: "1.5rem" }}>
              Developed and trained the AirQ prediction system as part of academic research into
              environmental machine learning and real-time data applications.
            </p>

            <p className="about-dev-desc" style={{ marginBottom: "1.5rem" }}>
              This project combines data science, meteorological modeling, and modern web development
              to address one of India&apos;s most pressing public health challenges. AirQ was designed,
              built, and trained entirely by Pratham — from feature engineering and model evaluation
              in Python to the full React/Tailwind frontend.
            </p>

            {/* Links */}
            <div className="about-links">
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="about-btn-gh"
              >
                <Github className="w-4 h-4" />
                GitHub
              </a>
              <a
                href="https://linkedin.com"
                target="_blank"
                rel="noopener noreferrer"
                className="about-btn-in"
              >
                <Linkedin className="w-4 h-4" />
                LinkedIn
              </a>
            </div>
          </div>

          <p className="about-techstack-footer">
            Built with React · Vite · Recharts · CSS · Python · Scikit-Learn
          </p>
        </div>
      </section>
    </>
  );
}
