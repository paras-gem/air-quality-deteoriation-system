
import { useEffect, useRef, useMemo } from "react";
import { ArrowRight, Zap, TrendingUp, BarChart2, Activity, AlertTriangle, CheckCircle, Database } from "lucide-react";
import { Link } from "react-router-dom";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  Cell,
} from "recharts";
import PredictionTrendChart from "./PredictionTrendChart";

const cities = [
  { name: "Delhi", aqi: 178, level: "Poor" },
  { name: "Hyderabad", aqi: 94, level: "Moderate" },
  { name: "Kolkata", aqi: 145, level: "Poor" },
  { name: "Mumbai", aqi: 62, level: "Satisfactory" },
  { name: "Chennai", aqi: 71, level: "Satisfactory" },
  { name: "Bangalore", aqi: 53, level: "Good" },
];

const aqiColorMap: Record<string, string> = {
  Good: "text-emerald-400",
  Satisfactory: "text-lime-400",
  Moderate: "text-yellow-400",
  Poor: "text-orange-400",
  "Very Poor": "text-red-400",
  Severe: "text-purple-400",
};

const features = [
  {
    icon: Zap,
    title: "1-Hour Micro-Forecasts",
    desc: "Get hyper-local AQI predictions for the next hour using real-time lag features and weather data.",
  },
  {
    icon: TrendingUp,
    title: "24-Hour Trend Analysis",
    desc: "Understand how air quality will evolve throughout the day with confidence intervals and trend lines.",
  },
  {
    icon: BarChart2,
    title: "Data-Driven Insights",
    desc: "Visualize feature importance, model accuracy, and historical patterns powering every prediction.",
  },
];

const featureImportance = [
  { feature: "AQI_lag24", importance: 0.41 },
  { feature: "AQI_lag1", importance: 0.28 },
  { feature: "Temperature", importance: 0.12 },
  { feature: "Wind Speed", importance: 0.09 },
  { feature: "Hour", importance: 0.05 },
  { feature: "Precipitation", importance: 0.03 },
  { feature: "Month", importance: 0.02 },
].sort((a, b) => a.importance - b.importance);

function generateScatterData() {
  return Array.from({ length: 80 }, (_, i) => {
    const actual = Math.round(50 + Math.random() * 250);
    const predicted = Math.round(actual + (Math.random() - 0.5) * 40);
    return { actual, predicted, hour: i % 24 };
  });
}

const scatterData = generateScatterData();

const BarTooltip = ({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { value: number }[];
}) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-200 border border-gray-200 rounded-xl p-3 shadow-2xl text-xs">
        <p className="text-emerald-400 font-semibold">
          Importance: {(payload[0].value * 100).toFixed(1)}%
        </p>
      </div>
    );
  }
  return null;
};

const ScatterTooltip = ({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { name: string; value: number }[];
}) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-200 border border-gray-200 rounded-xl p-3 shadow-2xl text-xs">
        <p className="text-gray-500">Actual: <span className="text-emerald-400 font-bold">{payload[0]?.value}</span></p>
        <p className="text-gray-500">Predicted: <span className="text-blue-400 font-bold">{payload[1]?.value}</span></p>
      </div>
    );
  }
  return null;
};

export default function HomeSection() {
  const sectionRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = sectionRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => entry.isIntersecting && el.classList.add("visible"),
      { threshold: 0.1 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const marqueeItems = [...cities, ...cities];

  const kpis = useMemo(
    () => [
      {
        icon: Activity,
        label: "Average Current AQI",
        value: "124",
        sub: "Across 4 cities",
        color: "text-yellow-400",
        bg: "bg-yellow-400/10",
        border: "border-yellow-400/20",
      },
      {
        icon: AlertTriangle,
        label: "Worst Performing City",
        value: "Delhi",
        sub: "AQI 178 · Poor",
        color: "text-red-400",
        bg: "bg-red-400/10",
        border: "border-red-400/20",
      },
      {
        icon: CheckCircle,
        label: "Best Performing City",
        value: "Mumbai",
        sub: "AQI 62 · Satisfactory",
        color: "text-emerald-400",
        bg: "bg-emerald-400/10",
        border: "border-emerald-400/20",
      },
      {
        icon: Database,
        label: "Data Points Analyzed",
        value: "84,320",
        sub: "Updated daily",
        color: "text-blue-400",
        bg: "bg-blue-400/10",
        border: "border-blue-400/20",
      },
    ],
    []
  );

  return (
    <section id="home" className="home-section">
      {/* Hero */}
      <div ref={sectionRef} className="section-fade home-hero">
        <div className="home-hero-bg">
          <div className="home-hero-glow" />
        </div>

        <div className="home-hero-content">
          <div className="home-badge">
            <span className="home-badge-dot" />
            Live Air Quality Monitoring · India
          </div>

          <h1 className="home-title">
            Stay Ahead of{" "}
            <span className="home-title-highlight">
              the Smog.
              <svg className="home-title-svg" viewBox="0 0 300 8" fill="none">
                <path d="M0 6 Q75 2 150 6 Q225 10 300 6" stroke="#10B981" strokeWidth="2" strokeLinecap="round" opacity="0.6" />
              </svg>
            </span>
          </h1>

          <p className="home-subtitle">
            AirQ uses advanced <span>Random Forest</span> machine
            learning to predict air quality deterioration in major Indian cities before it
            happens.
          </p>

          <div className="home-actions">
            <Link to="/predict" className="glow-btn home-btn-primary">
              Run a Prediction
              <ArrowRight />
            </Link>
            <button
              onClick={() => document.getElementById("analytics")?.scrollIntoView({ behavior: "smooth" })}
              className="home-btn-secondary"
            >
              View Analytics
            </button>
          </div>
        </div>
      </div>

      {/* Live Status Ticker */}
      <div className="home-ticker">
        <div className="home-ticker-inner">
          <div className="home-ticker-label">
            <span />
            Live AQI
          </div>
          <div className="home-ticker-scroll">
            <div className="marquee-inner">
              {marqueeItems.map((city, i) => (
                <span key={i} className="home-ticker-city">
                  <span className="home-ticker-cname">{city.name}</span>
                  <span className={`home-ticker-cval ${aqiColorMap[city.level]}`}>
                    AQI {city.aqi}
                  </span>
                  <span className={`home-ticker-cbadge ${aqiColorMap[city.level]}`}>
                    {city.level}
                  </span>
                  <span className="home-ticker-dot">•</span>
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Analytics Appended Content (Previously Dashboard) */}
      <div id="analytics" className="home-analytics">
        <PredictionTrendChart />

        <div className="home-section-header">
          <p className="home-section-eyebrow">Analytics Hub</p>
          <h2 className="home-section-title">
            Real-Time <span>Dashboard</span>
          </h2>
          <p className="home-section-desc">
            Live metrics, model performance, and air quality trends across major Indian cities.
          </p>
        </div>

        {/* KPI Cards */}
        <div className="home-grid-kpi">
          {kpis.map((kpi) => {
            const Icon = kpi.icon;
            return (
              <div key={kpi.label} className={`depth-card home-card-kpi border ${kpi.border}`}>
                <div className={`home-card-icon ${kpi.bg} ${kpi.border}`}>
                  <Icon className={kpi.color} />
                </div>
                <p className="home-card-label">{kpi.label}</p>
                <p className={`home-card-val ${kpi.color}`}>{kpi.value}</p>
                <p className="home-card-sub">{kpi.sub}</p>
              </div>
            );
          })}
        </div>

        {/* Charts: Feature Importance + Scatter Plot */}
        <div className="home-grid-charts">
          {/* Feature Importance */}
          <div className="depth-card home-chart-card">
            <div className="home-chart-glow-1"></div>
            <div className="home-chart-content">
              <h3 className="home-chart-title">Feature Importance</h3>
              <p className="home-chart-desc">Random Forest model feature weights</p>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart
                  data={featureImportance}
                  layout="vertical"
                  margin={{ left: 10, right: 20, top: 4, bottom: 4 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                  <XAxis
                    type="number"
                    tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    tick={{ fill: "#6B7280", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    tick={{ fill: "#6B7280", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    width={80}
                  />
                  <Tooltip content={<BarTooltip />} cursor={{ fill: "rgba(0,0,0,0.05)" }} />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {featureImportance.map((_, i) => (
                      <Cell
                        key={i}
                        fill={`rgba(16,185,129,${0.4 + (i / featureImportance.length) * 0.6})`}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Scatter: 24h Forecast Accuracy */}
          <div className="depth-card home-chart-card">
            <div className="home-chart-glow-2"></div>
            <div className="home-chart-content">
              <h3 className="home-chart-title">24-Hour Forecast Accuracy</h3>
              <p className="home-chart-desc">Actual vs Predicted scatter · closer to diagonal = better</p>
              <ResponsiveContainer width="100%" height={260}>
                <ScatterChart margin={{ left: 0, right: 10, top: 4, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                  <XAxis
                    dataKey="actual"
                    name="Actual"
                    type="number"
                    tick={{ fill: "#6B7280", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    label={{ value: "Actual AQI", position: "insideBottom", offset: -4, fill: "#6B7280", fontSize: 11 }}
                  />
                  <YAxis
                    dataKey="predicted"
                    name="Predicted"
                    type="number"
                    tick={{ fill: "#6B7280", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    label={{ value: "Predicted AQI", angle: -90, position: "insideLeft", fill: "#6B7280", fontSize: 11 }}
                  />
                  <Tooltip content={<ScatterTooltip />} cursor={{ strokeDasharray: "3 3" }} />
                  <Scatter data={scatterData} fill="#10B981" fillOpacity={0.6} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <div className="home-features">
        <p className="home-section-eyebrow" style={{ textAlign: 'center' }}>Platform Capabilities</p>
        <h2 className="home-section-title" style={{ textAlign: 'center', marginBottom: '3rem' }}>
          Why Choose <span>AirQ?</span>
        </h2>
        <div className="home-grid-feat">
          {features.map((f) => {
            const Icon = f.icon;
            return (
              <div key={f.title} className="depth-card home-feat-card">
                <div className="home-feat-icon">
                  <Icon />
                </div>
                <h3 className="home-feat-title">{f.title}</h3>
                <p className="home-feat-desc">{f.desc}</p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
