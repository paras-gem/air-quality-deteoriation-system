import { useState, useEffect } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceArea,
  ReferenceLine,
  Label,
} from "recharts";
import { Info, RefreshCw, TrendingUp } from "lucide-react";

interface DataPoint {
  time: string;
  aqi: number;
  isPredicted: boolean;
}

const cities = ["Delhi", "Mumbai", "Kolkata", "Hyderabad"];

export default function PredictionTrendChart() {
  const [selectedCity, setSelectedCity] = useState("Delhi");
  const [data, setData] = useState<DataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [forecastType, setForecastType] = useState<"1h" | "24h">("24h");
  const [lastSynced, setLastSynced] = useState<string>(new Date().toLocaleTimeString());

  const fetchData = async (city: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`http://localhost:8000/forecast?city=${city}`);
      if (!response.ok) throw new Error("Failed to fetch forecast data");
      const json = await response.json();
      setData(json.data);
      setLastSynced(new Date().toLocaleTimeString());
    } catch (err) {
      console.error(err);
      setError("Unable to sync with live weather grid.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData(selectedCity);
  }, [selectedCity]);

  // Find the index where prediction starts for styling
  const predictionStartIndex = data.findIndex((dp) => dp.isPredicted);
  const currentTimestamp = data[predictionStartIndex - 1]?.time;

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const isPred = payload[0].payload.isPredicted;
      return (
        <div className="backdrop-blur-md bg-black/60 border border-emerald-500/30 p-3 rounded-lg shadow-2xl">
          <p className="text-gray-400 text-xs mb-1">{payload[0].payload.time}</p>
          <div className="flex items-center gap-2">
            <span className={`text-lg font-bold ${isPred ? "text-cyan-400" : "text-emerald-400"}`}>
              AQI {payload[0].value}
            </span>
            <span className={`text-[10px] px-1.5 py-0.5 rounded-full uppercase tracking-tighter ${
              isPred ? "bg-cyan-500/20 text-cyan-400" : "bg-emerald-500/20 text-emerald-400"
            }`}>
              {isPred ? "Predicted" : "Actual"}
            </span>
          </div>
        </div>
      );
    }
    return null;
  };

  const currentAQI = data.filter(d => !d.isPredicted).slice(-1)[0]?.aqi;
  const nextHourAQI = data.filter(d => d.isPredicted)[0]?.aqi;
  const next24hAQI = data.filter(d => d.isPredicted).slice(-1)[0]?.aqi;

  const displayPrediction = forecastType === "1h" ? nextHourAQI : next24hAQI;

  return (
    <div className="pred-chart-card" style={{ 
      marginBottom: '3rem', 
      backgroundColor: 'rgba(255, 255, 255, 0.65)', 
      border: '1px solid rgba(255, 255, 255, 0.8)',
      boxShadow: '0 25px 50px -12px rgba(16, 185, 129, 0.12)',
      padding: '2rem'
    }}>
      <div className="pred-chart-glow-1" style={{ backgroundColor: 'rgba(16, 185, 129, 0.2)' }} />
      <div className="pred-chart-glow-2" style={{ backgroundColor: 'rgba(96, 165, 250, 0.15)' }} />

      <div className="pred-chart-content">
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 280px', gap: '2.5rem', alignItems: 'start' }}>
          {/* Left Column: Metrics & Header */}
          <div style={{ minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--color-emerald-500)', fontWeight: '800', fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.15em' }}>
                <TrendingUp size={16} />
                <span>AI Prediction Hub</span>
              </div>
              <div style={{ fontSize: '0.65rem', color: 'var(--color-gray-400)', fontWeight: '700' }}>
                SYNC: {lastSynced}
              </div>
            </div>

            <h3 className="pred-title" style={{ margin: 0, fontSize: '2rem', letterSpacing: '-0.02em', color: 'var(--color-gray-900)' }}>
              {selectedCity} <span>AQI Projection</span>
            </h3>
            
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem', marginTop: '1.5rem' }}>
              {cities.map((city) => (
                <button
                  key={city}
                  onClick={() => setSelectedCity(city)}
                  className="home-badge"
                  style={{ 
                    cursor: 'pointer',
                    backgroundColor: selectedCity === city ? 'var(--color-emerald-500)' : 'rgba(255,255,255,0.8)',
                    color: selectedCity === city ? 'white' : 'var(--color-gray-700)',
                    borderColor: selectedCity === city ? 'var(--color-emerald-500)' : 'var(--color-gray-200)',
                    padding: '0.5rem 1.25rem',
                    fontSize: '0.8rem',
                    fontWeight: '700',
                    boxShadow: selectedCity === city ? '0 8px 20px -5px rgba(16, 185, 129, 0.4)' : 'none',
                    transform: selectedCity === city ? 'scale(1.03)' : 'scale(1)',
                    transition: 'all 0.4s ease'
                  }}
                >
                  {city}
                </button>
              ))}
            </div>
          </div>

          {/* Right Column: Key Prediction Values */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
             <div style={{ 
               backgroundColor: 'white', 
               padding: '1.25rem', 
               borderRadius: '1.25rem', 
               border: '1px solid var(--color-emerald-500)',
               boxShadow: '0 15px 35px -10px rgba(16, 185, 129, 0.2)',
               position: 'relative'
             }}>
               <div style={{ display: 'flex', padding: '0.2rem', backgroundColor: 'var(--color-gray-100)', borderRadius: '0.6rem', marginBottom: '1rem' }}>
                  <button 
                    onClick={() => setForecastType("1h")}
                    style={{ flex: 1, padding: '0.3rem', border: 'none', borderRadius: '0.5rem', fontSize: '0.65rem', fontWeight: '800', cursor: 'pointer', backgroundColor: forecastType === "1h" ? 'white' : 'transparent', color: forecastType === "1h" ? 'var(--color-emerald-600)' : 'var(--color-gray-500)' }}
                  >1H</button>
                  <button 
                    onClick={() => setForecastType("24h")}
                    style={{ flex: 1, padding: '0.3rem', border: 'none', borderRadius: '0.5rem', fontSize: '0.65rem', fontWeight: '800', cursor: 'pointer', backgroundColor: forecastType === "24h" ? 'white' : 'transparent', color: forecastType === "24h" ? 'var(--color-emerald-600)' : 'var(--color-gray-500)' }}
                  >24H</button>
               </div>

               <p style={{ margin: 0, fontSize: '0.65rem', fontWeight: '800', color: 'var(--color-emerald-600)', textTransform: 'uppercase' }}>
                 Forecast Insight
               </p>
               <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.4rem' }}>
                 <p style={{ margin: 0, fontSize: '2.5rem', fontWeight: '900', color: 'var(--color-gray-900)' }}>
                    {displayPrediction ? Math.round(displayPrediction) : "--"}
                 </p>
                 <span style={{ fontSize: '0.9rem', fontWeight: '800', color: 'var(--color-emerald-500)' }}>AQI</span>
               </div>
             </div>

             <div style={{ 
               backgroundColor: 'rgba(255,255,255,0.5)', 
               padding: '1rem', 
               borderRadius: '1rem', 
               border: '1px solid var(--color-gray-200)',
               display: 'flex',
               justifyContent: 'space-between',
               alignItems: 'center'
             }}>
               <div>
                  <p style={{ margin: 0, fontSize: '0.6rem', fontWeight: '700', color: 'var(--color-gray-500)', textTransform: 'uppercase' }}>OBSERVED</p>
                  <p style={{ margin: 0, fontSize: '1.25rem', fontWeight: '900', color: 'var(--color-gray-700)' }}>{currentAQI ? Math.round(currentAQI) : "--"}</p>
               </div>
               <button onClick={() => fetchData(selectedCity)} style={{ background: 'none', border: 'none', color: 'var(--color-emerald-500)', cursor: 'pointer' }}>
                  <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
               </button>
             </div>
          </div>
        </div>

        {/* Visual Trend Chart */}
        {error ? (
          <div style={{ 
            height: '250px', 
            marginTop: '2rem',
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            border: '2px dashed var(--color-gray-200)', 
            borderRadius: '1rem'
          }}>
            <p style={{ color: 'var(--color-gray-400)', fontSize: '0.8rem', fontWeight: '600' }}>Cloud Sync Offline</p>
          </div>
        ) : (
          <div style={{ height: '300px', width: '100%', position: 'relative', marginTop: '2rem' }}>
            {loading && (
              <div style={{ 
                position: 'absolute', 
                inset: 0, 
                zIndex: 20, 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                backgroundColor: 'rgba(255,255,255,0.5)',
                backdropFilter: 'blur(4px)',
                borderRadius: '1rem'
              }}>
                <RefreshCw size={24} className="animate-spin" color="var(--color-emerald-500)" />
              </div>
            )}
            
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorAQI" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--color-emerald-500)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="var(--color-emerald-500)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="5 5" vertical={false} stroke="rgba(0,0,0,0.04)" />
                <XAxis
                  dataKey="time"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: 'var(--color-gray-500)', fontSize: 10, fontWeight: '700' }}
                  dy={10}
                />
                <YAxis
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: 'var(--color-gray-500)', fontSize: 10, fontWeight: '700' }}
                />
                <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'var(--color-emerald-400)', strokeWidth: 1.5, strokeDasharray: '3 3' }} />
                
                {predictionStartIndex !== -1 && (
                  <ReferenceArea
                    x1={data[predictionStartIndex]?.time}
                    x2={data[data.length-1]?.time}
                    fill="rgba(16, 185, 129, 0.03)"
                  />
                )}

                <Area
                  type="monotone"
                  dataKey={(d) => d.isPredicted ? null : d.aqi}
                  stroke="var(--color-emerald-500)"
                  strokeWidth={4}
                  fillOpacity={1}
                  fill="url(#colorAQI)"
                  animationDuration={2500}
                />
                
                <Area
                  type="monotone"
                  dataKey={(d) => d.isPredicted || (predictionStartIndex > 0 && d.time === data[predictionStartIndex-1]?.time) ? d.aqi : null}
                  stroke="var(--color-emerald-500)"
                  strokeWidth={3}
                  strokeDasharray="6 6"
                  fillOpacity={0.05}
                  fill="var(--color-emerald-500)"
                  animationDuration={2500}
                />

                {currentTimestamp && (
                  <ReferenceLine x={currentTimestamp} stroke="rgba(0,0,0,0.15)" strokeWidth={1.5}>
                    <Label value="NOW" position="insideTopLeft" fill="var(--color-gray-500)" fontSize={9} fontWeight="900" dy={15} dx={5} />
                  </ReferenceLine>
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}

        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '1.5rem', marginTop: '2rem', paddingTop: '1.5rem', borderTop: '1px solid var(--color-gray-200)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ width: '0.6rem', height: '0.6rem', borderRadius: '50%', backgroundColor: 'var(--color-emerald-500)', boxShadow: '0 0 10px var(--color-emerald-400)' }} />
            <span style={{ fontSize: '0.7rem', color: 'var(--color-gray-500)', fontWeight: '800', textTransform: 'uppercase' }}>Observed</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ width: '0.6rem', height: '0.6rem', borderRadius: '50%', border: '2px dashed var(--color-emerald-500)', backgroundColor: 'transparent' }} />
            <span style={{ fontSize: '0.7rem', color: 'var(--color-gray-500)', fontWeight: '800', textTransform: 'uppercase' }}>Predicted</span>
          </div>
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'var(--color-gray-400)', fontSize: '0.65rem', fontWeight: '700' }}>
            <Info size={12} />
            SYNCED • MODEL: RF-v6
          </div>
        </div>
      </div>
    </div>
  );
}
