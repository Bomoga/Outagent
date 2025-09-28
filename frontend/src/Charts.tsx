import { useEffect, useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";

type Series = {
  key: string;
  label: string;
  color: string;
  values: number[];
};

type RiskData = {
  horizon_hours: number[];
  risk_score: number[];
};

type LineChartProps = {
  title: string;
  series: Series[];
  xLabels: string[];
  width?: number;
  height?: number;
};

type ChartsProps = {
  selectedMetric: string;
  setSelectedMetric: React.Dispatch<React.SetStateAction<string>>;
};

const MARGIN = { top: 24, right: 24, bottom: 36, left: 48 };

function LineChart({
  title,
  series,
  xLabels,
  width = 560,
  height,
}: LineChartProps) {
  const calculatedHeight = height ?? 0;

  const plot = useMemo(() => {
    const chartHeight = calculatedHeight || 300;
    const innerW = width - MARGIN.left - MARGIN.right;
    const innerH = chartHeight - MARGIN.top - MARGIN.bottom;

    const allValues = series.flatMap((s) => s.values.filter((v) => !isNaN(v)));
    let yMin = Math.min(...allValues);
    let yMax = Math.max(...allValues);
    if (!isFinite(yMin) || !isFinite(yMax)) {
      yMin = 0;
      yMax = 1;
    }
    if (yMin === yMax) {
      yMin -= 1;
      yMax += 1;
    }

    const n = Math.max(1, Math.max(...series.map((s) => s.values.length)));
    const x = (i: number) =>
      MARGIN.left + (n === 1 ? 0 : (i / (n - 1)) * innerW);
    const y = (v: number) =>
      MARGIN.top + innerH - ((v - yMin) / (yMax - yMin)) * innerH;

    const polylines = series.map((s) => ({
      key: s.key,
      color: s.color,
      points: s.values
        .map((v, i) => (isNaN(v) ? null : `${x(i)},${y(v)}`))
        .filter(Boolean)
        .join(" "),
    }));

    const ticks = 5;
    const yTicks = Array.from({ length: ticks + 1 }, (_, i) => {
      const value = yMin + (i / ticks) * (yMax - yMin);
      return { value, y: y(value) };
    });

    const xTickEvery = Math.ceil(xLabels.length / 8) || 1;
    const xTicks = xLabels
      .map((lbl, i) => ({ i, lbl, x: x(i) }))
      .filter((t) => t.i % xTickEvery === 0);

    return { polylines, yTicks, xTicks, chartHeight };
  }, [series, xLabels, width, calculatedHeight]);

  return (
    <motion.div
      className="w-full h-full bg-white rounded-xl p-3 sm:p-6 shadow-md ring-1 ring-gray-200 flex flex-col"
      layout
      transition={{ duration: 0.6, ease: "easeInOut" }}
    >
      <div className="text-gray-900 font-semibold mb-2 text-base sm:text-lg">
        {title}
      </div>

      <svg
        width="100%"
        height="100%"
        viewBox={`0 0 ${width} ${plot.chartHeight}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={title}
        className="flex-1"
      >
        {/* Axes */}
        <line
          x1={MARGIN.left}
          y1={plot.chartHeight - MARGIN.bottom}
          x2={width - MARGIN.right}
          y2={plot.chartHeight - MARGIN.bottom}
          stroke="#9ca3af"
          strokeWidth={1}
        />
        <line
          x1={MARGIN.left}
          y1={MARGIN.top}
          x2={MARGIN.left}
          y2={plot.chartHeight - MARGIN.bottom}
          stroke="#9ca3af"
          strokeWidth={1}
        />

        {/* Y grid + labels */}
        {plot.yTicks.map((t, idx) => (
          <g key={`y-${idx}`}>
            <line
              x1={MARGIN.left}
              y1={t.y}
              x2={width - MARGIN.right}
              y2={t.y}
              stroke="#e5e7eb"
              strokeWidth={1}
            />
            <text
              x={MARGIN.left - 8}
              y={t.y}
              textAnchor="end"
              dominantBaseline="middle"
              fill="#6b7280"
              fontSize={11}
            >
              {t.value.toFixed(1)}
            </text>
          </g>
        ))}

        {/* X labels */}
        {plot.xTicks.map((t, idx) => (
          <text
            key={`x-${idx}`}
            x={t.x}
            y={plot.chartHeight - MARGIN.bottom + 16}
            textAnchor="middle"
            fill="#6b7280"
            fontSize={11}
          >
            {t.lbl}
          </text>
        ))}

        {/* Lines */}
        {plot.polylines.map((p) => (
          <polyline
            key={p.key}
            fill="none"
            stroke={p.color}
            strokeWidth={2}
            strokeDasharray={p.key.includes("forecast") ? "6 4" : "none"}
            strokeLinejoin="round"
            strokeLinecap="round"
            points={p.points}
          />
        ))}
      </svg>
    </motion.div>
  );
}

const COLORS = ["#2563eb", "#16a34a", "#dc2626", "#f59e0b"];

export default function Charts({
  selectedMetric,
  setSelectedMetric,
}: ChartsProps) {
  const [weatherData, setWeatherData] = useState<any>(null);
  const [weatherForecast, setWeatherForecast] = useState<any>(null);
  const [forecastData, setForecastData] = useState<number[]>([]);
  const [riskData, setRiskData] = useState<RiskData | null>(null);
  const [initialLoading, setInitialLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      const [weatherRes, forecastRes, loadRes, riskRes] = await Promise.all([
        fetch("http://127.0.0.1:8000/history/weather"),
        fetch("http://127.0.0.1:8000/forecast/weather"),
        fetch("http://127.0.0.1:8000/forecast/load"),
        fetch("http://127.0.0.1:8000/forecast/risk"),
      ]);

      if (weatherRes.ok) setWeatherData(await weatherRes.json());
      if (forecastRes.ok) setWeatherForecast(await forecastRes.json());
      if (loadRes.ok) setForecastData((await loadRes.json()).prediction_mw);
      if (riskRes.ok) setRiskData(await riskRes.json());

      if (initialLoading) setInitialLoading(false);
    }

    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const hours =
    weatherData?.timestamps?.map((t: string) =>
      new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    ) ?? [];

  const weatherSeries: Series[] = weatherData
    ? [
        {
          key: "ghi_kwhm2-history",
          label: "Insulation",
          color: COLORS[0],
          values: weatherData.ghi_kwhm2,
        },
        {
          key: "ghi_kwhm2-forecast",
          label: "Insulation Forecast",
          color: COLORS[0],
          values: [
            ...Array(weatherData.ghi_kwhm2.length).fill(NaN),
            ...(weatherForecast?.pred?.ghi_kwhm2 ?? []),
          ],
        },
        {
          key: "wind_mps-history",
          label: "Windspeed",
          color: COLORS[1],
          values: weatherData.wind_mps,
        },
        {
          key: "wind_mps-forecast",
          label: "Windspeed Forecast",
          color: COLORS[1],
          values: [
            ...Array(weatherData.wind_mps.length).fill(NaN),
            ...(weatherForecast?.pred?.wind_mps ?? []),
          ],
        },
        {
          key: "precip_mm-history",
          label: "Precipitation",
          color: COLORS[2],
          values: weatherData.precip_mm,
        },
        {
          key: "precip_mm-forecast",
          label: "Precipitation Forecast",
          color: COLORS[2],
          values: [
            ...Array(weatherData.precip_mm.length).fill(NaN),
            ...(weatherForecast?.pred?.precip_mm ?? []),
          ],
        },
        {
          key: "wet_bulb-history",
          label: "Wet Bulb @2m",
          color: COLORS[3],
          values: weatherData.wet_bulb,
        },
      ]
    : [];

  const powerSeries: Series[] = [
    {
      key: "load-actual",
      label: "Actual Demand",
      color: "#2563eb",
      values: Array.from({ length: hours.length }).map(
        () => Math.random() * 100
      ),
    },
    {
      key: "load-forecast",
      label: "Predicted Demand",
      color: "#2563eb",
      values: [...Array(hours.length).fill(NaN), ...forecastData],
    },
  ];

  const selectedSeries = weatherSeries.filter((s) =>
    s.label.startsWith(selectedMetric)
  );

  const highestRisk = useMemo(
    () => (riskData ? Math.max(...riskData.risk_score) : 0),
    [riskData]
  );
  const riskLevel =
    highestRisk < 0.3 ? "LOW" : highestRisk < 0.6 ? "MODERATE" : "HIGH";
  const riskColor =
    riskLevel === "LOW"
      ? "text-green-600"
      : riskLevel === "MODERATE"
      ? "text-yellow-600"
      : "text-red-600";

  if (initialLoading) {
    return (
      <div className="flex flex-col gap-6 h-full animate-pulse">
        <div className="h-[calc(50%-0.75rem)] bg-gray-200 rounded-xl" />
        <div className="h-[calc(50%-0.75rem)] bg-gray-200 rounded-xl" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 h-full">
      {/* Weather Chart */}
      <div className="h-[calc(50%-0.75rem)] bg-gray-50 rounded-xl p-4 shadow-lg flex flex-col">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">
          Weather Metrics
        </h2>
        <div className="flex flex-wrap gap-2 mb-4">
          {["Insulation", "Windspeed", "Precipitation", "Wet Bulb @2m"].map(
            (label) => (
              <button
                key={label}
                onClick={() => setSelectedMetric(label)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition ${
                  selectedMetric === label
                    ? "bg-blue-600 text-white shadow"
                    : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-100"
                }`}
              >
                {label}
              </button>
            )
          )}
        </div>
        <AnimatePresence mode="wait">
          <LineChart
            key={selectedMetric}
            title={`Weather: ${selectedMetric}`}
            series={selectedSeries}
            xLabels={[
              ...hours,
              ...(weatherForecast?.horizon_hours ?? []).map(
                (h: number) => `+${h}h`
              ),
            ]}
            height={200}
          />
        </AnimatePresence>

        {/* Risk Assessment Panel */}
        {riskData && (
          <div className="mt-2 px-3 py-2 rounded-md border border-gray-200 bg-gray-100 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className={`text-sm font-semibold ${riskColor}`}>
                Risk Level: {riskLevel} ({highestRisk.toFixed(2)})
              </span>
            </div>
            <span className="text-xs text-gray-700">
              {riskLevel === "HIGH"
                ? "🚨 Immediate action recommended"
                : riskLevel === "MODERATE"
                ? "⚠️ Monitor conditions closely"
                : "✅ Normal conditions"}
            </span>
          </div>
        )}
      </div>

      {/* Power Chart */}
      <div className="h-[calc(50%-0.75rem)] bg-gray-50 rounded-xl p-4 shadow-lg flex flex-col">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">
          Power Output (Actual + 12h Prediction)
        </h2>
        <LineChart
          title="MW per Hour"
          series={powerSeries}
          xLabels={[...hours, ...forecastData.map((_, i) => `+${i + 1}h`)]}
          height={250}
        />
      </div>
    </div>
  );
}
