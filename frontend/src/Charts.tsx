import { useMemo, useState } from "react";

type Series = {
  key: string;
  label: string;
  color: string;
  values: number[];
};

type LineChartProps = {
  title: string;
  series: Series[];
  xLabels: string[];
  width?: number;
  height?: number;
};

const MARGIN = { top: 24, right: 24, bottom: 36, left: 48 };

function LineChart({
  title,
  series,
  xLabels,
  width = 560,
  height,
}: LineChartProps) {
  const calculatedHeight = height ?? 0; // we’ll set the SVG height to 100% when height is undefined

  const plot = useMemo(() => {
    // If no fixed height provided, use a reasonable fallback for calculations (300px)
    const chartHeight = calculatedHeight || 300;
    const innerW = width - MARGIN.left - MARGIN.right;
    const innerH = chartHeight - MARGIN.top - MARGIN.bottom;

    const allValues = series.flatMap((s) => s.values);
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
      points: s.values.map((v, i) => `${x(i)},${y(v)}`).join(" "),
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

    return {
      innerW,
      innerH,
      x,
      y,
      yMin,
      yMax,
      polylines,
      yTicks,
      xTicks,
      chartHeight,
    };
  }, [series, xLabels, width, calculatedHeight]);

  return (
    <div className="w-full h-full bg-white rounded-xl p-3 sm:p-6 shadow-md ring-1 ring-gray-200 flex flex-col">
      <div className="text-gray-900 font-semibold mb-2 text-base sm:text-lg">
        {title}
      </div>
      <svg
        width="100%"
        height="100%" // fill parent height if no height provided
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
            strokeLinejoin="round"
            strokeLinecap="round"
            points={p.points}
          />
        ))}
      </svg>
    </div>
  );
}

const COLORS = ["#2563eb", "#16a34a", "#dc2626", "#f59e0b"];

function generateHours(len: number) {
  return Array.from({ length: len }, (_, i) => `${i}:00`);
}

export type ChartsProps = {
  hours?: string[];
  weatherSeries?: Series[];
  powerSeries?: Series[];
};

const Charts = ({
  hours: hoursProp,
}: ChartsProps) => {
  const hours = hoursProp ?? generateHours(24);

  const defaultWeather: Series[] = [
    {
      key: "Insulation",
      label: "Insulation",
      color: COLORS[0],
      values: hours.map((_, i) => +(12 + Math.sin(i / 2) * 3).toFixed(2)),
    },
    {
      key: "Windspeed",
      label: "Windspeed",
      color: COLORS[1],
      values: hours.map((_, i) => +(6 + Math.cos(i / 3) * 2).toFixed(2)),
    },
    {
      key: "Pecipitation",
      label: "Precipitation",
      color: COLORS[2],
      values: hours.map((_, i) => +Math.max(0, Math.sin(i / 4) * 2).toFixed(2)),
    },
    {
      key: "Wet Bulb/2m",
      label: "Wet Bulb @2m",
      color: COLORS[3],
      values: hours.map((_, i) => +(8 + Math.sin(i / 2.5) * 1.5).toFixed(2)),
    },
  ];

  const defaultPower: Series[] = [
    {
      key: "mwh",
      label: "MW per Hour",
      color: COLORS[0],
      values: hours.map((_, i) => +(50 + Math.sin(i / 3) * 15).toFixed(2)),
    },
  ];

  const [selectedMetric, setSelectedMetric] = useState<string>(
    defaultWeather[0].key
  );

  const selectedSeries = defaultWeather.filter((s) => s.key === selectedMetric);

  return (
    <div className="flex flex-col gap-6 h-full">
  {/* Weather Card */}
  <div className="h-[calc(50%-0.75rem)] bg-gray-50 rounded-xl p-4 sm:p-6 shadow-lg flex flex-col">
    <h2 className="text-lg font-semibold text-gray-900 mb-3">Weather Metrics</h2>
    <div className="flex flex-wrap gap-2 mb-4">
      {defaultWeather.map((s) => (
        <button
          key={s.key}
          onClick={() => setSelectedMetric(s.key)}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium transition ${
            selectedMetric === s.key
              ? "bg-blue-600 text-white shadow"
              : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-100"
          }`}
        >
          {s.label}
        </button>
      ))}
    </div>
    {/* Chart fills remaining space */}
    <div className="flex-1">
      <LineChart
        title={`Weather: ${selectedMetric}`}
        series={selectedSeries}
        xLabels={hours}
        height={200}
      />
    </div>
  </div>


      {/* Power Card */}
  <div className="h-[calc(50%-0.75rem)] bg-gray-50 rounded-xl p-4 sm:p-6 shadow-lg flex flex-col">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">
          Power Output
        </h2>
        <div className="flex-1">
          <LineChart
            title="MW per Hour"
            series={defaultPower}
            xLabels={hours}
            height={250}
          />
        </div>
      </div>
    </div>
  );
};

export default Charts;
