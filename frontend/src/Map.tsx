import { useState, useEffect, useMemo, useCallback } from "react";
import {
  GoogleMap,
  LoadScript,
  Circle,
  Marker,
  HeatmapLayerF,
} from "@react-google-maps/api";

type MapProps = {
  selectedMetric: string;
};

type WeatherData = {
  timestamps: string[];
  ghi_kwhm2: number[];
  wind_mps: number[];
  wind_dir_deg: number[];
  precip_mm: number[];
  wet_bulb: number[];
};

type WeatherForecast = {
  horizon_hours: number[];
  pred: {
    ghi_kwhm2?: number[];
    wind_mps?: number[];
    wind_dir_deg?: number[];
    precip_mm?: number[];
    wet_bulb?: number[];
  };
};

const containerStyle = { width: "100%", height: "100%" };
const center = { lat: 26.1, lng: -80.2 };
const LIBRARIES: ("visualization")[] = ["visualization"];

export default function Map({ selectedMetric }: MapProps) {
  const [mapReady, setMapReady] = useState(false);
  const [weatherData, setWeatherData] = useState<WeatherData | null>(null);
  const [forecast, setForecast] = useState<WeatherForecast | null>(null);

  useEffect(() => {
    async function fetchWeather() {
      try {
        const [histRes, forecastRes] = await Promise.all([
          fetch("http://127.0.0.1:8000/history/weather"),
          fetch("http://127.0.0.1:8000/forecast/weather"),
        ]);
        if (histRes.ok) setWeatherData(await histRes.json());
        if (forecastRes.ok) setForecast(await forecastRes.json());
      } catch (err) {
        console.error("Failed to fetch weather data", err);
      }
    }
    fetchWeather();
    const interval = setInterval(fetchWeather, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleMapLoad = useCallback((map: google.maps.Map) => {
    setMapReady(true);
    const bounds = new window.google.maps.LatLngBounds();
    bounds.extend({ lat: 27.5, lng: -79.5 });
    bounds.extend({ lat: 24.3, lng: -82.5 });
    map.fitBounds(bounds);
  }, []);

  const locations = useMemo(
    () => [
      { id: 1, name: "Miami", lat: 25.7617, lng: -80.1918 },
      { id: 2, name: "Fort Lauderdale", lat: 26.1224, lng: -80.1373 },
      { id: 3, name: "West Palm", lat: 26.7153, lng: -80.0534 },
      { id: 4, name: "Homestead", lat: 25.4687, lng: -80.4776 },
    ],
    []
  );

  const currentValues = useMemo(() => {
    if (!weatherData) return [];
    switch (selectedMetric) {
      case "Insulation":
        return weatherData.ghi_kwhm2;
      case "Windspeed":
        return weatherData.wind_mps;
      case "Precipitation":
        return weatherData.precip_mm;
      case "Wet Bulb @2m":
        return weatherData.wet_bulb;
      default:
        return [];
    }
  }, [selectedMetric, weatherData]);

  const forecastValues = useMemo(() => {
    if (!forecast) return [];
    switch (selectedMetric) {
      case "Insulation":
        return forecast.pred.ghi_kwhm2 ?? [];
      case "Windspeed":
        return forecast.pred.wind_mps ?? [];
      case "Precipitation":
        return forecast.pred.precip_mm ?? [];
      case "Wet Bulb @2m":
        return forecast.pred.wet_bulb ?? [];
      default:
        return [];
    }
  }, [selectedMetric, forecast]);

  const currentDirections = weatherData?.wind_dir_deg ?? [];
  const forecastDirections = forecast?.pred.wind_dir_deg ?? [];

  const heatmapPoints = useMemo(() => {
    if (!mapReady || selectedMetric !== "Insulation" || typeof window.google === "undefined")
      return [];
    return locations.map((p) => new window.google.maps.LatLng(p.lat, p.lng));
  }, [mapReady, selectedMetric, locations]);

  return (
    <div className="bg-white rounded-xl p-4 shadow-lg h-full flex flex-col min-h-[500px]">
      <h2 className="text-xl font-semibold text-black mb-2">South Florida Map</h2>
      <div className="flex-1 min-h-[300px] relative">
        <LoadScript
          googleMapsApiKey={import.meta.env.VITE_GOOGLE_MAPS_API_KEY}
          libraries={LIBRARIES}
        >
          <GoogleMap
            key={selectedMetric}
            mapContainerStyle={containerStyle}
            center={center}
            zoom={8}
            onLoad={handleMapLoad}
            options={{
              restriction: {
                latLngBounds: {
                  north: 27.5,
                  south: 24.3,
                  west: -82.5,
                  east: -79.5,
                },
                strictBounds: true,
              },
              streetViewControl: false,
              mapTypeControl: true,
            }}
          >
            {mapReady && currentValues.length > 0 && (
              <>
                {selectedMetric === "Insulation" && heatmapPoints.length > 0 && (
                  <HeatmapLayerF
                    data={heatmapPoints}
                    options={{ radius: 40, opacity: 0.6, dissipating: true }}
                  />
                )}

                {selectedMetric === "Precipitation" &&
                  locations.map((loc, i) => (
                    <Circle
                      key={`precip-${loc.id}`}
                      center={{ lat: loc.lat, lng: loc.lng }}
                      radius={(currentValues[i % currentValues.length] ?? 0) * 600}
                      options={{
                        fillColor: "blue",
                        fillOpacity: 0.3,
                        strokeWeight: 0,
                      }}
                    />
                  ))}

                {selectedMetric === "Windspeed" &&
                  locations.map((loc, i) => (
                    <Marker
                      key={`wind-${loc.id}`}
                      position={{ lat: loc.lat, lng: loc.lng }}
                      icon={{
                        path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
                        scale: 5,
                        fillColor: "#1e3a8a",
                        fillOpacity: 0.9,
                        strokeWeight: 1,
                        rotation: currentDirections[i % currentDirections.length] ?? 0,
                      }}
                      label={{
                        text: `${currentValues[i % currentValues.length]?.toFixed(1) ?? "0"} m/s`,
                        fontSize: "12px",
                        fontWeight: "bold",
                        color: "#1e3a8a",
                      }}
                    />
                  ))}

                {forecastValues.length > 0 &&
                  selectedMetric === "Precipitation" &&
                  locations.map((loc, i) => (
                    <Circle
                      key={`precip-forecast-${loc.id}`}
                      center={{ lat: loc.lat, lng: loc.lng }}
                      radius={(forecastValues[i % forecastValues.length] ?? 0) * 600}
                      options={{
                        fillColor: "blue",
                        fillOpacity: 0.15,
                        strokeWeight: 1,
                        strokeColor: "blue",
                        strokeOpacity: 0.5,
                      }}
                    />
                  ))}

                {forecastValues.length > 0 &&
                  selectedMetric === "Windspeed" &&
                  locations.map((loc, i) => (
                    <Marker
                      key={`wind-forecast-${loc.id}`}
                      position={{ lat: loc.lat + 0.02, lng: loc.lng }}
                      icon={{
                        path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
                        scale: 5,
                        fillColor: "#3b82f6",
                        fillOpacity: 0.5,
                        strokeWeight: 1,
                        rotation: forecastDirections[i % forecastDirections.length] ?? 0,
                      }}
                      label={{
                        text: `${forecastValues[i % forecastValues.length]?.toFixed(1) ?? "0"} m/s`,
                        fontSize: "11px",
                        color: "#3b82f6",
                      }}
                    />
                  ))}
              </>
            )}
          </GoogleMap>
        </LoadScript>
      </div>
    </div>
  );
}
