import { useState, useCallback } from "react";
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

const containerStyle = { width: "100%", height: "100%" };
const center = { lat: 26.1, lng: -80.2 }; // Centered roughly over South Florida
const LIBRARIES: ("visualization")[] = ["visualization"];

// 🌧 Precipitation data: distributed across several South FL cities
const precipitationData = [
  { id: 1, lat: 25.7617, lng: -80.1918, value: 12 }, // Miami
  { id: 2, lat: 26.1224, lng: -80.1373, value: 6 },  // Fort Lauderdale
  { id: 3, lat: 26.7153, lng: -80.0534, value: 10 }, // West Palm Beach
  { id: 4, lat: 25.4687, lng: -80.4776, value: 15 }, // Homestead
  { id: 5, lat: 26.3587, lng: -80.0831, value: 8 },  // Boca Raton
  { id: 6, lat: 25.0865, lng: -80.4473, value: 20 }, // Key Largo
  { id: 7, lat: 26.0112, lng: -80.1495, value: 5 },  // Hollywood
];

// 💨 Windspeed data with directions
const windData = [
  { id: 1, lat: 25.7617, lng: -80.1918, value: 12, direction: 90 },
  { id: 2, lat: 26.1224, lng: -80.1373, value: 18, direction: 45 },
  { id: 3, lat: 26.7153, lng: -80.0534, value: 8, direction: 135 },
  { id: 4, lat: 25.4687, lng: -80.4776, value: 15, direction: 200 },
  { id: 5, lat: 26.3587, lng: -80.0831, value: 10, direction: 300 },
  { id: 6, lat: 26.2712, lng: -80.2706, value: 16, direction: 120 }, // Coral Springs
];

// 🔆 Insulation data (dense for heatmap effect)
const insulationData = [
  { lat: 25.7617, lng: -80.1918 },
  { lat: 26.1224, lng: -80.1373 },
  { lat: 26.7153, lng: -80.0534 },
  { lat: 25.4687, lng: -80.4776 },
  { lat: 26.3587, lng: -80.0831 },
  { lat: 25.0865, lng: -80.4473 },
  { lat: 26.0112, lng: -80.1495 },
  { lat: 26.2712, lng: -80.2706 },
  { lat: 26.1420, lng: -81.7948 }, // Naples
];

// 🌡 Wet Bulb Data
const wetBulbData = [
  { id: 1, lat: 25.7617, lng: -80.1918, value: 26 },
  { id: 2, lat: 26.1224, lng: -80.1373, value: 30 },
  { id: 3, lat: 26.7153, lng: -80.0534, value: 27 },
  { id: 4, lat: 25.4687, lng: -80.4776, value: 28 },
  { id: 5, lat: 26.3587, lng: -80.0831, value: 25 },
  { id: 6, lat: 26.0112, lng: -80.1495, value: 29 },
];

const Map = ({ selectedMetric }: MapProps) => {
  const [mapReady, setMapReady] = useState(false);
  const handleMapLoad = useCallback(() => setMapReady(true), []);

  return (
    <div className="bg-white rounded-xl p-4 shadow-lg h-full flex flex-col min-h-[500px]">
      <h2 className="text-xl font-semibold text-black mb-4">South Florida Map</h2>
      <div className="flex-1 min-h-[300px]">
        <LoadScript
          googleMapsApiKey={import.meta.env.VITE_GOOGLE_MAPS_API_KEY}
          libraries={LIBRARIES}
        >
          <GoogleMap
            mapContainerStyle={containerStyle}
            center={center}
            zoom={8}
            onLoad={handleMapLoad}
          >
            {mapReady && (
              <>
                {selectedMetric === "Precipitation" &&
                  precipitationData.map((point) => (
                    <Circle
                      key={point.id}
                      center={{ lat: point.lat, lng: point.lng }}
                      radius={point.value * 600}
                      options={{
                        fillColor: "blue",
                        fillOpacity: 0.3,
                        strokeWeight: 0,
                      }}
                    />
                  ))}

                {selectedMetric === "Windspeed" &&
                  windData.map((point) => (
                    <Marker
                      key={point.id}
                      position={{ lat: point.lat, lng: point.lng }}
                      icon={{
                        path: window.google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
                        scale: 4,
                        strokeColor: point.value > 15 ? "red" : "green",
                        rotation: point.direction,
                      }}
                    />
                  ))}

                {selectedMetric === "Insulation" && (
                  <HeatmapLayerF
                    data={insulationData.map(
                      (p) => new window.google.maps.LatLng(p.lat, p.lng)
                    )}
                    options={{
                      radius: 40,
                      opacity: 0.6,
                      dissipating: true,
                    }}
                  />
                )}

                {selectedMetric === "Wet Bulb/2m" &&
                  wetBulbData.map((point) => (
                    <Circle
                      key={point.id}
                      center={{ lat: point.lat, lng: point.lng }}
                      radius={700}
                      options={{
                        fillColor:
                          point.value < 24
                            ? "green"
                            : point.value < 28
                            ? "orange"
                            : "red",
                        fillOpacity: 0.35,
                        strokeWeight: 0,
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
};

export default Map;
