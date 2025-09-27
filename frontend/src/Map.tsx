import { GoogleMap, LoadScript, Circle } from '@react-google-maps/api';

const center = { lat: 25.7617, lng: -80.1918 };

const containerStyle = {
  width: '100%',
  height: '100%',
};

// Example precipitation data
const precipitationData = [
  { id: 1, lat: 25.8, lng: -80.2, value: 5 }, // value = mm of rain
  { id: 2, lat: 25.6, lng: -80.3, value: 10 },
];

const Map = ({ showPrecipitation }: { showPrecipitation?: boolean }) => {
  return (
    <div className="bg-white rounded-xl p-4 shadow-lg h-full flex flex-col min-h-[500px]">
      <h2 className="text-xl font-semibold text-black mb-4">South Florida Map</h2>
      <div className="flex-1 min-h-[300px]">
        <LoadScript googleMapsApiKey={import.meta.env.VITE_GOOGLE_MAPS_API_KEY}>
          <GoogleMap
            mapContainerStyle={containerStyle}
            center={center}
            zoom={8}
          >
            {showPrecipitation &&
              precipitationData.map((point) => (
                <Circle
                  key={point.id}
                  center={{ lat: point.lat, lng: point.lng }}
                  radius={point.value * 500} // scale radius based on precipitation
                  options={{
                    fillColor: "blue",
                    fillOpacity: 0.3,
                    strokeWeight: 0,
                  }}
                />
              ))}
          </GoogleMap>
        </LoadScript>
      </div>
    </div>
  );
};

export default Map;
