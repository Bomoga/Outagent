import "./App.css";
import Charts from "./Charts";
import Map from "./Map";
import Header from "./Header";

function App() {
  return (
    <>
    <div className="bg-black text-slate-100 p-1">
      <div className="mx-auto max-w-7xl px-4 py-6">
        <Header/>

        <div className="mt-6 flex flex-col lg:flex-row gap-6 h-[calc(100vh-120px)]">
          {/* Left Side - Charts */}
          <div className="flex-1 flex flex-col gap-6">
            <Charts />
          </div>

          {/* Right Side - Map */}
          <div className="flex-1 h-[400px] lg:h-auto">
            <Map />
          </div>
        </div>
      </div>
    </div>
    </> 
  );
}

export default App;
