import logo from "./assets/Images/logo.png";

const Header = () => {
  return (
    <header className="flex items-center justify-between gap-4 h-10">
      <img
        src={logo}
        alt="Outagent logo"
        className="h-30 w-auto drop-shadow-md"
      />

      {/* Title + Subtitle */}
      <div className="flex items-center gap-4">
        <h1 className="text-2xl font-bold tracking-tight text-yellow-400 border-r border-gray-500 pr-4">
          Dashboard
        </h1>
        <p className="text-sm text-slate-400">
          Hourly weather metrics and power output
        </p>
      </div>
    </header>
  );
};

export default Header;
