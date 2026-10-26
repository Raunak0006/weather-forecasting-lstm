import React, { useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import {
  CloudSun,
  LayoutDashboard,
  LineChart,
  BrainCircuit,
  Database,
  Cpu,
  Menu,
  X,
  Sparkles,
} from 'lucide-react';

export const Layout: React.FC = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { name: 'Dashboard', to: '/dashboard', icon: LayoutDashboard },
    { name: 'Historical Data', to: '/history', icon: LineChart },
    { name: 'Forecast', to: '/forecast', icon: CloudSun },
    { name: 'Model Performance', to: '/performance', icon: BrainCircuit },
    { name: 'Dataset Management', to: '/dataset', icon: Database },
    { name: 'Model Training', to: '/training', icon: Cpu },
  ];

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-2xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-tr from-blue-600 to-sky-500 rounded-xl text-white shadow-sm">
                <CloudSun className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-slate-900 tracking-tight leading-tight">
                  Weather Forecasting Using LSTM
                </h1>
                <p className="text-xs text-blue-600 font-medium hidden sm:block">
                  Deep Learning Time-Series Predictive Intelligence
                </p>
              </div>
            </div>

            {/* Desktop header badges */}
            <div className="hidden md:flex items-center space-x-3">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-blue-50 text-blue-700 border border-blue-200">
                <Sparkles className="w-3.5 h-3.5 text-blue-600" />
                B.Tech AI Project
              </span>
              <span className="text-xs font-medium text-slate-500 bg-slate-100 px-2.5 py-1 rounded-md">
                FastAPI + PyTorch/TF + React
              </span>
            </div>

            {/* Mobile menu button */}
            <div className="flex md:hidden">
              <button
                type="button"
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="text-slate-600 hover:text-slate-900 p-2 rounded-lg"
              >
                {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="flex-1 flex max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 gap-6">
        {/* Desktop Sidebar Navigation */}
        <aside className="hidden md:block w-64 shrink-0">
          <nav className="bg-white rounded-2xl shadow-sm border border-slate-200/80 p-3 sticky top-24 space-y-1">
            <div className="px-3 py-2 text-2xs font-bold text-slate-400 uppercase tracking-wider">
              Navigation
            </div>
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    `flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                      isActive
                        ? 'bg-blue-600 text-white shadow-sm shadow-blue-500/20 font-semibold'
                        : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                    }`
                  }
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  <span>{item.name}</span>
                </NavLink>
              );
            })}
          </nav>
        </aside>

        {/* Mobile Navigation Drawer */}
        {mobileMenuOpen && (
          <div className="fixed inset-0 z-40 bg-black/40 md:hidden" onClick={() => setMobileMenuOpen(false)}>
            <div
              className="fixed inset-y-0 left-0 w-64 bg-white p-4 shadow-xl flex flex-col justify-between"
              onClick={(e) => e.stopPropagation()}
            >
              <div>
                <div className="flex items-center justify-between pb-4 border-b border-slate-200">
                  <div className="flex items-center space-x-2">
                    <CloudSun className="w-5 h-5 text-blue-600" />
                    <span className="font-bold text-slate-800 text-sm">Weather LSTM</span>
                  </div>
                  <button onClick={() => setMobileMenuOpen(false)}>
                    <X className="w-5 h-5 text-slate-500" />
                  </button>
                </div>
                <nav className="mt-4 space-y-1">
                  {navItems.map((item) => {
                    const Icon = item.icon;
                    return (
                      <NavLink
                        key={item.to}
                        to={item.to}
                        onClick={() => setMobileMenuOpen(false)}
                        className={({ isActive }) =>
                          `flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-sm font-medium ${
                            isActive
                              ? 'bg-blue-600 text-white font-semibold'
                              : 'text-slate-600 hover:bg-slate-100'
                          }`
                        }
                      >
                        <Icon className="w-4 h-4" />
                        <span>{item.name}</span>
                      </NavLink>
                    );
                  })}
                </nav>
              </div>
            </div>
          </div>
        )}

        {/* Main Content View */}
        <main className="flex-1 min-w-0">
          <Outlet />
        </main>
      </div>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-2">
          <p>© 2026 Weather Forecasting Using LSTM • Academic Project (B.Tech Computer Science / AI)</p>
          <p className="flex items-center space-x-2">
            <span>Model: Stacked LSTM RNN</span>
            <span>•</span>
            <span>Baseline: Persistence Lag-1</span>
          </p>
        </div>
      </footer>
    </div>
  );
};
