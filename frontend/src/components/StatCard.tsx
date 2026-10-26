import React from 'react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  trend?: string;
  trendPositive?: boolean;
}

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  trend,
  trendPositive,
}) => {
  return (
    <div className="bg-white rounded-xl shadow-md border border-slate-100 p-5 hover:shadow-lg transition-all duration-200">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-slate-500">{title}</p>
          <h3 className="text-2xl font-bold text-slate-800 mt-1">{value}</h3>
          {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
          {trend && (
            <span
              className={`inline-block text-xs font-semibold mt-2 px-2 py-0.5 rounded ${
                trendPositive ? 'bg-emerald-50 text-emerald-600' : 'bg-rose-50 text-rose-600'
              }`}
            >
              {trend}
            </span>
          )}
        </div>
        {icon && (
          <div className="p-3 bg-gradient-to-tr from-blue-600 to-sky-500 rounded-xl text-white shadow-sm">
            {icon}
          </div>
        )}
      </div>
    </div>
  );
};
