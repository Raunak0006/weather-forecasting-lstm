import React from 'react';

interface StatusBadgeProps {
  status: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status }) => {
  const getBadgeStyle = (st: string) => {
    switch (st.toLowerCase()) {
      case 'completed':
        return 'bg-emerald-100 text-emerald-800 border-emerald-300';
      case 'training':
        return 'bg-amber-100 text-amber-800 border-amber-300 animate-pulse';
      case 'failed':
        return 'bg-rose-100 text-rose-800 border-rose-300';
      case 'idle':
      default:
        return 'bg-slate-100 text-slate-700 border-slate-300';
    }
  };

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border ${getBadgeStyle(
        status
      )}`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full mr-1.5 ${
          status.toLowerCase() === 'completed'
            ? 'bg-emerald-500'
            : status.toLowerCase() === 'training'
            ? 'bg-amber-500'
            : status.toLowerCase() === 'failed'
            ? 'bg-rose-500'
            : 'bg-slate-400'
        }`}
      />
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
};
