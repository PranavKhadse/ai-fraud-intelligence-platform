import React from 'react';

interface LoadingSpinnerProps {
  message?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  message = 'Loading fraud intelligence telemetry...',
  size = 'md',
}) => {
  return (
    <div className={`spinner-container spinner-${size}`}>
      <div className="spinner" role="status" aria-label="Loading" />
      {message && <p className="spinner-text">{message}</p>}
    </div>
  );
};
