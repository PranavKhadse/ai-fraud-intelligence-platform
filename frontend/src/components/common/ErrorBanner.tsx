import React from 'react';
import { AlertTriangle } from 'lucide-react';

interface ErrorBannerProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({
  title = 'API Communication Error',
  message,
  onRetry,
}) => {
  return (
    <div className="error-banner" role="alert">
      <AlertTriangle className="error-banner-icon" size={20} />
      <div className="error-banner-content">
        <h4 className="error-banner-title">{title}</h4>
        <p className="error-banner-message">{message}</p>
        {onRetry && (
          <button type="button" className="btn-retry" onClick={onRetry}>
            Retry Request
          </button>
        )}
      </div>
    </div>
  );
};
