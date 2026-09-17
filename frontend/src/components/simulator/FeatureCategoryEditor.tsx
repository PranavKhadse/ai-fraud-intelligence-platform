import React, { useState } from 'react';
import { ChevronDown, ChevronRight, RotateCcw } from 'lucide-react';
import { CANONICAL_FEATURES, type FeatureMetadata } from '../../constants/featureCatalog.ts';

interface FeatureCategoryEditorProps {
  category: string;
  features: Record<string, any>;
  baselineFeatures: Record<string, any>;
  onChange: (featureName: string, value: any) => void;
  onResetFeature: (featureName: string) => void;
  defaultExpanded?: boolean;
}

export const FeatureCategoryEditor: React.FC<FeatureCategoryEditorProps> = ({
  category,
  features,
  baselineFeatures,
  onChange,
  onResetFeature,
  defaultExpanded = true,
}) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(defaultExpanded);

  const categoryFeatures: FeatureMetadata[] = CANONICAL_FEATURES.filter(
    (f) => f.category === category
  );

  const modifiedCount = categoryFeatures.filter((f) => {
    const curr = features[f.name];
    const base = baselineFeatures[f.name];
    return base !== undefined && curr !== base;
  }).length;

  return (
    <div className="feature-category-accordion">
      <button
        type="button"
        className="accordion-header"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
      >
        <div className="accordion-title-group">
          {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          <span className="accordion-category-name">{category}</span>
          <span className="accordion-count">({categoryFeatures.length} features)</span>
        </div>
        {modifiedCount > 0 && (
          <span className="modified-pill">
            {modifiedCount} modified
          </span>
        )}
      </button>

      {isExpanded && (
        <div className="accordion-body">
          <div className="feature-inputs-grid">
            {categoryFeatures.map((feat) => {
              const currentValue = features[feat.name] !== undefined ? features[feat.name] : feat.defaultValue;
              const baselineValue = baselineFeatures[feat.name];
              const isModified = baselineValue !== undefined && currentValue !== baselineValue;

              return (
                <div
                  key={feat.name}
                  className={`feature-input-card ${isModified ? 'is-modified' : ''}`}
                >
                  <div className="feature-input-header">
                    <label htmlFor={`input-${feat.name}`} className="feature-label" title={feat.name}>
                      {feat.label}
                    </label>
                    <div className="feature-header-actions">
                      {isModified && (
                        <button
                          type="button"
                          className="reset-feature-btn"
                          title="Revert to baseline value"
                          onClick={() => onResetFeature(feat.name)}
                        >
                          <RotateCcw size={12} />
                          <span>Revert</span>
                        </button>
                      )}
                    </div>
                  </div>

                  <div className="feature-input-control">
                    {feat.type === 'select' ? (
                      <select
                        id={`input-${feat.name}`}
                        className="feature-select"
                        value={String(currentValue)}
                        onChange={(e) => {
                          const val = feat.options && feat.options.includes('0') && feat.options.includes('1')
                            ? Number(e.target.value)
                            : e.target.value;
                          onChange(feat.name, val);
                        }}
                      >
                        {feat.options?.map((opt) => (
                          <option key={opt} value={opt}>
                            {opt}
                          </option>
                        ))}
                      </select>
                    ) : feat.type === 'slider' ? (
                      <div className="slider-control-group">
                        <input
                          id={`input-${feat.name}`}
                          type="range"
                          className="feature-slider"
                          min={feat.min ?? 0}
                          max={feat.max ?? 100}
                          step={feat.step ?? 1}
                          value={Number(currentValue)}
                          onChange={(e) => onChange(feat.name, Number(e.target.value))}
                        />
                        <input
                          type="number"
                          className="feature-slider-number"
                          min={feat.min ?? 0}
                          max={feat.max ?? 100}
                          step={feat.step ?? 1}
                          value={Number(currentValue)}
                          onChange={(e) => onChange(feat.name, Number(e.target.value))}
                        />
                      </div>
                    ) : (
                      <input
                        id={`input-${feat.name}`}
                        type="number"
                        className="feature-number-input"
                        min={feat.min}
                        max={feat.max}
                        step={feat.step ?? (Number.isInteger(currentValue) ? 1 : 0.01)}
                        value={currentValue}
                        onChange={(e) => {
                          const val = e.target.value === '' ? 0 : Number(e.target.value);
                          onChange(feat.name, val);
                        }}
                      />
                    )}
                  </div>

                  <div className="feature-footer">
                    <span className="feature-description">{feat.description}</span>
                    {isModified && baselineValue !== undefined && (
                      <span className="baseline-val-hint">
                        Base: <strong>{String(baselineValue)}</strong>
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
