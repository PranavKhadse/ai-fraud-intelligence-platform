import React, { useState, useEffect } from 'react';
import { User, Shield, ChevronDown, Check } from 'lucide-react';
import {
  getDevActorContext,
  setDevActorContext,
  isDevActorHeadersAllowed,
  type DevActorContext,
} from '../../api/caseApi.ts';
import type { AuditActorType } from '../../types/case.ts';

interface ActorContextSwitcherProps {
  onActorChange?: (actor: DevActorContext) => void;
}

const PRESET_ACTORS: { id: string; role: AuditActorType; label: string }[] = [
  { id: 'analyst_01', role: 'ANALYST', label: 'Analyst 01 (Triage Reviewer)' },
  { id: 'analyst_senior', role: 'ANALYST', label: 'Senior Analyst (Specialist)' },
  { id: 'admin_01', role: 'ADMIN', label: 'Admin 01 (Full Authority / Assign)' },
  { id: 'api_client_service', role: 'API_CLIENT', label: 'API Client Service (Read Only)' },
];

export const ActorContextSwitcher: React.FC<ActorContextSwitcherProps> = ({ onActorChange }) => {
  const [currentActor, setCurrentActor] = useState<DevActorContext>(getDevActorContext());
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [customId, setCustomId] = useState<string>('');
  const [customRole, setCustomRole] = useState<AuditActorType>('ANALYST');
  const [showCustom, setShowCustom] = useState<boolean>(false);

  useEffect(() => {
    setCurrentActor(getDevActorContext());
  }, []);

  // When dev actor headers are disabled (e.g. production mode), hide switcher completely
  if (!isDevActorHeadersAllowed()) {
    return null;
  }


  const handleSelectPreset = (actor: { id: string; role: AuditActorType }) => {
    const updated: DevActorContext = { actorId: actor.id, actorRole: actor.role };
    setDevActorContext(updated);
    setCurrentActor(updated);
    setIsOpen(false);
    setShowCustom(false);
    if (onActorChange) {
      onActorChange(updated);
    }
  };

  const handleApplyCustom = (e: React.FormEvent) => {
    e.preventDefault();
    if (!customId.trim()) return;
    const updated: DevActorContext = {
      actorId: customId.trim(),
      actorRole: customRole,
    };
    setDevActorContext(updated);
    setCurrentActor(updated);
    setIsOpen(false);
    setShowCustom(false);
    if (onActorChange) {
      onActorChange(updated);
    }
  };

  return (
    <div className="actor-switcher-container">
      <button
        type="button"
        className="actor-switcher-btn"
        onClick={() => setIsOpen(!isOpen)}
        title="Development/Test Actor Context (Headers Injection)"
        aria-expanded={isOpen}
      >
        <div className="actor-switcher-icon-wrap">
          {currentActor.actorRole === 'ADMIN' ? (
            <Shield size={13} className="text-warning" />
          ) : (
            <User size={13} className="text-primary" />
          )}
        </div>
        <div className="actor-switcher-info">
          <span className="actor-role-badge">{currentActor.actorRole}</span>
          <span className="actor-id-text font-mono">{currentActor.actorId}</span>
        </div>
        <ChevronDown size={12} className="text-muted" />
      </button>

      {isOpen && (
        <div className="actor-dropdown-menu" role="menu">
          <div className="actor-dropdown-header">
            <span className="dropdown-title">Dev Actor Context</span>
            <span className="dropdown-badge">Dev / Test Only</span>
          </div>

          <div className="actor-dropdown-list">
            {PRESET_ACTORS.map((preset) => {
              const isSelected =
                currentActor.actorId === preset.id && currentActor.actorRole === preset.role;
              return (
                <button
                  key={preset.id}
                  type="button"
                  className={`actor-dropdown-item ${isSelected ? 'selected' : ''}`}
                  onClick={() => handleSelectPreset(preset)}
                  role="menuitem"
                >
                  <div className="item-details">
                    <span className="item-label">{preset.label}</span>
                    <span className="item-sub font-mono">
                      {preset.id} &bull; {preset.role}
                    </span>
                  </div>
                  {isSelected && <Check size={14} className="text-primary" />}
                </button>
              );
            })}
          </div>

          {!showCustom ? (
            <div className="actor-dropdown-footer">
              <button
                type="button"
                className="btn-custom-actor-toggle"
                onClick={() => setShowCustom(true)}
              >
                + Custom Actor Identity
              </button>
            </div>
          ) : (
            <form onSubmit={handleApplyCustom} className="custom-actor-form">
              <div className="form-group-sm">
                <label className="form-label-sm">Actor ID:</label>
                <input
                  type="text"
                  value={customId}
                  onChange={(e) => setCustomId(e.target.value)}
                  placeholder="e.g. audit_inspector"
                  className="form-input-sm"
                  required
                />
              </div>
              <div className="form-group-sm">
                <label className="form-label-sm">Actor Role:</label>
                <select
                  value={customRole}
                  onChange={(e) => setCustomRole(e.target.value as AuditActorType)}
                  className="form-select-sm"
                >
                  <option value="ANALYST">ANALYST</option>
                  <option value="ADMIN">ADMIN</option>
                  <option value="API_CLIENT">API_CLIENT</option>
                  <option value="SYSTEM">SYSTEM</option>
                </select>
              </div>
              <div className="custom-form-actions">
                <button
                  type="button"
                  className="btn-sm-secondary"
                  onClick={() => setShowCustom(false)}
                >
                  Cancel
                </button>
                <button type="submit" className="btn-sm-primary" disabled={!customId.trim()}>
                  Apply
                </button>
              </div>
            </form>
          )}
        </div>
      )}
    </div>
  );
};
