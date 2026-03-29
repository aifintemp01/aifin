import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { PanelBottom, PanelLeft, PanelRight, Settings } from 'lucide-react';

export type AppMode = 'playground' | 'tree-query';

interface TopBarProps {
  isLeftCollapsed: boolean;
  isRightCollapsed: boolean;
  isBottomCollapsed: boolean;
  onToggleLeft: () => void;
  onToggleRight: () => void;
  onToggleBottom: () => void;
  onSettingsClick: () => void;
  activeMode: AppMode;
  onModeChange: (mode: AppMode) => void;
}

export function TopBar({
  isLeftCollapsed,
  isRightCollapsed,
  isBottomCollapsed,
  onToggleLeft,
  onToggleRight,
  onToggleBottom,
  onSettingsClick,
  activeMode,
  onModeChange,
}: TopBarProps) {
  return (
    <div className="absolute top-0 left-0 right-0 z-40 flex items-center h-9 px-2 bg-panel/80 border-b border-border">
      {/* Mode switcher — left side */}
      <div className="flex items-center gap-0.5 bg-ramp-grey-700/40 rounded-md p-0.5">
        <button
          onClick={() => onModeChange('playground')}
          className={cn(
            "px-3 py-1 text-xs font-medium rounded transition-all duration-150",
            activeMode === 'playground'
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          Playground
        </button>
        <button
          onClick={() => onModeChange('tree-query')}
          className={cn(
            "px-3 py-1 text-xs font-medium rounded transition-all duration-150",
            activeMode === 'tree-query'
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          Tree Query
        </button>
      </div>

      <div className="flex-1" />

      {/* Existing panel toggles — right side, untouched */}
      <div className="flex items-center gap-0">
        <Button
          variant="ghost"
          size="sm"
          onClick={onToggleLeft}
          className={cn(
            "h-8 w-8 p-0 text-muted-foreground hover:text-foreground hover:bg-ramp-grey-700 transition-colors",
            !isLeftCollapsed && "text-foreground"
          )}
          aria-label="Toggle left sidebar"
          title="Toggle Left Side Bar (⌘B)"
        >
          <PanelLeft size={16} />
        </Button>

        <Button
          variant="ghost"
          size="sm"
          onClick={onToggleBottom}
          className={cn(
            "h-8 w-8 p-0 text-muted-foreground hover:text-foreground hover:bg-ramp-grey-700 transition-colors",
            !isBottomCollapsed && "text-foreground"
          )}
          aria-label="Toggle bottom panel"
          title="Toggle Bottom Panel (⌘J)"
        >
          <PanelBottom size={16} />
        </Button>

        <Button
          variant="ghost"
          size="sm"
          onClick={onToggleRight}
          className={cn(
            "h-8 w-8 p-0 text-muted-foreground hover:text-foreground hover:bg-ramp-grey-700 transition-colors",
            !isRightCollapsed && "text-foreground"
          )}
          aria-label="Toggle right sidebar"
          title="Toggle Right Side Bar (⌘I)"
        >
          <PanelRight size={16} />
        </Button>

        <div className="w-px h-5 bg-ramp-grey-700 mx-1" />

        <Button
          variant="ghost"
          size="sm"
          onClick={onSettingsClick}
          className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground hover:bg-ramp-grey-700 transition-colors"
          aria-label="Open settings"
          title="Open Settings (⌘,)"
        >
          <Settings size={16} />
        </Button>
      </div>
    </div>
  );
}