import {
  Background,
  BackgroundVariant,
  ColorMode,
  Connection,
  Edge,
  EdgeChange,
  MarkerType,
  NodeChange,
  ReactFlow,
  addEdge,
  useEdgesState,
  useNodesState
} from '@xyflow/react';
import { useTheme } from 'next-themes';
import { useCallback, useEffect, useRef, useState } from 'react';

import '@xyflow/react/dist/style.css';

import { useFlowContext } from '@/contexts/flow-context';
import { useEnhancedFlowActions } from '@/hooks/use-enhanced-flow-actions';
import { useFlowHistory } from '@/hooks/use-flow-history';
import { useFlowKeyboardShortcuts, useKeyboardShortcuts } from '@/hooks/use-keyboard-shortcuts';
import { useToastManager } from '@/hooks/use-toast-manager';
import { AppNode } from '@/nodes/types';
import { edgeTypes } from '../edges';
import { nodeTypes } from '../nodes';
import { TooltipProvider } from './ui/tooltip';

type FlowProps = {
  className?: string;
};

export function Flow({ className = '' }: FlowProps) {
  const { theme, resolvedTheme } = useTheme();
  
  // Use the resolved theme for ReactFlow ColorMode
  const colorMode: ColorMode = resolvedTheme === 'light' ? 'light' : 'dark';
  
  const [nodes, setNodes, onNodesChange] = useNodesState<AppNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [isInitialized, setIsInitialized] = useState(false);
  const proOptions = { hideAttribution: true };
  
  // Get flow context for flow ID
  const { currentFlowId } = useFlowContext();
  
  // Get enhanced flow actions for complete state persistence
  const { saveCurrentFlowWithCompleteState } = useEnhancedFlowActions();
  
  // Get toast manager
  const { success, error } = useToastManager();

  // Initialize flow history (each flow maintains its own separate history)
  const { takeSnapshot, undo, redo, canUndo, canRedo, clearHistory } = useFlowHistory({ flowId: currentFlowId });

  // Create debounced auto-save function
  const autoSaveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastSavedFlowIdRef = useRef<number | null>(null);
  
  const autoSave = useCallback(async (flowIdToSave?: number | null) => {
    const targetFlowId = flowIdToSave !== undefined ? flowIdToSave : currentFlowId;
    
    if (autoSaveTimeoutRef.current) {
      clearTimeout(autoSaveTimeoutRef.current);
    }
    
    autoSaveTimeoutRef.current = setTimeout(async () => {
      if (!targetFlowId) return;
      if (targetFlowId !== currentFlowId) return;
      
      try {
        await saveCurrentFlowWithCompleteState();
        lastSavedFlowIdRef.current = targetFlowId;
      } catch (error) {
        console.error(`[Auto-save] Failed to save flow ${targetFlowId}:`, error);
      }
    }, 1000);
  }, [currentFlowId, saveCurrentFlowWithCompleteState]);

  const handleNodesChange = useCallback((changes: NodeChange<AppNode>[]) => {
    onNodesChange(changes);
    
    const shouldAutoSave = changes.some(change => {
      switch (change.type) {
        case 'add':    return true;
        case 'remove': return true;
        case 'position':
          return !change.dragging;
        default:
          return false;
      }
    });

    if (shouldAutoSave && isInitialized && currentFlowId) {
      const flowIdAtTimeOfChange = currentFlowId;
      autoSave(flowIdAtTimeOfChange);
    }
  }, [onNodesChange, autoSave, isInitialized, currentFlowId]);

  const handleEdgesChange = useCallback((changes: EdgeChange[]) => {
    onEdgesChange(changes);
    
    const shouldAutoSave = changes.some(change => change.type === 'remove');

    if (shouldAutoSave && isInitialized && currentFlowId) {
      const flowIdAtTimeOfChange = currentFlowId;
      autoSave(flowIdAtTimeOfChange);
    }
  }, [onEdgesChange, autoSave, isInitialized, currentFlowId]);

  // Delete an edge when clicked
  const handleEdgeClick = useCallback(
    (_: React.MouseEvent, edge: Edge) => {
      setEdges((eds) => eds.filter((e) => e.id !== edge.id));
      if (currentFlowId) {
        const flowIdAtTimeOfChange = currentFlowId;
        if (autoSaveTimeoutRef.current) clearTimeout(autoSaveTimeoutRef.current);
        setTimeout(async () => {
          if (flowIdAtTimeOfChange !== currentFlowId) return;
          try {
            await saveCurrentFlowWithCompleteState();
          } catch (err) {
            console.error('[Auto-save] Failed to save after edge deletion:', err);
          }
        }, 100);
      }
    },
    [setEdges, currentFlowId, saveCurrentFlowWithCompleteState]
  );

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (autoSaveTimeoutRef.current) clearTimeout(autoSaveTimeoutRef.current);
    };
  }, []);

  // Cancel pending auto-saves when flow changes
  useEffect(() => {
    if (autoSaveTimeoutRef.current) {
      clearTimeout(autoSaveTimeoutRef.current);
      autoSaveTimeoutRef.current = null;
    }
  }, [currentFlowId]);

  useEffect(() => {
    if (isInitialized && nodes.length === 0 && edges.length === 0) {
      takeSnapshot();
    }
  }, [isInitialized, takeSnapshot, nodes.length, edges.length]);

  useEffect(() => {
    if (!isInitialized) return;
    const timeoutId = setTimeout(() => { takeSnapshot(); }, 500);
    return () => clearTimeout(timeoutId);
  }, [nodes, edges, takeSnapshot, isInitialized]);

  useFlowKeyboardShortcuts(async () => {
    try {
      const savedFlow = await saveCurrentFlowWithCompleteState();
      if (savedFlow) {
        success(`"${savedFlow.name}" saved!`, 'flow-save');
      } else {
        error('Failed to save flow', 'flow-save-error');
      }
    } catch (err) {
      error('Failed to save flow', 'flow-save-error');
    }
  });

  useKeyboardShortcuts({
    shortcuts: [
      {
        key: 'z',
        ctrlKey: true,
        metaKey: true,
        callback: undo,
        preventDefault: true,
      },
      {
        key: 'z',
        ctrlKey: true,
        metaKey: true,
        shiftKey: true,
        callback: redo,
        preventDefault: true,
      },
    ],
  });
  
  const onInit = useCallback(() => {
    if (!isInitialized) setIsInitialized(true);
  }, [isInitialized]);

  const onConnect = useCallback(
    (connection: Connection) => {
      const newEdge: Edge = {
        ...connection,
        id: `edge-${Date.now()}`,
        markerEnd: { type: MarkerType.ArrowClosed },
      };
      setEdges((eds) => addEdge(newEdge, eds));
      
      if (currentFlowId) {
        const flowIdAtTimeOfChange = currentFlowId;
        if (autoSaveTimeoutRef.current) clearTimeout(autoSaveTimeoutRef.current);
        setTimeout(async () => {
          if (flowIdAtTimeOfChange !== currentFlowId) return;
          try {
            await saveCurrentFlowWithCompleteState();
          } catch (error) {
            console.error(`[Auto-save] Failed to save new connection for flow ${flowIdAtTimeOfChange}:`, error);
          }
        }, 100);
      }
    },
    [setEdges, currentFlowId, saveCurrentFlowWithCompleteState]
  );

  const backgroundStyle = { backgroundColor: 'hsl(var(--background))' };
  const gridColor = resolvedTheme === 'light'
    ? 'hsl(var(--foreground))'
    : 'hsl(var(--muted-foreground))';

  return (
    <div className={`w-full h-full ${className}`}>
      <TooltipProvider>
        <ReactFlow
          nodes={nodes}
          nodeTypes={nodeTypes}
          onNodesChange={handleNodesChange}
          edges={edges}
          edgeTypes={edgeTypes}
          onEdgesChange={handleEdgesChange}
          onConnect={onConnect}
          onEdgeClick={handleEdgeClick}
          onInit={onInit}
          colorMode={colorMode}
          proOptions={proOptions}
          deleteKeyCode={['Delete', 'Backspace']}
        >
          <Background 
            variant={BackgroundVariant.Dots}
            gap={13}
            color={gridColor}
            style={backgroundStyle}
          />
        </ReactFlow>
      </TooltipProvider>
    </div>
  );
}