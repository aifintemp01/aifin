import { getConnectedEdges, useReactFlow } from '@xyflow/react';
import { useMemo } from 'react';

import { useNodeContext } from '@/contexts/node-context';
import { extractBaseAgentKey } from '@/data/node-mappings';

export function useOutputNodeConnection(nodeId: string, flowId: string | null = null) {
  const { getAgentNodeDataForFlow, getOutputNodeDataForFlow } = useNodeContext();
  const { getNodes, getEdges } = useReactFlow();

  const agentNodeData = getAgentNodeDataForFlow(flowId);

  return useMemo(() => {
    const nodes = getNodes();
    const edges = getEdges();
    
    const connectedEdges = getConnectedEdges([{ id: nodeId }] as any, edges);
    const connectedAgentIds = connectedEdges
      .filter(edge => edge.target === nodeId)
      .map(edge => edge.source)
      .filter(sourceId => {
        const sourceNode = nodes.find(n => n.id === sourceId);
        return sourceNode?.type === 'agent-node';
      });

    const connectedPmId = (() => {
      // Step 1: direct PM connection
      const directPm = connectedAgentIds.find(
        agentId => extractBaseAgentKey(agentId) === 'portfolio_manager'
      );
      if (directPm) return directPm;

      // Step 2: find PM that our connected analysts feed INTO
      // Edges go: analyst → PM, so look for edges where source is one of our analysts
      // and target is a PM node
      const connectedSet = new Set(connectedAgentIds);
      for (const edge of edges) {
        if (!connectedSet.has(edge.source)) continue;
        const targetNode = nodes.find(n => n.id === edge.target);
        if (
          targetNode &&
          extractBaseAgentKey(targetNode.id) === 'portfolio_manager'
        ) {
          return targetNode.id;
        }
      }
      return null;
    })();

     const outputNodeData = (!flowId || !connectedPmId) 
      ? (() => {
          console.log('[Hook] Missing info:', { flowId, connectedPmId });
          return null;
        })()
      : (() => {
          const data = getOutputNodeDataForFlow(flowId, connectedPmId);
          console.log(`[Hook] Key: ${flowId}:${connectedPmId}, Found:`, !!data);
          return data;
        })();

    console.log('useOutputNodeConnection:', { 
      nodeId, 
      flowId, 
      connectedAgentIds, 
      connectedPmId 
    });
    console.log('outputNodeData:', outputNodeData);

    const isAnyAgentRunning = connectedAgentIds.some(agentId => 
      agentNodeData[agentId]?.status === 'IN_PROGRESS'
    );

    const isProcessing = isAnyAgentRunning;
    const isOutputAvailable = outputNodeData !== null && outputNodeData !== undefined;
    const isConnected = connectedAgentIds.length > 0;

    return {
      isProcessing,
      isAnyAgentRunning,
      isOutputAvailable,
      isConnected,
      connectedAgentIds: new Set(connectedAgentIds),
      connectedPmId,
      outputNodeData,
    };
  }, [nodeId, flowId, agentNodeData, getOutputNodeDataForFlow, getNodes, getEdges]);
}