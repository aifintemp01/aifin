import { NodeStatus, OutputNodeData, useNodeContext } from '@/contexts/node-context';
import { Agent } from '@/data/agents';
import { LanguageModel } from '@/data/models';
import { extractBaseAgentKey } from '@/data/node-mappings';
import { flowConnectionManager } from '@/hooks/use-flow-connection';
import {
  HedgeFundRequest
} from '@/services/types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = {
  getAgents: async (): Promise<Agent[]> => {
    try {
      const response = await fetch(`${API_BASE_URL}/hedge-fund/agents`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      return data.agents;
    } catch (error) {
      console.error('Failed to fetch agents:', error);
      throw error;
    }
  },

  getLanguageModels: async (): Promise<LanguageModel[]> => {
    try {
      const response = await fetch(`${API_BASE_URL}/language-models/`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      return data.models;
    } catch (error) {
      console.error('Failed to fetch models:', error);
      throw error;
    }
  },

  saveJsonFile: async (filename: string, data: any): Promise<void> => {
    try {
      const response = await fetch(`${API_BASE_URL}/storage/save-json`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename, data }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const result = await response.json();
      console.log(result.message);
    } catch (error) {
      console.error('Failed to save JSON file:', error);
      throw error;
    }
  },

  runHedgeFund: (
    params: HedgeFundRequest,
    nodeContext: ReturnType<typeof useNodeContext>,
    flowId: string | null = null
  ): (() => void) => {
    if (typeof params.tickers === 'string') {
      params.tickers = (params.tickers as unknown as string).split(',').map(t => t.trim());
    }

    const getAgentIds = () => params.graph_nodes.map(node => node.id);

    const backendParams = { ...params, flow_id: flowId };

    const controller = new AbortController();
    const { signal } = controller;

    fetch(`${API_BASE_URL}/hedge-fund/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(backendParams),
      signal,
    })
    .then(response => {
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const reader = response.body?.getReader();
      if (!reader) throw new Error('Failed to get response reader');

      const decoder = new TextDecoder();
      let buffer = '';

      const processStream = async () => {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split('\n\n');
            buffer = events.pop() || '';

            for (const eventText of events) {
              if (!eventText.trim()) continue;

              try {
                const eventTypeMatch = eventText.match(/^event: (.+)$/m);
                const dataMatch = eventText.match(/^data: (.+)$/m);

                if (eventTypeMatch && dataMatch) {
                  const eventType = eventTypeMatch[1];
                  const eventData = JSON.parse(dataMatch[1]);

                  console.log(`Parsed ${eventType} event:`, eventData);

                  switch (eventType) {
                    case 'start':
                      nodeContext.resetAllNodes(flowId);
                      break;

                    case 'progress':
                      if (eventData.agent) {
                        let nodeStatus: NodeStatus = 'IN_PROGRESS';
                        if (eventData.status === 'Done') nodeStatus = 'COMPLETE';

                        const baseAgentKey = eventData.agent.replace('_agent', '');
                        const uniqueNodeId =
                          getAgentIds().find(id => extractBaseAgentKey(id) === baseAgentKey) ||
                          baseAgentKey;

                        nodeContext.updateAgentNode(flowId, uniqueNodeId, {
                          status: nodeStatus,
                          ticker: eventData.ticker,
                          message: eventData.status,
                          analysis: eventData.analysis,
                          timestamp: eventData.timestamp,
                        });
                      }
                      break;

                    case 'complete':
                      if (eventData.data) {
                        const effectiveFlowId = eventData.data.flow_id || flowId;

                        // All PM node IDs from the graph
                        const pmNodeIds = params.graph_nodes
                          .filter(node => extractBaseAgentKey(node.id) === 'portfolio_manager')
                          .map(node => node.id);

                        console.log('PM node IDs:', pmNodeIds);
                        console.log('analyst_signals keys:', Object.keys(eventData.data.analyst_signals || {}));

                        if (pmNodeIds.length === 0) {
                          // No PM nodes — fallback (shouldn't happen in normal flows)
                          nodeContext.setOutputNodeData(
                            effectiveFlowId,
                            eventData.data as OutputNodeData
                          );
                        } else {
                          // Build direct analyst → PM map from graph edges
                          // (agents that have a direct edge to each PM)
                          const pmDirectAnalysts: Record<string, Set<string>> = {};
                          pmNodeIds.forEach(pmId => { pmDirectAnalysts[pmId] = new Set(); });

                          params.graph_edges.forEach(edge => {
                            const targetBase = extractBaseAgentKey(edge.target);
                            if (targetBase === 'portfolio_manager' && pmDirectAnalysts[edge.target]) {
                              pmDirectAnalysts[edge.target].add(edge.source);
                            }
                          });

                          // Store output data once per PM, with:
                          //   - decisions:       this PM's decisions only
                          //   - analyst_signals: only signals from analysts connected to this PM
                          pmNodeIds.forEach(pmId => {
                            // ── decisions ──────────────────────────────────
                            let pmDecisions: any;
                            const rawDecisions = eventData.data.decisions || {};

                            if (rawDecisions[pmId] !== undefined) {
                              // Backend keyed decisions by PM node ID (new format)
                              pmDecisions = rawDecisions[pmId];
                            } else if (pmNodeIds.length === 1) {
                              // Single PM — decisions may be flat (old format) or
                              // keyed by backend name like 'portfolio_manager_abc123'
                              const firstKey = Object.keys(rawDecisions)[0] || '';
                              pmDecisions = firstKey.startsWith('portfolio_manager')
                                ? rawDecisions[firstKey]
                                : rawDecisions;
                            } else {
                              // Multiple PMs but no matching key — use all as fallback
                              pmDecisions = rawDecisions;
                            }

                            // ── analyst_signals ────────────────────────────
                            // For multiple PMs, filter to only this PM's analysts.
                            // For single PM, all signals belong to it anyway.
                            let pmSignals = eventData.data.analyst_signals || {};
                            if (pmNodeIds.length > 1) {
                              const allowed = pmDirectAnalysts[pmId];
                              pmSignals = Object.fromEntries(
                                Object.entries(pmSignals).filter(([agentId]) => {
                                  // Include analyst if directly connected to this PM,
                                  // or if it's the risk_manager paired with this PM
                                  const suffix = pmId.split('_').pop();
                                  return (
                                    allowed.has(agentId) ||
                                    agentId === `risk_management_agent_${suffix}`
                                  );
                                })
                              );
                            }

                            console.log(
                              'Storing data for PM:', pmId,
                              'key:', `${effectiveFlowId}:${pmId}`,
                              'decisions keys:', Object.keys(pmDecisions || {}),
                              'signal agents:', Object.keys(pmSignals).length,
                            );

                            nodeContext.setOutputNodeData(
                              effectiveFlowId,
                              {
                                ...eventData.data,
                                decisions: pmDecisions,
                                analyst_signals: pmSignals,
                              } as OutputNodeData,
                              pmId,
                            );
                          });
                        }
                      }

                      nodeContext.updateAgentNodes(flowId, getAgentIds(), 'COMPLETE');
                      nodeContext.updateAgentNode(flowId, 'output', {
                        status: 'COMPLETE',
                        message: 'Analysis complete',
                      });

                      if (flowId) {
                        flowConnectionManager.setConnection(flowId, {
                          state: 'completed',
                          abortController: null,
                        });
                        setTimeout(() => {
                          const c = flowConnectionManager.getConnection(flowId);
                          if (c.state === 'completed') {
                            flowConnectionManager.setConnection(flowId, { state: 'idle' });
                          }
                        }, 30000);
                      }
                      break;

                    case 'error':
                      nodeContext.updateAgentNodes(flowId, getAgentIds(), 'ERROR');
                      if (flowId) {
                        flowConnectionManager.setConnection(flowId, {
                          state: 'error',
                          error: eventData.message || 'Unknown error occurred',
                          abortController: null,
                        });
                      }
                      break;

                    default:
                      console.warn('Unknown event type:', eventType);
                  }
                }
              } catch (err) {
                console.error('Error parsing SSE event:', err, 'Raw event:', eventText);
              }
            }
          }

          if (flowId) {
            const c = flowConnectionManager.getConnection(flowId);
            if (c.state === 'connected') {
              flowConnectionManager.setConnection(flowId, {
                state: 'completed',
                abortController: null,
              });
            }
          }
        } catch (error: any) {
          if (error.name !== 'AbortError') {
            console.error('Error reading SSE stream:', error);
            nodeContext.updateAgentNodes(flowId, getAgentIds(), 'ERROR');
            if (flowId) {
              flowConnectionManager.setConnection(flowId, {
                state: 'error',
                error: error.message || 'Connection error',
                abortController: null,
              });
            }
          }
        }
      };

      processStream();
    })
    .catch((error: any) => {
      if (error.name !== 'AbortError') {
        console.error('SSE connection error:', error);
        nodeContext.updateAgentNodes(flowId, getAgentIds(), 'ERROR');
        if (flowId) {
          flowConnectionManager.setConnection(flowId, {
            state: 'error',
            error: error.message || 'Connection failed',
            abortController: null,
          });
        }
      }
    });

    return () => {
      controller.abort();
      if (flowId) {
        flowConnectionManager.setConnection(flowId, {
          state: 'idle',
          abortController: null,
        });
      }
    };
  },
};