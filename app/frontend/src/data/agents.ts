import { api } from '@/services/api';

export interface Agent {
  key: string;
  display_name: string;
  description: string;
  investing_style: string;
  order: number;
}

// In-memory cache for agents to avoid repeated API calls
let agents: Agent[] | null = null;

/**
 * Get the list of agents from the backend API
 * Uses caching to avoid repeated API calls
 * @param forceRefresh - If true, will bypass cache and fetch fresh data
 */
export const getAgents = async (forceRefresh: boolean = false): Promise<Agent[]> => {
  if (agents && !forceRefresh) {
    return agents;
  }
  
  try {
    const freshAgents = await api.getAgents();
    agents = freshAgents;
    return freshAgents;
  } catch (error) {
    console.error('Failed to fetch agents:', error);
    throw error; // Let the calling component handle the error
  }
};

/**
 * Clear the agents cache - useful for testing or when you need to force a refresh
 */
export const clearAgentsCache = () => {
  agents = null;
};
