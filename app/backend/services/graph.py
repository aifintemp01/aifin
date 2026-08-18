import asyncio
import inspect
import json
import re
from collections import deque
from functools import partial

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app.backend.services.agent_service import create_agent_function
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.main import start
from src.utils.analysts import ANALYST_CONFIG
from src.graph.state import AgentState


def extract_base_agent_key(unique_id: str) -> str:
    parts = unique_id.split('_')
    if len(parts) >= 2:
        last_part = parts[-1]
        if len(last_part) == 6 and re.match(r'^[a-z0-9]+$', last_part):
            return '_'.join(parts[:-1])
    return unique_id


# ─────────────────────────────────────────────────────────────────────────────
# Layer detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_layers(agent_ids_set: set, graph_edges: list) -> dict[str, int]:
    """
    Compute layer depth for each agent via BFS from the input node.

    Layer 1 = agents directly reachable from the input node
              (edges whose source is NOT in agent_ids_set, i.e. the input node)
    Layer N = agents reachable in N hops from the input node
    """
    # Layer 1: agents with incoming edges from the input node (outside agent_ids_set)
    layer1 = set()
    for edge in graph_edges:
        if edge.source not in agent_ids_set and edge.target in agent_ids_set:
            layer1.add(edge.target)

    # Build agent-to-agent adjacency
    adjacency: dict[str, list] = {}
    for edge in graph_edges:
        if edge.source in agent_ids_set and edge.target in agent_ids_set:
            adjacency.setdefault(edge.source, []).append(edge.target)

    # BFS
    layers: dict[str, int] = {}
    queue: deque = deque()
    for agent_id in layer1:
        layers[agent_id] = 1
        queue.append(agent_id)

    while queue:
        current = queue.popleft()
        for neighbor in adjacency.get(current, []):
            if neighbor not in layers:
                layers[neighbor] = layers[current] + 1
                queue.append(neighbor)

    # Default Layer 1 for any agent not reached by BFS
    for agent_id in agent_ids_set:
        if agent_id not in layers:
            layers[agent_id] = 1

    return layers


def _compute_upstream(agent_ids_set: set, graph_edges: list) -> dict[str, list]:
    """Build map of agent_id -> list of directly upstream agent IDs."""
    upstream: dict[str, list] = {}
    for edge in graph_edges:
        if edge.source in agent_ids_set and edge.target in agent_ids_set:
            upstream.setdefault(edge.target, []).append(edge.source)
    return upstream


def _create_layered_agent(agent_func, agent_id, layer, is_last_hidden, upstream_agent_ids):
    """
    Wrap an agent function to inject layer metadata when the agent supports it,
    and to ensure the return dict always contains 'layer_context' for state merging.

    Agents that accept (layer, is_last_hidden, upstream_agent_ids) get those params.
    Agents that don't accept them (legacy agents) are called as before.
    """
    sig = inspect.signature(agent_func)
    supports_layers = 'layer' in sig.parameters

    if supports_layers:
        wrapped = partial(
            agent_func,
            agent_id=agent_id,
            layer=layer,
            is_last_hidden=is_last_hidden,
            upstream_agent_ids=upstream_agent_ids or [],
        )
    else:
        wrapped = partial(agent_func, agent_id=agent_id)

    # Ensure every agent return includes 'layer_context' so LangGraph
    # can merge it into state regardless of whether the agent uses it
    def _with_layer_context(state: AgentState):
        result = wrapped(state)
        if 'layer_context' not in result:
            result['layer_context'] = {}
        return result

    return _with_layer_context


# ─────────────────────────────────────────────────────────────────────────────
# Signal filter (multi-PM isolation)
# ─────────────────────────────────────────────────────────────────────────────

def _make_signal_filter(agent_func, agent_id: str, allowed_ids: set):
    """Wrap agent so it only sees analyst_signals from allowed_ids (multi-PM isolation)."""
    def _filtered(state: AgentState):
        filtered_signals = {
            k: v for k, v in state["data"]["analyst_signals"].items()
            if k in allowed_ids
        }
        patched = {
            **state,
            "data": {**state["data"], "analyst_signals": filtered_signals},
        }
        return agent_func(patched, agent_id=agent_id)
    return _filtered


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def create_graph(graph_nodes: list, graph_edges: list) -> StateGraph:
    """Create the workflow based on the React Flow graph structure."""
    graph = StateGraph(AgentState)
    graph.add_node("start_node", start)

    analyst_nodes = {
        key: (f"{key}_agent", config["agent_func"])
        for key, config in ANALYST_CONFIG.items()
    }

    agent_ids = [node.id for node in graph_nodes]
    agent_ids_set = set(agent_ids)

    portfolio_manager_nodes: set = set()
    for unique_agent_id in agent_ids:
        if extract_base_agent_key(unique_agent_id) == "portfolio_manager":
            portfolio_manager_nodes.add(unique_agent_id)

    # ── Build ALL edge maps first (one pass) ─────────────────────────────────
    nodes_with_incoming_edges: set = set()
    direct_to_portfolio_managers: dict[str, str] = {}
    graph_edges_to_add = []  # edges to add to graph after nodes are registered

    for edge in graph_edges:
        if edge.source in agent_ids_set and edge.target in agent_ids_set:
            source_base = extract_base_agent_key(edge.source)
            target_base = extract_base_agent_key(edge.target)
            nodes_with_incoming_edges.add(edge.target)

            if (
                source_base in ANALYST_CONFIG
                and source_base != "portfolio_manager"
                and target_base == "portfolio_manager"
            ):
                direct_to_portfolio_managers[edge.source] = edge.target
            else:
                graph_edges_to_add.append((edge.source, edge.target))

    # ── Compute layer metadata (needs edge maps) ──────────────────────────────
    layers = _compute_layers(agent_ids_set, graph_edges)
    upstream_agents = _compute_upstream(agent_ids_set, graph_edges)

    # ── Add analyst nodes ONCE with correct info ──────────────────────────────
    for unique_agent_id in agent_ids:
        base_agent_key = extract_base_agent_key(unique_agent_id)

        if base_agent_key == "portfolio_manager":
            continue
        if base_agent_key not in ANALYST_CONFIG:
            continue

        _, node_func = analyst_nodes[base_agent_key]
        layer = layers.get(unique_agent_id, 1)
        upstream_ids = upstream_agents.get(unique_agent_id, [])
        is_last_hidden = unique_agent_id in direct_to_portfolio_managers

        agent_function = _create_layered_agent(
            node_func, unique_agent_id, layer, is_last_hidden, upstream_ids
        )
        graph.add_node(unique_agent_id, agent_function)

    # ── Add graph edges between analyst nodes ─────────────────────────────────
    for source, target in graph_edges_to_add:
        graph.add_edge(source, target)

    # ── Build PM→analysts reverse map ────────────────────────────────────────
    pm_to_analysts: dict[str, set] = {pm_id: set() for pm_id in portfolio_manager_nodes}
    for analyst_id, pm_id in direct_to_portfolio_managers.items():
        pm_to_analysts[pm_id].add(analyst_id)

    multi_pm = len(portfolio_manager_nodes) > 1

    # ── Add PM + risk manager nodes ──────────────────────────────────────────
    risk_manager_nodes: dict[str, str] = {}

    for pm_id in portfolio_manager_nodes:
        analysts_for_pm = pm_to_analysts.get(pm_id, set())
        suffix = pm_id.split('_')[-1]
        risk_manager_id = f"risk_management_agent_{suffix}"
        risk_manager_nodes[pm_id] = risk_manager_id

        if multi_pm:
            rm_func = _make_signal_filter(risk_management_agent, risk_manager_id, analysts_for_pm)
            pm_allowed = analysts_for_pm | {risk_manager_id}
            pm_func = _make_signal_filter(portfolio_management_agent, pm_id, pm_allowed)
        else:
            rm_func = create_agent_function(risk_management_agent, risk_manager_id)
            pm_func = create_agent_function(portfolio_management_agent, pm_id)

        graph.add_node(risk_manager_id, rm_func)
        graph.add_node(pm_id, pm_func)

    # ── Connect start_node to entry analysts ─────────────────────────────────
    for agent_id in agent_ids:
        if agent_id not in nodes_with_incoming_edges:
            base = extract_base_agent_key(agent_id)
            if base in ANALYST_CONFIG and base != "portfolio_manager":
                graph.add_edge("start_node", agent_id)

    # ── Route analysts → risk managers → PMs → END ──────────────────────────
    for analyst_id, pm_id in direct_to_portfolio_managers.items():
        graph.add_edge(analyst_id, risk_manager_nodes[pm_id])

    for pm_id, risk_manager_id in risk_manager_nodes.items():
        graph.add_edge(risk_manager_id, pm_id)

    for pm_id in portfolio_manager_nodes:
        graph.add_edge(pm_id, END)

    graph.set_entry_point("start_node")
    return graph


async def run_graph_async(
    graph, portfolio, tickers, start_date, end_date,
    model_name, model_provider, request=None
):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: run_graph(graph, portfolio, tickers, start_date, end_date,
                          model_name, model_provider, request)
    )
    return result


def run_graph(
    graph: StateGraph,
    portfolio: dict,
    tickers: list[str],
    start_date: str,
    end_date: str,
    model_name: str,
    model_provider: str,
    request=None,
) -> dict:
    return graph.invoke(
        {
            "messages": [
                HumanMessage(content="Make trading decisions based on the provided data.")
            ],
            "data": {
                "tickers": tickers,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "analyst_signals": {},
            },
            "metadata": {
                "show_reasoning": False,
                "model_name": model_name,
                "model_provider": model_provider,
                "request": request,
            },
            "layer_context": {},  # initialise empty — agents write to this as they run
        }
    )


def parse_hedge_fund_response(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}\nResponse: {repr(response)}")
        return None