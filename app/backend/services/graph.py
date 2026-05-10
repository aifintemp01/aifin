import asyncio
import json
import re
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


def _make_signal_filter(agent_func, agent_id: str, allowed_ids: set):
    """
    Wrap an agent function so it only sees analyst_signals from allowed_ids.
    Ensures each PM (and its paired risk manager) operates on isolated signals
    when multiple PMs exist in the same flow.
    """
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

    portfolio_manager_nodes = set()

    # ── Add analyst nodes ────────────────────────────────────────────────────
    for unique_agent_id in agent_ids:
        base_agent_key = extract_base_agent_key(unique_agent_id)

        if base_agent_key == "portfolio_manager":
            portfolio_manager_nodes.add(unique_agent_id)
            continue

        if base_agent_key not in ANALYST_CONFIG:
            continue

        node_name, node_func = analyst_nodes[base_agent_key]
        agent_function = create_agent_function(node_func, unique_agent_id)
        graph.add_node(unique_agent_id, agent_function)

    # ── Build edge maps ──────────────────────────────────────────────────────
    nodes_with_incoming_edges = set()
    nodes_with_outgoing_edges = set()
    direct_to_portfolio_managers: dict[str, str] = {}   # analyst_id → pm_id

    for edge in graph_edges:
        if edge.source in agent_ids_set and edge.target in agent_ids_set:
            source_base = extract_base_agent_key(edge.source)
            target_base = extract_base_agent_key(edge.target)

            nodes_with_incoming_edges.add(edge.target)
            nodes_with_outgoing_edges.add(edge.source)

            if (
                source_base in ANALYST_CONFIG
                and source_base != "portfolio_manager"
                and target_base == "portfolio_manager"
            ):
                direct_to_portfolio_managers[edge.source] = edge.target
            else:
                graph.add_edge(edge.source, edge.target)

    # ── Build reverse map: PM → analysts that feed it ────────────────────────
    pm_to_analysts: dict[str, set] = {pm_id: set() for pm_id in portfolio_manager_nodes}
    for analyst_id, pm_id in direct_to_portfolio_managers.items():
        pm_to_analysts[pm_id].add(analyst_id)

    multi_pm = len(portfolio_manager_nodes) > 1

    # ── Add PM nodes and their paired risk managers ──────────────────────────
    risk_manager_nodes: dict[str, str] = {}   # pm_id → risk_manager_id

    for pm_id in portfolio_manager_nodes:
        analysts_for_pm = pm_to_analysts.get(pm_id, set())

        suffix = pm_id.split('_')[-1]
        risk_manager_id = f"risk_management_agent_{suffix}"
        risk_manager_nodes[pm_id] = risk_manager_id

        # Risk manager sees only this PM's analysts
        if multi_pm:
            rm_func = _make_signal_filter(
                risk_management_agent,
                risk_manager_id,
                analysts_for_pm,
            )
        else:
            rm_func = create_agent_function(risk_management_agent, risk_manager_id)
        graph.add_node(risk_manager_id, rm_func)

        # PM sees its analysts + its own risk manager's output
        if multi_pm:
            pm_allowed = analysts_for_pm | {risk_manager_id}
            pm_func = _make_signal_filter(
                portfolio_management_agent,
                pm_id,
                pm_allowed,
            )
        else:
            pm_func = create_agent_function(portfolio_management_agent, pm_id)
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
        lambda: run_graph(
            graph, portfolio, tickers, start_date, end_date,
            model_name, model_provider, request
        )
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
        }
    )


def parse_hedge_fund_response(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None