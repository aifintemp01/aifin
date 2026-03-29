import os
import re
import json
import math
import threading
import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from langchain.chat_models import init_chat_model
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from rich.console import Console
from rich.tree import Tree as RichTree

# -------------------------------
# CONFIG
# -------------------------------
DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", str(Path(__file__).parent / "chroma_db")))
MAX_NODES_DISPLAY = 8
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_CHUNKS = 10
CONFIDENCE_THRESHOLD = 0.25
console = Console()

# -------------------------------
# MODEL
# OPENAI_API_KEY loaded from environment — never hardcode
# -------------------------------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import chromadb

# Key is read from environment automatically by langchain_openai
# Set OPENAI_API_KEY in your .env file or Railway dashboard
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


# -------------------------------
# TREE NODE
# -------------------------------
@dataclass
class TreeNode:
    title: str
    node_id: str
    content: str
    summary: str = ""
    nodes: List["TreeNode"] = field(default_factory=list)

    def to_dict(self, include_content: bool = True) -> dict:
        result = {
            "title": self.title,
            "node_id": self.node_id,
            "summary": self.summary,
        }
        if include_content:
            result["content"] = self.content
        if self.nodes:
            result["nodes"] = [n.to_dict(include_content) for n in self.nodes]
        return result

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.nodes)


# -------------------------------
# SEARCH RESULT MODELS
# -------------------------------
class TreeSearchResult(BaseModel):
    thinking: str = Field(description="Reasoning")
    node_list: List[str] = Field(description="Relevant node IDs")


class ConfidenceCheck(BaseModel):
    has_enough: bool = Field(description="True if context is sufficient to answer the query")
    reasoning: str = Field(description="Why the context is or isn't sufficient")


search_model = model.with_structured_output(TreeSearchResult)
confidence_model = model.with_structured_output(ConfidenceCheck)

# -------------------------------
# PROMPTS
# -------------------------------
SUMMARY_PROMPT = """Summarize this document section in 2-3 sentences.
State facts directly.

Section: {title}

{content}
"""

TREE_SEARCH_PROMPT = """You are given a question and a tree structure.

Each node has:
- node_id
- title
- summary

Select the most relevant node IDs.

Question: {query}

Tree:
{tree_index}
"""

CONFIDENCE_PROMPT = """You are evaluating whether the provided context contains enough information to answer the query.

Query: {query}

Context from value-based search:
{context}

Does this context contain enough information to fully answer the query?
Reply with has_enough=true only if the answer can be directly derived from this context.
"""

ANSWER_PROMPT = """You are an expert financial analyst. Answer the question using the context below.
Be specific — cite numbers, names, dates where available.
If the exact answer is partially present, extract what you can and note what is missing.
Only say "I don't know" if the context contains absolutely nothing relevant.

Question: {query}

Context:
{context}

Answer:
"""

HEADER_LEVELS = {"title": 1, "section": 2, "subsection": 3}


# -------------------------------
# BUILD TREE
# -------------------------------
def build_tree(markdown: str) -> TreeNode:
    text = markdown.replace("<!-- page_break -->", "\n")

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "section"),
            ("###", "subsection"),
        ]
    )

    sections = splitter.split_text(text)

    root = TreeNode(title="Document", node_id="0000", content="")
    counter = 1
    stack = [(0, root)]

    for section in sections:
        level = 0
        section_title = "Untitled"

        for key in ["title", "section", "subsection"]:
            if key in section.metadata:
                level = HEADER_LEVELS[key]
                section_title = section.metadata[key]

        if level == 0:
            root.content += section.page_content
            continue

        node = TreeNode(
            title=section_title,
            node_id=f"{counter:04d}",
            content=section.page_content,
        )
        counter += 1

        while len(stack) > 1 and stack[-1][0] >= level:
            stack.pop()

        stack[-1][1].nodes.append(node)
        stack.append((level, node))

    return root


def clean_markdown(md: str) -> str:
    lines = md.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if len(line) < 3:
            continue
        line = " ".join(line.split())
        cleaned.append(line)
    return "\n".join(cleaned)


# -------------------------------
# HELPER
# -------------------------------
def _has_meaningful_content(text: str) -> bool:
    stripped = re.sub(r"^#+\s+.*$", "", text, flags=re.MULTILINE).strip()
    return len(stripped) > 20


# -------------------------------
# SUMMARIZATION
# -------------------------------
def summarize_tree(node: TreeNode):
    for child in node.nodes:
        summarize_tree(child)

    has_content = _has_meaningful_content(node.content)
    has_child_summaries = any(c.summary for c in node.nodes)

    if not has_content and not has_child_summaries:
        return

    if has_child_summaries:
        children_text = "\n".join(
            f"- {c.title}: {c.summary}" for c in node.nodes if c.summary
        )
        text = (
            f"{node.content}\n\nChild sections:\n{children_text}"
            if has_content
            else children_text
        )
    else:
        text = node.content

    console.print(f"[dim]Summarizing: {node.title}[/dim]")
    prompt = SUMMARY_PROMPT.format(title=node.title, content=text[:5000])
    node.summary = model.invoke(prompt).content.strip()


# -------------------------------
# DISPLAY TREE
# -------------------------------
def display_tree(node: TreeNode, parent: RichTree = None) -> RichTree:
    label = f"[bold]{node.title}[/bold] ({node.node_id})"

    if node.summary:
        short = node.summary[:100] + ("..." if len(node.summary) > 100 else "")
        label += f"\n[italic]{short}[/italic]"

    branch = parent.add(label) if parent else RichTree(label)

    for child in node.nodes:
        display_tree(child, branch)

    return branch


# -------------------------------
# SAVE / LOAD
# -------------------------------
def save_tree(tree: TreeNode, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tree.to_dict(), f, indent=2, ensure_ascii=False)


def _dict_to_node(d: dict) -> TreeNode:
    return TreeNode(
        title=d["title"],
        node_id=d["node_id"],
        content=d.get("content", ""),
        summary=d.get("summary", ""),
        nodes=[_dict_to_node(c) for c in d.get("nodes", [])],
    )


def load_tree(path: Path) -> TreeNode:
    with open(path, encoding="utf-8") as f:
        return _dict_to_node(json.load(f))


# -------------------------------
# NODE MAP
# -------------------------------
def create_node_map(root: TreeNode) -> Dict[str, TreeNode]:
    node_map = {}

    def dfs(node: TreeNode):
        node_map[node.node_id] = node
        for child in node.nodes:
            dfs(child)

    dfs(root)
    return node_map


# ============================================================
# HYBRID SEARCH
# ============================================================

def get_chroma_collection(doc_stem: str) -> chromadb.Collection:
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    safe_name = re.sub(r"[^a-zA-Z0-9\-]", "-", doc_stem)[:63]
    return client.get_or_create_collection(
        name=safe_name,
        metadata={"hnsw:space": "cosine"}
    )


def collection_is_indexed(collection: chromadb.Collection) -> bool:
    return collection.count() > 0


def build_chunk_index(root: TreeNode, collection: chromadb.Collection):
    if collection_is_indexed(collection):
        console.print("[dim]Chunk index already exists, skipping.[/dim]")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks = []
    all_ids = []
    all_metadata = []
    chunk_counter = 0

    def collect_chunks(node: TreeNode):
        nonlocal chunk_counter
        text = f"{node.title}\n{node.content}".strip()
        if not text or len(text) > 20:
            chunks = splitter.split_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_ids.append(f"chunk_{chunk_counter:06d}")
                all_metadata.append({
                    "node_id": node.node_id,
                    "node_title": node.title,
                })
                chunk_counter += 1
        for child in node.nodes:
            collect_chunks(child)

    collect_chunks(root)
    console.print(f"[dim]Total chunks to embed: {len(all_chunks)}[/dim]")

    if not all_chunks:
        console.print("[red]No chunks found — check tree content.[/red]")
        return

    BATCH = 100
    all_embeddings = []
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i: i + BATCH]
        embeddings = embedding_model.embed_documents(batch)
        all_embeddings.extend(embeddings)
        console.print(f"[dim]  Embedded {min(i + BATCH, len(all_chunks))}/{len(all_chunks)}[/dim]")

    for i in range(0, len(all_chunks), BATCH):
        collection.add(
            ids=all_ids[i: i + BATCH],
            documents=all_chunks[i: i + BATCH],
            embeddings=all_embeddings[i: i + BATCH],
            metadatas=all_metadata[i: i + BATCH],
        )

    console.print(f"[green]Chunk index built: {len(all_chunks)} chunks.[/green]")


def _compute_node_scores(
    chunk_ids: List[str],
    chunk_scores: List[float],
    chunk_metadatas: List[dict],
) -> Dict[str, float]:
    node_chunk_scores: Dict[str, List[float]] = {}
    for score, meta in zip(chunk_scores, chunk_metadatas):
        nid = meta["node_id"]
        node_chunk_scores.setdefault(nid, []).append(score)

    node_scores: Dict[str, float] = {}
    for nid, scores in node_chunk_scores.items():
        N = len(scores)
        node_scores[nid] = (1.0 / math.sqrt(N + 1)) * sum(scores)

    return node_scores


def value_based_search(
    query: str,
    collection: chromadb.Collection,
    top_k: int = TOP_K_CHUNKS,
) -> Tuple[List[str], float]:
    query_embedding = embedding_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"][0]:
        return [], 0.0

    distances = results["distances"][0]
    similarities = [1.0 - d for d in distances]
    metadatas = results["metadatas"][0]
    chunk_ids = results["ids"][0]

    node_scores = _compute_node_scores(chunk_ids, similarities, metadatas)
    ranked = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_node_ids = [nid for nid, _ in ranked[:MAX_NODES_DISPLAY]]
    top_score = ranked[0][1] if ranked else 0.0

    return ranked_node_ids, top_score


def tree_search(root: TreeNode, query: str) -> List[str]:
    def build_index(node: TreeNode, depth=0):
        indent = "  " * depth
        if not node.summary and node.title in ("Document", "Untitled"):
            text = ""
            for child in node.nodes:
                text += build_index(child, depth + 1)
            return text
        text = f"{indent}- [{node.node_id}] {node.title}\n"
        if node.summary:
            text += f"{indent}  {node.summary}\n"
        for child in node.nodes:
            text += build_index(child, depth + 1)
        return text

    tree_index = build_index(root)
    prompt = TREE_SEARCH_PROMPT.format(
        query=query,
        tree_index=tree_index[:40000],
    )
    result = search_model.invoke(prompt)
    return result.node_list[:MAX_NODES_DISPLAY]


def _check_confidence(query: str, context: str) -> bool:
    prompt = CONFIDENCE_PROMPT.format(query=query, context=context[:6000])
    result = confidence_model.invoke(prompt)
    return result.has_enough


def _build_context(node_ids: List[str], node_map: Dict[str, TreeNode]) -> str:
    parts = []
    for nid in node_ids:
        node = node_map.get(nid)
        if node:
            parts.append(f"### {node.title}\n{node.content}")
    return "\n\n".join(parts)


def hybrid_search(
    root: TreeNode,
    query: str,
    collection: chromadb.Collection,
) -> Tuple[List[str], str]:
    node_map = create_node_map(root)

    console.print("[dim]Running value-based search...[/dim]")
    value_node_ids, top_score = value_based_search(query, collection)
    console.print(f"[dim]Top node score: {top_score:.3f} (threshold: {CONFIDENCE_THRESHOLD})[/dim]")

    if value_node_ids and top_score >= CONFIDENCE_THRESHOLD:
        value_context = _build_context(value_node_ids, node_map)
        console.print("[dim]Checking if value-based context is sufficient...[/dim]")
        sufficient = _check_confidence(query, value_context)

        if sufficient:
            console.print("[green]Value-based search sufficient — skipping LLM search.[/green]")
            return value_node_ids, "value"

    console.print("[yellow]Running LLM tree search...[/yellow]")

    llm_results: List[str] = []

    def run_llm_search():
        try:
            llm_results.extend(tree_search(root, query))
        except Exception as e:
            console.print(f"[red]LLM search error: {e}[/red]")

    thread = threading.Thread(target=run_llm_search)
    thread.start()
    thread.join(timeout=30)

    merged_ids: List[str] = list(value_node_ids)
    seen = set(merged_ids)
    for nid in llm_results:
        if nid not in seen and nid in node_map:
            merged_ids.append(nid)
            seen.add(nid)

    final_ids = merged_ids[:MAX_NODES_DISPLAY + 2]
    return final_ids, "hybrid"


def retrieve_and_answer(
    root: TreeNode,
    query: str,
    collection: Optional[chromadb.Collection] = None,
) -> str:
    node_map = create_node_map(root)

    if collection is not None and collection_is_indexed(collection):
        node_ids, method = hybrid_search(root, query, collection)
        console.print(f"[dim]Search method: {method} | Nodes: {node_ids}[/dim]")
    else:
        console.print("[dim]No chunk index — using LLM tree search only.[/dim]")
        node_ids = tree_search(root, query)

    context = _build_context(node_ids, node_map)
    prompt = ANSWER_PROMPT.format(query=query, context=context[:12000])
    return model.invoke(prompt).content.strip()


def load_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# -------------------------------
# MAIN (local dev only)
# -------------------------------
if __name__ == "__main__":
    md_path = os.environ.get("TEST_MD_PATH", "test.md")
    tree_path = Path(md_path).with_suffix("").with_suffix(".tree.json")

    if tree_path.exists():
        console.print(f"[green]Loading saved tree from {tree_path}[/green]")
        tree = load_tree(tree_path)
    else:
        console.print("Loading markdown...")
        markdown_text = load_markdown(md_path)
        console.print("Building tree...")
        tree = build_tree(markdown_text)
        console.print("Summarizing tree...")
        summarize_tree(tree)
        save_tree(tree, tree_path)
        console.print(f"[green]Tree saved to {tree_path}[/green]")

    console.print(display_tree(tree))
    console.print(f"\n[bold]Tree has {tree.node_count()} nodes.[/bold]")

    doc_stem = Path(md_path).stem
    collection = get_chroma_collection(doc_stem)

    if not collection_is_indexed(collection):
        console.print("\n[yellow]Building chunk index...[/yellow]")
        build_chunk_index(tree, collection)
    else:
        console.print(f"[green]Chunk index loaded: {collection.count()} chunks.[/green]")

    console.print("\n[bold cyan]Ask anything. Type 'exit' to quit.[/bold cyan]\n")
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            break
        answer = retrieve_and_answer(tree, query, collection)
        console.print(f"\n[bold green]Answer:[/bold green] {answer}\n")