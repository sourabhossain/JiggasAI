from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from django.conf import settings
from .chroma_client import get_collection


class RAGState(TypedDict):
    query: str
    history: List[dict]
    retrieved_chunks: List[str]
    sources: List[str]
    answer: str
    needs_web_search: bool
    web_search_error: str


_agent = None


def retrieve_node(state: RAGState) -> dict:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.GOOGLE_API_KEY,
    )
    query_vector = embeddings.embed_query(state["query"])

    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=4,
    )

    chunks = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    sources = list({m.get("source", "") for m in metadatas if m.get("source")})

    return {
        "retrieved_chunks": chunks,
        "sources": sources,
    }


def router(state: RAGState) -> str:
    chunks = state.get("retrieved_chunks", [])

    try:
        total_chunks = get_collection().count()
    except Exception:
        total_chunks = 0

    if total_chunks == 0:
        print("[ROUTER] No documents in knowledge base - routing to no_docs")
        return "no_docs"

    if len(chunks) == 0:
        print("[ROUTER] Documents exist but no relevant chunks - routing to web search")
        return "web_search"

    total_content = " ".join(chunks)
    if len(total_content) < 200:
        print("[ROUTER] Chunks too short - routing to web search")
        return "web_search"

    print(f"[ROUTER] Found {len(chunks)} good chunks - routing to generate")
    return "generate"


def no_docs_node(state: RAGState) -> dict:
    print("[NO DOCS] No documents in knowledge base")
    return {
        "answer": (
            "কোনো document upload করা নেই।\n\n"
            "JiggasAI আপনার uploaded documents থেকে উত্তর দেয়। "
            "প্রথমে কিছু PDF upload করুন, তারপর প্রশ্ন করুন।\n\n"
            "**How to upload:**\n"
            "1. Sidebar এ Documents লিংকে ক্লিক করুন (admin only)\n"
            "2. Upload PDF বাটনে ক্লিক করুন\n"
            "3. আপনার PDF file select করুন\n"
            "4. Upload হলে এখানে এসে প্রশ্ন করুন"
        ),
        "sources": [],
    }


def web_search_node(state: RAGState) -> dict:
    from tavily import TavilyClient

    print(f"[WEB SEARCH] Searching web for: {state['query']}")

    try:
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        response = client.search(
            query=state["query"][:400],
            max_results=3,
            search_depth="basic",
        )

        chunks = []
        sources = []
        for result in response.get("results", []):
            chunks.append(result.get("content", ""))
            sources.append(result.get("url", "unknown"))

        print(f"[WEB SEARCH] Got {len(chunks)} results from web")

        return {
            "retrieved_chunks": chunks,
            "sources": sources,
            "needs_web_search": True,
            "web_search_error": "",
        }

    except Exception as e:
        print(f"[WEB SEARCH ERROR] {type(e).__name__}: {e}")
        return {
            "retrieved_chunks": [],
            "sources": [],
            "needs_web_search": True,
            "web_search_error": str(e),
        }


def generate_node(state: RAGState) -> dict:
    if state.get("web_search_error") and not state.get("retrieved_chunks"):
        return {
            "answer": "I couldn't find information in your documents or the web right now. "
                      "Please try again in a moment."
        }

    context = "\n\n".join(state["retrieved_chunks"])

    history_text = ""
    for turn in state["history"]:
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    used_web = state.get("needs_web_search", False)
    source_label = "web search results" if used_web else "uploaded documents"

    prompt = f"""You are a helpful assistant for JiggasAI.
Answer the question based on the {source_label} below.
If the answer is not in the context, say so clearly.

Context:
{context}
{f"Previous conversation:{chr(10)}{history_text}" if history_text else ""}
User: {state["query"]}
Assistant:"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.2,
    )
    response = llm.invoke(prompt)

    return {"answer": response.content}


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("no_docs", no_docs_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")

    graph.add_conditional_edges(
        "retrieve",
        router,
        {
            "generate": "generate",
            "web_search": "web_search",
            "no_docs": "no_docs",
        }
    )

    graph.add_edge("web_search", "generate")
    graph.add_edge("no_docs", END)
    graph.add_edge("generate", END)

    return graph.compile()


def run_agent(query: str, history: list = None) -> dict:
    global _agent
    if _agent is None:
        _agent = build_graph()

    try:
        result = _agent.invoke({
            "query": query,
            "history": (history or [])[-5:],
            "retrieved_chunks": [],
            "sources": [],
            "answer": "",
            "needs_web_search": False,
            "web_search_error": "",
        })
        return {
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except Exception as e:
        print(f"[AGENT ERROR] {type(e).__name__}: {e}")
        return {
            "answer": "Something went wrong. Please try again.",
            "sources": [],
        }


def run_agent_stream(query: str, history: list = None):
    if history is None:
        history = []

    state: RAGState = {
        "query": query,
        "history": history[-5:],
        "retrieved_chunks": [],
        "sources": [],
        "answer": "",
        "needs_web_search": False,
        "web_search_error": "",
    }

    try:
        state.update(retrieve_node(state))
    except Exception as e:
        print(f"[RETRIEVE ERROR] {type(e).__name__}: {e}")
        msg = "Couldn't reach the document index. Please try again in a moment."
        yield msg
        yield "__SOURCES__"
        yield f"__ANSWER__{msg}"
        return

    route = router(state)

    if route == "no_docs":
        answer = no_docs_node(state)["answer"]
        for word in answer.split(" "):
            yield word + " "
        yield "__SOURCES__"
        yield f"__ANSWER__{answer}"
        return

    if route == "web_search":
        state.update(web_search_node(state))

    if state.get("web_search_error") and not state.get("retrieved_chunks"):
        msg = ("I couldn't find information in your documents or the web right now. "
               "Please try again in a moment.")
        yield msg
        yield "__SOURCES__"
        yield f"__ANSWER__{msg}"
        return

    context = "\n\n".join(state["retrieved_chunks"])
    history_text = ""
    for turn in state["history"]:
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    used_web = state.get("needs_web_search", False)
    source_label = "web search results" if used_web else "uploaded documents"

    prompt = f"""You are a helpful assistant for JiggasAI.
Answer the question based on the {source_label} below.
If the answer is not in the context, say so clearly.

Context:
{context}
{f"Previous conversation:{chr(10)}{history_text}" if history_text else ""}
User: {state["query"]}
Assistant:"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.2,
    )

    sources = state.get("sources", [])
    full_answer = ""

    try:
        for chunk in llm.stream(prompt):
            token = chunk.content
            if token:
                full_answer += token
                yield token

    except Exception as e:
        error_type = type(e).__name__
        err_str = str(e).lower()
        print(f"[STREAM ERROR] {error_type}: {e}")

        if "quota" in err_str or "rate" in err_str or "resource_exhausted" in err_str:
            error_msg = "Google AI quota exceeded. Please try again later."
        elif "api_key" in err_str or "credentials" in err_str or "api key not valid" in err_str:
            error_msg = "AI service configuration error. Please contact admin."
        elif "timeout" in err_str:
            error_msg = "Request timed out. Please try again."
        else:
            error_msg = "AI service temporarily unavailable. Please try again."

        if full_answer:
            yield " [Note: response was cut short due to an error]"
        else:
            yield error_msg

        yield f"__SOURCES__{','.join(sources)}"
        yield f"__ANSWER__{full_answer or error_msg}"
        return

    if sources:
        yield f"__SOURCES__{','.join(sources)}"

    yield f"__ANSWER__{full_answer}"
