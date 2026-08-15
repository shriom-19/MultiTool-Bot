# pyrefly: ignore [missing-import]

from __future__ import annotations

import os
import sqlite3
import requests
import tempfile

from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file."
    )


# ============================================================
# 2. GEMINI LLM
# ============================================================
# Gemini is used ONLY as the LLM.
# No Gemini embeddings are used.
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)


# ============================================================
# 3. LOCAL HUGGING FACE EMBEDDINGS
# ============================================================
# Embeddings run locally.
# FAISS stores/searches these vectors.
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 4. PDF RETRIEVER STORAGE
# ============================================================

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(
    thread_id: Optional[str]
):
    """Fetch the FAISS retriever for a thread."""

    if not thread_id:
        return None

    return _THREAD_RETRIEVERS.get(
        str(thread_id)
    )


# ============================================================
# 5. PDF INGESTION
# ============================================================

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None
) -> dict:
    """
    Load a PDF, split it into chunks,
    create local embeddings, and store
    the retriever for the current thread.
    """

    if not file_bytes:
        raise ValueError(
            "No bytes received for ingestion."
        )

    thread_id = str(thread_id)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs:
            raise ValueError(
                "No readable content found in the PDF."
            )

        # ----------------------------------------------------
        # Split document into chunks
        # ----------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError(
                "PDF could not be split into text chunks."
            )

        # ----------------------------------------------------
        # Create FAISS vector store
        # ----------------------------------------------------

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        # ----------------------------------------------------
        # Create retriever
        # ----------------------------------------------------

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4
            }
        )

        # ----------------------------------------------------
        # Store retriever by thread
        # ----------------------------------------------------

        _THREAD_RETRIEVERS[thread_id] = retriever

        _THREAD_METADATA[thread_id] = {
            "filename": (
                filename
                or os.path.basename(temp_path)
            ),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": (
                filename
                or os.path.basename(temp_path)
            ),
            "documents": len(docs),
            "chunks": len(chunks),
        }

    finally:

        try:
            os.remove(temp_path)

        except OSError:
            pass



def delete_thread(thread_id: str) -> bool:
    """
    Delete a conversation/thread from the SQLite checkpointer
    and remove its associated RAG retriever and metadata.
    """

    thread_id = str(thread_id)

    try:
        # Delete RAG data from memory
        _THREAD_RETRIEVERS.pop(thread_id, None)
        _THREAD_METADATA.pop(thread_id, None)

        # Delete LangGraph checkpoints for this thread
        conn.execute(
            """
            DELETE FROM checkpoints
            WHERE thread_id = ?
            """,
            (thread_id,),
        )

        # Delete related checkpoint writes
        try:
            conn.execute(
                """
                DELETE FROM checkpoint_writes
                WHERE thread_id = ?
                """,
                (thread_id,),
            )
        except sqlite3.OperationalError:
            pass

        # Delete related checkpoint blobs
        try:
            conn.execute(
                """
                DELETE FROM checkpoint_blobs
                WHERE thread_id = ?
                """,
                (thread_id,),
            )
        except sqlite3.OperationalError:
            pass

        conn.commit()

        return True

    except Exception as e:
        print(f"Error deleting thread {thread_id}: {e}")
        return False
# ============================================================
# 6. TOOLS
# ============================================================

search_tool = DuckDuckGoSearchRun(
    region="us-en"
)


# ------------------------------------------------------------
# Calculator
# ------------------------------------------------------------

@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
) -> dict:
    """
    Perform basic arithmetic.

    Supported operations:
    add, sub, mul, div
    """

    try:

        if operation == "add":

            result = (
                first_num
                + second_num
            )

        elif operation == "sub":

            result = (
                first_num
                - second_num
            )

        elif operation == "mul":

            result = (
                first_num
                * second_num
            )

        elif operation == "div":

            if second_num == 0:

                return {
                    "error":
                    "Division by zero is not allowed"
                }

            result = (
                first_num
                / second_num
            )

        else:

            return {
                "error":
                f"Unsupported operation '{operation}'"
            }

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ------------------------------------------------------------
# Stock Price
# ------------------------------------------------------------

@tool
def get_stock_price(
    symbol: str
) -> dict:
    """
    Fetch latest stock price for a symbol.
    """

    alpha_key = os.getenv(
        "ALPHAVANTAGE_API_KEY"
    )

    if not alpha_key:

        return {
            "error":
            "ALPHAVANTAGE_API_KEY is not configured."
        }

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey={alpha_key}"
    )

    try:

        response = requests.get(
            url,
            timeout=15
        )

        response.raise_for_status()

        return response.json()

    except Exception as e:

        return {
            "error": str(e)
        }


# ------------------------------------------------------------
# RAG Tool
# ------------------------------------------------------------

@tool
def rag_tool(
    query: str,
    thread_id: Optional[str] = None
) -> dict:
    """
    Retrieve relevant information from
    the PDF associated with the current chat thread.
    """

    if not thread_id:

        return {
            "error":
            "thread_id is required for PDF retrieval.",
            "query": query,
        }

    thread_id = str(thread_id)

    retriever = _get_retriever(
        thread_id
    )

    if retriever is None:

        return {
            "error":
            "No document indexed for this chat. "
            "Upload a PDF first.",
            "query": query,
        }

    try:

        results = retriever.invoke(
            query
        )

        context = [
            doc.page_content
            for doc in results
        ]

        metadata = [
            doc.metadata
            for doc in results
        ]

        return {
            "query": query,
            "context": context,
            "metadata": metadata,
            "source_file": (
                _THREAD_METADATA
                .get(thread_id, {})
                .get("filename")
            ),
        }

    except Exception as e:

        return {
            "error": str(e),
            "query": query,
        }


# ============================================================
# 7. TOOL LIST
# ============================================================

tools = [
    search_tool,
    get_stock_price,
    calculator,
    rag_tool,
]

llm_with_tools = llm.bind_tools(
    tools
)


# ============================================================
# 8. LANGGRAPH STATE
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# 9. CHAT NODE
# ============================================================

def chat_node(
    state: ChatState,
    config=None
):
    """
    LLM node that can:
    - answer directly
    - call RAG
    - search the web
    - calculate
    - fetch stock prices
    """

    thread_id = None

    if config and isinstance(
        config,
        dict
    ):

        thread_id = (
            config
            .get("configurable", {})
            .get("thread_id")
        )

    system_message = SystemMessage(
        content=(
            "You are a helpful multi-purpose assistant.\n\n"

            "TOOLS:\n"
            "1. rag_tool - Use this for questions "
            "about an uploaded PDF.\n"
            "2. search_tool - Use this for web searches "
            "and current information.\n"
            "3. get_stock_price - Use this for stock prices.\n"
            "4. calculator - Use this for calculations.\n\n"

            f"Current thread_id: {thread_id}\n\n"

            "IMPORTANT:\n"
            "When using rag_tool, ALWAYS provide "
            "the current thread_id.\n"
            
            "For PDF questions, use rag_tool instead of "
            "answering from general knowledge.\n"

            "If no PDF is indexed and the user asks "
            "about a document, tell them to upload a PDF."
        )
    )

    messages = [
        system_message,
        *state["messages"]
    ]

    response = llm_with_tools.invoke(
        messages,
        config=config
    )

    return {
        "messages": [response]
    }


# ============================================================
# 10. TOOL NODE
# ============================================================

tool_node = ToolNode(
    tools
)


# ============================================================
# 11. SQLITE CHECKPOINTER
# ============================================================

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(
    conn=conn
)


# ============================================================
# 12. LANGGRAPH
# ============================================================

graph = StateGraph(
    ChatState
)

graph.add_node(
    "chat_node",
    chat_node
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_edge(
    START,
    "chat_node"
)

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

graph.add_edge(
    "tools",
    "chat_node"
)


# ============================================================
# 13. COMPILE GRAPH
# ============================================================

chatbot = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# 14. THREAD HELPERS
# ============================================================

def list_threads():
    """
    Return all thread IDs stored in SQLite.
    """

    all_threads = set()

    for thread in checkpointer.list(None):

        try:

            thread_id = (
                thread
                .config["configurable"]["thread_id"]
            )

            all_threads.add(
                str(thread_id)
            )

        except (
            KeyError,
            TypeError
        ):

            continue

    return list(all_threads)


def thread_document_metadata(
    thread_id: str
) -> dict:
    """
    Return metadata of the PDF
    indexed for the given thread.
    """

    return _THREAD_METADATA.get(
        str(thread_id),
        {}
    )