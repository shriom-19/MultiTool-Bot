import uuid
import streamlit as st

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from backend import (
    chatbot,
    list_threads,
    ingest_pdf,
    thread_document_metadata,
    delete_thread,
)

# ============================================================
# Utility Functions
# ============================================================

def generate_thread_id():
    """Generate a unique thread ID."""
    return str(uuid.uuid4())


def reset_chat():
    """Create a new chat thread."""
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    add_thread(thread_id)


def add_thread(thread_id):
    """Add thread to the session thread list."""
    thread_id = str(thread_id)

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    """
    Load conversation messages from LangGraph SQLite checkpointer.
    Only HumanMessage and AIMessage are displayed.
    Tool messages are ignored in the UI.
    """

    thread_id = str(thread_id)

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    messages = state.values.get("messages", [])

    conversation = []

    for msg in messages:

        # ----------------------------------------------------
        # User message
        # ----------------------------------------------------

        if isinstance(msg, HumanMessage):

            if isinstance(msg.content, str):

                conversation.append({
                    "role": "user",
                    "content": msg.content
                })

        # ----------------------------------------------------
        # AI message
        # ----------------------------------------------------

        elif isinstance(msg, AIMessage):

            content = msg.content

            if isinstance(content, str):

                if content.strip():

                    conversation.append({
                        "role": "assistant",
                        "content": content
                    })

            elif isinstance(content, list):

                text_parts = []

                for block in content:

                    if isinstance(block, str):

                        text_parts.append(block)

                    elif isinstance(block, dict):

                        if block.get("type") == "text":

                            text_parts.append(
                                block.get("text", "")
                            )

                text = "".join(text_parts)

                if text.strip():

                    conversation.append({
                        "role": "assistant",
                        "content": text
                    })

    return conversation


def extract_ai_text(content):
    """
    Safely extract text from Gemini/LangChain AIMessage content.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, str):

                text_parts.append(block)

            elif isinstance(block, dict):

                if block.get("type") == "text":

                    text_parts.append(
                        block.get("text", "")
                    )

        return "".join(text_parts)

    return ""


# ============================================================
# Session Initialization
# ============================================================

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = [
        str(thread_id)
        for thread_id in list_threads()
    ]

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}


# ============================================================
# Handle deleted current thread
# ============================================================

if st.session_state["thread_id"] is None:

    st.sidebar.info(
        "No active chat. Click 'New Chat' to start one."
    )

    st.title("Multi Utility Chatbot")

    st.chat_input(
        "Click 'New Chat' to start chatting",
        disabled=True
    )

    st.stop()


# ============================================================
# Current active thread
# ============================================================

add_thread(
    st.session_state["thread_id"]
)

thread_key = str(
    st.session_state["thread_id"]
)


# Current thread document cache
thread_docs = st.session_state[
    "ingested_docs"
].setdefault(
    thread_key,
    {}
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title(
    "LangGraph PDF Chatbot"
)

st.sidebar.markdown(
    f"**Thread ID:** `{thread_key}`"
)


# ============================================================
# New Chat
# ============================================================

if st.sidebar.button(
    "New Chat",
    use_container_width=True
):

    reset_chat()
    st.rerun()


# ============================================================
# Current PDF Status
# ============================================================

doc_meta = thread_document_metadata(
    thread_key
)


if doc_meta:

    st.sidebar.success(
        f"📄 Using `{doc_meta.get('filename')}`\n\n"
        f"Pages: {doc_meta.get('documents', 0)}\n\n"
        f"Chunks: {doc_meta.get('chunks', 0)}"
    )

else:

    st.sidebar.info(
        "No PDF indexed yet."
    )


# ============================================================
# PDF Upload
# ============================================================

st.sidebar.subheader(
    "📚 PDF RAG"
)


uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat",
    type=["pdf"],
)


if uploaded_pdf:

    filename = uploaded_pdf.name

    # --------------------------------------------------------
    # Check whether this PDF is already indexed
    # --------------------------------------------------------

    if (
        doc_meta
        and doc_meta.get("filename") == filename
    ):

        st.sidebar.info(
            f"`{filename}` is already processed "
            "for this chat."
        )

    else:

        # ----------------------------------------------------
        # Index button
        # ----------------------------------------------------

        if st.sidebar.button(
            "Index PDF",
            use_container_width=True
        ):

            try:

                with st.sidebar.status(
                    "Indexing PDF...",
                    expanded=True
                ) as status_box:

                    summary = ingest_pdf(
                        uploaded_pdf.getvalue(),
                        thread_id=thread_key,
                        filename=filename,
                    )

                    # Store locally in session state
                    thread_docs[filename] = summary

                    status_box.update(
                        label="✅ PDF indexed",
                        state="complete",
                        expanded=False,
                    )

                st.sidebar.success(
                    f"{summary.get('documents', 0)} pages | "
                    f"{summary.get('chunks', 0)} chunks"
                )

            except Exception as e:

                st.sidebar.error(
                    f"PDF indexing failed: {e}"
                )


# ============================================================
# Past Conversations
# ============================================================

st.sidebar.subheader("Past Conversations")

threads = [
    str(thread_id)
    for thread_id in st.session_state["chat_threads"]
][::-1]


if not threads:

    st.sidebar.write(
        "No past conversations yet."
    )

else:

    for saved_thread_id in threads:

        col1, col2 = st.sidebar.columns(
            [4, 1]
        )

        # -----------------------------------------------
        # Open conversation
        # -----------------------------------------------

        with col1:

            if st.button(
                saved_thread_id,
                key=f"open-{saved_thread_id}",
                use_container_width=True,
            ):

                st.session_state[
                    "thread_id"
                ] = saved_thread_id

                st.session_state[
                    "message_history"
                ] = load_conversation(
                    saved_thread_id
                )

                st.rerun()

        # -----------------------------------------------
        # Delete conversation
        # -----------------------------------------------

        with col2:

            if st.button(
                "🗑️",
                key=f"delete-{saved_thread_id}",
                help="Delete this conversation",
            ):

                deleted = delete_thread(saved_thread_id)

                if deleted:

                    # Remove thread from sidebar
                    if saved_thread_id in st.session_state["chat_threads"]:
                        st.session_state["chat_threads"].remove(
                            saved_thread_id
                        )

                    # Remove PDF/RAG session data
                    st.session_state["ingested_docs"].pop(
                        saved_thread_id,
                        None
                    )

                    # If the deleted thread is the current thread,
                    # clear the screen but DO NOT create a new thread.
                    if str(st.session_state["thread_id"]) == saved_thread_id:

                        st.session_state["thread_id"] = None
                        st.session_state["message_history"] = []

                    st.rerun()

                else:
                    st.sidebar.error(
                        "Could not delete conversation."
                    )


# ============================================================
# Main Layout
# ============================================================

st.title(
    "Multi Utility Chatbot"
)

st.caption(
    "Gemini + LangGraph + FAISS + "
    "Local Hugging Face Embeddings"
)


# ============================================================
# Chat Area
# ============================================================

for message in st.session_state[
    "message_history"
]:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ============================================================
# User Input
# ============================================================

user_input = st.chat_input(
    "Ask about your document or use tools"
)


if user_input:

    # ========================================================
    # Add user message to UI history
    # ========================================================

    st.session_state[
        "message_history"
    ].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):

        st.write(
            user_input
        )


    # ========================================================
    # LangGraph Configuration
    # ========================================================

    CONFIG = {
        "configurable": {
            "thread_id": thread_key
        },

        "metadata": {
            "thread_id": thread_key
        },

        "run_name": "chat_turn",
    }


    # ========================================================
    # Assistant Response
    # ========================================================

    with st.chat_message("assistant"):

        status_holder = {
            "box": None
        }


        def ai_only_stream():

            try:

                for message_chunk, metadata in chatbot.stream(

                    {
                        "messages": [
                            HumanMessage(
                                content=user_input
                            )
                        ]
                    },

                    config=CONFIG,

                    stream_mode="messages",
                ):

                    # ========================================
                    # Tool execution status
                    # ========================================

                    if isinstance(
                        message_chunk,
                        ToolMessage
                    ):

                        tool_name = getattr(
                            message_chunk,
                            "name",
                            "tool"
                        )

                        if status_holder["box"] is None:

                            status_holder[
                                "box"
                            ] = st.status(

                                f"🔧 Using `{tool_name}`...",
                                expanded=True,
                            )

                        else:

                            status_holder[
                                "box"
                            ].update(

                                label=(
                                    f"🔧 Using `{tool_name}`..."
                                ),

                                state="running",

                                expanded=True,
                            )


                    # ========================================
                    # AI response streaming
                    # ========================================

                    if isinstance(
                        message_chunk,
                        AIMessage
                    ):

                        text = extract_ai_text(
                            message_chunk.content
                        )

                        if text:

                            yield text


            except Exception as e:

                st.error(
                    f"Error while generating response: {e}"
                )


        # Stream response
        ai_message = st.write_stream(
            ai_only_stream()
        )


        # ====================================================
        # Tool finished
        # ====================================================

        if status_holder["box"] is not None:

            status_holder[
                "box"
            ].update(

                label="✅ Tool finished",

                state="complete",

                expanded=False,
            )


    # ========================================================
    # Save assistant response in UI history
    # ========================================================

    if ai_message:

        st.session_state[
            "message_history"
        ].append(
            {
                "role": "assistant",
                "content": ai_message,
            }
        )


    # ========================================================
    # Show current document information
    # ========================================================

    doc_meta = thread_document_metadata(
        thread_key
    )

    if doc_meta:

        st.caption(
            f"📄 Document indexed: "
            f"{doc_meta.get('filename')} | "
            f"Chunks: "
            f"{doc_meta.get('chunks', 0)} | "
            f"Pages: "
            f"{doc_meta.get('documents', 0)}"
        )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "RAG: Hugging Face all-MiniLM-L6-v2 + FAISS | "
    "LLM: Gemini | "
    "Memory: SQLite + LangGraph"
)