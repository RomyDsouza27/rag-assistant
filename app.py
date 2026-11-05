
import streamlit as st
from pathlib import Path

from rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="R",
    layout="wide"
)

st.title("RAG Research Assistant")

st.write(
    "Ask questions about the research documents. "
    "Answers are checked against retrieved source context."
)


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


def display_source_image(source):
    """Display a rendered PDF page when its image path is available."""
    image_path = source.get("image_path")

    if not image_path:
        st.caption("No associated page image is available in this source.")
        return

    path = Path(image_path)

    # Handle paths relative to the project directory.
    if not path.is_absolute():
        path = Path.cwd() / path

    if path.is_file():
        st.image(
            str(path),
            caption=(
                f"{source.get('source', 'Unknown source')} "
                f"— Page {source.get('page', 'Unknown')}"
            ),
            use_container_width=True
        )
    else:
        st.warning(
            "The associated page image could not be found. "
            f"Expected path: {path}"
        )


def display_hallucination_check(check):
    """Display the current baseline hallucination-check results."""
    if not check:
        st.info("No hallucination-check results are available.")
        return

    st.subheader("Hallucination Check")

    if check.get("supported", False):
        st.success(
            "No unsupported sentences detected by the baseline."
        )
    else:
        st.warning(
            check.get(
                "warning",
                "Potentially unsupported content detected."
            )
        )

    score = check.get("score")

    if score is not None:
        st.metric("Support Score", f"{score:.2f}")

    st.caption(
        "This is a rule-based lexical-overlap score, "
        "not a probability that the answer is correct."
    )

    unsupported = check.get("unsupported_sentences", [])

    if unsupported:
        st.write("**Potentially unsupported claims:**")

        for sentence in unsupported:
            st.write(f"- {sentence}")


def display_sources(sources):
    """Display retrieved text and associated page images."""
    if not sources:
        st.info("No sources were retrieved.")
        return

    st.subheader("Retrieved Sources")

    for i, source in enumerate(sources, start=1):
        source_name = source.get("source", "Unknown")
        page_number = source.get("page", "Unknown")

        with st.expander(
            f"{i}. {source_name} — Page {page_number}",
            expanded=False
        ):
            score = source.get("score")

            if score is not None:
                st.write(f"**Similarity score:** {score:.3f}")

            st.write(
                source.get("text", "No text available.")
            )

            # Show the page image if its path survived retrieval.
            display_source_image(source)


try:
    pipeline = load_pipeline()

except Exception as e:
    st.error("RAG system could not be loaded.")
    st.exception(e)
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


# Restore the previous conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant":
            display_hallucination_check(
                message.get("hallucination_check")
            )

            display_sources(
                message.get("sources", [])
            )


# Accept a new question.
question = st.chat_input(
    "Ask a question about the research documents..."
)


if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Searching documents and generating answer..."
        ):
            try:
                result = pipeline.ask(question)

                answer = result.get(
                    "answer",
                    "No answer was generated."
                )

                check = result.get(
                    "hallucination_check",
                    {}
                )

                sources = result.get("sources", [])

                st.subheader("Generated Answer")
                st.write(answer)

                display_hallucination_check(check)

                display_sources(sources)

                # Save everything needed to restore this answer.
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "hallucination_check": check,
                    "sources": sources
                })

            except Exception as e:
                st.error(
                    "An error occurred while processing your question."
                )
                st.exception(e)
