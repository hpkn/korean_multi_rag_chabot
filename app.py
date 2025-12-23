# Korean RAG Chatbot - ChromaDB + Ollama
import asyncio
import base64
import gc
import hashlib
import os
import re
import uuid
from pathlib import Path

import chromadb
import ollama
from dotenv import load_dotenv
from pypdf import PdfReader
import streamlit as st

from etl.embedding_generation import OllamaEmbedding
from _party import obsidian
from _party.jira_client import JiraClient, JiraConfig, extract_issues_content
from _party.notion_client import NotionClient, NotionConfig, extract_pages_content

# Load environment variables
load_dotenv()

# Configuration from .env
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gemma3:1b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# Obsidian default path
DEFAULT_OBSIDIAN_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "")

# Jira defaults
DEFAULT_JIRA_URL = os.getenv("JIRA_URL", "")
DEFAULT_JIRA_EMAIL = os.getenv("JIRA_EMAIL", "")
DEFAULT_JIRA_TOKEN = os.getenv("JIRA_API_TOKEN", "")

# Notion defaults
DEFAULT_NOTION_TOKEN = os.getenv("NOTION_API_TOKEN", "")

# Initialize clients
chroma_client = chromadb.Client()
embedding_generator = OllamaEmbedding()


# ============================================================================
# Core Functions
# ============================================================================

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    async def _generate():
        return await embedding_generator.generate_embeddings_batch(texts)
    return asyncio.run(_generate())


def clean_korean_text(text: str) -> str:
    """Clean and normalize Korean text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'- \d+ -', '', text)
    text = text.replace('．', '.').replace('，', ',')
    return text.strip()


def chunk_korean_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Chunk Korean text with awareness of sentence boundaries."""
    text = clean_korean_text(text)

    sentence_endings = re.compile(r'([.!?。]\s*|다\.\s*|요\.\s*|니다\.\s*|습니다\.\s*)')
    sentences = sentence_endings.split(text)

    full_sentences = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        if i + 1 < len(sentences) and sentence_endings.match(sentences[i + 1]):
            sentence += sentences[i + 1]
            i += 2
        else:
            i += 1
        if sentence.strip():
            full_sentences.append(sentence.strip())

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in full_sentences:
        sentence_len = len(sentence)

        if current_length + sentence_len > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            overlap_text = ' '.join(current_chunk)
            while len(overlap_text) > overlap and current_chunk:
                current_chunk.pop(0)
                overlap_text = ' '.join(current_chunk)
            current_length = len(overlap_text)

        current_chunk.append(sentence)
        current_length += sentence_len

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return [c for c in chunks if len(c) > 50]


def generate_collection_name(name: str, session_id: str) -> str:
    """Generate a valid ChromaDB collection name."""
    file_hash = hashlib.md5(name.encode()).hexdigest()[:8]
    return f"rag-{session_id[:8]}-{file_hash}"


# ============================================================================
# PDF Functions
# ============================================================================

def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file."""
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


# ============================================================================
# Indexing Functions
# ============================================================================

def index_documents(chunks: list[str], metadata: list[dict], collection_name: str):
    """Index chunks with embeddings into ChromaDB."""
    if not chunks:
        return None, 0

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    embeddings = get_embeddings(chunks)

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadata if metadata else None
    )

    return collection, len(chunks)


# ============================================================================
# LLM Query
# ============================================================================

def query_llm(prompt: str, context: str):
    """Query LLM with RAG context."""
    system_prompt = f"""당신은 문서 기반 질문 답변 도우미입니다.
아래 문서 내용을 바탕으로 질문에 답변하세요.
문서에 없는 내용은 "문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.

문서 내용:
{context}"""

    return ollama.chat(
        model=OLLAMA_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )


# ============================================================================
# UI Functions
# ============================================================================

def reset_chat():
    """Reset chat state."""
    st.session_state.messages = []
    st.session_state.collection = None
    gc.collect()


def display_pdf(file):
    """Display PDF preview."""
    file.seek(0)
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
        f'width="100%" height="400" type="application/pdf"></iframe>',
        unsafe_allow_html=True
    )


# ============================================================================
# Main App
# ============================================================================

# Initialize session state (each key separately to handle upgrades)
if "id" not in st.session_state:
    st.session_state.id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "obsidian_path" not in st.session_state:
    st.session_state.obsidian_path = DEFAULT_OBSIDIAN_PATH
if "source_type" not in st.session_state:
    st.session_state.source_type = None
if "jira_config" not in st.session_state:
    st.session_state.jira_config = None
    # Auto-configure Jira if credentials in .env
    if DEFAULT_JIRA_URL and DEFAULT_JIRA_EMAIL and DEFAULT_JIRA_TOKEN:
        st.session_state.jira_config = JiraConfig(
            url=DEFAULT_JIRA_URL,
            email=DEFAULT_JIRA_EMAIL,
            api_token=DEFAULT_JIRA_TOKEN
        )
if "jira_issues" not in st.session_state:
    st.session_state.jira_issues = []
if "notion_config" not in st.session_state:
    st.session_state.notion_config = None
    # Auto-configure Notion if token in .env
    if DEFAULT_NOTION_TOKEN:
        st.session_state.notion_config = NotionConfig(api_token=DEFAULT_NOTION_TOKEN)
if "notion_pages" not in st.session_state:
    st.session_state.notion_pages = []

session_id = st.session_state.id

# Page config
st.set_page_config(page_title="Korean RAG", layout="wide")

# Sidebar
with st.sidebar:
    st.header("📚 소스 선택")

    source_type = st.radio(
        "데이터 소스",
        ["PDF 업로드", "Obsidian 볼트", "Jira", "Notion"],
        key="source_selector"
    )

    st.divider()

    # ========== PDF Upload ==========
    if source_type == "PDF 업로드":
        st.subheader("📄 PDF 문서")
        uploaded_file = st.file_uploader("PDF 파일 선택", type="pdf")

        if uploaded_file:
            try:
                collection_name = generate_collection_name(uploaded_file.name, session_id)

                if st.session_state.collection is None or st.session_state.source_type != "pdf":
                    with st.spinner("문서 처리 중..."):
                        uploaded_file.seek(0)
                        text = extract_text_from_pdf(uploaded_file)
                        chunks = chunk_korean_text(text)

                        if not chunks:
                            st.error("텍스트를 추출할 수 없습니다.")
                            st.stop()

                        metadata = [{"source": uploaded_file.name} for _ in chunks]
                        collection, chunk_count = index_documents(chunks, metadata, collection_name)

                        st.session_state.collection = collection
                        st.session_state.source_type = "pdf"
                        st.success(f"✅ 준비 완료! ({chunk_count}개 청크)")

                st.markdown("### 미리보기")
                uploaded_file.seek(0)
                display_pdf(uploaded_file)

            except Exception as e:
                st.error(f"오류: {e}")

    # ========== Obsidian Vault ==========
    elif source_type == "Obsidian 볼트":
        st.subheader("🗃️ Obsidian 볼트")

        vault_path = st.text_input(
            "볼트 경로",
            value=st.session_state.obsidian_path,
            placeholder="C:/Users/username/Documents/Obsidian Vault"
        )

        if vault_path:
            st.session_state.obsidian_path = vault_path
            vault = Path(vault_path)

            if vault.exists():
                # Get folders using _party.obsidian
                folders = obsidian.get_vault_folders(vault)
                folder_names = [str(f.relative_to(vault)) if f != vault else "📁 전체 볼트" for f in folders]

                selected_folder_name = st.selectbox("폴더 선택", folder_names)

                if selected_folder_name == "📁 전체 볼트":
                    selected_folder = vault
                else:
                    selected_folder = vault / selected_folder_name

                # Show files in folder
                md_files = obsidian.get_markdown_files(selected_folder, recursive=False)

                if md_files:
                    st.markdown(f"**파일 ({len(md_files)}개)**")
                    for f in md_files[:10]:
                        st.text(f"  📝 {f.name}")
                    if len(md_files) > 10:
                        st.text(f"  ... 외 {len(md_files) - 10}개")

                # Options
                recursive = st.checkbox("하위 폴더 포함", value=True)

                # Index button
                if st.button("🔍 인덱싱 시작", type="primary"):
                    with st.spinner("인덱싱 중..."):
                        files = obsidian.get_markdown_files(selected_folder, recursive)

                        if not files:
                            st.warning("마크다운 파일이 없습니다.")
                        else:
                            # Extract content using _party.obsidian
                            chunks, metadata = obsidian.extract_vault_content(
                                files, vault, chunk_korean_text
                            )

                            if chunks:
                                collection_name = generate_collection_name(selected_folder_name, session_id)
                                collection, chunk_count = index_documents(chunks, metadata, collection_name)

                                st.session_state.collection = collection
                                st.session_state.source_type = "obsidian"
                                st.success(f"✅ 완료! {len(files)}개 파일, {chunk_count}개 청크")
                            else:
                                st.error("인덱싱 실패: 텍스트를 추출할 수 없습니다.")
            else:
                st.warning("경로가 존재하지 않습니다.")

    # ========== Jira ==========
    elif source_type == "Jira":
        st.subheader("🎫 Jira")

        # Connection settings
        with st.expander("연결 설정", expanded=st.session_state.jira_config is None):
            jira_url = st.text_input(
                "Jira URL",
                value=DEFAULT_JIRA_URL,
                placeholder="https://your-domain.atlassian.net"
            )
            jira_email = st.text_input(
                "이메일 / 사용자명",
                value=DEFAULT_JIRA_EMAIL,
                placeholder="user@example.com"
            )
            jira_token = st.text_input(
                "API 토큰",
                value=DEFAULT_JIRA_TOKEN,
                type="password",
                placeholder="API token or password"
            )

            if st.button("🔗 연결 테스트"):
                if jira_url and jira_email and jira_token:
                    config = JiraConfig(url=jira_url, email=jira_email, api_token=jira_token)
                    client = JiraClient(config)
                    success, message = client.test_connection()

                    if success:
                        st.success(message)
                        st.session_state.jira_config = config
                    else:
                        st.error(message)
                else:
                    st.warning("모든 필드를 입력해주세요.")

        # If connected, show project/search options
        if st.session_state.jira_config:
            client = JiraClient(st.session_state.jira_config)

            # Get projects
            projects = client.get_projects()

            if projects:
                project_options = ["직접 JQL 입력"] + [f"{p['key']} - {p['name']}" for p in projects]
                selected_project = st.selectbox("프로젝트 선택", project_options)

                if selected_project == "직접 JQL 입력":
                    jql = st.text_input(
                        "JQL 쿼리",
                        placeholder="project = MYPROJ AND status = Open"
                    )
                else:
                    project_key = selected_project.split(" - ")[0]
                    jql = f"project = {project_key} ORDER BY updated DESC"
                    st.code(jql, language="sql")

                max_issues = st.slider("최대 이슈 수", 10, 100, 50)

                if st.button("🔍 이슈 가져오기", type="primary"):
                    if jql:
                        with st.spinner("이슈 가져오는 중..."):
                            issues = client.search_issues(jql, max_results=max_issues)

                            if issues:
                                st.info(f"📋 {len(issues)}개 이슈 발견")

                                # Store issues for display
                                st.session_state.jira_issues = issues

                                # Extract and index content
                                chunks, metadata = extract_issues_content(
                                    issues, chunk_korean_text
                                )

                                if chunks:
                                    collection_name = generate_collection_name(
                                        f"jira-{jql[:20]}", session_id
                                    )
                                    collection, chunk_count = index_documents(
                                        chunks, metadata, collection_name
                                    )

                                    st.session_state.collection = collection
                                    st.session_state.source_type = "jira"
                                    st.success(f"✅ 완료! {len(issues)}개 이슈, {chunk_count}개 청크")
                                else:
                                    st.warning("이슈에서 텍스트를 추출할 수 없습니다.")
                            else:
                                if client.last_error:
                                    st.error(f"Jira API 오류: {client.last_error}")
                                else:
                                    st.warning(f"이슈를 찾을 수 없습니다. JQL: {jql}")
                    else:
                        st.warning("JQL 쿼리를 입력해주세요.")

                # Display fetched issues
                if st.session_state.jira_issues:
                    with st.expander(f"📋 이슈 목록 ({len(st.session_state.jira_issues)}개)", expanded=False):
                        for issue in st.session_state.jira_issues:
                            key = issue.get("key", "")
                            fields = issue.get("fields", {})
                            summary = fields.get("summary", "제목 없음")
                            status = fields.get("status", {}).get("name", "")
                            st.markdown(f"**{key}**: {summary} `{status}`")
            else:
                if client.last_error:
                    st.error(f"프로젝트 로드 오류: {client.last_error}")
                else:
                    st.warning("프로젝트를 가져올 수 없습니다. 연결을 확인해주세요.")

    # ========== Notion ==========
    else:  # Notion
        st.subheader("📝 Notion")

        # Connection settings
        with st.expander("연결 설정", expanded=st.session_state.notion_config is None):
            notion_token = st.text_input(
                "Integration Token",
                value=DEFAULT_NOTION_TOKEN,
                type="password",
                placeholder="secret_xxx..."
            )
            st.caption("🔗 [Notion Integrations](https://www.notion.so/my-integrations)에서 토큰을 생성하세요")

            if st.button("🔗 연결 테스트", key="notion_test"):
                if notion_token:
                    config = NotionConfig(api_token=notion_token)
                    client = NotionClient(config)
                    success, message = client.test_connection()

                    if success:
                        st.success(message)
                        st.session_state.notion_config = config
                    else:
                        st.error(message)
                else:
                    st.warning("Integration Token을 입력해주세요.")

        # If connected, show search options
        if st.session_state.notion_config:
            client = NotionClient(st.session_state.notion_config)

            search_query = st.text_input(
                "검색어 (선택)",
                placeholder="검색어 입력 또는 비워두면 전체 검색"
            )

            filter_options = ["전체", "페이지만", "데이터베이스만"]
            filter_choice = st.selectbox("필터", filter_options)

            filter_type = None
            if filter_choice == "페이지만":
                filter_type = "page"
            elif filter_choice == "데이터베이스만":
                filter_type = "database"

            max_pages = st.slider("최대 페이지 수", 10, 100, 50, key="notion_max")

            if st.button("🔍 페이지 가져오기", type="primary", key="notion_fetch"):
                with st.spinner("페이지 가져오는 중..."):
                    results = client.search(
                        query=search_query,
                        filter_type=filter_type,
                        page_size=max_pages
                    )

                    # Filter to only pages (not databases)
                    pages = [r for r in results if r.get("object") == "page"]

                    if pages:
                        st.info(f"📋 {len(pages)}개 페이지 발견")

                        # Store pages for display
                        st.session_state.notion_pages = pages

                        # Extract and index content
                        chunks, metadata = extract_pages_content(
                            client, pages, chunk_korean_text
                        )

                        if chunks:
                            collection_name = generate_collection_name(
                                f"notion-{search_query[:20] if search_query else 'all'}", session_id
                            )
                            collection, chunk_count = index_documents(
                                chunks, metadata, collection_name
                            )

                            st.session_state.collection = collection
                            st.session_state.source_type = "notion"
                            st.success(f"✅ 완료! {len(pages)}개 페이지, {chunk_count}개 청크")
                        else:
                            st.warning("페이지에서 텍스트를 추출할 수 없습니다.")
                    else:
                        if client.last_error:
                            st.error(f"Notion API 오류: {client.last_error}")
                        else:
                            st.warning("페이지를 찾을 수 없습니다.")

            # Display fetched pages
            if st.session_state.notion_pages:
                with st.expander(f"📋 페이지 목록 ({len(st.session_state.notion_pages)}개)", expanded=False):
                    for page in st.session_state.notion_pages:
                        from _party.notion_client import extract_page_title
                        title = extract_page_title(page)
                        url = page.get("url", "")
                        st.markdown(f"**{title}** [🔗]({url})" if url else f"**{title}**")

    st.divider()

    # Status
    if st.session_state.collection:
        st.success("💡 질문할 준비가 되었습니다!")
    else:
        st.info("📤 문서를 업로드하거나 볼트를 선택하세요.")

# Main area
col1, col2 = st.columns([6, 1])
with col1:
    st.header("💬 문서 Q&A")
with col2:
    st.button("초기화", on_click=reset_chat)

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("질문을 입력하세요..."):
    if st.session_state.collection is None:
        st.warning("먼저 문서를 선택해주세요.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Special handling for status-based Jira queries
    status_context = ""
    if st.session_state.source_type == "jira" and st.session_state.jira_issues:
        prompt_lower = prompt.lower()
        # Check for status keywords
        done_keywords = ["done", "완료", "closed", "resolved", "finished", "끝난"]
        progress_keywords = ["progress", "진행", "working", "in progress", "진행중"]
        pending_keywords = ["pending", "대기", "todo", "to do", "할일", "open", "backlog"]

        filtered_issues = []
        filter_type = None

        if any(kw in prompt_lower for kw in done_keywords):
            filter_type = "완료"
            for issue in st.session_state.jira_issues:
                status = issue.get("fields", {}).get("status", {})
                status_name = status.get("name", "").lower()
                status_cat = status.get("statusCategory", {}).get("name", "")
                if status_cat == "Done" or any(kw in status_name for kw in ["done", "완료", "closed", "resolved"]):
                    filtered_issues.append(issue)
        elif any(kw in prompt_lower for kw in progress_keywords):
            filter_type = "진행중"
            for issue in st.session_state.jira_issues:
                status = issue.get("fields", {}).get("status", {})
                status_name = status.get("name", "").lower()
                status_cat = status.get("statusCategory", {}).get("name", "")
                if status_cat == "In Progress" or any(kw in status_name for kw in ["progress", "진행"]):
                    filtered_issues.append(issue)
        elif any(kw in prompt_lower for kw in pending_keywords):
            filter_type = "대기"
            for issue in st.session_state.jira_issues:
                status = issue.get("fields", {}).get("status", {})
                status_name = status.get("name", "").lower()
                status_cat = status.get("statusCategory", {}).get("name", "")
                if status_cat == "To Do" or any(kw in status_name for kw in ["todo", "open", "backlog", "대기"]):
                    filtered_issues.append(issue)

        if filtered_issues:
            status_context = f"\n\n[{filter_type} 상태 이슈 {len(filtered_issues)}개]\n"
            for issue in filtered_issues:
                key = issue.get("key", "")
                fields = issue.get("fields", {})
                summary = fields.get("summary", "")
                status_name = fields.get("status", {}).get("name", "")
                status_context += f"- {key}: {summary} (상태: {status_name})\n"

    # Search - increase results for "all/전체/목록" queries
    all_keywords = ["all", "모든", "전체", "목록", "리스트", "list", "show me", "보여"]
    n_results = TOP_K_RESULTS
    if any(kw in prompt.lower() for kw in all_keywords):
        n_results = min(30, st.session_state.collection.count())  # Get more results

    query_embedding = get_embeddings([prompt])[0]
    results = st.session_state.collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"]
    )

    # Build context with source info
    context_parts = []
    if status_context:
        context_parts.append(status_context)  # Add filtered status info first
    if results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            source = meta.get("source", "unknown") if meta else "unknown"
            context_parts.append(f"[출처: {source}]\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    # Generate response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for chunk in query_llm(prompt, context):
            delta = chunk["message"]["content"]
            full_response += delta
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
