"""
Notion Integration
- Connect to Notion API
- Fetch pages and databases
- Extract text content for RAG
"""
import re
from dataclasses import dataclass
from typing import Optional
import requests


@dataclass
class NotionConfig:
    """Notion connection configuration."""
    api_token: str  # Integration token from https://www.notion.so/my-integrations

    def __post_init__(self):
        self.api_token = self.api_token.strip()

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }


class NotionClient:
    """Notion API client for fetching pages and databases."""

    BASE_URL = "https://api.notion.com/v1"

    def __init__(self, config: NotionConfig):
        self.config = config
        self.last_error = None

    def test_connection(self) -> tuple[bool, str]:
        """Test the Notion connection."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/users/me",
                headers=self.config.headers,
                timeout=10
            )
            if response.status_code == 200:
                user = response.json()
                name = user.get("name", user.get("bot", {}).get("owner", {}).get("user", {}).get("name", "Unknown"))
                return True, f"Connected as: {name}"
            else:
                return False, f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def search(
        self,
        query: str = "",
        filter_type: str = None,  # "page" or "database"
        page_size: int = 100
    ) -> list[dict]:
        """
        Search for pages and databases.

        Args:
            query: Search query string
            filter_type: Filter by "page" or "database"
            page_size: Number of results to return

        Returns:
            List of page/database objects
        """
        try:
            payload = {"page_size": page_size}
            if query:
                payload["query"] = query
            if filter_type:
                payload["filter"] = {"property": "object", "value": filter_type}

            response = requests.post(
                f"{self.BASE_URL}/search",
                headers=self.config.headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                self.last_error = None
                return response.json().get("results", [])
            else:
                self.last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                return []
        except Exception as e:
            self.last_error = f"Exception: {str(e)}"
            return []

    def get_page(self, page_id: str) -> Optional[dict]:
        """Get a page by ID."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/pages/{page_id}",
                headers=self.config.headers,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    def get_page_blocks(self, page_id: str) -> list[dict]:
        """Get all blocks (content) from a page."""
        try:
            blocks = []
            cursor = None

            while True:
                params = {"page_size": 100}
                if cursor:
                    params["start_cursor"] = cursor

                response = requests.get(
                    f"{self.BASE_URL}/blocks/{page_id}/children",
                    headers=self.config.headers,
                    params=params,
                    timeout=30
                )

                if response.status_code != 200:
                    self.last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                    break

                data = response.json()
                blocks.extend(data.get("results", []))

                if not data.get("has_more"):
                    break
                cursor = data.get("next_cursor")

            self.last_error = None
            return blocks
        except Exception as e:
            self.last_error = f"Exception: {str(e)}"
            return []

    def query_database(
        self,
        database_id: str,
        page_size: int = 100
    ) -> list[dict]:
        """Query all pages in a database."""
        try:
            pages = []
            cursor = None

            while True:
                payload = {"page_size": page_size}
                if cursor:
                    payload["start_cursor"] = cursor

                response = requests.post(
                    f"{self.BASE_URL}/databases/{database_id}/query",
                    headers=self.config.headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code != 200:
                    self.last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                    break

                data = response.json()
                pages.extend(data.get("results", []))

                if not data.get("has_more"):
                    break
                cursor = data.get("next_cursor")

            self.last_error = None
            return pages
        except Exception as e:
            self.last_error = f"Exception: {str(e)}"
            return []


def extract_rich_text(rich_text_array: list) -> str:
    """Extract plain text from Notion rich text array."""
    if not rich_text_array:
        return ""
    return "".join(item.get("plain_text", "") for item in rich_text_array)


def extract_block_text(block: dict) -> str:
    """Extract text from a single Notion block."""
    block_type = block.get("type", "")
    block_data = block.get(block_type, {})

    # Text blocks
    if block_type in ["paragraph", "heading_1", "heading_2", "heading_3",
                      "bulleted_list_item", "numbered_list_item", "quote",
                      "callout", "toggle"]:
        text = extract_rich_text(block_data.get("rich_text", []))
        if block_type.startswith("heading"):
            return f"\n{text}\n"
        elif block_type in ["bulleted_list_item", "numbered_list_item"]:
            return f"- {text}"
        return text

    # Code block
    elif block_type == "code":
        code = extract_rich_text(block_data.get("rich_text", []))
        language = block_data.get("language", "")
        return f"[Code ({language})]: {code}"

    # To-do
    elif block_type == "to_do":
        text = extract_rich_text(block_data.get("rich_text", []))
        checked = block_data.get("checked", False)
        status = "[x]" if checked else "[ ]"
        return f"{status} {text}"

    # Table of contents, divider, etc.
    elif block_type in ["table_of_contents", "divider", "breadcrumb"]:
        return ""

    # Child page/database
    elif block_type == "child_page":
        return f"[Page: {block_data.get('title', 'Untitled')}]"
    elif block_type == "child_database":
        return f"[Database: {block_data.get('title', 'Untitled')}]"

    # Bookmark, embed, etc.
    elif block_type in ["bookmark", "embed", "link_preview"]:
        url = block_data.get("url", "")
        return f"[Link: {url}]" if url else ""

    # Image, file, etc.
    elif block_type in ["image", "video", "file", "pdf"]:
        caption = extract_rich_text(block_data.get("caption", []))
        return f"[{block_type.title()}: {caption}]" if caption else f"[{block_type.title()}]"

    return ""


def extract_page_title(page: dict) -> str:
    """Extract title from a Notion page."""
    properties = page.get("properties", {})

    # Try common title property names
    for prop_name in ["title", "Title", "Name", "name", "이름", "제목"]:
        if prop_name in properties:
            prop = properties[prop_name]
            if prop.get("type") == "title":
                return extract_rich_text(prop.get("title", []))

    # Fallback: find any title property
    for prop in properties.values():
        if prop.get("type") == "title":
            return extract_rich_text(prop.get("title", []))

    return "Untitled"


def extract_page_content(client: NotionClient, page: dict) -> str:
    """
    Extract all text content from a Notion page.

    Returns combined text from title and all blocks.
    """
    parts = []
    page_id = page.get("id", "")

    # Title
    title = extract_page_title(page)
    if title:
        parts.append(f"# {title}")

    # Get page blocks
    blocks = client.get_page_blocks(page_id)

    for block in blocks:
        text = extract_block_text(block)
        if text:
            parts.append(text)

        # Handle nested blocks (children)
        if block.get("has_children"):
            child_blocks = client.get_page_blocks(block.get("id", ""))
            for child in child_blocks:
                child_text = extract_block_text(child)
                if child_text:
                    parts.append(f"  {child_text}")

    return "\n".join(parts)


def get_page_metadata(page: dict) -> dict:
    """Get metadata for a Notion page."""
    return {
        "source": extract_page_title(page),
        "type": "notion_page",
        "page_id": page.get("id", "Unknown"),
        "url": page.get("url", ""),
        "last_edited": page.get("last_edited_time", "")
    }


def extract_pages_content(
    client: NotionClient,
    pages: list[dict],
    chunk_fn: callable
) -> tuple[list[str], list[dict]]:
    """
    Extract and chunk content from multiple Notion pages.

    Args:
        client: NotionClient instance
        pages: List of Notion page objects
        chunk_fn: Function to chunk text into pieces

    Returns:
        Tuple of (chunks, metadata_list)
    """
    all_chunks = []
    all_metadata = []

    for page in pages:
        text = extract_page_content(client, page)
        if not text or len(text) < 50:
            continue

        chunks = chunk_fn(text)
        metadata = get_page_metadata(page)

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append(metadata.copy())

    return all_chunks, all_metadata
