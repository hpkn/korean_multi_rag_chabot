"""
Jira Integration
- Connect to Jira Cloud or Server
- Fetch issues, comments, descriptions
- Extract text content for RAG
"""
import re
from dataclasses import dataclass
from typing import Optional
import requests
from requests.auth import HTTPBasicAuth


@dataclass
class JiraConfig:
    """Jira connection configuration."""
    url: str  # e.g., "https://your-domain.atlassian.net" or "https://jira.company.com"
    email: str  # For Cloud: email, For Server: username
    api_token: str  # For Cloud: API token, For Server: password

    def __post_init__(self):
        # Clean up whitespace that may come from .env files
        self.url = self.url.strip().rstrip('/')
        self.email = self.email.strip()
        self.api_token = self.api_token.strip()

    @property
    def auth(self) -> HTTPBasicAuth:
        return HTTPBasicAuth(self.email, self.api_token)

    @property
    def headers(self) -> dict:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }


class JiraClient:
    """Jira API client for fetching issues."""

    def __init__(self, config: JiraConfig):
        self.config = config
        self.base_url = config.url  # Already cleaned in JiraConfig.__post_init__
        self.last_error = None

    def test_connection(self) -> tuple[bool, str]:
        """Test the Jira connection."""
        try:
            response = requests.get(
                f"{self.base_url}/rest/api/3/myself",
                headers=self.config.headers,
                auth=self.config.auth,
                timeout=10
            )
            if response.status_code == 200:
                user = response.json()
                return True, f"Connected as: {user.get('displayName', user.get('name', 'Unknown'))}"
            else:
                return False, f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def get_projects(self) -> list[dict]:
        """Get all accessible projects."""
        try:
            response = requests.get(
                f"{self.base_url}/rest/api/3/project",
                headers=self.config.headers,
                auth=self.config.auth,
                timeout=10
            )
            if response.status_code == 200:
                projects = response.json()
                self.last_error = None
                return [{"key": p["key"], "name": p["name"]} for p in projects]
            else:
                self.last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                return []
        except Exception as e:
            self.last_error = f"Exception: {str(e)}"
            return []

    def search_issues(
        self,
        jql: str,
        max_results: int = 50,
        fields: list[str] = None
    ) -> list[dict]:
        """
        Search issues using JQL (API v3).

        Args:
            jql: Jira Query Language string (e.g., "project = MYPROJ")
            max_results: Maximum number of issues to return
            fields: List of fields to include

        Returns:
            List of issue dictionaries
        """
        if fields is None:
            fields = ["summary", "description", "comment", "status", "assignee", "reporter", "created", "updated"]

        try:
            # Use the new v3 search/jql endpoint
            response = requests.get(
                f"{self.base_url}/rest/api/3/search/jql",
                headers=self.config.headers,
                auth=self.config.auth,
                params={
                    "jql": jql,
                    "maxResults": max_results,
                    "fields": ",".join(fields)
                },
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                self.last_error = None
                return data.get("issues", [])
            else:
                self.last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                return []
        except Exception as e:
            self.last_error = f"Exception: {str(e)}"
            return []

    def get_issue(self, issue_key: str) -> Optional[dict]:
        """Get a single issue by key."""
        try:
            response = requests.get(
                f"{self.base_url}/rest/api/3/issue/{issue_key}",
                headers=self.config.headers,
                auth=self.config.auth,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None


def extract_adf_text(adf_content: dict) -> str:
    """
    Extract plain text from Atlassian Document Format (ADF).
    Used in Jira API v3 for description and comments.
    """
    if not adf_content or not isinstance(adf_content, dict):
        return ""

    def extract_from_node(node):
        if isinstance(node, str):
            return node

        if not isinstance(node, dict):
            return ""

        node_type = node.get("type", "")
        text_parts = []

        # Handle text nodes
        if node_type == "text":
            return node.get("text", "")

        # Handle mentions
        if node_type == "mention":
            return node.get("attrs", {}).get("text", "")

        # Handle emoji
        if node_type == "emoji":
            return node.get("attrs", {}).get("shortName", "")

        # Recursively process content
        content = node.get("content", [])
        if isinstance(content, list):
            for child in content:
                child_text = extract_from_node(child)
                if child_text:
                    text_parts.append(child_text)

        result = " ".join(text_parts)

        # Add newlines for block elements
        if node_type in ("paragraph", "heading", "listItem", "tableCell"):
            result += "\n"
        if node_type in ("bulletList", "orderedList", "table"):
            result += "\n"

        return result

    return extract_from_node(adf_content).strip()


def clean_jira_markup(text: str) -> str:
    """
    Clean Jira wiki markup and convert to plain text.
    """
    if not text:
        return ""

    # If it's not a string (e.g., ADF dict), try to extract text
    if not isinstance(text, str):
        return extract_adf_text(text)

    # Remove code blocks
    text = re.sub(r'\{code[^}]*\}.*?\{code\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\{noformat\}.*?\{noformat\}', '', text, flags=re.DOTALL)

    # Remove panels
    text = re.sub(r'\{panel[^}]*\}', '', text)
    text = re.sub(r'\{panel\}', '', text)

    # Remove color/formatting
    text = re.sub(r'\{color[^}]*\}', '', text)
    text = re.sub(r'\{color\}', '', text)

    # Convert headers to plain text
    text = re.sub(r'^h[1-6]\.\s*', '', text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # Remove links but keep text
    text = re.sub(r'\[([^|]+)\|[^\]]+\]', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]', r'\1', text)

    # Remove images
    text = re.sub(r'![\w\-\.]+!', '', text)

    # Remove mentions
    text = re.sub(r'\[~[^\]]+\]', '', text)

    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


def extract_issue_text(issue: dict) -> str:
    """
    Extract all text content from a Jira issue.

    Returns combined text from summary, description, status, and comments.
    """
    parts = []
    fields = issue.get("fields", {})
    key = issue.get("key", "Unknown")

    # Status - include prominently for searchability
    status = fields.get("status", {}).get("name", "")
    status_category = fields.get("status", {}).get("statusCategory", {}).get("name", "")

    # Summary with status
    summary = fields.get("summary", "")
    if summary:
        status_text = f" (상태: {status})" if status else ""
        parts.append(f"[{key}]{status_text} {summary}")

    # Status info as searchable text
    if status:
        # Map common status names to Korean for better searchability
        status_info = f"이슈 상태: {status}"
        if status_category:
            status_info += f" ({status_category})"
        # Add keywords for different statuses
        if status_category == "Done" or status.lower() in ["done", "완료", "closed", "resolved"]:
            status_info += " - 완료된 작업, done task, completed"
        elif status_category == "In Progress" or status.lower() in ["in progress", "진행중", "진행 중"]:
            status_info += " - 진행중인 작업, in progress task, working"
        elif status_category == "To Do" or status.lower() in ["to do", "open", "대기", "할일"]:
            status_info += " - 대기중인 작업, pending task, todo"
        parts.append(status_info)

    # Assignee
    assignee = fields.get("assignee")
    if assignee:
        assignee_name = assignee.get("displayName", assignee.get("name", ""))
        if assignee_name:
            parts.append(f"담당자: {assignee_name}")

    # Description
    description = fields.get("description", "")
    if description:
        clean_desc = clean_jira_markup(description)
        if clean_desc:
            parts.append(clean_desc)

    # Comments
    comments = fields.get("comment", {}).get("comments", [])
    for comment in comments:
        body = comment.get("body", "")
        if body:
            clean_body = clean_jira_markup(body)
            if clean_body:
                parts.append(clean_body)

    return "\n\n".join(parts)


def get_issue_metadata(issue: dict) -> dict:
    """Get metadata for an issue."""
    fields = issue.get("fields", {})

    return {
        "source": issue.get("key", "Unknown"),
        "type": "jira_issue",
        "summary": fields.get("summary", "")[:100],
        "status": fields.get("status", {}).get("name", "Unknown"),
        "project": fields.get("project", {}).get("key", "Unknown") if fields.get("project") else "Unknown"
    }


def extract_issues_content(
    issues: list[dict],
    chunk_fn: callable
) -> tuple[list[str], list[dict]]:
    """
    Extract and chunk content from multiple issues.

    Args:
        issues: List of Jira issue dictionaries
        chunk_fn: Function to chunk text into pieces

    Returns:
        Tuple of (chunks, metadata_list)
    """
    all_chunks = []
    all_metadata = []

    for issue in issues:
        text = extract_issue_text(issue)
        if not text:
            continue

        chunks = chunk_fn(text)
        metadata = get_issue_metadata(issue)

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append(metadata.copy())

    return all_chunks, all_metadata


# Common JQL templates
JQL_TEMPLATES = {
    "project": "project = {project_key} ORDER BY updated DESC",
    "project_recent": "project = {project_key} AND updated >= -{days}d ORDER BY updated DESC",
    "my_issues": "assignee = currentUser() ORDER BY updated DESC",
    "mentioned": "text ~ {search_term} ORDER BY updated DESC",
}
