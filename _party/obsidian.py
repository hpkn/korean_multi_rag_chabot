"""
Obsidian Vault Integration
- Browse folders and files
- Extract text from markdown files
- Index content for RAG
"""
import re
from pathlib import Path
from typing import Optional


def extract_text_from_markdown(file_path: Path) -> str:
    """
    Extract clean text from Markdown file.
    Removes YAML frontmatter, links, code blocks, etc.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove YAML frontmatter
        content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
        # Remove markdown links but keep text
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        # Remove wiki-style links [[link]] -> link
        content = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', content)
        # Remove image links
        content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', content)
        # Remove code blocks
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        # Remove inline code
        content = re.sub(r'`[^`]+`', '', content)
        # Remove headers markdown but keep text
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        # Remove bold/italic markers
        content = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', content)
        content = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', content)
        # Remove blockquotes marker
        content = re.sub(r'^>\s*', '', content, flags=re.MULTILINE)
        # Remove horizontal rules
        content = re.sub(r'^[-*_]{3,}\s*$', '', content, flags=re.MULTILINE)
        # Remove list markers
        content = re.sub(r'^[\s]*[-*+]\s+', '', content, flags=re.MULTILINE)
        content = re.sub(r'^[\s]*\d+\.\s+', '', content, flags=re.MULTILINE)
        # Remove tags (#tag)
        content = re.sub(r'#[a-zA-Z가-힣][a-zA-Z0-9가-힣_/-]*', '', content)

        return content.strip()
    except Exception:
        return ""


def get_vault_folders(vault_path: Path) -> list[Path]:
    """
    Get all folders in Obsidian vault.
    Excludes hidden folders (starting with .)
    """
    if not vault_path.exists():
        return []

    folders = [vault_path]
    for item in vault_path.rglob('*'):
        if item.is_dir() and not any(part.startswith('.') for part in item.parts):
            folders.append(item)

    return sorted(folders)


def get_markdown_files(folder_path: Path, recursive: bool = False) -> list[Path]:
    """
    Get markdown files in a folder.

    Args:
        folder_path: Path to folder
        recursive: If True, include files from subfolders

    Returns:
        List of markdown file paths
    """
    if not folder_path.exists():
        return []

    if recursive:
        files = list(folder_path.rglob('*.md'))
    else:
        files = list(folder_path.glob('*.md'))

    # Exclude files in hidden folders
    files = [f for f in files if not any(part.startswith('.') for part in f.parts)]

    return sorted(files)


def get_file_metadata(file_path: Path, vault_root: Path) -> dict:
    """
    Get metadata for a file.

    Returns:
        Dict with source, folder, relative_path
    """
    try:
        rel_path = file_path.relative_to(vault_root)
        return {
            "source": file_path.name,
            "folder": file_path.parent.name,
            "path": str(rel_path)
        }
    except ValueError:
        return {
            "source": file_path.name,
            "folder": file_path.parent.name,
            "path": str(file_path)
        }


def extract_vault_content(
    files: list[Path],
    vault_root: Path,
    chunk_fn: callable
) -> tuple[list[str], list[dict]]:
    """
    Extract and chunk content from multiple files.

    Args:
        files: List of file paths
        vault_root: Root path of vault (for relative paths)
        chunk_fn: Function to chunk text into pieces

    Returns:
        Tuple of (chunks, metadata_list)
    """
    all_chunks = []
    all_metadata = []

    for file_path in files:
        text = extract_text_from_markdown(file_path)
        if not text:
            continue

        chunks = chunk_fn(text)
        metadata = get_file_metadata(file_path, vault_root)

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append(metadata.copy())

    return all_chunks, all_metadata


def get_folder_stats(folder_path: Path, recursive: bool = True) -> dict:
    """
    Get statistics about a folder.

    Returns:
        Dict with file_count, total_size, subfolders
    """
    files = get_markdown_files(folder_path, recursive)

    total_size = sum(f.stat().st_size for f in files if f.exists())
    subfolders = len([f for f in folder_path.iterdir() if f.is_dir() and not f.name.startswith('.')])

    return {
        "file_count": len(files),
        "total_size_kb": total_size // 1024,
        "subfolders": subfolders
    }
