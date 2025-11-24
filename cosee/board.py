from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class View:
    """Location of a note in a multi-page document."""

    page: int
    bbox: Optional[Tuple[int, int, int, int]] = None
    description: Optional[str] = None


@dataclass
class Cell:
    """One note stored on the shared Board."""

    id: int
    view: View
    content: str
    tags: List[str]
    author: str
    step: int


class Board:
    """Shared in-memory scratchpad of notes (cells) over a multi-page document."""

    def __init__(self) -> None:
        self._cells: Dict[int, Cell] = {}
        self._next_id: int = 1

    def add_cell(
        self,
        view: View,
        content: str,
        tags: Optional[List[str]] = None,
        author: str = "unknown",
        step: int = 0,
    ) -> int:
        """Add a new note to the Board and return its cell ID."""
        raw_tags = tags or []
        cell_tags = [t.strip().lower() for t in raw_tags if t is not None and t.strip()]
        cell_id = self._next_id
        self._next_id += 1

        self._cells[cell_id] = Cell(
            id=cell_id,
            view=view,
            content=content,
            tags=list(cell_tags),
            author=author,
            step=step,
        )
        return cell_id

    def get_cell(self, cell_id: int) -> Optional[Cell]:
        """Return the Cell with the given ID, or None if not found."""
        return self._cells.get(cell_id)

    def list_cells(self) -> List[Cell]:
        """Return all cells sorted in deterministic order (step, id)."""
        return sorted(self._cells.values(), key=lambda c: (c.step, c.id))

    def iter_cells(self) -> Iterable[Cell]:
        """
        Iterate over all cells in deterministic order (step, id).
        Convenience wrapper for query methods.
        """
        return iter(self.list_cells())

    def get_cells_by_page(self, page: int) -> List[Cell]:
        """Return cells whose view.page == page, sorted by (step, id)."""
        return [c for c in self.iter_cells() if c.view.page == page]

    def get_cells_by_author(self, author: str) -> List[Cell]:
        """Return cells authored by the given agent name, sorted by (step, id)."""
        return [c for c in self.iter_cells() if c.author == author]

    def get_cells_by_tags(self, tags: List[str]) -> List[Cell]:
        """
        Return cells whose tag list intersects with the given tags.
        Matching is case-insensitive with simple any-overlap semantics.
        """
        tag_set = {t.strip().lower() for t in tags if t is not None and t.strip()}
        return [c for c in self.iter_cells() if tag_set.intersection(c.tags)]

    def to_text(
        self,
        max_cells_per_page: int = 8,
        max_total_chars: int = 2000,
    ) -> str:
        """
        Return a deterministic, human-readable summary grouped by page.
        Enforces per-page and global length limits.
        """
        if not self._cells:
            return ""

        # Group cells by page with deterministic ordering.
        page_to_cells: Dict[int, List[Cell]] = {}
        for cell in self.list_cells():
            page_to_cells.setdefault(cell.view.page, []).append(cell)

        lines: List[str] = []
        for page in sorted(page_to_cells.keys()):
            cells = page_to_cells[page]
            if len(cells) > max_cells_per_page:
                cells = cells[-max_cells_per_page:]

            header = f"Page {page}:"
            lines.append(header)
            for cell in cells:
                tag_str = ",".join(cell.tags) if cell.tags else "-"
                desc = cell.view.description or ""
                bbox = (
                    f" bbox={cell.view.bbox}" if cell.view.bbox is not None else ""
                )
                location = f"{desc}{bbox}".strip()
                location_part = f" [{location}]" if location else ""
                line = (
                    f"  - [step={cell.step}][{cell.author}]"
                    f"[tags={tag_str}]{location_part} {cell.content}"
                )
                lines.append(line)

        # Enforce global character limit without breaking lines.
        output_lines: List[str] = []
        total_chars = 0
        for line in lines:
            added_length = len(line) + (1 if output_lines else 0)  # account for newline
            if output_lines:
                if total_chars + added_length > max_total_chars:
                    break
                total_chars += added_length
                output_lines.append(line)
            else:
                if len(line) > max_total_chars:
                    break
                total_chars += len(line)
                output_lines.append(line)

        return "\n".join(output_lines)

    def __len__(self) -> int:
        """Return the number of cells currently stored on the Board."""
        return len(self._cells)
