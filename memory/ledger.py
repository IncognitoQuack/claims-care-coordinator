"""
Medical Necessity Ledger — Shared Memory System

This is the core innovation: a structured shared memory that both the Clinical Agent
and Policy Agent read from and write to. When one agent discovers something, the other
agent's behavior changes in real-time.

Architecture:
- Thread-safe (asyncio locks)
- Event-driven (subscribers get notified on writes)
- Structured entries with typed fields
- Query interface for agents to search the ledger
"""

import asyncio
import uuid
from datetime import datetime
from typing import AsyncGenerator, Callable, Optional

from models.schemas import AgentSource, LedgerEntry, Severity


class MedicalNecessityLedger:
    def __init__(self):
        self._entries: list[LedgerEntry] = []
        self._lock = asyncio.Lock()
        self._subscribers: list[asyncio.Queue] = []
        self._context_cache: dict = {}  # agents can store derived context here

    async def write(
        self,
        source: AgentSource,
        event_type: str,
        message: str,
        data: dict = None,
        tags: list[str] = None,
        severity: Severity = Severity.NORMAL,
    ) -> LedgerEntry:
        """Write a new entry to the ledger. All subscribers are notified."""
        entry = LedgerEntry(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow(),
            source=source,
            event_type=event_type,
            message=message,
            data=data or {},
            tags=tags or [],
            severity=severity,
        )
        async with self._lock:
            self._entries.append(entry)
            # Notify all subscribers
            for queue in self._subscribers:
                await queue.put(entry)
        return entry

    async def read_all(self) -> list[LedgerEntry]:
        """Read all entries in the ledger."""
        async with self._lock:
            return list(self._entries)

    async def read_by_source(self, source: AgentSource) -> list[LedgerEntry]:
        """Read entries written by a specific agent."""
        async with self._lock:
            return [e for e in self._entries if e.source == source]

    async def read_by_tag(self, tag: str) -> list[LedgerEntry]:
        """Search entries by tag."""
        async with self._lock:
            return [e for e in self._entries if tag in e.tags]

    async def read_by_event_type(self, event_type: str) -> list[LedgerEntry]:
        """Read entries of a specific event type."""
        async with self._lock:
            return [e for e in self._entries if e.event_type == event_type]

    async def get_clinical_context(self) -> str:
        """
        Build a clinical context summary from all clinical entries.
        This is what the Policy Agent reads to adjust its search.
        """
        clinical_entries = await self.read_by_source(AgentSource.CLINICAL)
        if not clinical_entries:
            return "No clinical data available yet."

        parts = []
        for entry in clinical_entries:
            parts.append(f"[{entry.event_type}] {entry.message}")
            if entry.data:
                for k, v in entry.data.items():
                    if isinstance(v, list):
                        parts.append(f"  {k}: {', '.join(str(x) for x in v)}")
                    else:
                        parts.append(f"  {k}: {v}")
        return "\n".join(parts)

    async def get_policy_context(self) -> str:
        """Build a policy context summary from all policy entries."""
        policy_entries = await self.read_by_source(AgentSource.POLICY)
        if not policy_entries:
            return "No policy findings yet."

        parts = []
        for entry in policy_entries:
            parts.append(f"[{entry.event_type}] {entry.message}")
            if entry.data:
                for k, v in entry.data.items():
                    parts.append(f"  {k}: {v}")
        return "\n".join(parts)

    async def get_full_context(self) -> str:
        """Get the complete ledger as a formatted context string."""
        entries = await self.read_all()
        if not entries:
            return "Ledger is empty."

        parts = ["=== MEDICAL NECESSITY LEDGER ===\n"]
        for entry in entries:
            source_label = entry.source.value.upper()
            parts.append(
                f"[{entry.timestamp.strftime('%H:%M:%S.%f')[:-3]}] "
                f"({source_label}) [{entry.event_type}] {entry.message}"
            )
            if entry.tags:
                parts.append(f"  Tags: {', '.join(entry.tags)}")
            if entry.data:
                for k, v in entry.data.items():
                    parts.append(f"  {k}: {v}")
            parts.append("")
        return "\n".join(parts)

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to ledger updates. Returns a queue that receives new entries."""
        queue = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from ledger updates."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    async def set_context(self, key: str, value):
        """Store derived context that agents can share."""
        async with self._lock:
            self._context_cache[key] = value

    async def get_context(self, key: str, default=None):
        """Retrieve shared context."""
        async with self._lock:
            return self._context_cache.get(key, default)

    async def clear(self):
        """Clear the ledger for a new claim."""
        async with self._lock:
            self._entries.clear()
            self._context_cache.clear()
