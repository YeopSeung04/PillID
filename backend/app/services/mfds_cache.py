# app/services/mfds_cache.py
from __future__ import annotations

import time, re
from typing import List, Dict, Any, Optional

from app.services.pill_db import fetch_pill_page
from collections import defaultdict

def _tok(s: str) -> list[str]:
    return re.findall(r"[A-Z0-9\-]{2,}", (s or "").upper())

class MFDSCache:
    def __init__(self):
        self.items: List[Dict[str, Any]] = []
        self.loaded_at: Optional[float] = None
        self.loading: bool = False
        self.progress: dict = {"page": 0, "pages": 0, "rows": 0, "items": 0}
        self.print_index = defaultdict(list)  # token -> [item_idx]

    def is_ready(self) -> bool:
        return bool(self.items)

    def is_stale(self, ttl_seconds: int) -> bool:
        if self.loaded_at is None:
            return True
        return (time.time() - self.loaded_at) > ttl_seconds

    async def refresh(self, pages: int = 50, rows: int = 100):
        if self.loading:
            return
        self.loading = True
        self.progress = {"page": 0, "pages": pages, "rows": rows, "items": 0}

        t0 = time.time()
        try:
            all_items: List[Dict[str, Any]] = []
            for p in range(1, pages + 1):
                data = await fetch_pill_page(page_no=p, num_rows=rows)
                all_items.extend(data)

                self.progress["page"] = p
                self.progress["items"] = len(all_items)
                print(f"[MFDS] loading... {p}/{pages} pages | items={len(all_items)}", flush=True)

            self.items = all_items
            # 인덱스 구축
            self.print_index.clear()
            for i, it in enumerate(self.items):
                p = (it.get("PRINT_FRONT") or "") + " " + (it.get("PRINT_BACK") or "")
                for t in set(_tok(p)):
                    self.print_index[t].append(i)

            self.loaded_at = time.time()
            dt = self.loaded_at - t0
            print(f"[MFDS] cache loaded: {len(self.items)} items in {dt:.1f}s", flush=True)
        finally:
            self.loading = False


mfds_cache = MFDSCache()
