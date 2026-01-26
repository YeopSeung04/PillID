# app/services/mfds_cache.py
from __future__ import annotations

import time
import asyncio
from typing import List, Dict, Any, Optional

from app.services.pill_db import fetch_pill_page  # 네가 이미 성공시킨 fetch 코드 사용


class MFDSCache:
    def __init__(self):
        self.items: List[Dict[str, Any]] = []
        self.loaded_at: Optional[float] = None
        self.loading: bool = False

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
        try:
            all_items: List[Dict[str, Any]] = []
            for p in range(1, pages + 1):
                data = await fetch_pill_page(page_no=p, num_rows=rows)
                all_items.extend(data)

            self.items = all_items
            self.loaded_at = time.time()
        finally:
            self.loading = False


mfds_cache = MFDSCache()
