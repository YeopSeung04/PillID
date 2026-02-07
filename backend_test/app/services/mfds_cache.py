# app/services/mfds_cache.py
from __future__ import annotations

import time
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict

from app.services.pill_db import fetch_pill_page

_TOK_RE = re.compile(r"[A-Z0-9\-]{2,}")


def _tokens(s: str) -> list[str]:
    return _TOK_RE.findall((s or "").upper())


def _norm_token(t: str) -> str:
    # 하이픈 제거한 버전도 같이 인덱싱 (D-W == DW)
    return (t or "").upper().replace("-", "").strip()


def _iter_print_tokens(it: Dict[str, Any]) -> List[str]:
    p = f"{it.get('PRINT_FRONT') or ''} {it.get('PRINT_BACK') or ''}".strip()
    if not p:
        return []
    toks = _tokens(p)
    # stable unique
    return list(dict.fromkeys(toks))


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

            # ✅ 인덱스 구축 (원형 + 하이픈제거 버전 둘 다)
            self.print_index.clear()
            for i, it in enumerate(self.items):
                for t in _iter_print_tokens(it):
                    self.print_index[t].append(i)
                    nt = _norm_token(t)
                    if nt and nt != t:
                        self.print_index[nt].append(i)

            self.loaded_at = time.time()
            dt = self.loaded_at - t0
            print(f"[MFDS] cache loaded: {len(self.items)} items in {dt:.1f}s", flush=True)
        finally:
            self.loading = False


mfds_cache = MFDSCache()
