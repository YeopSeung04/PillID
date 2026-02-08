# app/services/mfds_cache.py
from __future__ import annotations

import os
import time
import re
import pickle
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
    return list(dict.fromkeys(toks))  # stable unique


class MFDSCache:
    def __init__(
        self,
        cache_dir: str = "cache",
        cache_file: str = "mfds_items.pkl",
    ):
        self.items: List[Dict[str, Any]] = []
        self.loaded_at: Optional[float] = None
        self.loading: bool = False
        self.progress: dict = {"page": 0, "pages": 0, "rows": 0, "items": 0}
        self.print_index = defaultdict(list)  # token -> [item_idx]

        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, cache_file)

    # -------------------------
    # status helpers
    # -------------------------
    def is_ready(self) -> bool:
        return bool(self.items)

    def is_stale(self, ttl_seconds: int) -> bool:
        if self.loaded_at is None:
            return True
        return (time.time() - self.loaded_at) > ttl_seconds

    # -------------------------
    # cache I/O
    # -------------------------
    def _ensure_cache_dir(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)

    def _build_index(self) -> None:
        self.print_index.clear()
        for i, it in enumerate(self.items):
            for t in _iter_print_tokens(it):
                self.print_index[t].append(i)
                nt = _norm_token(t)
                if nt and nt != t:
                    self.print_index[nt].append(i)

    def load_from_disk(self) -> bool:
        """로컬 pickle 캐시가 있으면 로드. 성공하면 True."""
        if not os.path.exists(self.cache_path):
            return False

        t0 = time.time()
        try:
            with open(self.cache_path, "rb") as f:
                payload = pickle.load(f)

            items = payload.get("items")
            loaded_at = payload.get("loaded_at")

            if not isinstance(items, list) or not items:
                return False

            self.items = items
            self.loaded_at = float(loaded_at) if loaded_at else time.time()
            self._build_index()

            dt = time.time() - t0
            print(f"[MFDS] cache file loaded: {len(self.items)} items in {dt:.2f}s ({self.cache_path})", flush=True)
            return True
        except Exception as e:
            print(f"[MFDS] cache file load failed: {e}", flush=True)
            return False

    def save_to_disk(self) -> None:
        """현재 items를 pickle로 저장."""
        self._ensure_cache_dir()
        payload = {
            "items": self.items,
            "loaded_at": self.loaded_at or time.time(),
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[MFDS] cache file saved: {len(self.items)} items -> {self.cache_path}", flush=True)

    # -------------------------
    # main refresh
    # -------------------------
    async def refresh(self, pages: int = 50, rows: int = 100, save_cache: bool = True):
        """MFDS에서 다시 긁어와서 items 업데이트 + 인덱스 재구축 + (옵션) 파일 저장"""
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
            self._build_index()

            self.loaded_at = time.time()
            dt = self.loaded_at - t0
            print(f"[MFDS] cache loaded: {len(self.items)} items in {dt:.1f}s", flush=True)

            if save_cache:
                self.save_to_disk()
        finally:
            self.loading = False

    # -------------------------
    # public entrypoint
    # -------------------------
    async def ensure_loaded(
        self,
        pages: int = 50,
        rows: int = 100,
        ttl_seconds: int = 7 * 24 * 3600,  # 기본 7일
    ):
        """
        서버 시작/요청 시 호출용:
        1) 메모리에 있으면 OK
        2) 없으면 디스크 캐시 먼저 로드 시도
        3) 디스크도 없거나 stale이면 MFDS에서 refresh 후 저장
        """
        # 1) 메모리에 이미 있으면
        if self.is_ready() and not self.is_stale(ttl_seconds):
            return

        # 2) 디스크 캐시 로드 시도
        if self.load_from_disk():
            # 디스크 로드 성공이면 TTL 검사
            if not self.is_stale(ttl_seconds):
                return

        # 3) stale/없음이면 새로 긁기
        await self.refresh(pages=pages, rows=rows, save_cache=True)


mfds_cache = MFDSCache()
