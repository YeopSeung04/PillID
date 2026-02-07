# app/services/pill_db.py

import os
import json
import httpx
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

SERVICE_KEY = os.getenv("MFDS_SERVICE_KEY")
BASE_URL = "https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03"
ENDPOINT = "/getMdcinGrnIdntfcInfoList03"


class MfdsApiError(RuntimeError):
    pass


def _safe_head(text: str, n: int = 600) -> str:
    text = text or ""
    return text[:n].replace("\r", "\\r").replace("\n", "\\n")


def _extract_items_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    공공데이터 응답 구조가 서비스마다 달라서 가능한 패턴을 모두 커버.
    최종적으로 items(list[dict]) 반환.
    """
    # 패턴 1) 최상위 header/body
    if "body" in payload and isinstance(payload.get("body"), dict):
        body = payload.get("body") or {}
    # 패턴 2) response.body
    elif "response" in payload and isinstance(payload.get("response"), dict):
        body = (payload.get("response") or {}).get("body") or {}
        if not isinstance(body, dict):
            body = {}
    else:
        body = {}

    # items 추출 (서비스별 키 차이 방어)
    items = body.get("items")
    if items is None:
        items = body.get("item")

    # 흔한 패턴: items = {"item": [...]}
    if isinstance(items, dict) and "item" in items:
        items = items["item"]

    # None/단일 dict 처리
    if items is None:
        return []
    if isinstance(items, dict):
        return [items]
    if isinstance(items, list):
        # list 내부가 dict가 아닐 수도 있어 방어
        return [x for x in items if isinstance(x, dict)]

    return []


def _extract_header(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    resultCode/resultMsg를 얻기 위한 header 추출.
    """
    if "header" in payload and isinstance(payload.get("header"), dict):
        return payload["header"]
    if "response" in payload and isinstance(payload.get("response"), dict):
        header = (payload["response"] or {}).get("header")
        if isinstance(header, dict):
            return header
    return {}


async def fetch_pill_page(
    page_no: int = 1,
    num_rows: int = 100,
    item_name: Optional[str] = None,
    entp_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    MFDS '의약품 낱알식별 정보' 목록을 페이지 단위로 가져온다.
    - serviceKey는 URL에 직접 붙여 이중 인코딩 방지 (이미 %3D 포함 가능)
    - _type/json vs type/json 자동 대응
    - 응답 구조(header/body vs response/body) 자동 파싱

    반환: items (list[dict])
    """
    if not SERVICE_KEY:
        raise MfdsApiError("MFDS_SERVICE_KEY is not set in .env")

    # serviceKey는 URL에 직접 붙여서 'encoded=true' 효과를 만든다.
    # (이미 인코딩된 키를 params로 넣으면 %가 %25로 이중 인코딩 될 수 있음)
    url = f"{BASE_URL}{ENDPOINT}?serviceKey={SERVICE_KEY}"

    base_params: Dict[str, Any] = {
        "pageNo": page_no,
        "numOfRows": num_rows,
    }
    if item_name:
        base_params["item_name"] = item_name
    if entp_name:
        base_params["entp_name"] = entp_name

    # 서비스마다 _type 또는 type을 씀. 둘 다 시도.
    candidates: List[Tuple[str, Dict[str, Any]]] = [
        ("_type", {**base_params, "_type": "json"}),
        ("type", {**base_params, "type": "json"}),
    ]

    last_err: Optional[Exception] = None

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        for mode, params in candidates:
            try:
                r = await client.get(url, params=params)

                # 200이 아닌 경우에도 body가 원인 메시지(XML/HTML)로 오는 케이스가 있어
                # raise_for_status 전에 내용을 확인할 수 있게 처리
                content_type = (r.headers.get("content-type") or "").lower()
                text_head = _safe_head(r.text)

                if r.status_code == 401:
                    raise MfdsApiError(
                        f"401 Unauthorized (mode={mode}). "
                        f"serviceKey 문제(미등록/오류/이중 인코딩) 가능성이 큼. "
                        f"content-type={content_type}, body_head={text_head}"
                    )

                if r.status_code >= 400:
                    raise MfdsApiError(
                        f"HTTP {r.status_code} (mode={mode}). "
                        f"content-type={content_type}, body_head={text_head}"
                    )

                # JSON 파싱 (content-type이 json이 아니어도 json으로 오는 경우가 있어 직접 시도)
                try:
                    payload = r.json()
                except json.JSONDecodeError:
                    raise MfdsApiError(
                        f"Non-JSON response (mode={mode}). "
                        f"content-type={content_type}, body_head={text_head}"
                    )

                header = _extract_header(payload)
                # resultCode/resultMsg가 있으면 힌트 제공
                result_code = header.get("resultCode") or header.get("result_code")
                result_msg = header.get("resultMsg") or header.get("result_msg")

                items = _extract_items_from_payload(payload)

                # 어떤 API는 items가 없고 header만 주기도 함 → 이 경우도 에러로 올려 원인 확인
                if not items and (result_code or result_msg):
                    # 정상 코드인데 0건일 수도 있으니 메시지는 포함하되 빈 리스트 반환
                    # 단, 명백한 에러 코드면 예외
                    if str(result_code).strip() not in ("", "00", "0", "SUCCESS"):
                        raise MfdsApiError(
                            f"MFDS error (mode={mode}): resultCode={result_code}, resultMsg={result_msg}"
                        )

                return items

            except Exception as e:
                last_err = e
                continue

    # 두 모드 모두 실패
    raise MfdsApiError(f"MFDS request failed for both _type/type. Last error: {last_err}")


