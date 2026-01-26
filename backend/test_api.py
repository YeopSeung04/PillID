import asyncio
from app.services.pill_db import fetch_pill_page

async def main():
    items = await fetch_pill_page(page_no=1, num_rows=5)
    print("count:", len(items))
    if items:
        print("keys:", list(items[0].keys()))
        print("sample:", items[0])

asyncio.run(main())
