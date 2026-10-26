"""演示页与静态资源测试。"""

from __future__ import annotations


async def test_demo_page_is_served(client) -> None:
    response = await client.get("/demo")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "企业级智能文档检索与问答平台" in response.text
    assert "/static/app.js" in response.text


async def test_static_assets_are_served(client) -> None:
    script = await client.get("/static/app.js")
    styles = await client.get("/static/app.css")

    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert "text/event-stream" not in script.text  # 前端只消费 SSE，不生产
    assert "/api/v1/query" in script.text
    assert styles.status_code == 200
    assert styles.headers["content-type"].startswith("text/css")
