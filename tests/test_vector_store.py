"""Milvus 向量库封装测试（注入 fake client，不连真库）。"""

from __future__ import annotations

import pytest

from app.services.vector_store import MilvusVectorStore, VectorRecord, VectorStoreError


class _Schema:
    def __init__(self) -> None:
        self.fields: list[dict] = []

    def add_field(self, name: str, dtype, **kwargs) -> None:
        self.fields.append({"name": name, "dtype": dtype, **kwargs})


class _IndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict] = []

    def add_index(self, **kwargs) -> None:
        self.indexes.append(kwargs)


class FakeMilvusClient:
    def __init__(self, existing_dim: int | None = None) -> None:
        self.existing_dim = existing_dim
        self.created: list[tuple] = []
        self.loads: list[str] = []
        self.upserts: list[tuple] = []
        self.deletes: list[tuple] = []
        self.queries: list[tuple] = []
        self.searches: list[dict] = []
        self.schemas: list[_Schema] = []
        self.index_params: list[_IndexParams] = []
        self.closed = False

    def has_collection(self, name: str) -> bool:
        return self.existing_dim is not None

    def create_schema(self, auto_id: bool, enable_dynamic_field: bool) -> _Schema:
        schema = _Schema()
        self.schemas.append(schema)
        return schema

    def prepare_index_params(self) -> _IndexParams:
        params = _IndexParams()
        self.index_params.append(params)
        return params

    def create_collection(self, name: str, schema: _Schema, index_params: _IndexParams) -> None:
        self.created.append((name, schema, index_params))

    def describe_collection(self, name: str) -> dict:
        return {"fields": [{"name": "embedding", "params": {"dim": self.existing_dim or 0}}]}

    def load_collection(self, name: str) -> None:
        self.loads.append(name)

    def upsert(self, collection_name: str, data: list[dict]) -> None:
        self.upserts.append((collection_name, data))

    def delete(self, collection_name: str, filter: str) -> None:
        self.deletes.append((collection_name, filter))

    def query(
        self,
        collection_name: str,
        filter: str,
        output_fields: list[str],
        consistency_level: str,
    ) -> list[dict]:
        self.queries.append((filter, output_fields, consistency_level))
        return [{"count(*)": 2}]

    def search(
        self,
        collection_name: str,
        data: list[list[float]],
        anns_field: str,
        limit: int,
        output_fields: list[str],
        search_params: dict,
        consistency_level: str = "Bounded",
        filter: str | None = None,
    ) -> list[list[dict]]:
        self.searches.append(
            {
                "collection_name": collection_name,
                "data": data,
                "anns_field": anns_field,
                "limit": limit,
                "output_fields": output_fields,
                "search_params": search_params,
                "consistency_level": consistency_level,
                "filter": filter,
            }
        )
        return [
            [
                {
                    "id": "c1",
                    "distance": 0.87,
                    "entity": {
                        "document_id": "d1",
                        "parent_id": "p1",
                        "text": "内容",
                        "section": "第一章",
                        "page": 2,
                    },
                }
            ]
        ]

    def close(self) -> None:
        self.closed = True


def _store(client: FakeMilvusClient) -> MilvusVectorStore:
    return MilvusVectorStore(uri="http://milvus:19530", collection="chunks", dim=4, client=client)


def _record() -> VectorRecord:
    return VectorRecord(
        id="c1",
        document_id="d1",
        parent_id="p1",
        text="内容",
        section=None,
        page=None,
        embedding=[0.1, 0.2, 0.3, 0.4],
    )


def test_ensure_collection_creates_schema_and_index_once() -> None:
    client = FakeMilvusClient()
    store = _store(client)

    store.ensure_collection()
    store.ensure_collection()

    assert len(client.created) == 1
    name, schema, index_params = client.created[0]
    assert name == "chunks"
    assert [field["name"] for field in schema.fields] == [
        "id",
        "document_id",
        "parent_id",
        "text",
        "section",
        "page",
        "embedding",
    ]
    embedding = next(field for field in schema.fields if field["name"] == "embedding")
    assert embedding["dim"] == 4
    assert index_params.indexes[0]["metric_type"] == "COSINE"
    assert index_params.indexes[0]["index_type"] == "HNSW"
    assert client.loads == ["chunks"]


def test_upsert_maps_payload_sentinels() -> None:
    client = FakeMilvusClient()
    store = _store(client)

    assert store.upsert_chunks([_record()]) == 1

    name, data = client.upserts[0]
    assert name == "chunks"
    assert data[0]["section"] == ""
    assert data[0]["page"] == -1
    assert data[0]["embedding"] == [0.1, 0.2, 0.3, 0.4]
    assert data[0]["document_id"] == "d1"


def test_empty_upsert_does_not_connect() -> None:
    client = FakeMilvusClient()
    store = _store(client)

    assert store.upsert_chunks([]) == 0
    assert client.created == []


def test_delete_and_count_filters() -> None:
    client = FakeMilvusClient()
    store = _store(client)
    store.ensure_collection()

    store.delete_document("d1")
    assert client.deletes[0] == ("chunks", 'document_id == "d1"')

    assert store.count("d1") == 2
    assert client.queries[0] == ('document_id == "d1"', ["count(*)"], "Strong")

    store.count()
    assert client.queries[1][0] == 'document_id != ""'


def test_dimension_mismatch_raises() -> None:
    store = _store(FakeMilvusClient(existing_dim=8))

    with pytest.raises(VectorStoreError, match="维度"):
        store.ensure_collection()


def test_close_releases_client() -> None:
    client = FakeMilvusClient()
    store = _store(client)
    store.ensure_collection()

    store.close()

    assert client.closed is True


def test_search_maps_hits_and_uses_bounded_consistency() -> None:
    client = FakeMilvusClient()
    store = _store(client)
    store.ensure_collection()

    hits = store.search([0.1, 0.2, 0.3, 0.4], 5)

    assert len(hits) == 1
    assert hits[0].id == "c1"
    assert hits[0].document_id == "d1"
    assert hits[0].page == 2
    assert hits[0].section == "第一章"
    assert hits[0].score == pytest.approx(0.87)
    request = client.searches[0]
    assert request["consistency_level"] == "Bounded"
    assert request["search_params"]["metric_type"] == "COSINE"
    assert request["limit"] == 5


def test_search_returns_empty_without_collection() -> None:
    client = FakeMilvusClient()
    store = _store(client)

    assert store.search([0.0, 0.0, 0.0, 0.0], 5) == []
    assert client.searches == []
