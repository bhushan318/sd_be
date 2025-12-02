"""
Tests for caching functionality
"""

from app.utils.cache import hash_model, ModelCache, get_cache
from app.models import Element, Link


def test_hash_model_consistency():
    """Test that hash_model produces consistent hashes for identical models"""
    elements = [
        Element(
            id="stock1",
            type="stock",
            name="Population",
            initial=100.0,
            equation="inflow - outflow",
        ),
        Element(id="flow1", type="flow", name="Inflow", equation="10"),
        Element(id="flow2", type="flow", name="Outflow", equation="5"),
    ]
    links = [
        Link(id="link1", source="flow1", target="stock1"),
        Link(id="link2", source="stock1", target="flow2"),
    ]

    hash1 = hash_model(elements, links)
    hash2 = hash_model(elements, links)

    # Same model should produce same hash
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 produces 64-character hex string


def test_hash_model_different_models():
    """Test that different models produce different hashes"""
    elements1 = [
        Element(
            id="stock1",
            type="stock",
            name="Population",
            initial=100.0,
            equation="inflow - outflow",
        )
    ]
    links1 = []

    elements2 = [
        Element(
            id="stock1",
            type="stock",
            name="Population",
            initial=200.0,
            equation="inflow - outflow",
        )
    ]
    links2 = []

    hash1 = hash_model(elements1, links1)
    hash2 = hash_model(elements2, links2)

    # Different initial values should produce different hashes
    assert hash1 != hash2


def test_hash_model_order_independent():
    """Test that hash is independent of element/link order"""
    elements1 = [
        Element(
            id="stock1",
            type="stock",
            name="Population",
            initial=100.0,
            equation="inflow - outflow",
        ),
        Element(id="flow1", type="flow", name="Inflow", equation="10"),
    ]
    links1 = [Link(id="link1", source="flow1", target="stock1")]

    # Same elements and links in different order
    elements2 = [
        Element(id="flow1", type="flow", name="Inflow", equation="10"),
        Element(
            id="stock1",
            type="stock",
            name="Population",
            initial=100.0,
            equation="inflow - outflow",
        ),
    ]
    links2 = [Link(id="link1", source="flow1", target="stock1")]

    hash1 = hash_model(elements1, links1)
    hash2 = hash_model(elements2, links2)

    # Should produce same hash regardless of order
    assert hash1 == hash2


def test_model_cache_basic():
    """Test basic cache operations"""
    cache = ModelCache(max_size=3)

    # Test put and get
    cache.put("hash1", {"elem1": "ast1"}, {"elem1": ["dep1"]}, ["elem1"])
    result = cache.get("hash1")

    assert result is not None
    assert result["asts"]["elem1"] == "ast1"
    assert result["dependency_graph"]["elem1"] == ["dep1"]
    assert result["evaluation_order"] == ["elem1"]


def test_model_cache_miss():
    """Test cache miss behavior"""
    cache = ModelCache(max_size=3)

    result = cache.get("nonexistent")
    assert result is None


def test_model_cache_lru_eviction():
    """Test LRU eviction when cache is full"""
    cache = ModelCache(max_size=3)

    # Fill cache
    cache.put("hash1", {}, {}, [])
    cache.put("hash2", {}, {}, [])
    cache.put("hash3", {}, {}, [])

    # All should be in cache
    assert cache.get("hash1") is not None
    assert cache.get("hash2") is not None
    assert cache.get("hash3") is not None

    # Access hash1 to make it most recently used
    cache.get("hash1")

    # Add new entry - should evict least recently used (hash2)
    cache.put("hash4", {}, {}, [])

    # hash2 should be evicted
    assert cache.get("hash2") is None
    # Others should still be present
    assert cache.get("hash1") is not None
    assert cache.get("hash3") is not None
    assert cache.get("hash4") is not None


def test_model_cache_stats():
    """Test cache statistics"""
    cache = ModelCache(max_size=3)

    # Initial stats
    stats = cache.get_stats()
    assert stats["size"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["evictions"] == 0

    # Add entry
    cache.put("hash1", {}, {}, [])

    # Miss
    cache.get("hash2")
    assert cache.get_stats()["misses"] == 1

    # Hit
    cache.get("hash1")
    assert cache.get_stats()["hits"] == 1

    # Fill cache and trigger eviction
    cache.put("hash2", {}, {}, [])
    cache.put("hash3", {}, {}, [])
    cache.put("hash4", {}, {}, [])  # Should evict hash1

    stats = cache.get_stats()
    assert stats["evictions"] >= 1


def test_model_cache_clear():
    """Test cache clearing"""
    cache = ModelCache(max_size=3)

    cache.put("hash1", {}, {}, [])
    cache.put("hash2", {}, {}, [])

    assert cache.get_stats()["size"] == 2

    cache.clear()

    assert cache.get_stats()["size"] == 0
    assert cache.get("hash1") is None
    assert cache.get("hash2") is None


def test_get_cache_singleton():
    """Test that get_cache returns the same instance"""
    cache1 = get_cache()
    cache2 = get_cache()

    assert cache1 is cache2


def test_cache_integration_with_evaluator():
    """Test cache integration with evaluator functions"""
    from app.evaluator import build_dependency_graph

    elements = [
        Element(id="var1", type="variable", name="Var1", equation="10"),
        Element(id="var2", type="variable", name="Var2", equation="var1 * 2"),
    ]
    links = []

    # First call - should miss cache
    deps1 = build_dependency_graph(elements, links, use_cache=True)

    # Second call - should hit cache
    deps2 = build_dependency_graph(elements, links, use_cache=True)

    # Results should be identical
    assert deps1 == deps2

    # Check cache stats show a hit
    cache = get_cache()
    stats = cache.get_stats()
    assert stats["hits"] >= 1
