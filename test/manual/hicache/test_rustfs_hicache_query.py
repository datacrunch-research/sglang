"""rustfs_hicache existence checks over prefix-query: known/pending keys are not asked,
queries are split by owner, hits are cached, a failed query falls back to HEADs,
cache-full writes are skipped, and the placement view is refreshed."""
import collections, concurrent.futures, threading, time
from sglang.private import rustfs_hicache as rh
from sglang.private.rustfs_nixl import RustfsError


class View:
    def owner_endpoint(self, key):  # two owners by first letter
        return "http://n1" if key < "m" else "http://n2"


class Client:
    def __init__(self, stored, fail=False):
        self.stored, self.fail, self.queries, self.heads, self.views = set(stored), fail, [], [], 0
    def placement(self, bucket):
        self.views += 1
        return View()
    def contains_many(self, bucket, keys, endpoint=None):
        self.queries.append((endpoint, list(keys)))
        if self.fail:
            raise RustfsError("prefix-query 500", 500)
        return [k in self.stored for k in keys]
    def head_size(self, bucket, key):
        self.heads.append(key)
        return 1 if key in self.stored else None
    def take_retries(self):
        return collections.Counter()


def storage(client, use_query=True):
    st = object.__new__(rh.RustfsHiCacheStorage)
    st.bucket, st.prefix, st.client, st.use_query = "kv", "p", client, use_query
    st._known, st._pending, st._pending_lock = rh._Lru(1 << 10), {}, threading.Lock()
    st._io = concurrent.futures.ThreadPoolExecutor(4)
    st._stats_lock, st._stats, st._last_report = threading.Lock(), collections.Counter(), time.time()
    st.placement_refresh, st._placement, st._placement_at, st._placement_lock = 60.0, None, 0.0, threading.Lock()
    return st


c = Client({"a", "c"})
st = storage(c)
st._known.put("k")
st._pending["w"] = threading.Event()
assert st._exists_many(["a", "b", "a", "c", "k", "w", "x", "y"]) == {
    "a": True, "b": False, "c": True, "k": True, "w": True, "x": False, "y": False}
assert sorted(c.queries) == [("http://n1", ["a", "b", "c"]), ("http://n2", ["x", "y"])] and not c.heads
assert c.views == 1  # fetched once, then reused
assert st._exists_many(["a", "c"]) == {"a": True, "c": True} and len(c.queries) == 2  # cached
assert st._stats["queries"] == 1 and st._stats["query_keys"] == 5
st._placement_at -= 61
assert st._owner("z") == "http://n2" and c.views == 2  # refreshed when older than placement_refresh
assert storage(Client(()), use_query=False)._owner("a") is None

c = Client({"a"}, fail=True)
st = storage(c)
assert st._exists_many(["a", "b"]) == {"a": True, "b": False}
assert sorted(c.heads) == ["a", "b"] and st._stats["query_errors"] == 1

c = Client({"a"})
st = storage(c, use_query=False)
assert st._exists_many(["a", "b"]) == {"a": True, "b": False} and not c.queries and sorted(c.heads) == ["a", "b"]

st = storage(Client(()))
st._put_failed("x", RustfsError("full", 503, cache_full=True))
st._put_failed("y", RustfsError("boom", 500))
assert st._stats["put_full"] == 1 and st._stats["put_errors"] == 1

assert storage(Client(()))._detect_query("auto") is True
assert storage(Client((), fail=True))._detect_query("auto") is False
assert storage(Client(()))._detect_query("false") is False
try:
    storage(Client((), fail=True))._detect_query("true"); raise SystemExit("expected ValueError")
except ValueError:
    pass
print("hicache query ok")
