"""kv-cache engine client paths: prefix-query chunking, and cache-full PUTs are not retried."""
import json
import sys
sys.argv = sys.argv[:1]
from sglang.private import rustfs_nixl as rn


class C(rn.RustfsNixlClient):
    def __init__(self, stored=()):
        import collections, threading
        self._retry_lock, self._retries = threading.Lock(), collections.Counter()
        self.stored, self.sent, self.answers = set(stored), [], []
    def request(self, method, bucket, key, headers=None, endpoint=None, query="", body=b""):
        self.sent.append((method, bucket, key, query, body))
        if self.answers:
            return self.answers.pop(0)
        keys = json.loads(body)["keys"]
        present = [k in self.stored for k in keys]
        hits = next((i for i, p in enumerate(present) if not p), len(present))
        return 200, {}, json.dumps({"hits": hits, "present": present}).encode()
    def _nixl_headers(self, addr, length):
        return {}


# Chunked by key count, answers concatenated in order.
rn.ENGINE_MAX_KEYS = 3
keys = [f"p/{i}" for i in range(8)]
c = C(stored={"p/0", "p/1", "p/5", "p/7"})
assert c.contains_many("kv", keys) == [True, True, False, False, False, True, False, True]
assert [len(json.loads(s[4])["keys"]) for s in c.sent] == [3, 3, 2]
assert all(s[:4] == ("POST", "kv", "", "x-rustfs-engine-op=prefix-query") for s in c.sent)
assert C().contains_many("kv", []) == []

# Chunked by body size.
rn.ENGINE_MAX_KEYS, rn.ENGINE_MAX_BODY = 4096, 100
c = C()
assert c.contains_many("kv", ["k" * 20] * 3) == [False] * 3 and [len(json.loads(s[4])["keys"]) for s in c.sent] == [2, 1]
assert all(len(s[4]) < rn.ENGINE_MAX_BODY for s in c.sent)

# A short answer is an error, not a silent miss.
c = C()
c.answers = [(200, {}, b'{"hits": 0, "present": [false]}')]
try:
    c.contains_many("kv", ["a", "b"]); raise SystemExit("expected RustfsError")
except rn.RustfsError:
    pass

# Cache full: one attempt, flagged; other 503s are retried.
full = (503, {}, b"<Error><Code>ServiceUnavailable</Code><Message>cache full</Message></Error>")
c = C()
c.answers = [full]
try:
    c.store_from("kv", "k", 0, 10); raise SystemExit("expected RustfsError")
except rn.RustfsError as e:
    assert e.status == 503 and e.cache_full
assert not c.take_retries() and len(c.sent) == 1
c = C()
c.answers = [(503, {}, b"slow down"), (200, {"x-amz-rdma-reply": "200", "x-amz-rdma-bytes-transferred": "10"}, b"")]
c.store_from("kv", "k", 0, 10)
assert c.take_retries()["PUT"] == 1
# Owner routing matches the server's pinned vector (crates/bucket-engine/src/placement.rs,
# owners_match_the_pinned_client_vector); endpoints map by host:port.
view = {"version": 1, "hash": "xxh3-64-rendezvous-v1", "replicas": 1, "nodes": [
    {"node": "10.0.0.1:9000", "drives": 4, "online": True},
    {"node": "10.0.0.2:9000", "drives": 4, "online": True},
    {"node": "10.0.0.3:9000", "drives": 2, "online": True}]}
pl = rn.Placement(view, ["http://10.0.0.1:9000", "http://10.0.0.2:9000/"])
for key, node in [("hicache/a", "10.0.0.1"), ("hicache/b", "10.0.0.2"), ("hicache/c", "10.0.0.1"),
                  ("hicache/e", "10.0.0.2"), ("hicache/f", "10.0.0.1")]:
    assert pl.owner_endpoint(key) == f"http://{node}:9000", key
assert pl.owner_endpoint("hicache/d") is None  # owner 10.0.0.3 is not a client endpoint
try:
    rn.Placement(dict(view, hash="other"), []); raise SystemExit("expected RustfsError")
except rn.RustfsError:
    pass
print("engine query ok")
