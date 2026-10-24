# sglang-server

`sglang-server` is SGLang's Rust HTTP frontend and request-processing pipeline. It exchanges typed requests and responses with the Python scheduler while keeping latency-sensitive work outside Python.

## Python and extension compatibility

Deploy the Python scheduler code and `_server` extension from the same revision.
Their `RUST_SERVER_WIRE_VERSION` values must agree; version 1 includes the
19-column generation header and version 2 adds exact abort targets. Python checks the loaded extension before launching
the frontend, and extension import checks the Python producer. An unversioned or
mismatched pair fails with a wire protocol error before accepting generation.
Increment both versions when making an incompatible positional-wire change.

Stop the entire engine before replacing either side. Install the matching Python
files and extension, verify import, and then restart. Never overwrite an extension
that is still mapped by a running process.

## Code review principles

1. **Use strongly typed boundaries.** Model every supported protocol shape with structs, enums, and validated newtypes; avoid opaque values such as `serde_json::Value` and `rmpv::Value` in production paths.
2. **Keep one canonical schema.** Rust and Python must derive their wire contracts from one source of truth, aligned with `io_struct.py`, instead of independently duplicating field names, order, defaults, or validation.
3. **Make protocol declarations minimal and declarative.** A reviewer should be able to understand the wire format from its type declarations alone, without tracing fillers, conversion code, macros, or repeated field lists.
4. **Use one representation per semantic stage.** Separate external input, normalized domain data, and wire data, and convert between them once at explicit boundaries; do not keep multiple overlapping representations of the same state.
5. **Design compatibility and safety explicitly.** Version protocols, reject unsupported or malformed inputs clearly, validate lengths and resource bounds before allocation, preserve invariants in types, and test compatibility across the real Rust and Python codecs.

`/abort_request` accepts an exact live public/internal request ID or a prefix
of at least 32 characters. Prefixes resolve to exact internal IDs before
cancellation reaches scheduler queues. `abort_all` is refused unless launched
with `--rust-server-allow-abort-all`. Dropped calls always abort their exact ID.
