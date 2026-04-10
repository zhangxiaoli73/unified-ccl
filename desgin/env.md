# UCCL Environment Variables

| Variable | Values | Default | Description |
|---|---|---|---|
| `UCCL_DEBUG` | `NONE`, `ERROR`, `WARN`, `INFO`, `TRACE` | `ERROR` | Controls log verbosity level. |
| `UCCL_ALGO` | `ring`, `one_shot`, `symmetric`, `copy_engine` / `ce` | Auto (topology-based) | Override algorithm selection for collectives. |
| `UCCL_EXEC_MODE` | `copy_engine` / `ce`, `hybrid` | `eu_only` | Override execution mode: EU-only, copy engine, or hybrid. |
| `UCCL_EU_COUNT` | Positive integer (≤ max available) | Hardware max | Override the number of EUs used for compute kernels. |
| `UCCL_NET_PLUGIN` | Path to `.so` file | Built-in UCX plugin | Load an external network plugin shared library. |
| `UCCL_FORCE_NET` | `1` | Unset (auto) | Force Net (NIC) transport even for intra-node peers. Useful for testing network path on a single machine. |