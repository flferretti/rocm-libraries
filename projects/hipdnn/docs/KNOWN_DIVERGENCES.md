# Known divergences: cuDNN frontend compatibility shim

This document describes intentional behavioral differences between NVIDIA cuDNN
frontend and hipDNN's cuDNN-shaped compatibility shim. The shim is source
compatibility, not ABI compatibility and not a full cuDNN backend implementation.

NVIDIA, CUDA, and cuDNN are trademarks and/or registered trademarks of NVIDIA
Corporation. hipDNN is not affiliated with or endorsed by NVIDIA.

## Scope

The shim covers two cuDNN frontend surfaces, both reached through
`<hipdnn_compatibility/cudnn/cudnn_frontend.h>`:

- the v9 graph API ([RFC 0012](rfcs/0012_CuDNN_Shim.md));
- the v0.x builders (the cuDNN v8 API) and the cuDNN backend descriptor C API beneath
  them, which `<hipdnn_compatibility/cudnn/cudnn.h>` declares
  ([RFC 0021](rfcs/0021_CuDNN_V8_Shim.md)).

The sections from [hipDNN extensions](#hipdnn-extensions) to [Logging](#logging) describe
the v9 graph API; [Heuristic modes](#heuristic-modes) and [Logging](#logging) also apply to
the v8 API. [cuDNN v8 API](#cudnn-v8-api) lists the v8 differences.

The legacy v7-style C API (`cudnnConvolutionForward`, `cudnnSetTensorNdDescriptor`, …) is
not provided.

## hipDNN extensions

The shim exposes one method with no cuDNN frontend equivalent:

`Graph::is_supported_ext(cudnnHandle_t, std::vector<HeurMode_t> = {FALLBACK})` is a
lightweight applicability probe forwarding to the native hipDNN method of the same
name. It returns `OK` when at least one engine can run the graph and
`GRAPH_NOT_SUPPORTED` when none can. Like the native method it auto-builds the
operation graph when needed; it neither creates nor invalidates execution plans, so
it may be called before or after `create_execution_plans`. A node-less graph has
nothing to probe and is reported supported.

Because it has no cuDNN counterpart, source that must also compile against real
cuDNN cannot use it.

## Heuristic modes

The shim accepts every cuDNN frontend `HeurMode_t` value (`A`, `B`, `FALLBACK`,
`OPENSOURCE`) and, on the v8 backend C API, every `CUDNN_HEUR_MODE_*` value (`INSTANT`,
`A`, `B`, `FALLBACK`). hipDNN does not expose matching cuDNN heuristic modes, so every mode
runs hipDNN's fallback/default engine selection. The first use of each non-`FALLBACK` mode
in a process logs one warning; later uses of the same mode, through either API, do not.

Impact: plan choice and performance may differ from cuDNN for the same requested
heuristic mode.

## Note filters

hipDNN exposes no per-plan numerical-note metadata and reports only its own
behavior notes. The shim therefore triages each note by what ignoring the
request could do. The tables below are the canonical triage; the shim's
`note_triage.h` switches and their table test follow them.

| Action | Effect |
|---|---|
| `no-op` | accepted silently |
| `warn` | logged as a warning on every call, otherwise ignored |
| `map` | honored by filtering applicable engines; logged at info |
| `error` | records `GRAPH_NOT_SUPPORTED`, returned by the next `validate()`, plan creation, support check, or plan build |

### Numerical notes

| Note | `select_numeric_notes` | `deselect_numeric_notes` | Why |
|---|---|---|---|
| `NOT_SET` | no-op | no-op | placeholder |
| `TENSOR_CORE` | warn | warn | performance only |
| `DOWN_CONVERT_INPUTS` | warn | error | excluding it is a precision guarantee the shim cannot check |
| `REDUCED_PRECISION_REDUCTION` | warn | error | excluding it is a precision guarantee the shim cannot check |
| `FFT` | warn | warn | algorithm class |
| `NONDETERMINISTIC` | warn | map | deselect keeps only engines that claim determinism (see [Determinism](#determinism)) |
| `WINOGRAD`, `WINOGRAD_TILE_4x4`, `WINOGRAD_TILE_6x6`, `WINOGRAD_TILE_13x13` | warn | warn | algorithm class |
| `STRICT_NAN_PROP` | error | warn | selecting it is a NaN-propagation guarantee the shim cannot check; excluding it only relaxes |
| out-of-range value | error | error | unclassifiable |

Deselecting an out-of-range value is an error here but a no-op upstream:
without numerical metadata the shim cannot show the exclusion is vacuous.

### Behavior notes

Notes hipDNN engines report (`isKnownBehaviorNote`) map for both select and
deselect: the shim filters applicable engines by their reported behavior notes.
Any other value, including cuDNN's CUDA-specific notes, errors on select (no
hipDNN engine can satisfy it) and warns on deselect (it excludes nothing).

| Note | `select_behavior_notes` | `deselect_behavior_notes` | Why |
|---|---|---|---|
| `NOT_SET` | no-op | no-op | placeholder |
| `RUNTIME_COMPILATION` | map | map | hipDNN note |
| `REQUIRES_FILTER_INT8x32_REORDER` | error | warn | cuDNN only |
| `REQUIRES_BIAS_INT8x32_REORDER` | error | warn | cuDNN only |
| `SUPPORTS_CUDA_GRAPH_NATIVE_API` | error | warn | cuDNN only; `populate_cuda_graph` is unsupported, and this is not stream capture (`SUPPORTS_GRAPH_CAPTURE`) |
| `CUBLASLT_DEPENDENCY` | error | warn | cuDNN only; not equivalent to `EXTERNAL_LIBRARY_DEPENDENCY` |
| `REQUIRES_LAYOUT_TRANSFORM` | map | map | hipDNN note |
| `SUPPORTS_GRAPH_CAPTURE` | map | map | hipDNN note |
| `EXTERNAL_LIBRARY_DEPENDENCY` | map | map | hipDNN note |
| `SUPPORTS_EXECUTION_PLAN_SERIALIZATION` | map | map | hipDNN note |
| out-of-range value | error | warn | reported by no hipDNN engine |

### Determinism

`deselect_numeric_notes({NONDETERMINISTIC})` keeps only engines that positively
claim deterministic results. hipDNN engines declare no numerical notes, so a
missing claim proves nothing. Until engine metadata carries the claim, the shim
keeps its own allowlist, today `MIOPEN_ENGINE_DETERMINISTIC`.

## Engine filtering

`deselect_engines`, the behavior-note filters, and the determinism filter are
applied to the applicable engines after native plan creation. A filter set after
plans were created takes effect at the next support check or plan build,
including for plans already created.

When the filters bar only some engines, the shim narrows the candidate set the
way cuDNN does: `build_plans(HEURISTICS_CHOICE)` retargets onto the top-ranked
surviving plan rather than failing because the top-ranked one was barred.

If no created plan survives the filters, the next plan creation
(`create_execution_plans`, `create_execution_plan`), support check
(`check_support()`), or plan build (`build_plans()`, `build_plan_at_index()`)
returns `GRAPH_NOT_SUPPORTED`; upstream returns
`GRAPH_EXECUTION_PLAN_CREATION_FAILED` from `check_support()`. This includes
`deselect_engines` alone barring every applicable engine.

## Engine IDs

cuDNN frontend presents integer engine IDs as dense graph-local indices bounded
by `get_engine_count()`.

hipDNN native engine IDs are stable hashes of engine names. The shim therefore
maintains a graph-local mapping:

```text
cuDNN-shaped engine index -> native hipDNN engine ID
```

`get_engine_count()` refreshes this map from hipDNN's ranked applicable engine
list. `get_knobs_for_engine`, `create_execution_plan`, and
`deselect_engines(vector<int64_t>)` translate cuDNN-shaped dense indices through
that map before calling native hipDNN APIs.

Impact: ordering follows hipDNN's ranked applicable engine list, not cuDNN's
backend ordering.

## Knobs

cuDNN frontend exposes knobs as a fixed `KnobType_t` enum plus integer
`minValue`, `maxValue`, and `stride` metadata. hipDNN native knobs use provider-
defined string IDs, variant-valued settings, descriptions, and richer constraints.
The shim uses explicit conversion rather than aliasing these incompatible models.

hipDNN knob IDs are namespaced (for example `global.workspace_size_limit`) while
cuDNN's `KnobType_t` is flat. The shim matches on the final dot-separated segment,
so both `tile_size` and `provider.tile_size` project onto the same cuDNN knob.
Matching is case-insensitive, and `workspace_size_limit` is the one bare name that
differs from its cuDNN counterpart (`workspace`).

When returning knobs from `get_knobs_for_engine`, native hipDNN knobs are projected
to cuDNN-shaped knobs only when all of the following are true:

- the native string knob ID maps to a known cuDNN `KnobType_t`
- the native knob value type is `int64`
- the native constraint can be represented as `minValue` / `maxValue` / `stride`

Native knobs that cannot be represented this way are omitted with a warning.

When creating a plan, each cuDNN `KnobType_t` key is resolved against the knobs the
target engine actually exposes, so a choice lands on the provider's own knob ID (for
example `miopen.tile_size`) rather than on a guessed bare name. A knob choice is
never silently dropped: `INVALID_VALUE` is returned for a key the engine does not
expose, for a key that more than one of the engine's knobs maps onto, and for a value
the native knob cannot hold (a string-valued knob, or a magnitude past 2^53 for a
float-valued knob). Integer choices are widened to `double` for float-valued knobs.

## Workspace and shared-memory caps

`deselect_workspace_greater_than` is forwarded to hipDNN and reapplied after plan
creation because native plan creation resets native filters.

`deselect_shared_mem_greater_than(0)` is a no-op. Any non-zero value records
`GRAPH_NOT_SUPPORTED`; hipDNN does not currently expose per-plan shared-memory
usage metadata for the shim to filter on.

## Plan and engine introspection

The shim forwards plan-name, plan-workspace, per-plan execution, knob query,
autotune, and current-plan behavior-note APIs where native hipDNN exposes an
equivalent.

`get_behavior_notes_for_plan_at_index` is best-effort. Native hipDNN exposes plan
name by index but not engine ID by index, so the shim resolves the plan name back
to an ID: built-in engine names by hash, and every other engine (including plugin
engines) through the `0x`-prefixed hexadecimal ID native reports for it.

`warmup` is implemented as `execute_plan_at_index(..., 0)`.

## CUDA graph capture

`populate_cuda_graph` and `update_cuda_graph` are compile-time-present runtime
error stubs. They return `GRAPH_NOT_SUPPORTED` because there is no in-scope
HIP-graph capture analogue in the shim.

## Execute overloads

The UID-map and tensor-attribute-map execute overloads forward to native hipDNN.
The flat pointer-array `execute(cudnnHandle_t, void**, int, void*)` overload is
present for source compatibility but returns `INVALID_VALUE`; the shim cannot
safely reconstruct the required tensor UID mapping from a flat pointer array.

## Logging

`cudnnCreate()` performs a best-effort bridge from cuDNN frontend logging
environment variables to hipDNN logging configuration before creating the hipDNN
handle:

- enabled cuDNN logging maps to `HIPDNN_LOG_LEVEL=info`
- file-path `CUDNN_FRONTEND_LOG_FILE` maps to `HIPDNN_LOG_FILE`
- `stdout` / `stderr` targets enable hipDNN logging but do not set `HIPDNN_LOG_FILE`
- disabled or no-target cases map to `HIPDNN_LOG_LEVEL=off`

The bridge is effective only if it runs before hipDNN backend logging initializes.
The `CUDNN_FE_LOG*` macros forward to hipDNN frontend logging macros; they do not
own a separate cuDNN frontend stream logger.

## cuDNN v8 API

The v0.x builders run on the emulated backend descriptor C API, so the C API subsections below
apply to builder code as well; [Builders](#builders) lists what differs only at the builder level.
Only the C identifiers that the v0.x headers, NVIDIA's legacy samples and PyTorch's v0.x consumers
name are declared; code that names any other cuDNN C identifier does not compile.

### Enum values

The C enums (`cudnnBackendAttributeName_t`, `cudnnDataType_t`,
`cudnnPointwiseMode_t`, …) have the cuDNN names but the shim's own values: numbered from
0 in alphabetical order of the enumerator name, with each `*_COUNT` one past the last
value. Values are append-only within a `hipdnn_frontend` major version.
`cudnnBackendTensorReordering_t` follows the order of the shim's `TensorReordering_t`.

Impact: source compatibility only. Code that stores raw cuDNN enum values produced by
real cuDNN, or compares against hard-coded numbers, sees different values.

### Last-error string

`cudnnGetLastErrorString` returns the most recent shim error on the calling thread, falling back to
hipDNN's last error when the shim has none. The buffer lives in the shim's header-only code, so each
DLL (Windows) or shared object (Linux) that includes the shim has its own copy.

Impact: an error raised by shim code in one module is not visible to `cudnnGetLastErrorString`
called from another module.

### Parameter descriptors

Parameter descriptors (`CONVOLUTION`, `POINTWISE`, `MATMUL`, `REDUCTION`, `RESAMPLE`, `RNG`) are
validated at `cudnnBackendFinalize`; their values reach hipDNN when the operation graph is lowered.
These differ from cuDNN:

| Case | Shim behavior |
|---|---|
| NaN propagation on `POINTWISE` and `RESAMPLE` | stored and ignored; hipDNN has no equivalent setting |
| `POINTWISE` `RELU_*_CLIP*`, `ELU_ALPHA`, `SOFTPLUS_BETA`, `SWISH_BETA` | narrowed to `float` when the graph is lowered; values beyond the float range saturate to ±`FLT_MAX`. Parameters left at their defaults are not sent, except the ReLU lower clip, which is always sent unless a slope is set (hipBLASLt's ReLU epilogue requires it) |
| `MATMUL` `PADDING_VALUE` | `NOT_SUPPORTED` at the matmul descriptor's finalize |
| `RESAMPLE` without `PADDING_MODE` | `BAD_PARAM` at the resample descriptor's finalize (upstream builders always send it) |
| `RNG` | finalizes and keeps its attributes for GET, but an `RNG` operation is `NOT_SUPPORTED` (see [Operations](#operations)) |

### Tensors

Tensor descriptors are validated at `cudnnBackendFinalize`. These differ from cuDNN:

| Case | Shim behavior |
|---|---|
| `BYTE_ALIGNMENT` below 16 | forwarded to hipDNN. A graph with such a tensor needs engine-plugin API 1.3.0 or later, which no current provider reports, so the graph has no engine: `ENGINE_GLOBAL_COUNT` is 0 and heuristics return no results, both with `SUCCESS` |
| `VECTOR_COUNT` other than 1; the vectorized data types `INT8x4`, `INT8x32` and `UINT8x4`; `REORDERING_MODE` other than `NONE` | `NOT_SUPPORTED` at tensor finalize |

### Operations

Operation descriptors are validated at `cudnnBackendFinalize`. These differ from cuDNN:

| Case | Shim behavior |
|---|---|
| `GEN_STATS`, `BN_FINALIZE_STATISTICS`, `BN_BWD_WEIGHTS`, `RNG`, `RESHAPE`, `PAGED_CACHE_LOAD` and other operation types the shim does not lower (including `BLOCK_SCALE_*` and `MOE_GROUPED_MATMUL*`, which hipDNN supports natively) | `NOT_SUPPORTED` at finalize, before any attribute check |
| Alpha ≠ 1 or beta ≠ 0 on convolution, pointwise (`ALPHA1`, `ALPHA2`) or resample | `NOT_SUPPORTED`; hipDNN operations have no scaling factors |
| Batch-norm inference and backward `EPSILON_DESC` | accepted but not used; hipDNN's inference takes inverse variance and its backward takes saved statistics, so neither needs epsilon |
| Batch-norm backward without `MEAN_DESC` / `INV_VARIANCE_DESC` | `BAD_PARAM`; the saved statistics are required |
| Running-statistic descriptors outside batch-norm training; `MEAN_DESC` on RMS norm | `NOT_SUPPORTED` |
| Layer-norm forward without `BIAS_DESC` | `BAD_PARAM`; hipDNN's layer norm requires a bias |
| Norm `INSTANCE`, `GROUP`, `ADA_LAYER` modes; peer statistics | `NOT_SUPPORTED` |
| Max-pool backward resample without `IDXDESC` | `BAD_PARAM`; hipDNN needs the index tensor |
| Backward resample `XDESC` / `YDESC` | `NOT_SUPPORTED`; hipDNN's backward resample has no such inputs |
| Matmul M/N/K override descriptors | `NOT_SUPPORTED` |
| A pointwise tensor slot the mode does not use (for example `TDESC` on a non-select mode, or `YDESC` on a backward activation) | `BAD_PARAM`; upstream builders never set these |

### Operation graphs

`OPERATIONGRAPH` finalize lowers the graph to hipDNN; it needs no device and runs no heuristics.
These differ from cuDNN:

| Case | Shim behavior |
|---|---|
| No `HANDLE` | `BAD_PARAM`; hipDNN needs a handle to build the graph (cuDNN 9.8 and later make it optional) |
| `IS_DYNAMIC_SHAPE_ENABLED` or `IS_OVERRIDE_SHAPE_ENABLED` set to true | `NOT_SUPPORTED` at finalize; the v8 surface has no variant-pack shape overrides |
| Norm forward / backward compute type | cuDNN norm operations carry none; the shim sends `FLOAT` to hipDNN |
| Convolution with `HALF` or `BFLOAT16` tensors and `COMP_TYPE` `FLOAT` | forwarded as `FLOAT`, but the engine chooses the accumulation precision: the MIOpen provider may run a reduced-precision kernel (a 3x3 stride-1 `HALF` forward on gfx1151 runs the Winograd solver `ConvWinoFuryRxS<2-3>` with fp16 accumulation), so results can differ from cuDNN by more than fp32 accumulation would. They match PyTorch's own MIOpen convolution |
| Several `TENSOR` descriptors with one UID | accepted when they agree; `BAD_PARAM` naming the UID when data type, dimensions, strides, virtualness, by-value or constant value, or ragged offset differ. Differing byte alignments use the smallest |
| One UID written by two operations; a cycle | `BAD_PARAM` |
| `CONSTANT_VALUE` on a tensor whose type is not `FLOAT`, `DOUBLE`, `HALF`, `BFLOAT16`, `UINT8`, `INT32`, `INT64` or `BOOLEAN` | `NOT_SUPPORTED` at graph finalize |
| hipDNN operation validation | Graph finalize runs hipDNN's own checks. Beyond cuDNN's: epsilon and the exponential-average factor must be single-element `IS_BY_VALUE` tensors; batch-norm scale, bias and statistics must be per-channel `[1,C,1,…]` (no per-activation mode) with N·spatial > 1 when training; a reduction must reduce at least one dimension; a max-pool index must be `INT8` or `INT32`; RMS-norm scale must match the trailing dimensions of X. A rejection is `BAD_PARAM` naming the operation type, except "no engine supports this graph", which is `NOT_SUPPORTED`. Backend failures during lowering, allocation included, also report `BAD_PARAM` with hipDNN's message |
| `ENGINE_GLOBAL_COUNT` | computed on the first value GET by running hipDNN heuristics, which needs a HIP device; a failure returns its status and the next GET retries. It counts only engines hipDNN reports as applicable whose default configuration finalizes; an engine that fails is logged and left out, so every counted index builds. The result is a per-graph snapshot: later changes to the heuristic configuration or autotune cache do not affect an existing graph |
| Engine indices | `0 … ENGINE_GLOBAL_COUNT-1` enumerate the applicable hipDNN engines in ascending (signed) hipDNN engine-ID order, not cuDNN's global engine numbering. They change if the set of applicable engines changes (provider added, removed or upgraded, different GPU) |

### Engines, plans and execution

These differ from cuDNN:

| Case | Shim behavior |
|---|---|
| `ENGINEHEUR_MODE` | every mode returns the same results: the graph's engines in hipDNN's fallback rank. See [Heuristic modes](#heuristic-modes) |
| `ENGINEHEUR_SM_COUNT_TARGET`, `ENGINEHEUR_DEVICEPROP` | `NOT_SUPPORTED` when set |
| `ENGINEHEUR_RESULTS` | configurations are finalized when returned and are shared: two fills of the same result return the same configuration |
| Matmul on gfx115x under Windows | hipBLASLt declines every matmul there (a hipBLASLt GEMM crash workaround), and no other provider runs matmul, so a v8 matmul graph has no engine |
| Engine indices in errata and in consumer code | errata rules and hard-coded engine numbers (for example PyTorch's cuDNN engines 5, 23, 58 and 63) name NVIDIA engines. Under the shim the same number selects an unrelated hipDNN engine, or none. At `CUDNN_VERSION` 92200 only the 3-D `ConvBwdData` engine-23 rule is active, and it applies only to a graph with more than 23 applicable engines |
| `ENGINE_NUMERICAL_NOTE` | `NONDETERMINISTIC` for every engine except those known to be deterministic (today `MIOPEN_ENGINE_DETERMINISTIC`); no other numerical note is reported, so filters on notes such as `DOWN_CONVERT_INPUTS` or `TENSOR_CORE` never match. Deterministic-only selection keeps only the known-deterministic engines |
| `ENGINE_BEHAVIOR_NOTE` | hipDNN behavior notes with a cuDNN counterpart, matched by name; others (for example `SUPPORTS_EXECUTION_PLAN_SERIALIZATION`) are not reported |
| `ENGINE_KNOB_INFO` | lists only hipDNN knobs that project onto a cuDNN knob type (see [Knobs](#knobs): int64 value, plain minimum/maximum/stride, no list of valid values). A cuDNN type that two of an engine's knobs map onto is not listed (the v9 `get_knobs_for_engine` lists both) |
| When `ENGINE_BEHAVIOR_NOTE` and `ENGINE_KNOB_INFO` are read | from hipDNN on the first GET, not at `ENGINE` finalize; a failure is returned by that GET and a later GET retries |
| `KNOB_INFO` finalized by the caller | `BAD_PARAM`; knob-information descriptors are filled by `ENGINE_KNOB_INFO` only |
| `ENGINECFG_KNOB_CHOICES` | checked at `ENGINECFG` finalize: a type not listed in the engine's `KNOB_INFO` is `NOT_SUPPORTED`; a type chosen twice, or a value outside the knob's range or off its stride, is `BAD_PARAM`. A failure building the hipDNN configuration is reported at `EXECUTION_PLAN` finalize. A choice of `-1` means "unset" (upstream's `EngineConfigBuilder` sends `-1` for every knob it was not asked to set once one is set): it is ignored (not checked, not applied, not counted as a repeat, not in the plan JSON `"knobs"`) unless the knob is listed and its range accepts `-1`. The `KNOB_CHOICES` GET still returns every choice as set. A configuration whose choices are all ignored uses the engine's default hipDNN configuration |
| `EXECUTION_PLAN_HANDLE` | optional for a plan built from `ENGINE_CONFIG` (upstream's `ExecutionPlanBuilder` never sets it): unset or null, the plan runs on the operation graph's `HANDLE`; when set, `BAD_PARAM` at finalize unless it is that handle (hipDNN builds the plan on the graph's handle). Required for a plan loaded from `JSON_REPRESENTATION` (`BAD_PARAM` otherwise), as upstream's `loadFromJson` sets it |
| `EXECUTION_PLAN_KERNEL_CACHE` | `NOT_SUPPORTED` when set |
| `EXECUTION_PLAN_JSON_REPRESENTATION` format | the shim's own versioned JSON, not cuDNN's: engine index, engine ID (a decimal string), engine name, behavior notes, knob choices and, when hipDNN can serialize the plan, a base64 payload. Notes and knob types are the shim's enum values. JSON written by cuDNN or by another tool is `BAD_PARAM` at finalize |
| Loading a plan without `"payload"` | `NOT_SUPPORTED` at finalize; engines whose plans hipDNN cannot serialize cannot be reloaded |
| Loaded plans | the engine ID must match the one inside the payload (`BAD_PARAM` otherwise). Engine index, name, behavior notes and knob choices are reported from the JSON as saved; the index refers to the engine set of the process that saved it. The loaded plan's engine has no `OPERATION_GRAPH` or `KNOB_INFO` (`NOT_SUPPORTED`), and its configuration cannot build a new plan (`NOT_SUPPORTED`) |
| `cudnnBackendExecute` handle | `BAD_PARAM` unless it is the handle the plan was built on (the graph's handle) or, for a loaded plan, the `HANDLE` it was loaded with |
| `VARIANT_PACK` | the hipDNN variant pack is built at finalize; a missing or empty `DATA_POINTERS` or `UNIQUE_IDS`, or the two with different element counts, is `BAD_PARAM` at finalize |
| Per-call overhead | `VariantPackBuilder::build()` plus `cudnnBackendExecute` costs about 1–3 µs more than building the hipDNN variant pack and calling `hipdnnBackendExecute` on the same plan; `cudnnBackendExecute` alone adds well under 1 µs |
| `ENGINE_LAYOUT_INFO`, `ENGINECFG_WORKSPACE_SIZE`, `VARIANT_PACK_INTERMEDIATES` | not declared (no supported consumer uses them); code that names them does not compile |

### Builders

The v0.x builders (`TensorBuilder`, `ConvDescBuilder`, `PointWiseDescBuilder`, `MatMulDescBuilder`,
`ReductionDescBuilder`, `ResampleDescBuilder`, `RngDescBuilder`, `OperationBuilder`,
`OperationGraphBuilder`, `VariantPackBuilder`, `EngineHeuristicsBuilder`, `EngineBuilder`,
`EngineConfigBuilder`, `EngineFallbackListBuilder`, `ExecutionPlanBuilder`) and the plan helpers
(`get_heuristics_list`, `cudnnGetPlan`, `time_sorted_plan`, `cudnnFindPlan`,
`cudnnFindPlanAndCache`) are upstream's, running on the C API above, so every C API row applies to
their `build()`. These differ as well:

| Case | Shim behavior |
|---|---|
| Builder error text | `cudnnException::what()` ends in `cudnn_status: HIPDNN_STATUS_…`: `to_string(cudnnStatus_t)` returns `cudnnGetErrorString`, which reports hipDNN's name for the translated status. `getCudnnStatus()` and `get_status()` still carry the cuDNN status |
| nlohmann serialization of frontend enums | `DataType_t`, `PointwiseMode_t` and the other frontend enums are hipDNN's enums, and the shim defines no nlohmann serializer for them, so `json j = DataType_t::FLOAT;` stores an integer, not upstream's `"FLOAT"`. `describe()` is unaffected: it renders enums through the shim's own name table, byte for byte as upstream |
| Descriptor log lines | the body of a builder's log line (`describe()`) is printed only when `CUDNN_FRONTEND_LOG_INFO` is above 0, as upstream, and the line itself goes through hipDNN INFO logging. With hipDNN INFO logging on but `CUDNN_FRONTEND_LOG_INFO` unset, the line is `[cudnn_frontend] ` with nothing after it |
| `getStream()` / `getLogger()` | no shim code calls them. A consumer call opens `CUDNN_FRONTEND_LOG_FILE` for writing, truncating it as upstream does; the `cudnnCreate()` logging bridge may also route hipDNN's log to that file, so the two writers can overwrite each other |
| `NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING` | ignored: the shim never loads cuDNN at run time, and `NV_CUDNN_FE_DYNAMIC_CHECK_*` expand to nothing, so the run-time cuDNN version checks they guard are skipped |
| `get_environment()`, `getLogLevel()` | `get_environment` returns a pointer into a per-thread buffer that the next call on the same thread overwrites (upstream returns `getenv`'s pointer), and returns null for a variable that is set but empty. `getLogLevel` returns 0 when the `NV_CUDNN_FRONTEND_DISABLE_LOGGING` environment variable is set to anything but `0`; upstream honors only the macro |
| Operation and operation-graph `build()` errors | an operation the C API rejects at finalize fails `OperationBuilder::build()` with that row's status, and a graph rejection, such as a graph built without `setHandle`, fails `OperationGraphBuilder::build()` the same way: see [Operations](#operations) and [Operation graphs](#operation-graphs). `GenStats`, `BNFinalize`, `Dgrad_Drelu_BN_Bwd`, `Rng`, paged-cache-load and `Reshape` operations always fail with `NOT_SUPPORTED` |
| PyTorch fused convolutions | `cudnn_convolution_relu` passes `alpha2 = 0` to the pointwise add, so it always gets `NOT_SUPPORTED`; `cudnn_convolution_add_relu` does whenever its alpha is not 1 (see the scaling row under [Operations](#operations)) |
| Heuristic mode strings | every mode returns the same list (see [Heuristic modes](#heuristic-modes)), so a mode list such as `{"heuristics_instant", "heuristics_fallback"}` given to `get_heuristics_list` or `EngineConfigGenerator` yields every engine configuration once per mode. `cudnnGetPlan` and `time_sorted_plan` build and time the duplicates, and they count against a plan or benchmark limit |
| `get_fallback_engine_list` | returns upstream's table of NVIDIA engine numbers; under the shim those numbers select unrelated hipDNN engines or none (see the engine-indices row under [Engines, plans and execution](#engines-plans-and-execution)) |
| `setKernelCache`, `setDeviceProperties`, `setSMCount` | a non-null kernel cache or device properties, or an SM count of 0 or more, makes `build()` fail with `NOT_SUPPORTED` |
| `cudnnReorderFilterAndBiasInt8x32` | checks its arguments as upstream, then returns `NOT_SUPPORTED`: the shim has no int8x32 reordering |
| Plan tags | the `_k<type>=<value>` segments of `ExecutionPlan::getTag()` use the shim's knob-type integers, so errata `"knob"` entries written against cuDNN's integers do not match |
| `time_sorted_plan` timing | measured with hipDNN profiling events on the handle's stream: device time between the start and stop of each timed execution, as upstream's CUDA events |
| `time_sorted_plan` ties | plans with equal times are all kept; upstream keeps only one of them |
| `cudnnFindPlanAndCache` with no plan that ran | throws `cudnnException` with `NOT_SUPPORTED`; with `NV_CUDNN_DISABLE_EXCEPTION` it returns an unbuilt plan whose status is `BAD_PARAM` (no `EXECUTION_PLAN_ENGINE_CONFIG`). Upstream reads the front of an empty list |
| PyTorch quantized operations across threads | PyTorch's quantized convolution, linear and add operations cache plans process-wide, so a plan cached on one thread and executed with another thread's handle is `BAD_PARAM` (see the `cudnnBackendExecute` handle row). PyTorch's `Conv_v8` caches are per-thread and unaffected |
| `ConvDescBuilder` default mode | upstream's default is `CONVOLUTION`; MIOpen runs only `CROSS_CORRELATION`, so a graph that keeps the default has no engine. Call `setMathMode(CUDNN_CROSS_CORRELATION)` |
| `PointWiseDescBuilder` Swish | β = 1 is upstream's default and is not sent (see [Parameter descriptors](#parameter-descriptors)), so hipBLASLt's Swish epilogue (which requires β present and equal to 1) is unreachable from v8; MIOpen's fused convolution rejects a Swish β of any value |
