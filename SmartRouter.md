# FCC SmartRouter

SmartRouter chooses the best currently usable model from the configured FCC
pool. It is a policy layer above provider clients: it does not make provider
requests while ranking, and it never changes a route's health during ranking.

## What is configured

The candidate inventory comes from `MODEL`, the model-role overrides, and
`MODEL_FALLBACKS`. The model registry is rebuilt from that inventory when a
request is routed. The configured pool is the source of truth; provider model
catalogs are used for discovery and diagnostics, not as an instruction to call
every model.

## Ranking policy

Routes are evaluated in this order:

1. The route must be registered and meet the request's capability requirements.
2. Routes in blocked, active backoff, quarantine, or exhausted shared-quota
   states are removed.
3. Remaining routes are ordered by capability tier, capability score, provider
   diversity, prior failures, and observed first-token latency.

Lower tier numbers are better:

| Tier | Meaning | Examples |
| --- | --- | --- |
| Tier 1 | VIP / highest capability | 400B+, 550B, 700B, 1.3T |
| Tier 2 | Strong general-purpose | 100B–399B and selected families |
| Tier 3 | Useful mid-size models | 27B–99B |
| Tier 4 | Smaller or unknown models | under 27B or no size evidence |

Known model families override size inference. Otherwise explicit parameter
markers such as `550B`, `120B`, or `1.3T` provide a conservative coarse tier.

`!health` / `fcc-health` and the router-status API are read-only views of the
same registry and health state used by routing. `fcc-health` also reports the
configured primary versus the last successful route, parameter-size/tier
labels, active/limited/exhausted/blocked counts, probe progress, and every
provider reset window observed from quota metadata or persisted quarantine
state (minute/hour/day/week/month/provider-defined when available).

The foreground objective is configurable with `SMART_ROUTER_POLICY`:

| Value | Behavior |
| --- | --- |
| `quality` (default) | Keep the strongest healthy capability tier first; use measured latency within that tier. |
| `fastest` | Prefer the lowest-latency proven healthy route, while still respecting health, quota, and request requirements. |

The same setting is available in Admin → Models. It is read at server startup,
so restart `fcc-server` after changing it.

## Health state machine

| State | Meaning | Routing behavior |
| --- | --- | --- |
| `unknown` | No recent evidence | Eligible for a first attempt |
| `available` | Meaningful output was received | Eligible and latency is recorded |
| `backoff` | Temporary 5xx, timeout, overload, or transient 429 | Skipped until retry time and rehabilitation probes pass |
| `quarantined` | Explicit quota/free-allocation exhaustion | Skipped until reset/expiry |
| `blocked` | Authentication, permission, or billing failure | Not selected automatically |

Health is persisted under `~/.fcc/route-health.json`, so a newly started FCC
process does not have to rediscover the same dead routes through user traffic.
Provider quota buckets can quarantine several routes together when the limit is
account-wide.

## Background rehabilitation

When a route enters transient backoff, FCC's low-volume background probe service waits for
its cooldown and sends a tiny streamed `ping` canary. It probes at most two
routes per cycle. At startup it also tests at most two never-observed routes per
cycle; this warms unknown routes without firing hundreds of requests at once.
It does not probe routes that are quarantined or blocked. A recovering route
must produce valid output in three separate probes before it
is promoted back to `AVAILABLE`; until then it remains excluded from user
traffic. A failed canary resets the probation counter and starts a new normal
backoff.

Canaries are deliberately small (`max_tokens` 4, reasoning off) and use the
real provider client, credentials, streaming path, and model identifier. They
prove basic reachability and response production without spending a full user
request. Real requests remain authoritative: a pre-response production failure
still triggers immediate fallback even if the most recent canary succeeded.

## Request lifecycle

1. FCC maps the incoming Claude model name to the configured route pool.
2. SmartRouter ranks the pool without network calls.
3. FCC tries the first route.
4. If the provider fails before output begins, FCC records the failure and
   moves through the ranked candidates.
5. Once meaningful output begins, that route is considered proven healthy. A
   later stream failure is diagnostic and does not demote the route for the
   current request.

Generation requests use one initial provider attempt plus one recovery probe.
This is intentionally separate from catalog discovery. A broken VIP route can
therefore recover briefly, but cannot consume five exponential-backoff attempts
while the user waits. Provider-side 5xx responses fast-fail to the next route;
the background canary owns recovery for those routes. Timeouts and ordinary
retryable throttles retain one recovery attempt. Explicit quota, payment,
authentication, and permission failures do not retry.

The provider progress watchdog is 45 seconds by default: if a route produces no
meaningful event for that interval, FCC records a timeout and falls back. This
prevents a dead upstream from holding Claude indefinitely while still allowing
normal reasoning streams to continue.

## Latency strategy

Do not generate a test request against all hundreds of models at startup. That
would consume quotas, trigger throttles, and make startup less predictable.
The pragmatic strategy is:

- discover catalogs once;
- reuse persisted route health;
- probe only a small bounded set per cycle in the background;
- prefer a proven route over an untested peer within the same tier;
- among proven same-tier routes, prefer the lower observed first-token latency;
- keep capability tier ahead of latency for complex coding work.

Set `SMART_ROUTER_POLICY=fastest` when interactive latency matters more than
capability rank. For tool-heavy or complex work, keep the default `quality`
policy so Tier 1 stays ahead of lower-tier speed. Both policies require health,
quota, and request-capability checks first.

## Free/paid policy

`FCC_VERIFIED_FREE_MODELS` records explicit free verification and is exposed in
health diagnostics. The current compatibility behavior allows a configured
route whose eligibility is `unknown`; operators should therefore keep paid or
uncertain routes out of `MODEL`/`MODEL_FALLBACKS` unless they intentionally want
to test them. A strict hard-zero mode can be added later by making
`UNKNOWN` an exclusion in `SmartRouter._evaluate`.

## Client integrations

All Claude surfaces must use the same environment:

```text
ANTHROPIC_BASE_URL=http://127.0.0.1:<FCC_PORT>
ANTHROPIC_AUTH_TOKEN=<FCC_TOKEN>
CLAUDE_CODE_USE_GATEWAY=1
```

Do not set `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY`. Native Claude Code can
stall after `/v1/models` when that compatibility path is enabled; FCC removes
the variable from terminal launches and from the VS Code integration.

After changing the port, reconnect the VS Code integration from FCC's Admin UI
or reload the extension. Restart the client after changing settings.

## Operational checks

```text
fcc-health
```

Confirm that the selected route is `available`, that VIP routes are not in
`backoff` or `quarantined`, and that the observed latency belongs to a route
that has actually emitted output. A `500` before output means “temporarily
unhealthy route”, not “the model spent tokens thinking”.
