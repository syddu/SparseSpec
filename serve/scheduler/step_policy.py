"""GRASS: keep the speculation cycles of all active requests in step (``batch_policy="step"``).

With the engine's own policies a request keeps the phase of the speculation cycle it was admitted
or restored with, so in every iteration some requests draft, some verify and some wait for their
verify result, and every iteration needs a forward pass. Under this policy all active requests
share one phase: they draft together, verify together, and the iteration in which they wait needs
no forward pass.

A phase is ``(status, cur_spec_idx)``: DRAFT 0 .. DRAFT k-1, VERIFY k, STALL 0. The target is the
phase most active speculating requests are in, taken at the start of an iteration, when a request's
status is the one it executes next. Three rules bring a request to the target:

1. A request at the start of a cycle (DRAFT, no drafts in flight) takes the target's index when the
   target is a DRAFT phase. Its cycle is shortened to the drafts that are left, which the engine
   supports already (it is how ``greedy`` spreads new requests over the phases).
2. A request at the start of a cycle while the target is VERIFY or STALL is set aside for the one
   or two iterations until the target is DRAFT 0. A request that has just finished prefill is not
   set aside (its first token is still on its way from the GPU); it runs one cycle out of step.
3. A restored request with drafts in flight waits until the target is its own phase, at most one
   cycle.

A request that waits has all its KV on the GPU. It is kept in the scheduler's ``offload_requests``
list, marked by ``restore_wait`` (iterations waited so far), so every invariant the scheduler has on
its request lists holds unchanged. After two cycles it rejoins in any case.

This module has no imports, so its logic can be tested without the engine's dependencies.
"""


def phase(request):
    """``(status name, cur_spec_idx)``, or None for a request that is not speculating."""
    name = request.status.name
    return None if name == "NORMAL" else (name, request.cur_spec_idx)


def target_phase(active_requests):
    """The phase most active speculating requests are in, or None if there are none."""
    counts = {}
    for request in active_requests:
        p = phase(request)
        if p is not None:
            counts[p] = counts.get(p, 0) + 1
    return max(counts, key=counts.get) if counts else None


def at_cycle_start(request):
    return (
        request.status.name == "DRAFT"
        and request.cur_spec_idx == request.initial_idx
        and request.len_draft_probs == 0
    )


def try_join(request, target):
    """Whether ``request`` runs in step from this iteration on. Sets its index under rule 1."""
    p = phase(request)
    if p is None or target is None or p == target:
        return True
    if at_cycle_start(request) and target[0] == "DRAFT":
        request.cur_spec_idx = request.initial_idx = target[1]
        return True
    return False


def is_waiting(request):
    return getattr(request, "restore_wait", None) is not None


def keep_in_step(active_requests, offload_requests, max_batch_size, spec_stride):
    """Apply the rules at the start of an iteration. Both lists are changed in place.

    Returns the number of requests that wait after the call.
    """
    target = target_phase(active_requests)
    if target is not None:
        for request in list(active_requests):
            if phase(request) is None or try_join(request, target):
                continue
            if at_cycle_start(request) and not hasattr(request, "token_indptr"):
                active_requests.remove(request)  # rule 2
                offload_requests.append(request)
                request.restore_wait = 0
    limit = 2 * (spec_stride + 2)
    waiting = 0
    for request in [r for r in offload_requests if is_waiting(r)]:
        fits = len(active_requests) < max_batch_size
        if fits and (try_join(request, target) or request.restore_wait >= limit):
            offload_requests.remove(request)
            request.restore_wait = None
            active_requests.append(request)
        else:
            request.restore_wait += 1
            waiting += 1
    return waiting
