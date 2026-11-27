"""Tests of serve/scheduler/step_policy.py against a small copy of the scheduler's state machine.

Run with ``python -m pytest tests/test_step_policy.py``. The module under test has no imports, so
it is loaded from its file and the test needs neither torch nor a GPU.
"""
import enum
import importlib.util
import random
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "step_policy", Path(__file__).resolve().parents[1] / "serve/scheduler/step_policy.py"
)
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)


class S(enum.Enum):
    NORMAL = 0
    DRAFT = 1
    VERIFY = 2
    STALL = 3


class Req:
    def __init__(self, rid):
        self.rid = rid
        self.status = S.NORMAL
        self.cur_spec_idx = 0
        self.initial_idx = 0
        self.len_draft_probs = 0
        self.drafted = 0  # draft tokens over the whole run
        self.cycles = 0


class Engine:
    """The parts of BaseScheduler / SpecScheduler that touch the phase of a request."""

    def __init__(self, k, max_batch_size, step=True, seed=0):
        self.k, self.max_batch_size, self.step = k, max_batch_size, step
        self.active, self.offloaded = [], []
        self.rng = random.Random(seed)
        self.forwards = self.iterations = self.mixed = 0

    def admit(self, request):
        request.cur_spec_idx = request.initial_idx = 0 if self.step else self.rng.randint(0, self.k - 1)
        self.active.append(request)

    def offload(self, request):  # the engine never offloads a STALL request
        assert request.status is not S.STALL
        self.active.remove(request)
        self.offloaded.append(request)

    def restore(self):
        really = [r for r in self.offloaded if not sp.is_waiting(r)]
        waiting = len(self.offloaded) - len(really)
        if not really or len(self.active) + waiting >= self.max_batch_size:
            return
        request = really[-1]
        if self.step and not sp.try_join(request, sp.target_phase(self.active)):
            request.restore_wait = 0
        else:
            self.offloaded.remove(request)
            self.active.append(request)

    def iterate(self):
        if self.step:
            sp.keep_in_step(self.active, self.offloaded, self.max_batch_size, self.k)
        assert len(self.active) <= self.max_batch_size
        executed = [r for r in self.active if r.status is not S.STALL]
        kinds = {r.status for r in executed}
        self.mixed += S.DRAFT in kinds and S.VERIFY in kinds
        self.forwards += bool(executed)
        self.iterations += 1
        # update_metadata_host: runs only with a forward pass, for requests in both lists
        if executed:
            for r in self.active + self.offloaded:
                if hasattr(r, "token_indptr") and r.status is not S.STALL:
                    del r.token_indptr
        for r in self.active:
            if r.status is S.DRAFT:  # prepare_forward_inputs
                r.len_draft_probs += 1
            if r.status is S.STALL and hasattr(r, "token_indptr"):  # verify_outputs
                del r.token_indptr
            elif r.status is not S.STALL:
                r.token_indptr = True
            if r.status is S.NORMAL:
                r.status = S.DRAFT
            elif r.status is S.DRAFT:
                r.cur_spec_idx += 1
                r.drafted += 1
                assert r.cur_spec_idx <= self.k
                assert r.cur_spec_idx - r.initial_idx == r.len_draft_probs
                if r.cur_spec_idx == self.k:
                    r.status = S.VERIFY
            elif r.status is S.VERIFY:
                assert r.cur_spec_idx - r.initial_idx == r.len_draft_probs
                assert self.k + 1 - r.initial_idx >= 2  # at least one draft and the bonus token
                r.cur_spec_idx = r.initial_idx = 0
                r.cycles += 1
                r.status = S.STALL
            else:
                r.len_draft_probs = 0
                r.status = S.DRAFT


def phases(engine):
    return {sp.phase(r) for r in engine.active} - {None}


def test_requests_admitted_one_per_iteration_end_up_in_one_phase():
    for k in (2, 6, 10):
        e = Engine(k, 64)
        for i in range(64):
            e.admit(Req(i))
            e.iterate()
        for _ in range(2 * (k + 2)):
            e.iterate()
        e.forwards = e.iterations = e.mixed = 0
        for _ in range(10 * (k + 2)):
            e.iterate()
            assert len(phases(e)) == 1 and len(e.active) == 64 and not e.offloaded
        assert e.mixed == 0
        assert e.forwards == e.iterations - 10  # the wait iteration of every cycle has no pass


def test_without_the_policy_every_iteration_has_a_forward_pass():
    e = Engine(6, 64, step=False)
    for i in range(64):
        e.admit(Req(i))
        e.iterate()
    e.forwards = e.iterations = 0
    for _ in range(80):
        e.iterate()
    assert e.forwards == e.iterations and e.mixed > 0


def test_restored_requests_come_back_in_step_and_nothing_is_lost():
    k = 6
    e = Engine(k, 32, seed=1)
    for i in range(32):
        e.admit(Req(i))
        e.iterate()
    rng = random.Random(2)
    longest_wait = 0
    for it in range(4000):
        if rng.random() < 0.15:
            victims = [r for r in e.active if r.status is not S.STALL]
            if victims:
                e.offload(rng.choice(victims))
        if rng.random() < 0.3:
            e.restore()
        e.iterate()
        longest_wait = max([longest_wait] + [r.restore_wait for r in e.offloaded if sp.is_waiting(r)])
        if it == 3 * (k + 2):
            e.mixed = 0  # the first cycle of a request can be out of step
        if it > 3 * (k + 2):
            assert len(phases(e)) <= 1
    assert longest_wait <= k + 2
    assert len(e.active) + len(e.offloaded) == 32
    assert e.mixed == 0
    # every request keeps making progress
    assert min(r.cycles for r in e.active + e.offloaded) > 50


def test_a_request_with_drafts_in_flight_is_never_given_a_new_index():
    class R:
        pass

    r = R()
    r.status, r.cur_spec_idx, r.initial_idx, r.len_draft_probs = S.DRAFT, 3, 1, 2
    assert not sp.try_join(r, ("DRAFT", 5)) and r.cur_spec_idx == 3
    assert sp.try_join(r, ("DRAFT", 3))
    r.cur_spec_idx, r.initial_idx, r.len_draft_probs = 0, 0, 0
    assert not sp.try_join(r, ("VERIFY", 6)) and not sp.try_join(r, ("STALL", 0))
    assert sp.try_join(r, ("DRAFT", 4)) and (r.cur_spec_idx, r.initial_idx) == (4, 4)
