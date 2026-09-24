from lifecycle import Status, Gates, InvalidTransition, transition

def test_registration_requires_all_gates():
    good = Gates(True, True, True, True)
    assert transition(Status.PROVISIONAL, Status.REGISTERED, good) is Status.REGISTERED

def test_registration_rejects_failed_gate():
    for field in ("utility", "stability", "integrity", "cost"):
        vals = dict(utility=True, stability=True, integrity=True, cost=True)
        vals[field] = False
        try:
            transition(Status.PROVISIONAL, Status.REGISTERED, Gates(**vals))
        except InvalidTransition:
            pass
        else:
            raise AssertionError(f"failed gate {field} was accepted")

def test_registered_cannot_be_created_from_other_states():
    good = Gates(True, True, True, True)
    for s in (Status.EXPERIMENTAL, Status.QUARANTINED, Status.RETIRED):
        try:
            transition(s, Status.REGISTERED, good)
        except InvalidTransition:
            pass
        else:
            raise AssertionError(f"invalid registration transition from {s}")

def test_basic_lifecycle():
    s = transition(Status.EXPERIMENTAL, Status.PROVISIONAL)
    s = transition(s, Status.QUARANTINED)
    s = transition(Status.QUARANTINED, Status.EXPERIMENTAL)
    s = transition(s, Status.PROVISIONAL)
    s = transition(s, Status.REGISTERED, Gates(True, True, True, True))
    assert s is Status.REGISTERED
