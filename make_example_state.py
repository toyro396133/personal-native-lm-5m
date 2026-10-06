from personal_state import PersonalState, StateSlot, CORE_DIM, POLICY_DIM, WORLD_DIM, ROUTING_DIM

def vec(n, index, value=0.8):
    x = [0.0] * n
    x[index] = value
    return x

state = PersonalState(
    user_id="example-user",
    core=StateSlot(vec(CORE_DIM, 2), confidence=0.90, evidence_count=12),
    policies=StateSlot(vec(POLICY_DIM, 9), confidence=0.84, evidence_count=8),
    worlds={
        "coding": StateSlot(vec(WORLD_DIM, 4), confidence=0.92, evidence_count=16),
        "writing": StateSlot(vec(WORLD_DIM, 17), confidence=0.73, evidence_count=6),
    },
    routing=StateSlot(vec(ROUTING_DIM, 3), confidence=0.81, evidence_count=7),
)
state.to_json("example_personal_state.json")
print("wrote example_personal_state.json")
