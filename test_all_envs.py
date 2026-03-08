print("Testing all three environments...")
print("=" * 50)

# Test 1 — DiplomacyNegotiationEnv
from envs.diplomacy_env import DiplomacyNegotiationEnv
env1 = DiplomacyNegotiationEnv()
obs, info = env1.reset()
print("DiplomacyNegotiationEnv:")
env1.render()
print(f"MRO: {[c.__name__ for c in DiplomacyNegotiationEnv.__mro__]}")
print("✅ Diplomacy OK\n")

# Test 2 — ContractorNegotiationEnv
from envs.contractor_env import ContractorNegotiationEnv
env2 = ContractorNegotiationEnv()
obs, info = env2.reset()
print("ContractorNegotiationEnv:")
env2.render()
print(f"MRO: {[c.__name__ for c in ContractorNegotiationEnv.__mro__]}")
print("✅ Contractor OK\n")

# Test 3 — HumanImitationEnv
from envs.human_imitation_env import HumanImitationEnv
env3 = HumanImitationEnv()
obs, info = env3.reset()
print("HumanImitationEnv:")
env3.render()
print(f"MRO: {[c.__name__ for c in HumanImitationEnv.__mro__]}")
print("✅ HumanImitation OK\n")

print("=" * 50)
print("All 3 environments passed smoke test.")
print("Ready for Phase 2 training.")

print("Testing all three environments...")
print("=" * 50)

# Test 1 — DiplomacyNegotiationEnv
from envs.diplomacy_env import DiplomacyNegotiationEnv
env1 = DiplomacyNegotiationEnv()
obs, info = env1.reset()
print("DiplomacyNegotiationEnv:")
env1.render()
print(f"MRO: {[c.__name__ for c in DiplomacyNegotiationEnv.__mro__]}")
print("✅ Diplomacy OK\n")

# Test 2 — ContractorNegotiationEnv
from envs.contractor_env import ContractorNegotiationEnv
env2 = ContractorNegotiationEnv()
obs, info = env2.reset()
print("ContractorNegotiationEnv:")
env2.render()
print(f"MRO: {[c.__name__ for c in ContractorNegotiationEnv.__mro__]}")
print("✅ Contractor OK\n")

# Test 3 — HumanImitationEnv
from envs.human_imitation_env import HumanImitationEnv
env3 = HumanImitationEnv()
obs, info = env3.reset()
print("HumanImitationEnv:")
env3.render()
print(f"MRO: {[c.__name__ for c in HumanImitationEnv.__mro__]}")
print("✅ HumanImitation OK\n")

print("=" * 50)
print("All 3 environments passed smoke test.")
print("Ready for Phase 2 training.")

