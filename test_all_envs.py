"""OpenEnv compliance tests for DiplomacyNegotiationEnv, ContractorNegotiationEnv, HumanImitationEnv."""
import sys

def check_env(name, env_class, has_different_resets=True):
    from openenv.env import Env
    passed = 0
    failed = []

    # MRO confirms inheritance from openenv.env.Env
    mro_names = [c.__name__ for c in env_class.__mro__]
    if "Env" in mro_names and env_class.__mro__[0] == env_class:
        print(f"  {name} MRO inherits from openenv.env.Env: PASS")
        passed += 1
    else:
        print(f"  {name} MRO inherits from openenv.env.Env: FAIL (MRO={mro_names})")
        failed.append("MRO")

    env = env_class()

    # reset() returns (obs, info) where obs.shape == (384,)
    obs, info = env.reset()
    if isinstance(obs, __import__("numpy").ndarray) and obs.shape == (384,):
        print(f"  {name} reset() obs.shape == (384,): PASS")
        passed += 1
    else:
        sh = getattr(obs, "shape", type(obs).__name__)
        print(f"  {name} reset() obs.shape == (384,): FAIL (obs.shape={sh})")
        failed.append("reset_obs_shape")

    # step(action) returns (obs, reward, done, info) with reward as real float
    try:
        obs2, reward, done, info2 = env.step("test action")
        if isinstance(reward, (int, float)) and not isinstance(reward, bool):
            print(f"  {name} step() returns (obs, reward, done, info) reward float: PASS")
            passed += 1
        else:
            print(f"  {name} step() reward is real float: FAIL (type={type(reward).__name__})")
            failed.append("step_reward_float")
    except Exception as e:
        print(f"  {name} step() returns 4-tuple: FAIL ({e})")
        failed.append("step")

    # render() returns non-empty string
    out = env.render()
    if isinstance(out, str) and len(out.strip()) > 0:
        print(f"  {name} render() non-empty string: PASS")
        passed += 1
    else:
        print(f"  {name} render() non-empty string: FAIL (type={type(out).__name__}, len={len(out) if isinstance(out, str) else 'N/A'})")
        failed.append("render")

    # Each reset() gives DIFFERENT output (not hardcoded) — only for envs that support it
    if has_different_resets:
        obs_a, _ = env.reset()
        obs_b, _ = env.reset()
        diff = (obs_a != obs_b)
        if isinstance(diff, __import__("numpy").ndarray):
            different = diff.any()
        else:
            different = obs_a != obs_b
        if different:
            print(f"  {name} reset() gives different output: PASS")
            passed += 1
        else:
            print(f"  {name} reset() gives different output: FAIL (observations identical)")
            failed.append("reset_different")

    return passed, failed


def main():
    print("Testing all three environments (OpenEnv compliance)")
    print("=" * 50)

    from envs.diplomacy_env import DiplomacyNegotiationEnv
    from envs.contractor_env import ContractorNegotiationEnv
    from envs.human_imitation_env import HumanImitationEnv

    total_passed = 0
    total_failed = []

    for label, env_class, different_resets in [
        ("DiplomacyNegotiationEnv", DiplomacyNegotiationEnv, True),
        ("ContractorNegotiationEnv", ContractorNegotiationEnv, True),
        ("HumanImitationEnv", HumanImitationEnv, True),
    ]:
        print(f"\n--- {label} ---")
        p, f = check_env(label, env_class, has_different_resets=different_resets)
        total_passed += p
        total_failed.extend([(label, x) for x in f])

    print("\n" + "=" * 50)
    if total_failed:
        print(f"RESULT: {total_passed} checks PASSED, {len(total_failed)} FAILED")
        sys.exit(1)
    print("All OpenEnv compliance checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
