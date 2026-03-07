class OpenEnvBase:
    """
    Minimal OpenEnv-style base environment interface.

    Compatible with the expected OpenEnv 0.2.1 API:
      - reset() -> observation, info
      - step(action) -> observation, reward, done, info
      - render() -> optional visualization / text
      - close() -> cleanup
    """

    def reset(self):
        """Reset the environment to an initial state and return (observation, info)."""
        raise NotImplementedError

    def step(self, action):
        """Run one environment step given an action and return (observation, reward, done, info)."""
        raise NotImplementedError

    def render(self):
        """Render the current environment state (optional)."""
        raise NotImplementedError

    def close(self):
        """Clean up any resources held by the environment."""
        raise NotImplementedError

