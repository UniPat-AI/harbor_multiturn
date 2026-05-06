from harbor.metrics.base import BaseMetric


class Min(BaseMetric[dict[str, float | int]]):
    def compute(
        self, rewards: list[dict[str, float | int] | None]
    ) -> dict[str, float | int]:
        values = []

        for reward in rewards:
            if reward is None:
                values.append(0)
            elif len(reward) == 1:
                values.extend(reward.values())
            elif "reward" in reward:
                values.append(reward["reward"])
            else:
                raise ValueError(
                    f"Expected exactly one key or a 'reward' key in reward "
                    f"dictionary, got keys: {list(reward.keys())}"
                )

        return {"min": min(values)}
