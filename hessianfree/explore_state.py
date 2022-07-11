"""Here, I just play around with the `state` attribute of the optimizer in order
to understand, how it works and what I can do with it."""


import torch


class MyOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, damping):

        defaults = dict(
            lr=lr,
            damping=damping,
        )
        super(MyOptimizer, self).__init__(params, defaults)
        self._params = self.param_groups[0]["params"]

    def step(self):

        print("\n===== STEP =====")
        print("INIT self.state = ", self.state)

        # Example: A global parameter
        # NOTE: If we access the key "example_global", this will not throw a key
        # error but instead set `self.state[example_global] = 0`.
        self.state.setdefault("example_global", 0)

        for p in self._params:

            # Dummy parameter update
            p.data = p.data + 1

            # Example: Save parameter-specific information by using `p` as key
            state = self.state[p]  # Creates an empty dict if key not known
            if len(state) == 0:
                state["example"] = 10.0

            state["example"] *= 0.1

            blubb = state.get("blubb")
            print("blubb = ", blubb)
            # This also works and yields 'None'

        self.state["example_global"] += 1

        print("FINAL self.state = ", self.state)


if __name__ == "__main__":

    init_params = torch.rand((2))
    params = [init_params.clone().detach().requires_grad_(True)]

    opt = MyOptimizer(params, lr=1.0, damping=0.1)
    opt.step()
    opt.step()

    print("\nopt.state_dict() = ", opt.state_dict())
