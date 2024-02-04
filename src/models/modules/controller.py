import math


class BasePIController():
    def __init__(self,
                 expected_kl: float,
                 init_beta: float = 0.,
                 beta_min: float = 0.,
                 Kp: float = 0.01,
                 Ki: float = 0.0001):
        assert init_beta >= beta_min, \
            f"beta(0) ({init_beta}) has to be bigger than beta_min {beta_min}"
        self.expected_kl = expected_kl
        self.beta_min = beta_min
        self.Kp = Kp
        self.Ki = Ki
        self.beta = init_beta

    def __call__(self, actual_kl: float):
        """PI controller algorithm

        Args:
            actual_kl (float): Actual KL-divergence loss.

        Returns:
            beta (float): weight of KL-div in VAE's objective function.
        """
        raise NotImplementedError


class PositionalPIController(BasePIController):

    def __init__(self,
                 expected_kl: float,
                 init_beta: float = 0.,
                 beta_min: float = 0.,
                 beta_max: float = 1.,
                 Kp: float = 0.01,
                 Ki: float = 0.0001):
        super().__init__(expected_kl, init_beta, beta_min, Kp, Ki)
        self.beta_max = beta_max
        self.I = 0.

    def __call__(self, actual_kl: float):
        """Positional PI controller algorithm from paper
        `ControlVAE: Controllable Variational Autoencoder`

        Args:
            actual_kl (float): Actual KL-divergence loss.

        Returns:
            beta (float): weight of KL-div in VAE's objective function.
        """
        error = self.expected_kl - actual_kl
        P = self.Kp / (1. + math.exp(error))

        if self.beta_min <= self.beta <= self.beta_max:
            self.I = self.I - self.Ki * error

        self.beta = P + self.I
        if self.beta > self.beta_max:
            self.beta = self.beta_max
        elif self.beta < self.beta_min:
            self.beta = self.beta_min

        return self.beta
