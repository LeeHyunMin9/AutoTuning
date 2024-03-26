import numpy as np
from sklearn.metrics import PredictionErrorDisplay

def metric(y_true, y_pred, type = 'MAE'):
    if type == 'MAE':
        return np.abs(y_true - y_pred)
    
    elif type == 'MAPE':
        return np.abs((y_true - y_pred) / y_true) * 100
    
    elif type == 'MSLE':
        return (np.log1p(y_true) - np.log1p(y_pred))**2



# Display the Regression Results
class PredictionErrorDisPlayDistances(PredictionErrorDisplay):

    def __init__(self, y_true, y_pred, distance = 'MAE'):
        #super().__init__()
        self.y_true = y_true
        self.y_pred = y_pred
        self.distance = distance  

    def plot(
        self,
        ax=None,
        *,
        kind="residual_vs_predicted",
        scatter_kwargs=None,
        line_kwargs=None,
    ):
        """Plot visualization.

        Extra keyword arguments will be passed to matplotlib's ``plot``.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        kind : {"actual_vs_predicted", "residual_vs_predicted"}, \
                default="residual_vs_predicted"
            The type of plot to draw:

            - "actual_vs_predicted" draws the observed values (y-axis) vs.
              the predicted values (x-axis).
            - "residual_vs_predicted" draws the residuals, i.e. difference
              between observed and predicted values, (y-axis) vs. the predicted
              values (x-axis).

        scatter_kwargs : dict, default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.scatter`
            call.

        line_kwargs : dict, default=None
            Dictionary with keyword passed to the `matplotlib.pyplot.plot`
            call to draw the optimal line.

        Returns
        -------
        display : :class:`~sklearn.metrics.PredictionErrorDisplay`

            Object that stores computed values.
        """

        expected_kind = ("actual_vs_predicted", "residual_vs_predicted")
        if kind not in expected_kind:
            raise ValueError(
                f"`kind` must be one of {', '.join(expected_kind)}. "
                f"Got {kind!r} instead."
            )

        import matplotlib.pyplot as plt

        if scatter_kwargs is None:
            scatter_kwargs = {}
        if line_kwargs is None:
            line_kwargs = {}

        default_scatter_kwargs = {"color": "tab:blue", "alpha": 0.8}
        default_line_kwargs = {"color": "black", "alpha": 0.7, "linestyle": "--"}

        scatter_kwargs = {**default_scatter_kwargs, **scatter_kwargs}
        line_kwargs = {**default_line_kwargs, **line_kwargs}

        if ax is None:
            _, ax = plt.subplots()

        if kind == "actual_vs_predicted":
            max_value = max(np.max(self.y_true), np.max(self.y_pred))
            min_value = min(np.min(self.y_true), np.min(self.y_pred))
            self.line_ = ax.plot(
                [min_value, max_value], [min_value, max_value], **line_kwargs
            )[0]

            x_data, y_data = self.y_pred, self.y_true
            xlabel, ylabel = "Predicted values", "Actual values"

            self.scatter_ = ax.scatter(x_data, y_data, **scatter_kwargs)

            # force to have a squared axis
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks(np.linspace(min_value, max_value, num=5))
            ax.set_yticks(np.linspace(min_value, max_value, num=5))
        else:  # kind == "residual_vs_predicted"
            self.line_ = ax.plot(
                [np.min(self.y_pred), np.max(self.y_pred)],
                [0, 0],
                **line_kwargs,
            )[0]
            self.scatter_ = ax.scatter(
                self.y_pred, metric(self.y_true , self.y_pred, self.distance), **scatter_kwargs
            )
            xlabel, ylabel = "Predicted values", f"Residuals {self.distance}(actual, predicted)"

        ax.set(xlabel=xlabel, ylabel=ylabel)

        self.ax_ = ax
        self.figure_ = ax.figure

        return self