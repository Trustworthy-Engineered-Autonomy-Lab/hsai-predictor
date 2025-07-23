import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import numpy as np


class ResultGraph:
    def __init__(self):
        """
        Constructor for ThreeAxisPlotter.
        You can add any configuration or style properties here.
        """
        pass

    def plot_lines(self, data_list, labels=None):
        """
        Plots multiple lines in 3D space, each with a distinct color and label.

        Parameters:
        -----------
        data_list : list of tuples/lists
            A list of data entries for each line. Each element should be:
               (x_values, y_values, z_values)
            where x_values, y_values, and z_values are lists or arrays of points.

        labels : list of strings, optional
            A list of labels for the lines. If provided, must have the same length
            as data_list. If None, default labels "Line 1", "Line 2", etc. are used.
        """
        # Create a new figure and add a 3D subplot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Generate distinct colors from a colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(data_list)))

        for i, (x_vals, y_vals, z_vals) in enumerate(data_list):
            # Determine the label for this line
            if labels and i < len(labels):
                label = labels[i]
            else:
                label = f"Line {i + 1}"

            # Plot this line with a unique color
            ax.plot(x_vals, y_vals, z_vals, color=colors[i], label=label)

        # Set axis labels
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Accuracy")
        ax.set_zlabel("F1 Score")

        # Place a legend in the best location
        ax.legend(loc="best")

        # Display the plot
        plt.show()


if __name__ == "__main__":
    # Example usage
    # Create some sample 3D line data
    t = np.linspace(0, 2 * np.pi, 100)
    line1 = (np.sin(t), np.cos(t), t)  # x=sin(t), y=cos(t), z=t
    line2 = (2 * np.sin(t), 2 * np.cos(t), 2 * t)  # x=2sin(t), y=2cos(t), z=2t
    line3 = (np.sin(2 * t), np.cos(2 * t), t)  # x=sin(2t), y=cos(2t), z=t
    line4 = ([1, 2], [1, 2], [1, 2])
    data = [line4]
    labels = ["Line"]

    # Create an instance of our plotter and draw
    plotter = ResultGraph()
    plotter.plot_lines(data_list=data, labels=labels)
