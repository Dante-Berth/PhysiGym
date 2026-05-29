import pandas as pd
import matplotlib.pyplot as plt


def create_plot_from_csv(csv_path, output_filename="output_plot.pdf"):
    """
    Reads a CSV file and creates a scatter plot of x and y,
    color-coded by the 'type' column.
    """
    try:
        # Load the dataset
        df = pd.read_csv(csv_path)

        # Initialize the plot
        plt.figure(figsize=(10, 7))

        # Iterate through unique types to plot them with different colors/labels
        # This automatically handles categories like 'tumor', 'healthy', etc.
        for category, group in df.groupby("type"):
            plt.scatter(
                group["x"], group["y"], label=category, alpha=0.7, edgecolors="w"
            )

        # Add labels and title using LaTeX formatting
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Initial Condition")

        # Add a legend and grid
        plt.legend(title="Category")
        plt.grid(True, linestyle="--", alpha=0.5)

        # Save the image
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        plt.close()  # Close plot to free up memory

        print(f"Successfully saved plot to {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    L = [0, 1, 5, 7]
    for i in L:
        create_plot_from_csv(
            csv_path=f"data/async_sac_tip_1_img_mc_cells_substrates_1767578650/env1/episode0000000{i}/cells_1.csv",
            output_filename=f"output_plot_{i}.pdf",
        )
