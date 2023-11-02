import matplotlib.pyplot as plt

def plot_time_series(coord, true_data, predicted_data):
    plt.figure(figsize=(15,6))
    plt.plot(true_data, label="True", alpha=0.7)
    plt.plot(predicted_data, label="Predicted", linestyle='--', alpha=0.7)
    plt.title(f"Time Series for Coordinate: {coord}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

coords, sources, targets = next(iter(test_loader))
predictions = model(sources)
selected_coord = coords[0]  # Select the first coordinate for demonstration
plot_time_series(selected_coord, targets[0].view(-1).tolist(), predictions[0].view(-1).tolist())

import seaborn as sns

def plot_heatmap(data, title=""):
    plt.figure(figsize=(10,8))
    sns.heatmap(data, cmap='viridis')
    plt.title(title)
    plt.show()

# For demonstration
selected_true_tensor = targets[0].view(8,6)
selected_predicted_tensor = predictions[0].view(8,6)
plot_heatmap(selected_true_tensor, title="True Values")
plot_heatmap(selected_predicted_tensor, title="Predicted Values")

errors = targets.view(-1) - predictions.view(-1)
plt.figure(figsize=(10,6))
sns.histplot(errors.detach().numpy(), bins=50, kde=True)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(targets.view(-1).detach().numpy(), predictions.view(-1).detach().numpy(), alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values')
plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r')  # x=y line
plt.show()
