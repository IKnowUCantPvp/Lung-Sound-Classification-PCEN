import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss


def load_and_process_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Create mapping for true classes - matches the order in the directory structure
    class_mapping = {
        'Bronchiectasis': 0,  # First directory
        'Bronchiolitis': 1,  # Second directory
        'COPD': 2,  # Third directory
        'Healthy': 3,  # Fourth directory
        'Pneumonia': 4,  # Fifth directory
        'URTI': 5  # Sixth directory
    }

    df['true_class_numeric'] = df['true_class'].map(class_mapping)
    return df, class_mapping


def calculate_metrics(df, class_mapping):
    # List of models
    models = ['conv2doldpcen', 'conv2dpcen', 'conv2doldpcen_trial_1',
              'conv2dspec', 'mfcc_cnn', 'conv2doldpcen_trial_2']

    # Get class names for better readability
    class_names = list(class_mapping.keys())

    results = []

    for model in models:
        pred_col = f'{model}_pred'
        conf_col = f'{model}_conf'

        # Calculate accuracy
        accuracy = accuracy_score(df['true_class_numeric'], df[pred_col])

        # Calculate confusion matrix
        cm = confusion_matrix(df['true_class_numeric'], df[pred_col])

        # Calculate per-class accuracy
        per_class_acc = {}
        for class_name, class_num in class_mapping.items():
            mask = df['true_class_numeric'] == class_num
            if mask.sum() > 0:  # Only calculate if we have samples for this class
                class_acc = accuracy_score(
                    df[mask]['true_class_numeric'],
                    df[mask][pred_col]
                )
                per_class_acc[class_name] = class_acc

        # Calculate custom loss using confidence scores with error handling
        correct_mask = df['true_class_numeric'] == df[pred_col]
        confidences = df[conf_col].where(correct_mask, 1 - df[conf_col])
        # Clip values to avoid log(0)
        confidences = np.clip(confidences, 1e-15, 1.0)
        loss = np.mean(-np.log(confidences))

        results.append({
            'model': model,
            'accuracy': accuracy,
            'loss': loss,
            'confusion_matrix': cm,
            'per_class_accuracy': per_class_acc
        })

    return pd.DataFrame(results)


def plot_results(results, class_mapping):
    plt.style.use('default')

    # Plot 1: Overall Performance
    plt.figure(figsize=(15, 8))

    # Prepare data
    models = results['model']
    accuracies = results['accuracy']
    losses = results['loss']

    # Create bar positions
    x = np.arange(len(models))
    width = 0.35

    # Create plot
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Plot accuracy bars
    bars1 = ax1.bar(x - width / 2, accuracies, width, label='Accuracy', color='royalblue')
    ax1.set_ylabel('Accuracy', color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')

    # Create second y-axis for loss
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, losses, width, label='Loss', color='lightcoral')
    ax2.set_ylabel('Loss', color='lightcoral')
    ax2.tick_params(axis='y', labelcolor='lightcoral')

    # Set x-axis labels
    plt.xticks(x, models, rotation=45, ha='right')

    # Add title and adjust layout
    plt.title('Model Performance Comparison')
    fig.tight_layout()

    # Save figure
    plt.savefig('model_performance.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.tight_layout()
    plt.savefig('model_performance.png')

    # Plot 2: Confusion Matrices
    class_names = list(class_mapping.keys())

    # Create one figure per model for confusion matrices
    for idx, row in results.iterrows():
        plt.figure(figsize=(10, 8))
        plt.imshow(row['confusion_matrix'], cmap='Blues')

        # Add numbers to cells
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, str(row['confusion_matrix'][i, j]),
                         ha="center", va="center")

        # Add labels
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.yticks(range(len(class_names)), class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{row["model"]} Confusion Matrix')

        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{row["model"]}.png', bbox_inches='tight', dpi=300)
        plt.close()

    # Plot 3: Per-class accuracies
    for idx, row in results.iterrows():
        plt.figure(figsize=(12, 6))

        # Prepare data
        classes = list(row['per_class_accuracy'].keys())
        accuracies = list(row['per_class_accuracy'].values())

        # Create bars
        plt.bar(classes, accuracies)

        # Customize plot
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Accuracy')
        plt.title(f'{row["model"]} Per-Class Accuracy')

        # Add value labels on top of bars
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

        plt.tight_layout()
        plt.savefig(f'per_class_accuracy_{row["model"]}.png', bbox_inches='tight', dpi=300)
        plt.close()


def main():
    # Load and process the data
    df, class_mapping = load_and_process_data('test_results.csv')

    # Calculate metrics
    results = calculate_metrics(df, class_mapping)

    # Print detailed results
    print("\nDetailed Results:")
    print("================")
    for _, row in results.iterrows():
        print(f"\nModel: {row['model']}")
        print(f"Overall Accuracy: {row['accuracy']:.4f}")
        print(f"Loss: {row['loss']:.4f}")
        print("\nPer-Class Accuracies:")
        for class_name, acc in row['per_class_accuracy'].items():
            print(f"{class_name}: {acc:.4f}")
        print("\nConfusion Matrix:")
        print(row['confusion_matrix'])

    # Create visualizations
    plot_results(results, class_mapping)


if __name__ == "__main__":
    main()