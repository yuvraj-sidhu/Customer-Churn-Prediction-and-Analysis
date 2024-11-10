import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    sns.barplot(x=importances, y=feature_names)
    plt.title("Feature Importance")
    plt.savefig("visuals/feature_importance.png")

if __name__ == "__main__":
    plot_feature_importance(model, X_train.columns)
