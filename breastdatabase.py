import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

while True:
    print("\nMenu de Visualização:")
    print("1. Heatmap - Correlação entre todas as Features")
    print("2. Histogramas - Feactures individualmente")
    print("3. Boxplot - Todas as Features")
    print("4. Violin Plot - Todas as Features")
    print("5. Scatter Plot - Agrupado pelas Classes")
    print("0. Sair")
    
    choice = input("Escolha uma opção: ")
    
    if choice == '0':
        print("Até mais.")
        break
    elif choice == '1':
        def show_correlation_heatmap():
            plt.figure(figsize=(12, 8))
            correlation_matrix = df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
            plt.title("Heatmap (Correlação entre Features)")
            plt.tight_layout()
            plt.show()
            
        show_correlation_heatmap()
    elif choice == '2':
        def show_histograms():
            plt.figure(figsize=(12, 10))
            df.drop('target', axis=1).hist(bins=20, figsize=(12, 10))
            plt.suptitle("Histogramas de cada Features Individualmente")
            plt.tight_layout()
            plt.show()
            
        show_histograms()
    elif choice == '3':
        def show_boxplot():
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=df.drop('target', axis=1), orient="h")
            plt.title("Boxplot de Todas as Features")
            plt.xlabel("Valores das Features")
            plt.tight_layout()
            plt.show()
            
        show_boxplot()
    elif choice == '4':
        def show_violin_plot():
            plt.figure(figsize=(12, 8))
            sns.violinplot(data=df.drop('target', axis=1), orient="h")
            plt.title("Violin Plot de Todas as Features")
            plt.xlabel("Valores das Features")
            plt.tight_layout()
            plt.show()
            
        show_violin_plot()
    elif choice == '5':
        def show_scatter_plot():
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=df, x='mean radius', y='mean texture', hue='target')
            plt.title("Scatter Plot (Agrupado pelas Classes)")
            plt.tight_layout()
            plt.show()
            
        show_scatter_plot()
    else:
        print("Opção inválida. Escolha novamente.")
