import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar e filtrar os dados
df = pd.read_excel('barrettII_eyes_clustering.xlsx')
df['Correto'] = df['Correto'].map({'S': 1, 'N': 0})
df = df[df['Correto'] == 1]

# 2. Selecionar colunas e normalizar
colunas = ['AL', 'ACD', 'WTW', 'K1', 'K2']
X = df[colunas]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Aplicar PCA (redução para 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. K-Means com k=2, k=3, k=4
kmeans_2 = KMeans(n_clusters=2)
labels_2 = kmeans_2.fit_predict(X_scaled)

kmeans_3 = KMeans(n_clusters=3)
labels_3 = kmeans_3.fit_predict(X_scaled)

kmeans_4 = KMeans(n_clusters=4)
labels_4 = kmeans_4.fit_predict(X_scaled)

# 5. Criar DataFrames com os rótulos
df_pca_2 = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca_2['Cluster'] = labels_2

df_pca_3 = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca_3['Cluster'] = labels_3

df_pca_4 = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca_4['Cluster'] = labels_4

# 6. Adicionar os clusters ao df original
df['Cluster_2'] = labels_2
df['Cluster_3'] = labels_3
df['Cluster_4'] = labels_4

# 7. Calcular perfis médios
perfis_2 = df.groupby('Cluster_2')[colunas].mean()
perfis_3 = df.groupby('Cluster_3')[colunas].mean()
perfis_4 = df.groupby('Cluster_4')[colunas].mean()

print("Perfis de cada cluster com k=2:")
print(perfis_2)
print("\nPerfis de cada cluster com k=3:")
print(perfis_3)
print("\nPerfis de cada cluster com k=4:")
print(perfis_4)

# 8. Gráficos de barras para visualização dos perfis
# k=2
perfis_2.T.plot(kind='bar', figsize=(10, 6))
plt.title('Perfis Médios por Cluster (k=2)')
plt.ylabel('Média dos Atributos')
plt.xticks(rotation=0)
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# k=3
perfis_3.T.plot(kind='bar', figsize=(10, 6))
plt.title('Perfis Médios por Cluster (k=3)')
plt.ylabel('Média dos Atributos')
plt.xticks(rotation=0)
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# k=4
perfis_4.T.plot(kind='bar', figsize=(10, 6))
plt.title('Perfis Médios por Cluster (k=4)')
plt.ylabel('Média dos Atributos')
plt.xticks(rotation=0)
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Plotagem lado a lado dos clusters no plano PCA
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

sns.scatterplot(data=df_pca_2, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', ax=axes[0])
axes[0].set_title('K-Means com k=2')

sns.scatterplot(data=df_pca_3, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', ax=axes[1])
axes[1].set_title('K-Means com k=3')

sns.scatterplot(data=df_pca_4, x='PCA1', y='PCA2', hue='Cluster', palette='Set3', ax=axes[2])
axes[2].set_title('K-Means com k=4')

plt.tight_layout()
plt.show()
