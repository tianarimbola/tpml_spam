"""
Script pour filtrer le dataset et garder seulement les messages en français
"""
import pandas as pd
import os

# Charger le dataset brut
input_file = 'dataset_raw.csv'
output_file = 'dataset_french.csv'

print("Chargement du dataset brut...")
df = pd.read_csv(input_file)

print(f"Total de messages dans le dataset: {len(df)}")
print(f"Langues présentes: {df['lang'].unique()}")

# Filtrer seulement les messages français
df_french = df[df['lang'] == 'french'].copy()

print(f"\nMessages en français trouvés: {len(df_french)}")
print(f"Distribution des labels:")
print(df_french['labels'].value_counts())

# Réinitialiser les indices
df_french.reset_index(drop=True, inplace=True)

# Sauvegarder le dataset filtré
df_french.to_csv(output_file, index=False)
print(f"\nDataset français sauvegardé dans: {output_file}")

# Afficher quelques exemples
print("\n=== Exemples de messages SPAM ===")
spam_examples = df_french[df_french['labels'] == 'spam'].head(3)
for idx, row in spam_examples.iterrows():
    print(f"{idx+1}. {row['text'][:80]}...")

print("\n=== Exemples de messages HAM (non-spam) ===")
ham_examples = df_french[df_french['labels'] == 'ham'].head(3)
for idx, row in ham_examples.iterrows():
    print(f"{idx+1}. {row['text'][:80]}...")
