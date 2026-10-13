# Mini-BERT : Modèle de langage Transformer from scratch

Implémentation d'un modèle de type BERT simplifié à partir de zéro avec **PyTorch**.

L'objectif de ce projet est de comprendre le fonctionnement des modèles NLP modernes basés sur les Transformers en implémentant :
- une architecture Transformer Encoder
- le mécanisme d'attention
- le Masked Language Modeling (MLM)
- le transfert d'apprentissage et le fine-tuning

Aucun modèle Transformer pré-entraîné de HuggingFace n'a été utilisé.

---

## Description du projet

Le projet est réalisé en deux étapes :

### 1. Pré-entraînement (MLM)

Un modèle Mini-BERT est entraîné sur le jeu de données **AG News**.

Le modèle apprend à retrouver des mots masqués dans une phrase :

> "The cat sat on the [MASK]"

grâce à l'objectif de Masked Language Modeling.

### 2. Fine-tuning (Classification)

L'encodeur pré-entraîné est ensuite réutilisé pour une tâche de classification de sentiments sur la base de données **IMDB**.

Les performances sont comparées entre :
- un Mini-BERT pré-entraîné
- un modèle initialisé aléatoirement

afin d'étudier l'impact du pré-entraînement.

---

## Architecture

```
Embeddings des tokens
          +
Embeddings positionnels
          |
          v
Transformer Encoder
          |
          v
Tête de prédiction / Classification
```

Principaux paramètres :
- Framework : PyTorch
- Dimension des embeddings : 128
- Nombre de couches Transformer : 2-4
- Nombre de têtes d'attention : 4
- Longueur maximale des séquences : 256 tokens

---

## Bibliothèques utilisées

- PyTorch
- NumPy
- Matplotlib
- HuggingFace Datasets / Tokenizers (uniquement pour le traitement des données)
