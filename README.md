# Poisonous Mushroom Prediction

A project from my postgraduate course *Big Data Technologies* (CS982, University of Strathclyde), analysing the [UCI Mushroom dataset](https://archive.ics.uci.edu/dataset/73/mushroom) — 8,124 gilled mushroom samples from the Agaricus and Lepiota families, each described by 22 categorical, observable attributes (cap shape/surface/colour, odour, gill properties, stalk features, spore print colour, habitat, etc.) — to see how well toxicity can be predicted from physical traits alone.

The full EDA, preprocessing, and modelling live in [`mushroom_toxicity_prediction.ipynb`](./mushroom_toxicity_prediction.ipynb). This README summarizes the approach and findings.

## Dataset

Classes are fairly balanced (~51.8% edible, ~48.2% poisonous). The only missing values live in `stalk-root` (~30.5% marked `?`), which were kept as their own category rather than imputed or dropped.

## Exploratory Data Analysis

- A custom **ordinal encoding** was applied to features with a natural order (`bruises?`, `gill-spacing`, `gill-size`, `ring-number`, `population`) to make a correlation heatmap against toxicity meaningful — bruising and gill size showed moderate negative correlations with toxicity.
- Count plots across all 22 categorical features showed that **odour** and **spore print colour** were unusually discriminative — most odour categories map almost exclusively to one class (e.g. a mushroom with no scent, almond, or anise smell is reliably edible), and a buff gill colour is almost always poisonous. This hinted early on that a decision tree would find very pure splits.
- A parallel-categories plot confirmed the same pattern visually: `odor` and `spore-print-color` cleanly separate edible from poisonous, while features like `veil-color` carry little discriminative signal.

## Feature Engineering

Preprocessing combined the ordinal encoding above with **one-hot encoding** for the remaining nominal features (shapes, colours, surfaces, etc.), expanding the feature space to 107 columns, followed by a 70/30 train-test split.

## Unsupervised Analysis

Even though the dataset is labelled, clustering was used to see whether toxicity emerges as a natural structure in the data, independent of the class labels.

- **Agglomerative clustering** (Euclidean and Hamming distance) and **K-Means** were both tried; K-Means was carried forward for its easier interpretability and better scaling, with results comparable to agglomerative clustering.
- K-Means was run before and after PCA (107 features → 2 principal components):

| Metric | Before PCA | After PCA |
|---|---|---|
| Silhouette Score | 0.10 | 0.59 |
| Calinski-Harabasz Index | 720.96 | 6932.83 |
| Homogeneity | 0.59 | 0.57 |
| Completeness | 0.61 | 0.60 |

PCA sharply improved cluster compactness and separation, at a small cost to homogeneity/completeness from the information lost in compressing 107 features into 2. No single feature dominated either principal component (the top contributor to each explained only ~12% of variance), and the K-Means clusters only loosely tracked the true edible/poisonous labels — several poisonous mushrooms were grouped with the edible cluster. **Conclusion: the data doesn't separate into toxicity-aligned clusters on its own** — clustering isn't a reliable way to flag poisonous mushrooms here.

## Supervised Analysis

Five classifiers were grid-search tuned (5-fold CV) and compared: Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest, and Naïve Bayes.

**Results:** Logistic Regression, KNN, Decision Tree, and Random Forest all reached perfect scores (100% accuracy, precision, recall, F1) on the held-out test set. Naïve Bayes was the outlier, with a tendency toward false positives and one false negative — a meaningful failure mode for a poison classifier, where a single missed case matters.

The **Decision Tree** (max depth 10, min samples split 2, chosen via grid search) was carried forward for analysis over the other equally-accurate models, since it doubles as an interpretability tool — it visualises its own decision rules and ranks feature importance directly:

| Feature | Importance |
|---|---|
| odor | 67.1% |
| stalk-root | 25.4% |
| spore-print-color | 3.3% |
| stalk-color-above-ring | 1.9% |
| stalk-surface-below-ring | 1.3% |

Odour alone accounts for two-thirds of the model's decision-making, consistent with the EDA — confirming that observable smell is the single most reliable field cue for whether a mushroom is safe to eat.

## Key takeaways

- Perfect classification metrics are a signal to check *why*, not just report the number — here it traces back to specific features (odour, stalk-root) with near-total class purity, visible from the EDA before any model was fit.
- Model choice isn't only about accuracy: with several classifiers tied at 100%, the deciding factor was interpretability (decision rules + feature importances) over black-box performance.
- Unsupervised and supervised results told different stories on the same data — clustering found only loose structure, while a decision tree found near-perfect separation. Labelled ground truth mattered a lot for this dataset.
- In a health/safety context, error *type* matters as much as error *rate* — Naïve Bayes' false negative was flagged as disqualifying even though its aggregate accuracy was only slightly lower than the top models.

## Running it

```bash
pip install numpy pandas seaborn matplotlib plotly scikit-learn
jupyter notebook mushroom_toxicity_prediction.ipynb
```

The notebook expects the raw dataset as `agaricus-lepiota.data` in the same directory — download it from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/73/mushroom) (not redistributed here).
