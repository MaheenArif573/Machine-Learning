import numpy as np
import pandas as pd
from collections import Counter

def calculate_entropy(labels):
    counts = np.array(list(Counter(labels).values()))
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))


def calculate_weighted_entropy_categorical(df, feature, label_column):
    total = len(df)
    weighted = 0.0
    for val in df[feature].unique():
        subset = df[df[feature] == val][label_column]
        weight = len(subset) / total
        weighted += weight * calculate_entropy(subset)
    return weighted


def best_numeric_split(df, feature, label_column):
    """
    For numeric columns: try candidate thresholds (midpoints between
    sorted unique values) and return the one with lowest weighted entropy.
    This removes the need to manually bin continuous columns.
    """
    values = np.sort(df[feature].unique())
    if len(values) < 2:
        return None, np.inf

    thresholds = (values[:-1] + values[1:]) / 2
    total = len(df)
    best_thresh, best_entropy = None, np.inf

    # Subsample thresholds for speed on large unique-value columns
    if len(thresholds) > 20:
        thresholds = np.percentile(thresholds, np.linspace(0, 100, 20))

    for t in thresholds:
        left  = df[df[feature] <= t][label_column]
        right = df[df[feature] >  t][label_column]
        if len(left) == 0 or len(right) == 0:
            continue
        w = (len(left) / total) * calculate_entropy(left) + \
            (len(right) / total) * calculate_entropy(right)
        if w < best_entropy:
            best_entropy, best_thresh = w, t

    return best_thresh, best_entropy


def calculate_information_gain(df, feature, label_column):
    initial = calculate_entropy(df[label_column])

    if pd.api.types.is_numeric_dtype(df[feature]):
        _, weighted = best_numeric_split(df, feature, label_column)
    else:
        weighted = calculate_weighted_entropy_categorical(df, feature, label_column)

    return initial - weighted


def find_best_feature(df, features, label_column):
    gains = {f: calculate_information_gain(df, f, label_column) for f in features}
    return max(gains, key=gains.get), gains


#  DECISION TREE  
def build_tree(df, features, label_column, max_depth=None, depth=0, min_samples_split=2):
    labels = df[label_column]

    if labels.nunique() == 1:
        return labels.iloc[0]

    if not features or (max_depth is not None and depth >= max_depth) or len(df) < min_samples_split:
        return labels.mode()[0]

    best_feature, _ = find_best_feature(df, features, label_column)
    remaining = [f for f in features if f != best_feature]
    is_numeric = pd.api.types.is_numeric_dtype(df[best_feature])

    if is_numeric:
        thresh, _ = best_numeric_split(df, best_feature, label_column)
        if thresh is None:
            return labels.mode()[0]

        left  = df[df[best_feature] <= thresh]
        right = df[df[best_feature] >  thresh]
        if len(left) == 0 or len(right) == 0:
            return labels.mode()[0]

        return {
            best_feature: {
                "__numeric__": True,
                "threshold": thresh,
                "<=": build_tree(left,  remaining, label_column, max_depth, depth + 1, min_samples_split),
                ">":  build_tree(right, remaining, label_column, max_depth, depth + 1, min_samples_split),
            }
        }
    else:
        tree = {best_feature: {"__numeric__": False}}
        for val in df[best_feature].unique():
            subset = df[df[best_feature] == val]
            tree[best_feature][val] = build_tree(
                subset, remaining, label_column, max_depth, depth + 1, min_samples_split
            )
        return tree


def predict_one(tree, sample):
    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))
    branch = tree[feature]
    val = sample.get(feature)

    if branch.get("__numeric__"):
        if val is None or pd.isna(val):
            # Missing value: walk both and vote
            l = predict_one(branch["<="], sample)
            r = predict_one(branch[">"], sample)
            return l
        next_node = branch["<="] if val <= branch["threshold"] else branch[">"]
        return predict_one(next_node, sample)
    else:
        options = {k: v for k, v in branch.items() if k != "__numeric__"}
        if val not in options:
            leaves = [predict_one(child, sample) for child in options.values()]
            return Counter(leaves).most_common(1)[0][0]
        return predict_one(options[val], sample)


def predict_proba_one(tree, sample, classes):
    """Returns class -> probability dict by walking down to leaf."""
    leaf = predict_one(tree, sample)
    return {c: (1.0 if c == leaf else 0.0) for c in classes}


def predict_tree(tree, df):
    return df.apply(lambda row: predict_one(tree, row.to_dict()), axis=1)


def bootstrap_sample_balanced(df, label_column):
    counts = df[label_column].value_counts()
    minority_count = counts.min()

    parts = []
    for cls in counts.index:
        cls_df = df[df[label_column] == cls]
        parts.append(cls_df.sample(n=minority_count, replace=True,
                                   random_state=np.random.randint(0, 10_000)))
    return pd.concat(parts).sample(frac=1).reset_index(drop=True)


def bootstrap_sample_plain(df):
    return df.sample(n=len(df), replace=True, random_state=np.random.randint(0, 10_000))

#  METRICS
def classification_report(actual, preds, pos_label="Yes"):
    actual = np.array(actual)
    preds  = np.array(preds)

    tp = np.sum((preds == pos_label) & (actual == pos_label))
    fp = np.sum((preds == pos_label) & (actual != pos_label))
    fn = np.sum((preds != pos_label) & (actual == pos_label))
    tn = np.sum((preds != pos_label) & (actual != pos_label))

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (tp + tn) / len(actual)

    print(f"\n{'─'*35}")
    print(f"  Confusion Matrix")
    print(f"{'─'*35}")
    print(f"  {'':12s}  Pred Yes  Pred No")
    print(f"  {'Actual Yes':12s}  {tp:>8}  {fn:>7}")
    print(f"  {'Actual No':12s}  {fp:>8}  {tn:>7}")
    print(f"{'─'*35}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"{'─'*35}\n")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

#  RANDOM FOREST
def random_feature_subset(features, n_features=None):
    if n_features is None:
        n_features = max(1, int(np.sqrt(len(features))))
    return list(np.random.choice(features, size=n_features, replace=False))


class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, n_features=None,
                 min_samples_split=5, balance_classes=True, oob_score=True):
        """
        n_trees           : number of trees — 100+ gives smoother, more stable votes
        max_depth         : caps overfitting; tune via validation
        n_features        : features tried per split (None = sqrt heuristic)
        min_samples_split : minimum rows required to split a node
        balance_classes   : balanced bootstrap to counter class imbalance
        oob_score         : compute out-of-bag accuracy as free validation
                             (no need to hold out a separate val set)
        """
        self.n_trees           = n_trees
        self.max_depth         = max_depth
        self.n_features        = n_features
        self.min_samples_split = min_samples_split
        self.balance_classes   = balance_classes
        self.oob_score_enabled = oob_score
        self.trees, self.feature_cols, self.oob_indices = [], [], []

    def fit(self, df, label_column):
        self.label_column = label_column
        self.classes_ = sorted(df[label_column].unique())
        all_features = [c for c in df.columns if c != label_column]
        self.trees, self.feature_cols, self.oob_indices = [], [], []

        df = df.reset_index(drop=True)

        for _ in range(self.n_trees):
            if self.balance_classes:
                sample = bootstrap_sample_balanced(df, label_column)
                used_idx = set()  # balanced sampling breaks simple OOB tracking; skip
            else:
                sample = bootstrap_sample_plain(df)
                used_idx = set(sample.index) if sample.index.is_unique else set()

            oob_idx = list(set(df.index) - used_idx) if not self.balance_classes else []

            features = random_feature_subset(all_features, self.n_features)
            tree = build_tree(sample.reset_index(drop=True), features, label_column,
                              self.max_depth, min_samples_split=self.min_samples_split)

            self.trees.append(tree)
            self.feature_cols.append(features)
            self.oob_indices.append(oob_idx)

        if self.oob_score_enabled and not self.balance_classes:
            self._compute_oob_score(df)

        return self

    def _compute_oob_score(self, df):
        """Free validation: for each row, predict only using trees that
        did NOT see it during training (out-of-bag), then check accuracy."""
        votes = {i: [] for i in df.index}

        for tree, oob_idx in zip(self.trees, self.oob_indices):
            if not oob_idx:
                continue
            oob_df = df.loc[oob_idx]
            preds = predict_tree(tree, oob_df)
            for i, p in zip(oob_idx, preds):
                votes[i].append(p)

        correct, total = 0, 0
        for i, v in votes.items():
            if not v:
                continue
            pred = Counter(v).most_common(1)[0][0]
            total += 1
            correct += int(pred == df.loc[i, self.label_column])

        self.oob_score_ = correct / total if total else None

    def predict_proba(self, df):
        """Soft voting: average vote fraction per class across all trees.
        More informative than hard labels — lets you tune the decision threshold."""
        df = df.reset_index(drop=True)
        all_preds = np.array([
            predict_tree(tree, df).to_numpy(dtype=object)
            for tree in self.trees
        ])  # shape (n_trees, n_samples)

        n_samples = all_preds.shape[1]
        proba = np.zeros((n_samples, len(self.classes_)))

        for j, cls in enumerate(self.classes_):
            proba[:, j] = (all_preds == cls).mean(axis=0)

        return pd.DataFrame(proba, columns=self.classes_)

    def predict(self, df, threshold=None, pos_label=None):
        """
        threshold + pos_label: optional — flag pos_label if its probability
        exceeds threshold (e.g. threshold=0.35 to boost recall on minority class).
        Default behavior (no threshold) = standard majority vote.
        """
        proba = self.predict_proba(df)

        if threshold is not None and pos_label is not None:
            other_label = [c for c in self.classes_ if c != pos_label][0]
            return np.where(proba[pos_label] >= threshold, pos_label, other_label)

        return proba.idxmax(axis=1).to_numpy(dtype=object)

    def score(self, df, label_column=None):
        label_column = label_column or self.label_column
        preds  = self.predict(df)
        actual = df[label_column].to_numpy(dtype=object)
        return np.mean(preds == actual)

    def evaluate(self, df, pos_label="Yes", threshold=None):
        preds  = self.predict(df, threshold=threshold, pos_label=pos_label if threshold else None)
        actual = df[self.label_column].to_numpy(dtype=object)
        return classification_report(actual, preds, pos_label=pos_label)

    def feature_importance(self, df):
        all_features = [c for c in df.columns if c != self.label_column]
        importance = {f: [] for f in all_features}

        for features in self.feature_cols:
            gains = {f: calculate_information_gain(df, f, self.label_column) for f in features}
            for f, g in gains.items():
                importance[f].append(g)

        return {f: np.mean(vals) if vals else 0.0 for f, vals in importance.items()}

def train_test_split(df, test_size=0.2, random_state=42):
    shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split = int(len(shuffled) * (1 - test_size))
    return shuffled.iloc[:split], shuffled.iloc[split:]

if __name__ == "__main__":
    np.random.seed(42)
    n = 1000

    df = pd.read_csv('/content/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    df = df[
        [
            'Contract',
            'InternetService',
            'PaymentMethod',
            'Partner',
            'SeniorCitizen',
            'Dependents',
            'tenure',
            'MultipleLines',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies',
            'PaperlessBilling',
            'MonthlyCharges',
            'Churn'
        ]
    ]

    train_df, test_df = train_test_split(df, test_size=0.2)
    print(f"Train size : {len(train_df)}  |  Test size: {len(test_df)}")
    print(f"Class dist : {df['Churn'].value_counts().to_dict()}\n")

    rf = RandomForest(n_trees=100, max_depth=10, min_samples_split=5,
                      balance_classes=True, oob_score=False)
    rf.fit(train_df, label_column="Churn")

    print("── Default threshold (0.5) ──")
    rf.evaluate(test_df, pos_label="Yes")

    print("Feature Importances:")
    for feat, imp in sorted(rf.feature_importance(train_df).items(), key=lambda x: -x[1]):
        print(f"  {feat:20s}: {imp:.4f}")