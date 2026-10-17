"""
Loading, cleaning, feature construction, and deterministic data splits.

The source CSV is a joined bina.az export.  The cleaning step keeps
listing-time property characteristics, removes join/web metadata, and removes
columns that are derived from the regression target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/house_sale.csv"

# Columns that LEAK the target (derived from price). Drop these before training.
# total_price is a duplicate of price. unit_price is total_price/area, so either column leaks the target into regression
LEAKAGE_COLUMNS: list[str] = [
    "unit_price", 
    "total_price"
]


# Identifiers, URLs, free text, scrape metadata, and duplicated join fields are
# not stable and contribute almost no valaue, therefore they would create noisy/high-cardinality features
NON_FEATURE_COLUMNS: list[str] = [
    "id_x", "id_y", "estate_id", "estate_details_id_x", "estate_details_id_y",
    "rel_url", "estate_rel_url_x", "estate_rel_url_y", "estate_rel_url",
    "datetime_scrape_x", "datetime_scrape_y", "day_x", "day_y", "hour_x",
    "hour_y", "currency_x", "currency_y", "img_url", "description",
    "owner_name", "owner_title", "shop_name", "shop_title", "address",
    "updated", "extra_info", "attributes", "city_when", "views",
    # English/second-table aliases of the Azerbaijani property fields kept
    # below, plus their raw unit-formatted source columns.
    "repair", "bill_of_sale", "mortgage", "Mərtəbə", "Sahə", "Torpaq sahəsi",
    "Binanın növü",
]

AZERBAIJANI_RENAME = {
    "Otaq sayı": "rooms",
    "Təmir": "renovation",
    "Çıxarış": "deed_status",
    "İpoteka": "mortgage_status",
    "Kateqoriya": "category",
}

APARTMENT_CATEGORIES = {"Köhnə tikili", "Yeni tikili"}


def _number_from_text(series: pd.Series) -> pd.Series:
    """Extract a decimal number from values such as ``145 m²``."""
    return pd.to_numeric(
        series.astype("string").str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )


def load_raw(path: str = DATA_PATH):
    """Read the raw CSV into a DataFrame (keep UTF-8). Return it unmodified."""
    return pd.read_csv(path, encoding="utf-8")


def clean(df):
    """
    Clean the raw frame for price regression and classification.

    Numeric fields are parsed from their Azerbaijani unit-formatted strings.
    Missing floor indicators are retained; missing apartment floors are filled
    with the median floor within the apartment category. Non-apartment floor
    fields are encoded as zero (an explicit not-applicable sentinel), with
    their missing indicators preserved. Other predictor missing values are
    handled in ``make_features``. Only rows without a valid positive target
    are removed.
    Exact duplicate rows are removed.  Price outliers are removed using
    1.5*IQR rule, which is less affected by this right-skewed target
    than a mean/std rule.  Target leakage and web/join metadata are dropped.
    """
    cleaned = df.copy()
    if "price" not in cleaned:
        raise KeyError("The input frame must contain the regression target 'price'.")

    cleaned["price"] = _number_from_text(cleaned["price"])
    for currency_column in ("currency_x", "currency_y"):
        if currency_column in cleaned:
            currencies = cleaned[currency_column].dropna().astype("string").str.upper()
            if not currencies.empty and not currencies.eq("AZN").all():
                raise ValueError(f"Unsupported currency values in {currency_column}.")
    if "Sahə" in cleaned:
        cleaned["area_m2"] = _number_from_text(cleaned["Sahə"])
    if "Torpaq sahəsi" in cleaned:
        cleaned["land_area_sot"] = _number_from_text(cleaned["Torpaq sahəsi"])
    if "Otaq sayı" in cleaned:
        cleaned["Otaq sayı"] = pd.to_numeric(cleaned["Otaq sayı"], errors="coerce")
    if "Mərtəbə" in cleaned:
        floors = cleaned["Mərtəbə"].astype("string").str.extract(
            r"^\s*(\d+)\s*/\s*(\d+)\s*$"
        )
        cleaned["floor"] = pd.to_numeric(floors[0], errors="coerce")
        cleaned["total_floors"] = pd.to_numeric(floors[1], errors="coerce")
        # Missing floors are expected for houses, land, garages, and other
        # non-apartment listings. Preserve that fact explicitly. For apartment
        # listings, use the median floor of the same apartment category.
        cleaned["floor_missing"] = cleaned["floor"].isna().astype(float)
        cleaned["total_floors_missing"] = cleaned["total_floors"].isna().astype(float)
        if "Kateqoriya" in cleaned:
            is_apartment = cleaned["Kateqoriya"].isin(APARTMENT_CATEGORIES)
            for column in ("floor", "total_floors"):
                category_median = cleaned.loc[is_apartment].groupby(
                    cleaned.loc[is_apartment, "Kateqoriya"]
                )[column].transform("median")
                cleaned.loc[is_apartment, column] = (
                    cleaned.loc[is_apartment, column].fillna(category_median)
                )
                # Zero is outside the valid apartment range and means
                # "not applicable" for houses, land, garages, and other
                # non-apartment categories. The indicator carries that fact.
                cleaned.loc[~is_apartment, column] = (
                    cleaned.loc[~is_apartment, column].fillna(0)
                )

    cleaned = cleaned.dropna(subset=["price"])
    cleaned = cleaned[cleaned["price"] > 0].drop_duplicates()

    q1, q3 = cleaned["price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    if pd.notna(iqr) and iqr > 0:
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        cleaned = cleaned[cleaned["price"].between(lower, upper)]

    columns_to_drop = set(LEAKAGE_COLUMNS + NON_FEATURE_COLUMNS)
    cleaned = cleaned.drop(
        columns=[column for column in columns_to_drop if column in cleaned],
        errors="ignore",
    )
    return cleaned.rename(columns=AZERBAIJANI_RENAME).reset_index(drop=True)


def make_features(df):
    """
    Build a numeric design matrix and return ``(X, y, feature_names)``.

    Nominal columns are one-hot encoded, with a separate missing-value
    category. Numeric missing values are median-imputed using this frame.
    ``unit_price`` and ``total_price`` are removed again defensively.
    """
    processor, X, y = fit_feature_processor(df)
    return X, y, processor.feature_names


@dataclass
class FeatureProcessor:
    """Training-fitted encoding and imputation metadata."""

    categorical_columns: list[str]
    binary_categorical: list[str]
    multi_categorical: list[str]
    feature_names: list[str]
    numeric_medians: pd.Series


def _model_frame_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "price" not in df:
        raise KeyError("The cleaned frame must contain the regression target 'price'.")

    model_frame = df.drop(
        columns=[column for column in LEAKAGE_COLUMNS if column in df],
        errors="ignore",
    ).copy()
    y = pd.to_numeric(model_frame.pop("price"), errors="coerce")
    valid_rows = y.notna() & (y > 0)
    return model_frame.loc[valid_rows].copy(), y.loc[valid_rows].astype(float)


def _encode_frame(
    model_frame: pd.DataFrame,
    processor: FeatureProcessor,
) -> pd.DataFrame:
    frame = model_frame.copy()
    frame = frame.drop(
        columns=[column for column in frame.columns if frame[column].isna().all()],
        errors="ignore",
    )
    categorical = [
        column for column in processor.categorical_columns if column in frame
    ]
    numeric = frame.drop(columns=categorical, errors="ignore")
    encoded_parts = [numeric]

    for columns, drop_first in (
        (processor.binary_categorical, True),
        (processor.multi_categorical, False),
    ):
        columns = [column for column in columns if column in frame]
        if columns:
            encoded_parts.append(pd.get_dummies(
                frame[columns],
                columns=columns,
                drop_first=drop_first,
                dummy_na=True,
                dtype=float,
            ))

    X = pd.concat(encoded_parts, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    if processor.feature_names:
        X = X.reindex(columns=processor.feature_names, fill_value=0.0)
        for column, median in processor.numeric_medians.items():
            if column in X:
                X[column] = X[column].fillna(median)
    return X.astype(float)


def fit_feature_processor(
    df: pd.DataFrame,
) -> tuple[FeatureProcessor, pd.DataFrame, pd.Series]:
    """Fit feature rules on one frame, normally the training frame only."""
    model_frame, y = _model_frame_and_target(df)
    categorical = model_frame.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    binary = [
        column for column in categorical
        if model_frame[column].dropna().nunique() <= 2
    ]
    multi = [column for column in categorical if column not in binary]
    provisional = FeatureProcessor(
        categorical, binary, multi, [], pd.Series(dtype=float)
    )
    encoded = _encode_frame(model_frame, provisional)
    encoded = encoded.drop(columns=encoded.columns[encoded.nunique(dropna=False) <= 1])
    medians = encoded.median(numeric_only=True)
    feature_names = encoded.columns.tolist()
    processor = FeatureProcessor(categorical, binary, multi, feature_names, medians)
    X = _encode_frame(model_frame, processor)
    return processor, X, y


def transform_features(
    df: pd.DataFrame,
    processor: FeatureProcessor,
) -> tuple[pd.DataFrame, pd.Series]:
    """Transform another frame using processor rules fitted on training data."""
    model_frame, y = _model_frame_and_target(df)
    return _encode_frame(model_frame, processor), y


def make_tier_label(y_price, threshold=None):
    """
    Create the binary price-tier label.

    When threshold is None, calculate it as the median of the supplied
    prices. For leakage-free evaluation, call this first with y_train only,
    then pass the returned threshold when creating validation and test labels.

    premium = 1 when price > threshold
    standard = 0 otherwise

    Return:
        y_tier: binary pandas Series
        threshold: the median or supplied threshold
    """
    prices = pd.Series(y_price)
    threshold = float(prices.median()) if threshold is None else float(threshold)
    return (prices > threshold).astype(int), threshold


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED, stratify=None):
    """
    Deterministic split into train / validation / test.
    Return (X_tr, y_tr, X_val, y_val, X_te, y_te).
    """
    if not 0 < val_size < 1 or not 0 < test_size < 1:
        raise ValueError("val_size and test_size must be between 0 and 1.")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    n_test = int(round(len(X) * test_size))
    n_val = int(round(len(X) * val_size))
    rng = np.random.default_rng(seed)
    if stratify is None:
        indices = rng.permutation(len(X))
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]
    else:
        groups = np.asarray(stratify)
        if len(groups) != len(X):
            raise ValueError("stratify must contain one label per row.")
        test_parts, val_parts, train_parts = [], [], []
        for group in np.unique(groups):
            group_indices = np.flatnonzero(groups == group)
            group_indices = rng.permutation(group_indices)
            group_test = int(round(len(group_indices) * test_size))
            group_val = int(round(len(group_indices) * val_size))
            test_parts.append(group_indices[:group_test])
            val_parts.append(group_indices[group_test:group_test + group_val])
            train_parts.append(group_indices[group_test + group_val:])
        test_idx = rng.permutation(np.concatenate(test_parts))[:n_test]
        val_idx = rng.permutation(np.concatenate(val_parts))[:n_val]
        train_idx = rng.permutation(np.concatenate(train_parts))

    def select(values, index):
        return values.iloc[index] if hasattr(values, "iloc") else values[index]

    return (
        select(X, train_idx), select(y, train_idx),
        select(X, val_idx), select(y, val_idx),
        select(X, test_idx), select(y, test_idx),
    )


def standardize(X_tr, *others):
    """
    Standardize features using TRAIN statistics only (mean/std from X_tr),
    then apply the same transform to the other splits.
    Return the scaled arrays in the same order as given.
    (Needed for the SVM; trees don't require it.)
    """
    train = np.asarray(X_tr, dtype=float)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0

    def transform(values):
        return (np.asarray(values, dtype=float) - mean) / std

    return (transform(X_tr), *(transform(values) for values in others))
