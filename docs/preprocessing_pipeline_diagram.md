# Data Preprocessing and Feature Engineering Pipeline

```mermaid
flowchart TD
    A["Raw CSV train-data.csv"] --> B["Load and Parse 14 Columns"]

    B --> C["Deterministic Cleaning Function"]

    C --> D["Parse Numeric Values Remove Commas"]
    D --> E["New Price Unit Fix Times 100"]
    E --> F["Invalid to NaN Seats Mileage"]
    F --> G["Drop Missing Target dropna Price"]
    G --> H["Fill Categorical Unknown"]

    H --> I["Feature Engineering add features"]

    I --> J1["Technique One vehicle age"]
    I --> J2["Technique Two mileage per year"]
    I --> J3["Technique Three power per cc"]
    I --> J4["Technique Four owner rank ordinal"]
    I --> J5["Technique Five age mileage interaction"]

    J1 --> K["Engineered Dataset 28 Features"]
    J2 --> K
    J3 --> K
    J4 --> K
    J5 --> K

    K --> L["Train Test Split 80 20"]

    L --> M1["Training Set 80 Percent"]
    L --> M2["Held out Test 20 Percent"]

    M1 --> N["ColumnTransformer Fit On Train"]

    N --> N1["Numeric Path SimpleImputer Median"]
    N --> N2["Categorical Path OneHot Encode"]

    N1 --> O["Preprocessed Features"]
    N2 --> O

    O --> P1["Model Training log1p Target"]
    P1 --> Q["Ridge With StandardScaler"]
    P1 --> R["Random Forest"]
    P1 --> S["Gradient Boosting"]

    Q --> T["Five Fold CV Compare MAE"]
    R --> T
    S --> T

    T --> U["Best Model Gradient Boosting"]
    U --> V["Randomized Search CV"]
    V --> W["Refit On Full 80 Percent"]

    W --> X["Evaluate Held Out 20 Percent"]

    M2 --> X

    W --> Y["Save Pipeline joblib"]

    Y --> Z["FastAPI Inference"]

    style A fill:#e1f5fe
    style H fill:#fff3e0
    style I fill:#f3e5f5
    style J1 fill:#fce4ec
    style J2 fill:#fce4ec
    style J3 fill:#fce4ec
    style J4 fill:#fce4ec
    style J5 fill:#fce4ec
    style K fill:#e8f5e9
    style L fill:#fff8e1
    style N fill:#e3f2fd
    style O fill:#f1f8e9
    style P1 fill:#fff3e0
    style T fill:#fce4ec
    style U fill:#e8f5e9
    style V fill:#fff8e1
    style W fill:#e8f5e9
    style X fill:#fce4ec
    style Y fill:#e1f5fe
    style Z fill:#f3e5f5
```

## Key Stages Summary

| Stage | Description | Leakage Prevention |
|-------|-------------|-------------------|
| **1. Raw Load** | Read CSV, assign 14 canonical columns | - |
| **2. Deterministic Cleaning** | Parse numbers, fix units, invalid to NaN, drop missing target, fill categoricals | No fitted statistics |
| **3. Feature Engineering** | 5 derived features (vehicle_age, mileage_per_year, power_per_cc, owner_rank, age_mileage_interaction) | Pure row-wise math, no data fitting |
| **4. Train/Test Split** | 80/20 split BEFORE any statistical fitting | Held-out set untouched until final eval |
| **5. Preprocessing Pipeline** | Median imputation (numeric), most-frequent plus one-hot (categorical) | Imputer/encoder fit ONLY on 80% training fold |
| **6. Model Training** | Ridge, RandomForest, GradientBoosting on log1p(Price) | 5-fold CV on training fold only |
| **7. Hyperparameter Tuning** | RandomizedSearchCV on best candidate | CV inside training fold only |
| **8. Final Evaluation** | Single evaluation on held-out 20% | Test set never used during fitting |
| **9. Serialization** | joblib pipeline plus metadata for FastAPI | Feature engineering applied before pipeline at inference |