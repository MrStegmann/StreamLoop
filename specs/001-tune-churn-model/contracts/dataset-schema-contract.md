# Data Schema Contract: Telco Customer Churn Ingestion

**Branch**: `001-tune-churn-model`  
**Contract Type**: Tabular Ingestion Schema  
**Dataset**: IBM Telco Customer Churn  

---

## Column Specification

| Column | Data Type | Nullable | Domain / Enum / Constraints | Description |
|---|---|---|---|---|
| `customerID` | string | No | Non-empty alphanumeric string (e.g. `7590-VHVEG`) | Subscriber ID. Excluded from feature matrix $X$. |
| `gender` | string | No | `{"Female", "Male"}` | Subscriber gender. |
| `SeniorCitizen` | integer | No | `{0, 1}` | Binary flag indicating if subscriber is senior citizen. |
| `Partner` | string | No | `{"Yes", "No"}` | Subscriber has a partner. |
| `Dependents` | string | No | `{"Yes", "No"}` | Subscriber has dependents. |
| `tenure` | integer | No | Integer $\ge 0$ | Number of months subscriber has stayed with company. |
| `PhoneService` | string | No | `{"Yes", "No"}` | Subscriber has phone service. |
| `MultipleLines` | string | No | `{"Yes", "No", "No phone service"}` | Subscriber has multiple phone lines. |
| `InternetService` | string | No | `{"DSL", "Fiber optic", "No"}` | Subscriber internet service provider. |
| `OnlineSecurity` | string | No | `{"Yes", "No", "No internet service"}` | Subscriber has online security add-on. |
| `OnlineBackup` | string | No | `{"Yes", "No", "No internet service"}` | Subscriber has online backup add-on. |
| `DeviceProtection` | string | No | `{"Yes", "No", "No internet service"}` | Subscriber has device protection add-on. |
| `TechSupport` | string | No | `{"Yes", "No", "No internet service"}` | Subscriber has technical support add-on. |
| `StreamingTV` | string | No | `{"Yes", "No", "No internet service"}` | Subscriber has streaming TV add-on. |
| `StreamingMovies` | string | No | `{"Yes", "No", "No internet service"}` | Subscriber has streaming movies add-on. |
| `Contract` | string | No | `{"Month-to-month", "One year", "Two year"}` | Contract terms. |
| `PaperlessBilling` | string | No | `{"Yes", "No"}` | Subscriber enrolled in paperless billing. |
| `PaymentMethod` | string | No | `{"Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"}` | Payment method. |
| `MonthlyCharges` | float | No | Float $> 0$ | Amount charged monthly. |
| `TotalCharges` | string / float | Yes* | Float $\ge 0$ or whitespace `" "` | Total amount charged. Whitespace `" "` occurs when `tenure == 0` (11 rows). Coerced to `0.0`. |
| `Churn` | string | No | `{"Yes", "No"}` | Target label. Coerced to binary int (`1` for "Yes", `0` for "No"). |

---

## Ingestion Validation Rules

1. **Row Count**: Raw dataset contains exactly 7,043 rows. Ingestion MUST verify $N > 1000$.
2. **Column Presence**: Ingestion MUST verify all 21 columns exist in the incoming DataFrame.
3. **Missing Value Coercion**:
   - `TotalCharges`: Whitespace string `" "` MUST be converted to `0.0` or numeric float without dropping records.
   - Any unanticipated NaN values in numerical columns MUST be imputed with the training median inside the pipeline.
   - Any unanticipated unseen categories in categorical columns MUST be ignored by `OneHotEncoder(handle_unknown='ignore')`.
4. **Stratification Constraint**: Target distribution has approximately 26.5% positive churn cases. Train/test split MUST preserve this exact ratio across partitions using `stratify=y`.
