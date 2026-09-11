# Behavioral Feature Catalog (Phase 3)

> **Platform:** AI-Powered Fraud Detection & Risk Intelligence Platform  
> **Source Schema:** Canonical 15-Field Transaction Schema  
> **Total Output Columns:** 62 (15 Canonical + 47 Engineered Behavioral Features)  
> **Point-in-Time Constraint:** Strict Inequality ($	ext{timestamp}_j < 	ext{timestamp}_i$)  
> **Target Leakage Protocol:** Zero usage of `is_fraud` labels in feature generation.  

---

## 1. Catalog Summary by Feature Group

| # | Feature Group | Feature Count | Primary Business / Risk Motivation | Historical Lookback |
|---|---|---|---|---|
| **1** | **Temporal** | 11 | Diurnal fraud surges (22:00–04:00) and weekly payment rhythms | Current timestamp ($t_i$) |
| **2** | **Velocity** | 7 | Rapid carding bursts & high-frequency transaction spikes | $1	ext{h}, 6	ext{h}, 24	ext{h}, 7	ext{d}, 30	ext{d}$ |
| **3** | **Spending** | 8 | Short-term balance depletion and high-dollar bursts | $1	ext{h}, 24	ext{h}, 7	ext{d}, 30	ext{d}$ |
| **4** | **Spending Deviation** | 5 | Normalizing individual account spend baselines ($Z$-scores & ratios) | Lifetime expanding ($t_j < t_i$) |
| **5** | **Account History** | 6 | Lifetime account maturity, total volume, and entity diversity | Lifetime expanding ($t_j < t_i$) |
| **6** | **Merchant Interaction** | 6 | Account familiarity with merchants/categories & global volume | Lifetime expanding ($t_j < t_i$) |
| **7** | **Geographic / Travel** | 4 | Impossible travel speeds between consecutive transactions | Immediate prior ($t_{	ext{prev}} < t_i$) |
| **Total** | **7 Groups** | **47 Features** | **Comprehensive Multi-Dimensional Fraud Defense** | **Zero-Leakage Point-in-Time** |

---

## 2. Exhaustive Feature Dictionary (47 Engineered Features)

### Group 1: Temporal Features (11 Features)

| Feature Name | Data Type | Formula / Definition | Cold-Start Policy | Leakage Considerations |
|---|---|---|---|---|
| `transaction_hour` | `int32` | $	ext{Hour of day } (0 \dots 23)$ | N/A (current transaction) | Derived solely from current timestamp. |
| `day_of_week` | `int32` | $0=	ext{Monday} \dots 6=	ext{Sunday}$ | N/A (current transaction) | Derived solely from current timestamp. |
| `day_of_month` | `int32` | Day of month ($1 \dots 31$) | N/A (current transaction) | Derived solely from current timestamp. |
| `month` | `int32` | Month ($1 \dots 12$) | N/A (current transaction) | Derived solely from current timestamp. |
| `week_of_year` | `int32` | ISO week number ($1 \dots 53$) | N/A (current transaction) | Derived solely from current timestamp. |
| `is_weekend` | `int8` | $1 	ext{ if } 	ext{day\_of\_week} \in [5, 6] 	ext{ else } 0$ | N/A (current transaction) | Zero lookahead. |
| `is_night` | `int8` | $1 	ext{ if } 	ext{hour} \in [22, 23, 0, 1, 2, 3, 4] 	ext{ else } 0$ | N/A (current transaction) | Captures empirical 4× nighttime fraud surge. |
| `hour_sin` | `float32` | $\sin(2\pi 	imes 	ext{hour} / 24.0)$ | N/A (current transaction) | Smooth cyclic boundary at midnight ($23 	o 0$). |
| `hour_cos` | `float32` | $\cos(2\pi 	imes 	ext{hour} / 24.0)$ | N/A (current transaction) | Smooth cyclic boundary at midnight. |
| `day_of_week_sin` | `float32` | $\sin(2\pi 	imes 	ext{dow} / 7.0)$ | N/A (current transaction) | Smooth cyclic boundary across weeks ($6 	o 0$). |
| `day_of_week_cos` | `float32` | $\cos(2\pi 	imes 	ext{dow} / 7.0)$ | N/A (current transaction) | Smooth cyclic boundary across weeks. |

---

### Group 2: Velocity Features (7 Features)

| Feature Name | Data Type | Group Key | Window | Cold-Start | Description |
|---|---|---|---|---|---|
| `txn_count_1h` | `int32` | `account_id` | $(t_i - 3600	ext{s}, t_i)$ | `0` | Count of strictly prior transactions for this account in the past 1 hour. |
| `txn_count_6h` | `int32` | `account_id` | $(t_i - 21600	ext{s}, t_i)$ | `0` | Count of strictly prior transactions for this account in the past 6 hours. |
| `txn_count_24h` | `int32` | `account_id` | $(t_i - 86400	ext{s}, t_i)$ | `0` | Count of strictly prior transactions for this account in the past 24 hours. |
| `txn_count_7d` | `int32` | `account_id` | $(t_i - 604800	ext{s}, t_i)$ | `0` | Count of strictly prior transactions for this account in the past 7 days. |
| `txn_count_30d` | `int32` | `account_id` | $(t_i - 2592000	ext{s}, t_i)$ | `0` | Count of strictly prior transactions for this account in the past 30 days. |
| `time_since_prev_txn_seconds` | `float32` | `account_id` | Strict previous | `0.0` | Elapsed seconds since immediately previous transaction with $t_j < t_i$. |
| `is_first_account_txn` | `int8` | `account_id` | Lifetime | `1` | Binary indicator ($1$ if account has 0 prior transactions, else $0$). |

---

### Group 3: Spending Features (8 Features)

| Feature Name | Data Type | Group Key | Window | Cold-Start | Description |
|---|---|---|---|---|---|
| `amt_sum_1h` | `float32` | `account_id` | $(t_i - 3600	ext{s}, t_i)$ | `0.0` | Sum of amounts for strictly prior transactions in past 1 hour. |
| `amt_sum_24h` | `float32` | `account_id` | $(t_i - 86400	ext{s}, t_i)$ | `0.0` | Sum of amounts for strictly prior transactions in past 24 hours. |
| `amt_sum_7d` | `float32` | `account_id` | $(t_i - 604800	ext{s}, t_i)$ | `0.0` | Sum of amounts for strictly prior transactions in past 7 days. |
| `amt_sum_30d` | `float32` | `account_id` | $(t_i - 2592000	ext{s}, t_i)$ | `0.0` | Sum of amounts for strictly prior transactions in past 30 days. |
| `amt_mean_24h` | `float32` | `account_id` | $(t_i - 86400	ext{s}, t_i)$ | `0.0` | Average amount of transactions in past 24h: `amt_sum_24h / txn_count_24h`. |
| `amt_mean_7d` | `float32` | `account_id` | $(t_i - 604800	ext{s}, t_i)$ | `0.0` | Average amount of transactions in past 7d: `amt_sum_7d / txn_count_7d`. |
| `amt_max_24h` | `float32` | `account_id` | $(t_i - 86400	ext{s}, t_i)$ | `0.0` | Maximum single transaction amount observed in past 24 hours. |
| `amt_median_30d` | `float32` | `account_id` | $(t_i - 2592000	ext{s}, t_i)$ | `0.0` | Median transaction amount observed in past 30 days. |

---

### Group 4: Spending Deviation Features (5 Features)

| Feature Name | Data Type | Formula | Cold-Start | Business Meaning |
|---|---|---|---|---|
| `historical_amount_mean` | `float32` | $\mu_{< t_i} = rac{1}{N}\sum_{j: t_j < t_i} A_j$ | `0.0` | Expanding lifetime average transaction amount for this cardholder. |
| `historical_amount_std` | `float32` | $\sigma_{< t_i} = \sqrt{rac{1}{N-1}\sum (A_j - \mu)^2}$ | `0.0` | Expanding sample standard deviation ($N \ge 2$). |
| `historical_amount_median` | `float32` | $	ext{Median}(\{A_j : t_j < t_i\})$ | `0.0` | Expanding median spend for this cardholder. |
| `amount_zscore` | `float32` | $(A_i - \mu_{< t_i}) / \sigma_{< t_i}$ | `0.0` | Standardized spending deviation score. Highly discriminative for fraud spikes. |
| `amount_ratio_to_historical_mean` | `float32` | $A_i / \mu_{< t_i}$ | `1.0` | Multiple of typical spend (neutral cold start = 1.0). |

---

### Group 5: Account History Features (6 Features)

| Feature Name | Data Type | Group Key | Cold-Start | Description |
|---|---|---|---|---|
| `account_txn_count_before` | `int32` | `account_id` | `0` | Total lifetime transaction count for account prior to $t_i$. |
| `account_total_spend_before` | `float32` | `account_id` | `0.0` | Total lifetime dollar volume transacted by account prior to $t_i$. |
| `account_avg_amount_before` | `float32` | `account_id` | `0.0` | Lifetime mean dollar amount spent by account prior to $t_i$. |
| `account_max_amount_before` | `float32` | `account_id` | `0.0` | Lifetime maximum transaction amount observed for account prior to $t_i$. |
| `account_unique_merchant_count_before` | `int32` | `account_id` | `0` | Count of distinct `merchant_id` entities transacted with prior to $t_i$. |
| `account_unique_category_count_before` | `int32` | `account_id` | `0` | Count of distinct `merchant_category` codes transacted in prior to $t_i$. |

---

### Group 6: Merchant / Category Interaction Features (6 Features)

| Feature Name | Data Type | Group Key | Cold-Start | Description |
|---|---|---|---|---|
| `account_merchant_txn_count_before` | `int32` | `(account_id, merchant_id)` | `0` | Number of times this account transacted at this specific merchant before $t_i$. |
| `account_category_txn_count_before` | `int32` | `(account_id, merchant_category)` | `0` | Number of times this account transacted in this category before $t_i$. |
| `account_merchant_spend_before` | `float32` | `(account_id, merchant_id)` | `0.0` | Total dollars spent by this account at this specific merchant before $t_i$. |
| `account_category_spend_before` | `float32` | `(account_id, merchant_category)` | `0.0` | Total dollars spent by this account in this category before $t_i$. |
| `merchant_txn_count_before` | `int32` | `merchant_id` | `0` | Global transaction volume at this merchant across all accounts before $t_i$. |
| `category_txn_count_before` | `int32` | `merchant_category` | `0` | Global transaction volume in this category across all accounts before $t_i$. |

---

### Group 7: Geographic / Travel Features (4 Features)

| Feature Name | Data Type | Definition | Cold-Start | Behavioral Threshold / Rules |
|---|---|---|---|---|
| `cardholder_merchant_distance_km` | `float32` | $	ext{Haversine}((	ext{card\_lat}, 	ext{card\_lon}), (	ext{merch\_lat}, 	ext{merch\_lon}))$ | Current row coords | Static distance in km. |
| `distance_from_prev_merchant_km` | `float32` | $	ext{Haversine}((	ext{prev\_merch\_lat}, 	ext{prev\_merch\_lon}), (	ext{curr\_merch\_lat}, 	ext{curr\_merch\_lon}))$ | `0.0` | Distance between consecutive merchant locations for same account ($t_{	ext{prev}} < t_i$). |
| `implied_travel_speed_kmh` | `float32` | $	ext{distance\_from\_prev\_merchant\_km} / (\Delta t / 3600.0)$ | `0.0` | Raw calculated speed in km/h. Stored uncapped whenever numerically safe. |
| `is_impossible_travel_speed` | `int8` | $1 	ext{ if } 	ext{implied\_travel\_speed\_kmh} > 800.0 	ext{ else } 0$ | `0` | Triggers when implied travel exceeds commercial aircraft speed (800 km/h). |

---

## 3. Canonical Base Columns (15 Columns)

Every output Parquet dataset retains all 15 canonical transaction fields from Phase 1:
1. `transaction_id` (str)
2. `account_id` (str)
3. `timestamp` (datetime64[ns])
4. `unix_time` (int64)
5. `amount` (float64)
6. `currency` (str)
7. `merchant_id` (str)
8. `merchant_category` (str)
9. `cardholder_lat` (float64)
10. `cardholder_long` (float64)
11. `merchant_lat` (float64)
12. `merchant_long` (float64)
13. `city_pop` (int64)
14. `job_category` (str)
15. `is_fraud` (int64)
