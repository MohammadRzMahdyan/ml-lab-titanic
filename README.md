# Titanic XGBoost – Quick Look

* **دیتاست:** `data/Titanic-Dataset.csv`
* **ترنسفورمرها:** تو `transformers/custom_transformers.py` (PClass, Name, Sex, Sibsp/Parch, Ticket, Fare, Age, Embarked)
* **نوت‌بوک‌ها:**

  * EDA اولیه و pipeline basics
  * `xgb_final_pipeline.ipynb` → مدل نهایی
* **مدل:** `outputs/models/titanic_xgb_model.pkl` (XGBoost با preprocessing)
* **نمودارها:** `outputs/figures/learning_curve.png`
* **چی می‌بینی:** همه preprocessing‌ها و مدل تو یه pipeline جمع شده، ROC-AUC تست ~0.87، آماده استفاده.

