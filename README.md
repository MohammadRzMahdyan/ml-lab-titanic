# Titanic Preprocessing Pipeline

فولدر  آماده‌سازی دیتاست تایتانیک. هدف آماده سازی دیتاست اولیه برا استخراج داده های بهتر  .


```
├── data
│   └── Titanic-Dataset.csv
├── notebooks
│   ├── eda_pipeline_intro.ipynb
│   └── titanic_pipeline_basics.ipynb
├── requirements.txt
└── transformers
    └── custom_transformers.py
```

### در هر فایل:

1. **custom_transformer.py**  
   - کلاس‌های Transformer شخصی سازی شده:
     - `PClassEncoder` – کدگذاری کلاس مسافر  
     - `NameExtractor` – گرفتن عنوان و اسم خانوادگی  
     - `SexEncoder` – کدگذاری جنسیت  
     - `SibspBinning` و `ParchBinning` – دسته‌بندی اندازه خانواده  
     - `TicketExtractorAdvanced` – گرفتن پیشوند و شماره بلیط  
     - `FareBinning` – دسته‌بندی قیمت بلیط  
     - `AgeImmputer` – پر کردن سن‌های خالی با میانگین  
     - `EmbarkedEncoder` – پر کردن بندرهای خالی و تبدیل به باینری  

2. **eda_pipeline_intro.ipynb**  
   - تحلیل فیچرها:
     - بررسی توزیع و نرخ بقا بر اساس `pclass`, `sex`, `sibsp`, `parch`, `fare`, `age`  
     - جدول و نمودارهای ویولین  
     - دسته‌بندی فیچرها و آماده کردن برای pipeline  

## روند کار با داده‌ها

1. **بارگذاری داده‌ها**
```python
import pandas as pd
data = pd.read_csv('data/Titanic-Dataset.csv')
data.columns = data.columns.str.lower()
X = data.drop(columns=['survived'])
y = data['survived']
```

2. **تقسیم به train و test**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

3. **استفاده از ColumnTransformer با ترنسفورمرهای شخصی سازه شده**
```python
from sklearn.compose import ColumnTransformer
from transformers.custom_transformer import *

preprocessor = ColumnTransformer(
    transformers=[
        ('pclass', PClassEncoder(), ['pclass']),
        ('name', NameExtractor(), ['name']),
        ('sex', SexEncoder(), ['sex']),
        ('sibsp', SibspBinning(), ['sibsp']),
        ('parch', ParchBinning(), ['parch']),
        ('ticket', TicketExtractorAdvanced(top_k=5), ['ticket']),
        ('fare', FareBinning(method='quantile', q=4), ['fare']),
        ('age', AgeImmputer(), ['age']),
        ('embarked', EmbarkedEncoder(), ['embarked']),
    ],
    remainder='drop'
)

preprocessor.fit(X_train)
X_train_preprocessed = preprocessor.transform(X_train)
X_test_preprocessed  = preprocessor.transform(X_test)
```

