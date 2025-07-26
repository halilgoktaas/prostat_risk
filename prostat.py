import  matplotlib
from IPython.core.pylabtools import figsize
from docutils.nodes import legend
from pyspark.ml.connect.classification import LogisticRegression

matplotlib.use('TkAgg')
import  pandas
import  numpy
import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  confusion_matrix, classification_report

## veri setini tanıma ve ön işleme #####
df = pd.read_csv('data/synthetic_prostate_cancer_risk.csv')
df.info()
df.shape
df.describe()
df.columns
df.isnull().sum()
df['alcohol_consumption'].isnull().sum()
df[df.isnull().any(axis = 1)].sum()
df.isnull().mean().sort_values(ascending=False)
df['alcohol_consumption']
df['alcohol_consumption'].value_counts(dropna=False)
df['alcohol_consumption'].fillna('unknown', inplace = True)
df.isnull().values.any()

df.select_dtypes(include='object').nunique()
df.select_dtypes(include='number').hist(figsize=(12,6))

## Görselleştirme ##
sns.countplot(data = df, x='smoker', hue='risk_level')
plt.title('Sigara kullanımına göre risk dağılımı')
plt.xlabel('sigara kullanım')
plt.ylabel('risk seviyesi')
plt.xticks(rotation = 45)
plt.legend(title= 'risk seviyesi')
plt.show()

sns.countplot(data = df, x='diet_type', hue= 'risk_level')
plt.title('Diyet Kalitesine göre Risk Dağılımı')
plt.xlabel('Diyet Kalitesi')
plt.ylabel('Risk seviyesi')
plt.tight_layout()
plt.show()

sns.countplot(data =df, x='family_history', hue='risk_level')
plt.title('Aile geçmişine göre risk dağılımı')
plt.xlabel('Aile Geçmişi')
plt.ylabel('Risk seviyesi')
plt.xticks(rotation = 0)
plt.tight_layout()
plt.show()

sns.countplot(data =df, x='alcohol_consumption', hue='risk_level')
plt.title('Alkol tüketimine göre risk dağılımı')
plt.xticks(rotation = 0)
plt.tight_layout()
plt.show()

sns.boxplot(data =df, x='risk_level', y='age')
plt.title('Yaş arttıkça risk artıyor mu')
plt.xlabel('Yaş')
plt.ylabel('risk')
plt.tight_layout()
plt.show()

sns.boxplot(data=df, x='risk_level', y='bmi')
plt.tight_layout()
plt.show()

sns.boxplot(data=df, x='risk_level',y='sleep_hours')
plt.tight_layout()
plt.show()

# tüm sayısal değişkenler tek grafik ###
fig, axes = plt.subplots(3,1, figsize = (12,6))
sns.boxplot(data = df, x = 'risk_level', y='age', ax = axes[0], palette= 'Set1')
axes[0].set_title('Yaşa göre Risk analizi')

sns.boxplot(data = df, x= 'risk_level', y='sleep_hours', ax = axes[1], palette='Set2')
axes[1].set_title('Uyku süresine göre Risk Analizi')

sns.boxplot(data = df, x='risk_level', y='bmi', ax = axes[2], palette ='Set3')
axes[2].set_title('Boy kilo endeksine göre risk analizi')

plt.tight_layout()
plt.show()

## Modelleme ##
le = LabelEncoder()
df['risk_level_encoded'] = le.fit_transform(df['risk_level'])


kategorik_veriler = df.select_dtypes(include = 'object').columns.drop('risk_level')

for col in kategorik_veriler:
    df[col + '_enc'] =  LabelEncoder().fit_transform(df[col])

ozellik_secimi = [
    'age', 'bmi', 'sleep_hours', 'smoker_enc', 'alcohol_consumption_enc', 'diet_type_enc', 'physical_activity_level_enc',
    'family_history_enc', 'mental_stress_level_enc', 'regular_health_checkup_enc', 'prostate_exam_done_enc'
]

X = df[ozellik_secimi]
y = df['risk_level_encoded']

## eğitim ve test ##
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(max_iter=1000,class_weight = 'balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)





