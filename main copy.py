import pandas as pd

df = pd.read_csv('train.csv')
#df.info()
df.dropna()
df.drop(['id','occupation_name', 'career_start', 'career_end', 'last_seen'], axis = 1, inplace = True)

def life_main(lm):
    if lm == 'False':
        return 0
    else:
        return int(lm)
df['life_main'] = df['life_main'].apply(life_main)

def bdate(d):
    d = str(d).split('.')
    if len(d) == 3:
        return int(d[2])
    else:
        return 1988
df['bdate'] = df['bdate'].apply(bdate)

def city(c):
    s_main = ['Moscow']
    if c in s_main:
        return 1
    else:
        return 0
df['city'] = df['city'].apply(city)


def occupation_type(t):
    if t == 'univercity':
        return 1
    else:
        return 0
df['occupation_type'] = df['occupation_type'].apply(occupation_type)

def education_form(ef):
    if ef == 'Full-time':
        return 1
    elif ef == 'Distance Learning':
        return 2
    elif ef == 'Part-time':
        return 3
    elif ef == 'External':
        return 4
    else:
        return 0
df['education_form'] = df['education_form'].apply(education_form)

def education_status(es):
    if es == 'Alumnus (Specialist)':
        return 1
    elif es == 'Student (Specialist)':
        return 2
    elif es == "Student (Bachelor's)":
        return 3
    elif es == "Alumnus (Bachelor's)":
        return 4
    elif es == "Alumnus (Master's)":
        return 5
    elif es == 'PhD':
        return 6
    elif es == "Student (Master's)":
        return 7
    elif es == "Undergraduate applicant":
        return 8
    elif es == "Candidate of Sciences":
        return 9
    else:
        return 0
df['education_status'] = df['education_status'].apply(education_status)

def people_main(pm):
    if pm == 'False':
        return 0
    else:
        return int(pm)
df['people_main'] = df['people_main'].apply(people_main)

def langs(l):
    return len(l.split(';'))
df['langs'] = df['langs'].apply(langs)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X = df.drop('result', axis = 1)
y = df['result']

sc = StandardScaler()
classifier = KNeighborsClassifier(n_neighbors = 9)

best = 0
best_classifier = None
for i in range(50):
    print('Попытка', i+1, 'из 50')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    res = accuracy_score(y_test, y_pred) * 100
    if best < res:
        best = res
        best_classifier = classifier
print('Лучший процент правильно предсказанных исходов:', best)






df2 = pd.read_csv('test.csv')
df2.drop(['occupation_name', 'career_start', 'career_end', 'last_seen'], axis = 1, inplace = True)

df2['life_main'] = df2['life_main'].apply(life_main)
df2['bdate'] = df2['bdate'].apply(bdate)
df2['city'] = df2['city'].apply(city)
df2['occupation_type'] = df2['occupation_type'].apply(occupation_type)
df2['education_status'] = df2['education_status'].apply(education_status)
df2['education_form'] = df2['education_form'].apply(education_form)
df2['langs'] = df2['langs'].apply(langs)
df2['people_main'] = df2['people_main'].apply(people_main)

df2.info()

x_test = df2.drop('id', axis = 1)
x_test = sc.transform(x_test)
y_pred = classifier.predict(x_test)

final_itog = pd.DataFrame({'ID' : df2['id'], 'result' : y_pred})
final_itog.to_csv('final_itog.csv', index = False)
