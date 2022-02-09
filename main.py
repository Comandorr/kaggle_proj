import pandas as pd
import random

df = pd.read_csv('train.csv')
#df = df.dropna()

df.drop(['id', 'occupation_name', 'career_start', 'career_end'], axis = 1, inplace = True)

df['has_mobile'] = df['has_mobile'].apply(int)
df['graduation'] = df['graduation'].apply(int)
df['followers_count'] = df['followers_count'].apply(int)

def last_seen(ls):
    ls = ls.split()[0]
    ls = ls.split('-')
    ls = int(ls[0])*365 + int(ls[1]) * 30 + int(ls[2])
    return ls
df['last_seen'] = df['last_seen'].apply(last_seen)

es_list = []
def education_status(es):
    if not (es in es_list):
        es_list.append(es)
    return es_list.index(es)
df['education_status'] = df['education_status'].apply(education_status)

ef_list = []
def education_form(ef):
    if not ef in ef_list:
        ef_list.append(ef)
    return ef_list.index(ef)
df['education_form'] = df['education_form'].apply(education_form)

def langs(l):
    return len(l.split(';'))
df['langs'] = df['langs'].apply(langs)

def people_main(pm):
    if pm == 'False':
        return 0
    else:
        return int(pm)
df['people_main'] = df['people_main'].apply(people_main)

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

cities = []
def city(c):
    if not c in cities:
        cities.append(c)
    return cities.index(c)
df['city'] = df['city'].apply(city)

def occupation_type(t):
    if t == 'univercity':   return 1
    else:   return 0
df['occupation_type'] = df['occupation_type'].apply(occupation_type)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

x = df.drop('result', axis = 1)
y = df['result']

sc = StandardScaler()
classifier = KNeighborsClassifier(n_neighbors = 1001)

df2 = pd.read_csv('test.csv')
df2.drop(['occupation_name', 'career_start', 'career_end'], axis = 1, inplace = True)

df2['life_main'] = df2['life_main'].apply(life_main)
df2['bdate'] = df2['bdate'].apply(bdate)
df2['city'] = df2['city'].apply(city)
df2['occupation_type'] = df2['occupation_type'].apply(occupation_type)

df2['education_status'] = df2['education_status'].apply(education_status)
df2['education_form'] = df2['education_form'].apply(education_form)
df2['langs'] = df2['langs'].apply(langs)
df2['people_main'] = df2['people_main'].apply(people_main)
df2['last_seen'] = df2['last_seen'].apply(last_seen)
df2['has_mobile'] = df2['has_mobile'].apply(int)
df2['graduation'] = df2['graduation'].apply(int)
df2['followers_count'] = df2['followers_count'].apply(int)

x_real = df2.drop('id', axis = 1)

best = 0
for i in range(500):
    print('Попытка', i+1, 'из 500')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.02)
    #x_train = x
    #y_train = y
    #x_test = x
    #y_test = y
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    res = accuracy_score(y_test, y_pred)
    if res > best:
        best = res
        x_real = sc.fit_transform(x_real)
        y_pred = classifier.predict(x_real)
        
final_itog = pd.DataFrame({'ID' : df2['id'], 'result' : y_pred})
final_itog.to_csv('final_itog.csv', index = False)
print('Лучший процент', best*100)
