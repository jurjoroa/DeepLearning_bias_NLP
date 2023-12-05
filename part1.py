

#pipenv install git+https://github.com/dssg/aequitas.git#egg=aequitas 

import pandas as pd

import pandas as pd
import seaborn as sns
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot


df_compas = pd.read_csv("https://raw.githubusercontent.com/dssg/aequitas/master/examples/data/compas_for_aequitas.csv")

#Check if I have repited rows

df_compas.duplicated().sum()


#Check score column

df_compas.score.value_counts()


df_compas_raw = pd.read_csv("Compas data/compas-scores-raw.csv")


df_compas_filt = pd.read_csv("Compas data/cox-violent-parsed_filt.csv")



df_compas_parsec = pd.read_csv("Compas data/cox-violent-parsed.csv")


#drop -1 in v_decile_score in df_compas_parsec

df_compas_parsec_w = df_compas_parsec[df_compas_parsec.v_decile_score != -1]

#create a new column with the score_bin. 1 equals in v_score_text Medium or High, 0 equals Low

df_compas_parsec_w['score_bin'] = df_compas_parsec_w.v_score_text.apply(lambda x: 1 if x in ['Medium', 'High'] else 0)

#give me the races unique

df_compas.race.value_counts()

aq_palette = sns.diverging_palette(225, 35, n=2)


by_race = sns.countplot(x="race", hue="v_decile_score", data=df_compas_parsec[df_compas_parsec.race.isin(['African-American', 'Caucasian', 'Hispanic', "Other", "Asian", "Native American"])], palette=aq_palette)


#unique values is_recid

df_compas_parsec.is_recid.value_counts()

#erase -1 in is_recid

df_compas_parsec_w = df_compas_parsec_w[df_compas_parsec_w.is_recid != -1]

df_compas_parsec_w = df_compas_parsec_w[df_compas_parsec_w.v_decile_score != -1]

#keep this columns: v_decile_score, race, sex, age_cat, is_recid, id from df_compas_parsec_w


# Assuming df_compas_parsec_w is your original DataFrame
selected_columns = ['score_bin', 'race', 'sex', 'age_cat', 'is_recid', 'id']

# Creating a new DataFrame with only the selected columns
df_compas_parsec_final = df_compas_parsec_w[selected_columns]

#give me unique values on is_recid

df_compas_parsec_final.score_bin.value_counts()

#rename score_bin to score

df_compas_parsec_final.rename(columns={'score_bin': 'score'}, inplace=True)

#rename is_recid to label_value

df_compas_parsec_final.rename(columns={'is_recid': 'label_value'}, inplace=True)

g = Group()
xtab, _ = g.get_crosstabs(df_compas_parsec_final)

xtab

absolute_metrics = g.list_absolute_metrics(xtab)

xtab[[col for col in xtab.columns if col not in absolute_metrics]]

xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(2)







