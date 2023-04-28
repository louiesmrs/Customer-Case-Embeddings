# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = ['get']
product = {
    "nb": "outputs/output.ipynb",
    "data": "outputs/clean_data.csv"
}

# %% [raw]
# import numpy as np #linear algebra
# import pandas as pd #data manipulation and analysis
# import seaborn as sns #data visualization
# import matplotlib.pyplot as plt #data visualization
# import sklearn.preprocessing as skp #machine learning (preprocessing)
# import sklearn.cluster as skc #machine learning (clustering)
# import warnings # ignore warnings
# warnings.filterwarnings('ignore')

# %%



# %%
import numpy as np #linear algebra
import pandas as pd #data manipulation and analysis
from sklearn import preprocessing

# %% [markdown]
# ## INTRO TO DATA

# %%
df = pd.read_csv(upstream['get']['data'])

# %%
df.info()

# %%
df.describe()

# %% [markdown]
# ### Drop unnecessary Columns

# %%
# Specify the cols to drop
to_drop = [
    'parts_only', 'dps_classification', "dispatch_number"
]

# Drop them
df.drop(to_drop, axis='columns', inplace=True)

# %% [markdown]
# ### Distribution of Columns

# %%
# # Visualize the distribution of each variable.
# plt.figure(figsize=(12,16))
# for i, j in enumerate(df.describe().columns):
#     plt.subplot(5,2, i+1)
#     sns.distplot(x=df[j])
#     plt.xlabel(j)
#     plt.title('{} Distribution'.format(j))
#     plt.subplots_adjust(wspace=.2, hspace=.5)
#     plt.tight_layout()
# plt.show()

# %% [markdown]
# ## OUTLIERS FOR SYSTEM_AGE

# %%

df.loc[df['system_age'] > 10000

       ]

# %%
df.system_age[df['system_age'] > 10000] = None

# %% [markdown]
# ### Filling missing data with mean

# %%
df.system_age[df['system_age'] < 0] = None
df.system_age.fillna(df.system_age.mean(), inplace=True)


# %%
df.loc[df['system_age'].isnull()
       ]

# %% [markdown]
# ## DROPPING DUPLICATE ROWS

# %%
df.duplicated().sum()

# %%
df.loc[df.
       duplicated(), :]

# %%
df = df.drop_duplicates(keep=False)

# %%
df.info()

# %% [markdown]
# ## CONVERTING TO DATETIME + CREATING NEW COLUMN

# %%
dft = df.copy()
dft["close_dts"] = pd.to_datetime(df.close_dts, format="%H:%M:%S", errors='raise')
dft["close_dts"] =dft["close_dts"].dt.time

# %%
dft.head(10)

# %%
dft["created_dts"] = pd.to_datetime(df.created_dts, format = "%H:%M:%S" , errors='raise')
dft["created_dts"] =dft["created_dts"].dt.time

# %%
dft.head(10
        )

# %%

df["created_date"] = pd.to_datetime(df.created_date, format = "%m/%d/%Y %H:%M" , errors='raise')


# %% [markdown]
# ## DEALING WITH MISSING DATA

# %%
df.head(10)


# %%

df["created_dts"] = pd.to_datetime(df.created_dts, format = "%H:%M:%S" , errors='raise')
df["close_dts"] = pd.to_datetime(df.close_dts, format="%H:%M:%S", errors='raise')


# %%
df['journey_time'] = df['close_dts'] - df["created_dts"]
df.journey_time[df['journey_time'].dt.days != 0] = None

# %%
df.head(10)

# %%

df["seconds"] = df.journey_time.dt.total_seconds()
df.head(10)

# %%
df.info()

# %%
missing_data = pd.DataFrame({'total_missing': df.isnull().sum(), 'perc_missing': (df.isnull().sum()/18146)*100})
missing_data

# %%
df.info()

# %% [markdown]
# ### Filling missing values for countries with "No Country"

# %%
df.country_name_dawn.fillna("No Country", inplace=True)

# %%
df.total_activity_count.fillna(df.total_activity_count.mean(), inplace=True)
df.total_inbounf_count.fillna(df.total_inbounf_count.mean(), inplace=True)

# %%
df.journey_time.fillna(df.journey_time.mean(), inplace=True)
df.close_dts.fillna(df.close_dts.mean(), inplace=True)

# %%
df.seconds.fillna(df.seconds.mean(), inplace= True)

# %%
df.country_name_dawn.isnull().sum()

# %% [markdown]
# ## GRAPHS + CONVERTING 6 COLUMNS TO CATEGORY DTYPE

# %% [markdown]
# ### Checking possible columns to convert to category dtype

# %% [markdown]
# ### Convert to category (commented due to adding more graphs)

# %%
#df.commodity_desc = df.commodity_desc.astype('category')
#df.prod_desc = df.prod_desc.astype('category')
#df.contact_method_name = df.contact_method_name.astype('category')
#df.contact_method = df.contact_method.astype('category')
#df.new_warranty_type = df.new_warranty_type.astype('category')
#df.warranty_status = df.warranty_status.astype('category')

# %%
df.info()

# %% [markdown]
# ### Comparing similar columns

# %% [markdown]
# ### Adding Continent Column

# %%
asia = ['Afghanistan', 'Bahrain', 'United Arab Emirates','Saudi Arabia', 'Kuwait', 'Qatar', 'Oman',
       'Sultanate of Oman','Lebanon', 'Iraq', 'Yemen', 'Pakistan', 'Lebanon', 'Philippines', 'Jordan', 'Taiwan',
       'Indonesia', 'Singapore', 'China', 'Korea', 'Hong Kong', 'Malaysia', 'India', 'APJ Support', 'Japan', 'Thailand']
europe = ['Germany','Spain', 'France', 'Italy', 'Netherlands', 'Norway', 'Sweden','Czech Republic', 'Finland',
          'Denmark', 'Switzerland', 'United Kingdom', 'Ireland', 'Poland', 'Greece','Austria',
          'Bulgaria', 'Hungary', 'Luxembourg', 'Romania' , 'Slovakia', 'Estonia', 'Slovenia','Portugal',
          'Croatia', 'Lithuania', 'Latvia','Serbia', 'Estonia', 'Iceland', 'Western Europe' , 'CEE eCis' , 'Belgium']
sa =['Chile', 'Peru', 'Argentina', 'Colombia', 'Mexico', 'Brazil']
na = ['United States', 'Canada']
africa = ['South Africa']
oceania = ['Australia', 'New Zealand']


def GetConti(country):
    if country in asia:
        return "Asia"
    elif country in europe:
        return "Europe"
    elif country in africa:
        return "Africa"
    elif country in na:
        return "North America"
    elif country in sa:
        return "South America"
    elif country in oceania:
        return "Oceania"
    else:
        return "No country"


df['Continent'] = df['country_name_dawn'].apply(lambda x: GetConti(x))

# %%
df.head(10)

# %% [markdown]
# ### Encoding Continents

# %%
from sklearn.preprocessing import OneHotEncoder


le = preprocessing.LabelEncoder()
df['continent_cat'] = le.fit_transform(df['Continent'])
df['country_cat'] = le.fit_transform(df['country_name_dawn'])

# %%
df.info()

# %%
df['date_delta'] = (df['created_date'] - df['created_date'].min())  / np.timedelta64(1,'D')

# %%
df.head(10)

# %% [markdown]
#

# %% [markdown]
#

# %%
to_drop2 = [
    "last_updated_dts", "close_dts",
    "created_dts", "created_date", "tag_id",  "contact_method", "contact_method_name",
    "journey_time", "seconds"
]

# Drop them
df.drop(to_drop2, axis='columns', inplace=True)

# %%
df.to_csv(str(product['data']), index=False)

# %%
