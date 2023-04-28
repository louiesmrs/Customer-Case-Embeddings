# %% tags=["parameters"]
upstream = None
product = None


# %%
import pandas as pd
df = pd.read_csv("20K_cases_data_anonymized_label_encoded_v3.csv")
# %%
df.to_csv(product['data'], index=False)