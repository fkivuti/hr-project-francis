# hr-project-francis
hr DS project week 5

# Import numpy and pandas libraries
import pandas as pd
import numpy as np

# load datafile and preview first few records
hr_df = pd.read_csv('https://bit.ly/2ODZvLCHRDataset')
hr_df.head()

# load glossary dataframe and view the records
glossary_df = pd.read_csv('https://bit.ly/2Wz3sWcGlossary')
glossary_df

# check if there is a column with null values
hr_df.isnull().sum()

# Instead of replacing null values in the previous year rating column with zero, we opt to delete these rows i.e. 4124 rows
hr_df = hr_df.dropna(axis=0, subset=['previous_year_rating'])
hr_df.isnull().sum()

# Iterate through the columns in the dataframe and find unique elements for non numeric columns. We will take a set  of the column values and 
# thus the set within the index within the set will be the new numerical value or id of that non numerical observation.

# create a function that gets the columns and interate through them

def handle_non_numerical_data(hr_df):
    columns = hr_df.columns.values
    for column in columns:

# Embed a function that converts the parameter value to the any value of that item (as Key) from the text_digit_val dictionary

        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

# During iteration through the columns, check and pick columns which are not int64 or float64 and then convert the column to list of its values
        if hr_df[column].dtype != np.int64 and hr_df[column].dtype != np.float64:
            column_contents = hr_df[column].values.tolist()

# Take a set of the columns and extract the unique values only.            
            unique_elements = set(column_contents)

# Create a new dictionary key for each of the unique values found with avalye of a new number.
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

# Use the map function to perform mapping of the new values into the columns
            hr_df[column] = list(map(convert_to_int, hr_df[column]))

    return hr_df


# Call our handle_non_numerical_data function and preview the newly converted data frame

hr_df = handle_non_numerical_data(hr_df)
print(hr_df.head())

# import RandomForestRegressor as follows
from sklearn.ensemble import RandomForestRegressor

# Defining features and target
features =  hr_df.drop(['employee_id', 'is_promoted'], axis=1)
target = hr_df['is_promoted']

# Create a regressor object with random state set to Zero and n_estimators set to 3
random_regressor = RandomForestRegressor(random_state = 0, n_estimators=3)

# Train the model
random_regressor.fit(features, target)

# Define sample data that will be used to predict the 'is_promoted' outcome.

features =  hr_df.drop(['employee_id', 'is_promoted'], axis=1)
new_features = pd.DataFrame(
    [
        [4, 12, 3 , 0, 1, 2, 50, 4.0, 11, 1, 0, 65],
        [3, 6, 1 , 1, 1, 2, 27, 0, 6, 1, 0, 30],
    ],
    columns=features.columns
)

# Predict if this employee will be promoted

is_promoted = random_regressor.predict(new_features)  
print(is_promoted)

# check model's accuracy level
accuracy_score = random_regressor.score(features, target)
print(accuracy_score)

# import RandomForestRegressor as follows
from sklearn.ensemble import RandomForestRegressor

# Defining features and target
features =  hr_df.drop(['employee_id', 'is_promoted'], axis=1)
target = hr_df['is_promoted']

# Create a regressor object with random state set to Zero and n_estimators set to 3
random_regressor = RandomForestRegressor(random_state = 0, n_estimators=3)

# Train the model
random_regressor.fit(features, target)

# Define sample data that will be used to predict the 'is_promoted' outcome.

features =  hr_df.drop(['employee_id', 'is_promoted'], axis=1)
new_features = pd.DataFrame(
    [
        [4, 12, 3 , 0, 1, 2, 50, 4.0, 11, 1, 0, 65],
        [3, 6, 1 , 1, 1, 2, 27, 0, 6, 1, 0, 30],
    ],
    columns=features.columns
)

# Predict if this employee will be promoted

is_promoted = random_regressor.predict(new_features)  
print(is_promoted)

# check model's accuracy level
accuracy_score = random_regressor.score(features, target)
print(accuracy_score)

