

# read in the training and testing data into Pandas.DataFrame objects
input_df = pd.read_csv('data/raw/train.csv', header=0)
submit_df = pd.read_csv('data/raw/test.csv', header=0)

# merge the two DataFrames into one
df = pd.concat([input_df, submit_df])

# re-number the combined data set so there aren't duplicate indexes
df.reset_index(inplace=True)

# reset_index() generates a new column that we don't want, so let's get rid of it
df.drop('index', axis=1, inplace=True)

# the remaining columns need to be reindexed so we can access the first column at '0' instead of '1'
df = df.reindex_axis(input_df.columns, axis=1)

df['Cabin'][df.Cabin.isnull()] = 'U0'

df['Fare'][np.isnan(df['Fare']) ] = df['Fare'].median()