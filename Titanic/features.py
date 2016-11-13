try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split

def load_data(df, feature_cols, target):
    X = df.ix[:, feature_cols]
    y = df.ix[:, target]
    return train_test_split(X, y, test_size=0.2, random_state=42) 
