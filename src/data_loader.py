import pandas as pd
def load_data():
    df = pd.read_csv("data/data.csv")
    print(df.columns)
    print(df.shape)
    print(df['Sentiment'].value_counts())
    print(df.head())
    label_map = {"negative": 0,"neutral": 1, "positive":2}
    df['label'] = df['Sentiment'].map(label_map)
    print("label distribution:")
    print(df['label'].value_counts())
    return df

if __name__ == "__main__":
    load_data()
