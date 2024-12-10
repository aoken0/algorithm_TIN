import pandas as pd
import numpy as np

df = pd.DataFrame(np.empty((6, 4)), columns=['id', 'x', 'y', 'h'])

df['id'] = np.arange(1, 7)

rng = np.random.default_rng()
df['x'] = np.round(rng.uniform(0, 10, 6), 2)
df['y'] = np.round(rng.uniform(0, 10, 6), 2)
df['h'] = np.round(rng.uniform(0, 10, 6), 2)

print(df)

df.to_csv('./csv/random_points.csv', index=False)