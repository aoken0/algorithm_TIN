import pandas as pd
import numpy as np

num = 300

df = pd.DataFrame(np.empty((num, 4)), columns=['id', 'x', 'y', 'h'])

df['id'] = np.arange(0, num)

rng = np.random.default_rng()
df['x'] = np.round(rng.uniform(0, 100, num), 2)
df['y'] = np.round(rng.uniform(0, 100, num), 2)
df['h'] = np.round(rng.uniform(0, 10, num), 2)

df.to_csv(f'./csv/random_{num}points.csv', index=False)
