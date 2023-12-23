import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler

df = pd.read_csv("data.csv")

"""
                RC1         RC2          RC3
duration_ms   0.18989823 -0.71261775 -0.205509129
danceability  0.07433553  0.34804843 -0.444383487
energy        0.36852082 -0.07709154  0.055373170
loudness      0.33360653 -0.05071244  0.027325302
acousticness -0.33253609  0.08085599 -0.001308598
valence       0.07584853  0.43870500 -0.050250414
tempo         0.05258095  0.13075919  0.823864578
"""
#df = df.drop(columns=['key','mode','speechiness','liveness','time_signature','instrumentalness'])
#df["duration_mins"] = df["duration_ms"]/60000
df = df.drop(columns=['key','mode','speechiness','liveness','time_signature','instrumentalness','track_genre'])
scaler = StandardScaler()
df[['duration_ms','danceability','energy','loudness','acousticness','valence','tempo']] = \
    scaler.fit_transform(df[['duration_ms','danceability','energy','loudness','acousticness','valence','tempo']])

'''
scaler = MinMaxScaler()
df[['duration_ms','danceability','energy','loudness','acousticness','valence','tempo']] = \
    scaler.fit_transform(df[['duration_ms','danceability','energy','loudness','acousticness','valence','tempo']])
'''

df['RC1'] = 0.18989823*df['duration_ms'] + 0.07433553*df['danceability'] + 0.36852082*df['energy'] +\
    0.33360653*df['loudness']+ (-0.33253609)*df['acousticness'] + 0.07584853*df['valence'] + 0.05258095*df['tempo']
df['RC2'] = (-0.71261775)*df['duration_ms'] + 0.34804843*df['danceability'] + (-0.07709154)*df['energy'] +\
    (-0.05071244)*df['loudness'] + 0.08085599*df['acousticness'] + 0.43870500*df['valence'] + 0.13075919*df['tempo']
df['RC3'] = (-0.205509129)*df['duration_ms'] + (-0.444383487)*df['danceability'] + 0.055373170*df['energy'] +\
    0.027325302*df['loudness'] + (-0.001308598)*df['acousticness'] + (-0.050250414)*df['valence'] + 0.823864578*df['tempo']

df = df.drop(columns=['duration_ms','danceability','energy','loudness','acousticness','valence','tempo'])

df['explicit'] = df['explicit'].astype(str)
#df= pd.get_dummies(df, columns=['explicit','track_genre'], prefix='Category')
df= pd.get_dummies(df, columns=['explicit'], prefix='Category')

df.loc[((df.popularity >= 0) & (df.popularity <= 40)), "popularity_level" ] = 0
df.loc[((df.popularity > 40)), "popularity_level" ] = 1
df = df.reindex(sorted(df.columns), axis=1)
df.to_csv("final_dataset2.csv",index=False)