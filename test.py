import pandas as pd 
df= pd.read_csv('~/Downloads/yellow_tripdata_2020-01.csv')
df= df.dropna()
passenger_count= []
for i in range(31*48):
    passenger_count.append(0)


for i in range(6339567):
    x= 10*int(df['tpep_pickup_datetime'][i][8]) + int(df['tpep_pickup_datetime'][i][9])
    time= 60*(10*int(df['tpep_pickup_datetime'][i][-8]) + int(df['tpep_pickup_datetime'][i][-7])) + 10*int(df['tpep_pickup_datetime'][i][-5]) + int(df['tpep_pickup_datetime'][i][-4])
    slot_= int(time/30)
    passenger_count[(x-1)*48 + slot_-1]= passenger_count[(x-1)*48 + slot_-1] + int(df['passenger_count'][i])

print(passenger_count, '\n')
print('length', len(passenger_count))
sr= pd.Series(passenger_count)
sr.to_csv('Passenger.csv')

    


