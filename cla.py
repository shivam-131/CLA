import pandas as pd
import numpy as np
import datetime

df= pd.read_csv('~/Desktop/Passenger.csv', names= ["Values"])
def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo/expo_sum
min=364
max=13512
# for i in range(1488):
#     if df["Values"][i]<min:
#         min= df["Values"][i]
#d     if df['Values'][i]>max:
#         max= df["Values"][i]
# print(min, max)
from nupic.encoders.multi import MultiEncoder

encoder= MultiEncoder()
encoder.addMultipleEncoders(fieldEncodings= {
    'dateTime': dict(fieldname= 'dateTime', type='DateEncoder', timeOfDay= (5, 5)),
    'passengers': dict(fieldname= 'passengers', type= 'ScalarEncoder', name= 'passengers' , minval= min-100, maxval= max+100, w=25, n= 1500, clipInput= True),
})
encoded_data= []
start_date= datetime.datetime(2020, 1, 1, 0, 0)
end_date= datetime.datetime(2020, 1, 31, 23, 30)
delta = datetime.timedelta(minutes=30)
i=0
while start_date <= end_date:
    encoded_data.append(encoder.encode({'dateTime': start_date, 'passengers': df["Values"][i],}))
    start_date += delta
    i=i+1
from nupic.algorithms.spatial_pooler import SpatialPooler
sp= SpatialPooler(inputDimensions=(1524,), columnDimensions=(2048, ))
sdr= []
for j in range(len(encoded_data)):
    d= np.full((2048,), 0)
    sp.compute(encoded_data[j], learn= True, activeArray= d)
    sdr.append(d)
print(len(sdr))
from nupic.algorithms.temporal_memory import TemporalMemory

tm= TemporalMemory(columnDimensions=(2048,), cellsPerColumn=32, activationThreshold=15, initialPermanence=0.21, connectedPermanence=0.5, maxSegmentsPerCell=128, maxSynapsesPerSegment=128, maxNewSynapseCount=32)
likelihood=0
for time in range(len(sdr)-1):
    thislist= []
    for k in range(len(sdr[time])):
        if sdr[time][k]==1:
            thislist.append(k)
    tm.compute(thislist, learn= True)
    predicted_cells= tm.getPredictiveCells()
    thisSet= set()
    for cell in predicted_cells:
        thisSet.add(int(cell/(32)))
    #INVERSE SQUARE PERCENTAGE ERROR
    y= np.full((2048, ), 0)
    for x in thisSet:
        y[x]= 1
    MAPE= (np.linalg.norm(sdr[time+1]- y))/(np.linalg.norm(sdr[time+1]))
    print(MAPE)
    #CROSS ENTRPOY LOSS
    if time>0:
        prob= softmax(y)
        likelihood= (np.log(np.dot(prob, sdr[time+1])) + (time)*likelihood)/(time+1)
        #print(likelihood)
