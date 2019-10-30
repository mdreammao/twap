import influxdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from joblib import dump, load, Parallel, delayed
import dateutil.parser as dtparser
import datetime
import torch
import os

import warnings
warnings.filterwarnings("ignore") 
import time
from sklearn.linear_model import SGDClassifier
import sklearn
from numpy.lib.stride_tricks import as_strided
import lightgbm as lgb


# GLOBAL PART
database='MaoTickFactors20190831'
INFLUXDBHOST='192.168.58.71'
LOCALDATAPATH=r'd:/BTP/LocalDataBase'
LOCALFeatureDATAPATH=r'd:/Data'
#LOCALDATAPATH=r'/home/public/mao/BTP/LocalDataBase'
#database='MaoTickFactors20191027'
#INFLUXDBHOST='192.168.38.2'
file=os.path.join(LOCALDATAPATH,'normalization20190712.h5')
with pd.HDFStore(file,'r',complib='blosc:zstd',append=True,complevel=9) as store:
    mynormalization=store['data']
FEATURE_COLUMNS =list(mynormalization['name'])
#TARGET_COLUMNS = ['buyPriceIncreaseNext15s','sellPriceIncreaseNext15s']
TARGET_COLUMNS = ['midIncreaseNext1m']
USEFUL_COLUMNS=FEATURE_COLUMNS+['realData']
All_COLUMNS=USEFUL_COLUMNS+TARGET_COLUMNS

model_save_path=os.path.join(LOCALDATAPATH,'lightgbmModel.txt')
BATCH_SIZE = 200
SEQ_LENGTH = 10
VALIDATION_SIZE = 5
INLOOP_SIZE = 10000
PREPARE_JOBS = 8   
startDate=20170101
endDate=20171231
testStart=20180101
testEnd=20180131

def getCodes(type=1000):
    if (type==300):
        localFileStr=os.path.join(LOCALDATAPATH,'stockCodes300.h5')
    elif (type==500):
        localFileStr=os.path.join(LOCALDATAPATH,'stockCodes500.h5')
    else:
        localFileStr=os.path.join(LOCALDATAPATH,'stockCode.h5')
    with pd.HDFStore(localFileStr,'r',complib='blosc:zstd',append=False,complevel=9) as store:
        stockCodes=store['data']
    return list(stockCodes)
    pass
def getTradedays(startDate,endDate):
    localFileStr=os.path.join(LOCALDATAPATH,'tradedays.h5')
    startDate=str(startDate)
    endDate=str(endDate)
    store = pd.HDFStore(localFileStr,'r')
    mydata=store.select(max(store.keys()))
    store.close()
    mydata=mydata.loc[(mydata['date']>=startDate) &(mydata['date']<=endDate),'date']
    return (list(mydata))
    pass
def getDataFromInfluxdb(code,date,database,columns=[]):
    client = influxdb.DataFrameClient(host=INFLUXDBHOST, port=8086, username='root', password='root', database=database)
    measure=code
    b = dtparser.parse(str(date))+ datetime.timedelta(hours=0)
    e = dtparser.parse(str(date)) + datetime.timedelta(hours=24)
    colstr=''
    if len(columns)>0:
        for col in columns:
            colstr=colstr+f""" "{col}", """
        colstr=colstr[:-2]
    else:
        colstr='*'
    query=f""" select {colstr} from "{database}"."autogen"."{measure}" where time >= {int(b.timestamp() * 1000 * 1000 * 1000)} and time < {int(e.timestamp() * 1000 * 1000* 1000)} """
    result=client.query(query)
    if result!={}:
        data=pd.DataFrame(result[measure])
    else:
        data=pd.DataFrame()
    return data
def getDataList(codes,startDate,endDate):
    days=getTradedays(startDate,endDate)
    mylist=[]
    for code in codes:
        for day in days:
            mylist.append({'code':code,'date':day})
    return mylist
    pass
def dataNormalization(dataAll,normalization):
    cols=list(normalization['name'])
    dataAll=dataAll.dropna(axis=0,how='any')
    i=0
    for col in cols:
        data=dataAll[col].values
        logif=normalization['log'].iloc[i]
        if logif==True:
            b=normalization['logConstant'].iloc[i]
            data=np.log(data+b)
        mad=normalization['mad'].iloc[i]
        middle=normalization['mean'].iloc[i]
        data=np.clip(data,a_min=middle-mad,a_max=middle+mad)
        data=(data-middle)/mad
        
        #import pdb
        #pdb.set_trace()
        
        dataAll.loc[:,col]=data
        #dataAll[col]=data
        i=i+1
    
    return dataAll
    pass

def getDataFromH5(code,date,columns):
    code=str(code)
    date=str(date)
    #code=code.replace('.','_')
    file=os.path.join(LOCALFeatureDATAPATH,'features',date,code+".h5")
    if (os.path.isfile(file)==True):
        with pd.HDFStore(file,'r',complib='blosc:zstd',append=False,complevel=9) as store:
            data=store['data']
        return data[columns]
    else:
        return pd.DataFrame()

def get_tick_data_fromh5(code,date,columns=All_COLUMNS):
    try:
        tick_data = getDataFromH5(code,date,columns)
        if tick_data.shape[0]==0:
            #print(f'data of {code} in {date} from {database} has error!!!')
            return np.zeros((0, len(FEATURE_COLUMNS))), np.zeros((0, len(TARGET_COLUMNS))), np.zeros((0, 1)),[]
        ##鏍囧噯鍖?        tick_data=dataNormalization(tick_data,mynormalization)
        # replace inf to na
        tick_data = tick_data.replace(np.inf, np.nan)
        tick_data = tick_data.replace(-np.inf, np.nan)
        ##鍘绘帀寮€澶村拰缁撳熬鐨刵an
        startIndex=tick_data[tick_data.isna().sum(axis=1)==0].index[0]
        endIndex=tick_data[tick_data.isna().sum(axis=1)==0].index[-1]
        tick_data=tick_data.loc[startIndex:endIndex,:]
        mytime=tick_data.index
        ## split X, y 
        X = tick_data[FEATURE_COLUMNS].values
        padding = np.zeros((SEQ_LENGTH-1, X.shape[1]))
        y = 100*tick_data[TARGET_COLUMNS].values
        #y=np.clip(y,a_min=-3,a_max=3)/6
        flag = tick_data['realData'].values
        return np.concatenate((padding, X), axis=0), y, flag,mytime
    except:
        #print(f'data of {code} in {date} from {database} has error!!!')
        return np.zeros((0, len(FEATURE_COLUMNS))), np.zeros((0, len(TARGET_COLUMNS))), np.zeros((0, 1)),[]

def get_tick_data(code,date,database,columns=All_COLUMNS):
    try:
        tick_data = getDataFromInfluxdb(code,date,database,columns)
        if tick_data.shape[0]==0:
            #print(f'data of {code} in {date} from {database} has error!!!')
            return np.zeros((0, len(FEATURE_COLUMNS))), np.zeros((0, len(TARGET_COLUMNS))), np.zeros((0, 1)),[]
        ##鏍囧噯鍖?        tick_data=dataNormalization(tick_data,mynormalization)
        # replace inf to na
        tick_data = tick_data.replace(np.inf, np.nan)
        tick_data = tick_data.replace(-np.inf, np.nan)
        ##鍘绘帀寮€澶村拰缁撳熬鐨刵an
        startIndex=tick_data[tick_data.isna().sum(axis=1)==0].index[0]
        endIndex=tick_data[tick_data.isna().sum(axis=1)==0].index[-1]
        tick_data=tick_data.loc[startIndex:endIndex,:]
        mytime=tick_data.index
        ## split X, y 
        X = tick_data[FEATURE_COLUMNS].values
        padding = np.zeros((SEQ_LENGTH-1, X.shape[1]))
        y = 100*tick_data[TARGET_COLUMNS].values
        #y=np.clip(y,a_min=-3,a_max=3)/6
        flag = tick_data['realData'].values
        return np.concatenate((padding, X), axis=0), y, flag,mytime
    except:
        #print(f'data of {code} in {date} from {database} has error!!!')
        return np.zeros((0, len(FEATURE_COLUMNS))), np.zeros((0, len(TARGET_COLUMNS))), np.zeros((0, 1)),[]

def modifyData(batch_X, batch_y, batch_flag):
    timeSeries=batch_X.shape[0]
    factors=batch_X.shape[1]
    modifiedTimeSeries=timeSeries-SEQ_LENGTH+1
    modifiedFactors=factors*SEQ_LENGTH
    batch_seq_X = np.zeros((modifiedTimeSeries,modifiedFactors))
    batch_seq_y = batch_y
    batch_seq_flag = batch_flag
    na_idx=np.zeros(batch_seq_X.shape[0])
    for i in range(batch_seq_X.shape[0]):
        batch_seq_X[i,:]=batch_X[i:i+SEQ_LENGTH,:].copy().flatten()
        if ((np.isnan(batch_seq_X[i,:]).any()==True) | (np.isnan(batch_seq_y[i]).any()==True)| (batch_seq_flag[i]==0)):
            na_idx[i]=1
    #batch_seq_X = batch_seq_X[na_idx == 0]
    #batch_seq_y = batch_seq_y[na_idx == 0]
    batch_seq_X[np.isnan(batch_seq_X)]=0
    batch_seq_y[np.isnan(batch_seq_y)]=0
    return batch_seq_X,batch_seq_y.flatten(),na_idx



def statistics(code,date,model):
   # batch_X, batch_y, batch_flag,mytime=get_tick_data_fromh5(code,date,All_COLUMNS)
    batch_X, batch_y, batch_flag,mytime=get_tick_data(code,date,database,All_COLUMNS)
    if (batch_X.shape[0]<1000):
        return 
    
    input,target,flag=  modifyData( batch_X, batch_y, batch_flag) 
    predict=gbm.predict(input)
    predictDf=pd.DataFrame(index=mytime)
    predictDf['predict']=predict
    predictDf['target']=target
    predictDf['flag']=flag
    #tickData=getDataFromH5(code,date,['B1','S1'])
    tickData=getDataFromInfluxdb(code,date,database,['B1','S1'])
    #if tickData['B1'].mean()<10:
    #    return
    tickData[['predict','target','flag']]=predictDf[['predict','target','flag']]
    tick=tickData.values
    buy=0
    sell=0
    mybuy=0
    mysell=0
    num=0
    step=20
    for i in range(0,tick.shape[0]-1,step):
        buy+=tick[i][1]
        sell+=tick[i][0]
        num=num+1
        #如果要跌，等等再买
        if (tick[i][2]<-0.015) & (tick[i][3]<1) & (i<(tick.shape[0]-2*step)):
            mybuy+=tick[i+step][1]
        else:
            mybuy+=tick[i][1]
        #如果要涨，等等再卖
        if (tick[i][2]>0.015) & (tick[i][3]<1) & (i<(tick.shape[0]-2*step)):
            mysell+=tick[i+step][0]
        else:
            mysell+=tick[i][0]
    buy=np.round(buy/num,8)
    sell=np.round(sell/num,8)
    mybuy=np.round(mybuy/num,8)
    mysell=np.round(mysell/num,8)
    mid=(buy+sell)/2
    print(code,date,np.round(r2_score(target[batch_flag == 0],predict[batch_flag == 0]),4),np.round(np.corrcoef(target[batch_flag == 0],predict[batch_flag == 0])[0][1],4))
    #print(buy,sell,mid,mybuy,mybuy)
    buyimprove=np.round((buy-mybuy)/buy,4)
    sellimprove=np.round((mysell-sell)/sell,4)
    buymidimprove=np.round((mid-mybuy)/mid,4)
    sellmidimprove=np.round((mysell-mid)/mid,4)
    print(buyimprove,sellimprove,buymidimprove,sellmidimprove)
    print("==============================================================================")
    pass
#statistics('000021.SZ',20180111,gbm)
stocks=getCodes(500)
stocks=['600000.SH']
test_list=getDataList(stocks,20180403,20180403)
model_save_path=os.path.join(LOCALDATAPATH,'lightgbmModel','20180403.txt')
gbm = lgb.Booster(model_file=model_save_path)
for item in test_list:
    code=item['code']
    date=item['date']
    statistics(code,date,gbm)
    #print(code,date)
    #break
    pass



