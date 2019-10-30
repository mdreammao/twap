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
os.environ['NUMEXPR_MAX_THREADS'] = '8'
#LOCALDATAPATH=r'/home/public/mao/BTP/LocalDataBase'
#LOCALFeatureDATAPATH=r'/home/maoheng/Data'
#database='MaoTickFactors20191027'
#INFLUXDBHOST='192.168.38.2'

file=os.path.join(LOCALDATAPATH,'normalization20190712.h5')
with pd.HDFStore(file,'r',complib='blosc:zstd',append=True,complevel=9) as store:
    mynormalization=store['data']
FEATURE_COLUMNS =list(mynormalization['name'])
#TARGET_COLUMNS = ['midIncreaseNext1m','buyPriceIncreaseNext15s','sellPriceIncreaseNext15s']
TARGET_COLUMNS = ['midIncreaseNext1m']
USEFUL_COLUMNS=FEATURE_COLUMNS+['realData']
All_COLUMNS=USEFUL_COLUMNS+TARGET_COLUMNS+['B1','S1']

model_save_path=os.path.join(LOCALDATAPATH,'lightgbmModel.txt')
BATCH_SIZE = 200
SEQ_LENGTH = 10
VALIDATION_SIZE = 5
INLOOP_SIZE = 10000
PREPARE_JOBS = -1   
startDate=20180901
endDate=20191025



def getCodes():
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


def get_tick_data(code,date,database,columns=All_COLUMNS):
    try:
        tick_data = getDataFromInfluxdb(code,date,database,columns)
        if tick_data.shape[0]==0:
            #print(f'data of {code} in {date} from {database} has error!!!')
            return np.zeros((0, len(FEATURE_COLUMNS))), np.zeros((0, len(TARGET_COLUMNS))), np.zeros((0, 1)),[]
        ##标准化
        tick_data=dataNormalization(tick_data,mynormalization)
        # replace inf to na
        tick_data = tick_data.replace(np.inf, np.nan)
        tick_data = tick_data.replace(-np.inf, np.nan)
        ##去掉开头和结尾的nan
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

def pathCreate(path):
    if os.path.exists(path)==False:
        #logger.info(f'{path} is not exists! {path} will be created!')
        try:
            os.makedirs(path)
        except:
            print(f'create {path} error!')
            pass
def saveDataFromInfluxdb(code,date,database,columns):
    data=getDataFromInfluxdb(code,date,database,columns)
    if data.shape[0]<100:
        return
    #code=code.replace('.','_')
    path=os.path.join(LOCALFeatureDATAPATH,'features',date)
    pathCreate(path)
    files=os.path.join(path,code+".h5")
    if (os.path.isfile(files)==False):
        
        with pd.HDFStore(files,'a',complib='blosc:zstd',append=False,complevel=9) as store:
            store.append('data',data,append=False,format="table",data_columns=data.columns)
    else:
        print(f'data of {code} in {date} has already exits!')
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
stocks=getCodes()
dataList=getDataList(stocks,20180901,20180915)
#Parallel(n_jobs=PREPARE_JOBS, verbose=0)(delayed(saveDataFromInfluxdb)(o['code'], o['date'], database, All_COLUMNS) for o in dataList)
data=getDataFromH5('600000.SH',20180903,All_COLUMNS)  
print('ok!')
#gbm = lgb.Booster(model_file=model_save_path)

