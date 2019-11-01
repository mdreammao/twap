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
LOCALDATAPATH=r'F:/BTP/LocalDataBase'
LOCALFeatureDATAPATH=r'F:/Data'
#LOCALDATAPATH=r'/home/public/mao/BTP/LocalDataBase'
#database='MaoTickFactors20191027'
#INFLUXDBHOST='192.168.38.2'
#LOCALFeatureDATAPATH=r'/home/maoheng/Data'
file=os.path.join(LOCALDATAPATH,'normalization20190712.h5')
with pd.HDFStore(file,'r',complib='blosc:zstd',append=True,complevel=9) as store:
    mynormalization=store['data']
FEATURE_COLUMNS =list(mynormalization['name'])
#TARGET_COLUMNS = ['midIncreaseNext1m','buyPriceIncreaseNext15s','sellPriceIncreaseNext15s']
TARGET_COLUMNS = ['midIncreaseNext1m']
USEFUL_COLUMNS=FEATURE_COLUMNS+['realData']
All_COLUMNS=USEFUL_COLUMNS+TARGET_COLUMNS
DataSource='h5'



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


def pathCreate(path):
    if os.path.exists(path)==False:
        #logger.info(f'{path} is not exists! {path} will be created!')
        try:
            os.makedirs(path)
        except:
            print(f'create {path} error!')
            pass
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

def getPredictDataFromH5(code,date,savepath):
    code=str(code)
    date=str(date)
    file=os.path.join(LOCALFeatureDATAPATH,'predict',savepath,date,code+".h5")
    if (os.path.isfile(file)==True):
        with pd.HDFStore(file,'r',complib='blosc:zstd',append=False,complevel=9) as store:
            data=store['data']
        return data
    else:
        return pd.DataFrame()
    pass

def savePredictDataToH5(code,date,savepath,data):
    code=str(code)
    date=str(date)
    file=os.path.join(LOCALFeatureDATAPATH,'predict',savepath,date,code+".h5")
    pathCreate(os.path.join(LOCALFeatureDATAPATH,'predict',savepath,date))
    if data.shape[0]>3000:
        with pd.HDFStore(file,'a',complib='blosc:zstd',append=False,complevel=9) as store:
            store.append('data',data,append=False,format="table",data_columns=data.columns)
    pass
def getData(code,date,columns):
    if (DataSource=='h5'):
        return getDataFromH5(code,date,columns)
    else:
        return getDataFromInfluxdb(code,date,database,columns)
    pass
#def get_tick_data_fromh5(code,date,columns=All_COLUMNS):
#    try:
#        tick_data = getDataFromH5(code,date,columns)
#        if tick_data.shape[0]==0:
#            #print(f'data of {code} in {date} from {database} has error!!!')
#            return np.zeros((0, len(FEATURE_COLUMNS))), np.zeros((0, len(TARGET_COLUMNS))), np.zeros((0, 1)),[]
#        tick_data=dataNormalization(tick_data,mynormalization)
#        # replace inf to na
#        tick_data = tick_data.replace(np.inf, np.nan)
#        tick_data = tick_data.replace(-np.inf, np.nan)
#        ##鍘绘帀寮€澶村拰缁撳熬鐨刵an
#        startIndex=tick_data[tick_data.isna().sum(axis=1)==0].index[0]
#        endIndex=tick_data[tick_data.isna().sum(axis=1)==0].index[-1]
#        tick_data=tick_data.loc[startIndex:endIndex,:]
#        mytime=tick_data.index
#        ## split X, y 
#        X = tick_data[FEATURE_COLUMNS].values
#        padding = np.zeros((SEQ_LENGTH-1, X.shape[1]))
#        y = 100*tick_data[TARGET_COLUMNS].values
#        #y=np.clip(y,a_min=-3,a_max=3)/6
#        flag = tick_data['realData'].values
#        return np.concatenate((padding, X), axis=0), y, flag,mytime
#    except:
#        #print(f'data of {code} in {date} from {database} has error!!!')
#        return np.zeros((0, len(FEATURE_COLUMNS))), np.zeros((0, len(TARGET_COLUMNS))), np.zeros((0, 1)),[]

def get_tick_data(code,date,database,columns=All_COLUMNS):
    try:
        tick_data = getData(code,date,columns)
        if tick_data.shape[0]==0:
            #print(f'data of {code} in {date} from {database} has error!!!')
            return np.zeros((0, len(FEATURE_COLUMNS))), np.zeros((0, len(TARGET_COLUMNS))), np.zeros((0, 1)),[]
        tick_data=dataNormalization(tick_data,mynormalization)
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
            #print(f'num:{i}, {batch_seq_flag[i]}')
            na_idx[i]=1
    #batch_seq_X = batch_seq_X[na_idx == 0]
    #batch_seq_y = batch_seq_y[na_idx == 0]
    batch_seq_X[np.isnan(batch_seq_X)]=0
    batch_seq_y[np.isnan(batch_seq_y)]=0
    return batch_seq_X,batch_seq_y.flatten(),na_idx





#def statistics(code,date,model):
#   # batch_X, batch_y, batch_flag,mytime=get_tick_data_fromh5(code,date,All_COLUMNS)
#    batch_X, batch_y, batch_flag,mytime=get_tick_data(code,date,database,All_COLUMNS)
#    if (batch_X.shape[0]<1000):
#        return 
    
#    input,target,flag=  modifyData( batch_X, batch_y, batch_flag) 
#    predict=gbm.predict(input)
#    predictDf=pd.DataFrame(index=mytime)
#    predictDf['predict']=predict
#    predictDf['target']=target
#    predictDf['flag']=flag
#    #tickData=getDataFromH5(code,date,['B1','S1'])
#    tickData=getData(code,date,['B1','S1'])
#    #if tickData['B1'].mean()<10:
#    #    return
#    tickData[['predict','target','flag']]=predictDf[['predict','target','flag']]
#    tick=tickData.values
#    buy=0
#    sell=0
#    mybuy=0
#    mysell=0
#    num=0
#    step=20
#    for i in range(0,tick.shape[0]-1,step):
#        buy+=tick[i][1]
#        sell+=tick[i][0]
#        num=num+1
#        #如果要跌，等等再买
#        if (tick[i][2]<-0.015) & (tick[i][3]<1) & (i<(tick.shape[0]-2*step)):
#            mybuy+=tick[i+step][1]
#        else:
#            mybuy+=tick[i][1]
#        #如果要涨，等等再卖
#        if (tick[i][2]>0.015) & (tick[i][3]<1) & (i<(tick.shape[0]-2*step)):
#            mysell+=tick[i+step][0]
#        else:
#            mysell+=tick[i][0]
#    buy=np.round(buy/num,8)
#    sell=np.round(sell/num,8)
#    mybuy=np.round(mybuy/num,8)
#    mysell=np.round(mysell/num,8)
#    mid=(buy+sell)/2
#    print(code,date,np.round(r2_score(target[flag == 0],predict[flag == 0]),4),np.round(np.corrcoef(target[flag == 0],predict[flag == 0])[0][1],4))
#    #print(buy,sell,mid,mybuy,mybuy)
#    buyimprove=np.round((buy-mybuy)/buy,4)
#    sellimprove=np.round((mysell-sell)/sell,4)
#    buymidimprove=np.round((mid-mybuy)/mid,4)
#    sellmidimprove=np.round((mysell-mid)/mid,4)
#    print(buyimprove,sellimprove,buymidimprove,sellmidimprove)
#    print("==============================================================================")
#    pass

def myPredictSave(code,date,model_save_path,savepath):
    batch_X, batch_y, batch_flag, mytime = get_tick_data(code, date, database, All_COLUMNS)
    if (batch_X.shape[0] < 1000):
        return
    #model = lgb.Booster(model_file=model_save_path)
    model=model_save_path
    input, target, flag = modifyData(batch_X, batch_y, batch_flag)
    predict = model.predict(input)
    predictDf = pd.DataFrame(index=mytime)
    predictDf['predict'] = predict
    predictDf['target'] = target
    predictDf['flag'] = flag
    # tickData=getDataFromH5(code,date,['B1','S1'])
    tickData = getData(code, date, ['B1', 'S1'])
    tickData[['predict', 'target', 'flag']] = predictDf[['predict', 'target', 'flag']]
    tickData['code']=code
    tickData['date']=date
    savePredictDataToH5(code,date,savepath,tickData)
    print(f'predict of {code} in date {date} finished!!')
    pass


def myPredictSaveByCode(code,startDate,endDate,model_save_path,savepath):
    days=getTradedays(startDate,endDate)
    gbm = lgb.Booster(model_file=model_save_path)
    for date in days:
        myPredictSave(code,date,gbm,savepath)

def mystrategy(code, date, model_save_path,savepath):
    #batch_X, batch_y, batch_flag, mytime = get_tick_data(code, date, database, All_COLUMNS)
    #if (batch_X.shape[0] < 1000):
    #    return
    ##model = lgb.Booster(model_file=model_save_path)
    #model=model_save_path
    #input, target, flag = modifyData(batch_X, batch_y, batch_flag)
    #predict = model.predict(input)
    #predictDf = pd.DataFrame(index=mytime)
    #predictDf['predict'] = predict
    #predictDf['target'] = target
    #predictDf['flag'] = flag
    ## tickData=getDataFromH5(code,date,['B1','S1'])
    #tickData = getData(code, date, ['B1', 'S1'])
    #tickData[['predict', 'target', 'flag']] = predictDf[['predict', 'target', 'flag']]
    
    tickData=getPredictDataFromH5(code,date,savepath)
    if (tickData.shape[0]<=0):
        return pd.DataFrame()
    tick = tickData.values
    target=tick[:,3]
    predict=tick[:,2]
    flag=tick[:,4]
    buy = 0
    sell = 0
    mybuy = 0
    mysell = 0
    num = 0
    step = 20
    for i in range(0, tick.shape[0] - 1, step):
        buy += tick[i][1]
        sell += tick[i][0]
        num = num + 1

        buyPriceNow=tick[i][1]
        sellPriceNow=tick[i][0]
        # 如果要跌，等等再买
        if (tick[i][2] < -0.005) & (tick[i][3] < 1) & (i < (tick.shape[0] - 2 * step)):
            allreadybuy=False
            for j in range(step):
                if (tick[i+j][1]<buyPriceNow*0.998):
                    mybuy += tick[i + j][1]
                    allreadybuy=True
                    break
            if allreadybuy==False:
                mybuy += tick[i + step][1]
        else:
            mybuy += tick[i][1]
        # 如果要涨，等等再卖
        if (tick[i][2] > 0.005) & (tick[i][3] < 1) & (i < (tick.shape[0] - 2 * step)):
            allreadysell=False
            for j in range(step):
                if (tick[i+j][0]>sellPriceNow*1.002):
                    mysell += tick[i + j][0]
                    allreadysell=True
                    break
            if allreadysell==False:
                mysell += tick[i + step][0]
        else:
            mysell += tick[i][0]
    buy = np.round(buy / num, 8)
    sell = np.round(sell / num, 8)
    mybuy = np.round(mybuy / num, 8)
    mysell = np.round(mysell / num, 8)
    mid = (buy + sell) / 2
    r2 = np.round(r2_score(target[flag == 0], predict[flag == 0]), 4)
    mycorr = np.round(np.corrcoef(list(target[flag == 0]), list(predict[flag == 0]))[0][1], 4)
    # print(code, date, np.round(r2_score(target[flag == 0], predict[flag == 0]), 4),np.round(np.corrcoef(target[flag == 0], predict[flag == 0])[0][1], 4))
    # print(buy,sell,mid,mybuy,mybuy)
    buyimprove = np.round((buy - mybuy) / buy, 4)
    sellimprove = np.round((mysell - sell) / sell, 4)
    buymidimprove = np.round((mid - mybuy) / mid, 4)
    sellmidimprove = np.round((mysell - mid) / mid, 4)
    print(buyimprove, sellimprove, buymidimprove, sellmidimprove)
    print("==============================================================================")
    result = [
        {'code': code, 'date': date, 'buyimprove': buyimprove, 'sellimprove': sellimprove, 'r2': r2, 'corr': mycorr}]
    result = pd.DataFrame(data=result)

    return result

def mystrategyByCode(code,startDate,endDate,model_save_path,savepath):
    days=getTradedays(startDate,endDate)
    gbm = lgb.Booster(model_file=model_save_path)
    resultList=[]
    for date in days:
        result=mystrategy(code,date,gbm,savepath)
        resultList.append(result)
    resultList=pd.concat(resultList)
    return resultList
    pass


def myPredictSaveAll(stocks, mystart, myend, modelpath,savepath):
    Parallel(n_jobs=PREPARE_JOBS, verbose=0)(delayed(myPredictSaveByCode)(code, mystart,myend, modelpath,savepath) for code in stocks)
    pass

def mybacktest(stocks, mystart, myend, modelpath,savepath):
    #backtestList = getDataList(stocks, mystart, myend)
    #result = Parallel(n_jobs=PREPARE_JOBS, verbose=0)(delayed(mystrategy)(o['code'], o['date'], model) for o in backtestList)
    
    result = Parallel(n_jobs=PREPARE_JOBS, verbose=0)(delayed(mystrategyByCode)(code, mystart,myend, modelpath,savepath) for code in stocks)
    
    
    result = pd.concat(result)
    return result
    pass

stocks=getCodes(300)
#stocks=['600000.SH']
days=getTradedays(20180103,20191025)
mystart=20190201
myend=20190228
fileName='dart05020180102.txt'
savepath='dart05020180102'
model_save_path = os.path.join(LOCALDATAPATH, 'lightgbmModel', fileName)
PREPARE_JOBS=-1
gbm = lgb.Booster(model_file=model_save_path)


t0=time.time()
#myPredictSaveAll(stocks,mystart,myend,model_save_path,savepath)
t1=time.time()
print(t1-t0)




t0=time.time()
#data=mybacktest(stocks,mystart,myend,model_save_path)
data=mybacktest(stocks,mystart,myend,model_save_path,savepath)
t1=time.time()
print(t1-t0)

print(data['buyimprove'].mean())


