import influxdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from joblib import dump, load, Parallel, delayed
import dateutil.parser as dtparser
import datetime
import torch
import os
os.environ['NUMEXPR_MAX_THREADS'] = '4'
import warnings
warnings.filterwarnings("ignore") 
import time
from sklearn.linear_model import SGDClassifier
import sklearn
from numpy.lib.stride_tricks import as_strided

# GLOBAL PART
#device = torch.device("cuda:0")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
file=os.path.join(r'd:/BTP/LocalDataBase','normalization20190712.h5')
with pd.HDFStore(file,'r',complib='blosc:zstd',append=True,complevel=9) as store:
    mynormalization=store['data']
FEATURE_COLUMNS =list(mynormalization['name'])
#TARGET_COLUMNS = ['buyPriceIncreaseNext15s','sellPriceIncreaseNext15s']
TARGET_COLUMNS = ['midIncreaseNext1m']
USEFUL_COLUMNS=FEATURE_COLUMNS+['realData']
#startDate=20180401
#endDate=20190628
database='MaoTickFactors20190831'
INFLUXDBHOST='192.168.58.71'
LOCALDATAPATH=r'd:/BTP/LocalDataBase'
BATCH_SIZE = 100
SEQ_LENGTH = 10
VALIDATION_SIZE = 5
INLOOP_SIZE = 10000
PREPARE_JOBS = 8   
startDate=20170101
endDate=20171231
testStart=20180101
testEnd=20180131

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

All_COLUMNS=USEFUL_COLUMNS+TARGET_COLUMNS
def get_tick_data(code,date,database,columns=All_COLUMNS):
    try:

        tick_data = getDataFromInfluxdb(code,date,database,columns)
        if tick_data.shape[0]==0:
            #print(f'data of {code} in {date} from {database} has error!!!')
            return np.zeros((0, len(FEATURE_COLUMNS))), np.zeros((0, len(TARGET_COLUMNS))), np.zeros((0, 1))
        ##标准化
        tick_data=dataNormalization(tick_data,mynormalization)
        # replace inf to na
        tick_data = tick_data.replace(np.inf, np.nan)
        tick_data = tick_data.replace(-np.inf, np.nan)
        ##去掉开头和结尾的nan
        startIndex=tick_data[tick_data.isna().sum(axis=1)==0].index[0]
        endIndex=tick_data[tick_data.isna().sum(axis=1)==0].index[-1]
        tick_data=tick_data.loc[startIndex:endIndex,:]
        ## split X, y 
        X = tick_data[FEATURE_COLUMNS].values
        padding = np.zeros((SEQ_LENGTH-1, X.shape[1]))
        y = 100*tick_data[TARGET_COLUMNS].values
        #y=np.clip(y,a_min=-3,a_max=3)/6
        flag = tick_data['realData'].values
        return np.concatenate((padding, X), axis=0), y, flag
    except:
        #print(f'data of {code} in {date} from {database} has error!!!')
        return np.zeros((0, len(FEATURE_COLUMNS))), np.zeros((0, len(TARGET_COLUMNS))), np.zeros((0, 1))


stocks=getCodes()
train_list=getDataList(stocks,startDate,endDate)
test_list=getDataList(stocks,testStart,testEnd)

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
    batch_seq_X = batch_seq_X[na_idx == 0]
    batch_seq_y = batch_seq_y[na_idx == 0]
    return batch_seq_X,batch_seq_y

load_index_list = np.random.permutation(len(train_list))
test_index_list=np.random.permutation(len(test_list))
batch_start, batch_end = 0, 0
max_batch = len(train_list)
start_time = time.time()
current_loop = 1
test_loop=1
while batch_start != max_batch:
    batch_end = min(max_batch, batch_start + BATCH_SIZE)
    batch_idx = load_index_list[batch_start: batch_end]
    prepared_data = Parallel(n_jobs=PREPARE_JOBS, verbose=0)(delayed(get_tick_data)(o['code'], o['date'], database, All_COLUMNS) for o in [train_list[z] for z in batch_idx])
    sample_count = 0
    inputs=np.zeros((0, len(FEATURE_COLUMNS)*SEQ_LENGTH))
    targets=np.zeros((0, 1))
    for loop_idx, (batch_X, batch_y, batch_flag) in enumerate(prepared_data):
        if batch_X.shape[0] < SEQ_LENGTH:
            continue
        [batch_seq_X,batch_seq_y]=modifyData(batch_X, batch_y, batch_flag)
        #sample_count += batch_X.shape[0]
        #timeSeries=batch_X.shape[0]
        #factors=batch_X.shape[1]
        #modifiedTimeSeries=timeSeries-SEQ_LENGTH+1
        #modifiedFactors=factors*SEQ_LENGTH
        #batch_seq_X = np.zeros((modifiedTimeSeries,modifiedFactors))
        #batch_seq_y = batch_y
        #batch_seq_flag = batch_flag
        ##break
        ## remove na (must remove before forward)
        #na_idx=np.zeros(batch_seq_X.shape[0])
        #for i in range(batch_seq_X.shape[0]):
        #    batch_seq_X[i,:]=batch_X[i:i+SEQ_LENGTH,:].copy().flatten()
        #    if ((np.isnan(batch_seq_X[i,:]).any()==True) | (np.isnan(batch_seq_y[i]).any()==True)| (batch_seq_flag[i]==0)):
        #        na_idx[i]=1
        ##seq_batch_size = batch_seq_X.shape[0]
        ##na_X = torch.isnan(batch_seq_X.contiguous().view(seq_batch_size, -1)).sum(dim=1)
        ##na_y = torch.isnan(batch_seq_y.contiguous().view(seq_batch_size, -1)).sum(dim=1)
        ##na_idx = na_X + na_y + (1 - batch_seq_flag.long())
        #batch_seq_X = batch_seq_X[na_idx == 0]
        #batch_seq_y = batch_seq_y[na_idx == 0]
        if len(batch_seq_X) == 0:
            continue
        if inputs.shape[0]==0:
            inputs=batch_seq_X
            targets=batch_seq_y
        else:
            inputs=np.concatenate([inputs,batch_seq_X])
            targets=np.concatenate([targets,batch_seq_y])
    

    #开始训练模型
    clf = sklearn.linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
    clf.partial_fit(inputs,targets)
    testInputs=np.zeros((0, len(FEATURE_COLUMNS)*SEQ_LENGTH))
    testTargets=np.zeros((0, 1))
    while testInputs.shape[0]==0:
        batch_X, batch_y, batch_flag=get_tick_data(test_list[test_index_list[test_loop]]['code'], test_list[test_index_list[test_loop]]['date'],database, All_COLUMNS)
        if (batch_X.shape[0]>1000):
            testInputs,testTargets=modifyData(batch_X, batch_y, batch_flag)
            test_loop+=1
    print("times: {} ".format(current_loop))  # 当前次数
    print("scores: {} ".format(clf.score(testInputs,testTargets)))  # 在测试集上看效果
    batch_start=batch_start+BATCH_SIZE
    current_loop+=1






