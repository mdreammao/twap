import influxdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from joblib import dump, load, Parallel, delayed
import dateutil.parser as dtparser
import datetime
import torch
import os
os.environ['NUMEXPR_MAX_THREADS'] = '6'
import warnings
warnings.filterwarnings("ignore") 
import time
from sklearn.linear_model import SGDClassifier
import sklearn
from numpy.lib.stride_tricks import as_strided
import lightgbm as lgb

# GLOBAL PART
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
BATCH_SIZE = 200
SEQ_LENGTH = 10
VALIDATION_SIZE = 5
INLOOP_SIZE = 10000
PREPARE_JOBS = 12   
startDate=20180101
endDate=20181231
testStart=20190103
testEnd=20190103
model_save_path=os.path.join(LOCALDATAPATH,'lightgbmModel.txt')
All_COLUMNS=USEFUL_COLUMNS+TARGET_COLUMNS

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
        y=np.clip(y,a_min=-1,a_max=1)/6
        flag = tick_data['realData'].values
        return np.concatenate((padding, X), axis=0), y, flag
    except:
        #print(f'data of {code} in {date} from {database} has error!!!')
        return np.zeros((0, len(FEATURE_COLUMNS))), np.zeros((0, len(TARGET_COLUMNS))), np.zeros((0, 1))




def modifyData(batch_X, batch_y, batch_flag):
    timeSeries=batch_X.shape[0]
    factors=batch_X.shape[1]
    modifiedTimeSeries=timeSeries-SEQ_LENGTH+1
    modifiedFactors=factors*SEQ_LENGTH
    if timeSeries<100:
        return np.zeros((0,modifiedFactors)),np.zeros((0,1))
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
    return batch_seq_X,batch_seq_y.flatten()

def getDataAssumble(prepared_data):
    inputs=np.zeros((0, len(FEATURE_COLUMNS)*SEQ_LENGTH))
    targets=np.zeros((0, 1))
    data_list = Parallel(n_jobs=PREPARE_JOBS, verbose=0)(delayed(modifyData)(item[0],item[1],item[2]) for item in prepared_data)
    batch_seq_X=[]
    batch_seq_y=[]
    hasData=False
    for item in data_list:
        if item[0].shape[0]>1000:
            batch_seq_X.append(item[0])
            batch_seq_y.append(item[1])
            hasData=True
    if hasData==False:
        return [],[]
    inputs=np.concatenate(batch_seq_X)
    targets=np.concatenate(batch_seq_y)
    return inputs,targets
    pass

def mytrain(stocks,startDate,endDate,BATCH_SIZE1,BATCH_SIZE2):
    model_save_path=os.path.join(LOCALDATAPATH,'lightgbmModel',endDate+'.txt')
    train_list=getDataList(stocks,startDate,endDate)
    batch_start, batch_end = 0, 0
    max_batch = len(train_list)
    current_loop = 1
    test_loop=1
    gbm = None
    params = {
            'task': 'train',
            'application': 'regression',  # 目标函数
            #'application': 'quantile',  # 目标函数
            #'boosting_type': 'gbdt',  # 设置提升类型
            'boosting_type': 'dart',  # 设置提升类型
            'drop_rate':0.8,
            'learning_rate': 0.01,  # 学习速率
            'num_leaves': 1000,  # 叶子节点数
            'tree_learner': 'serial',
            'min_data_in_leaf': 10,
            'metric': ['l1', 'l2', 'rmse'],  # l1:mae, l2:mse  # 评估函数
            'max_bin': 255,
            'num_trees':1000,
            'max_depth':50,
            'num_threads':8,
            #'feature_fraction':1,
            'verbose':1
    }
    while batch_start != max_batch:
        batch_middle=min(max_batch-2, batch_start + BATCH_SIZE1)
        batch_end = min(max_batch, batch_start + BATCH_SIZE1+BATCH_SIZE2)
        #train_list_now = train_list[batch_start:batch_end ]
        #prepared_data = Parallel(n_jobs=PREPARE_JOBS, verbose=0)(delayed(get_tick_data)(o['code'], o['date'], database, All_COLUMNS) for o in train_list_now)
        #inputsAll,targetsAll=getDataAssumble(prepared_data)
        #inputs,testInputs,targets, testTargets = train_test_split(inputsAll, targetsAll, test_size=0.2)
        train_list_now = train_list[batch_start:batch_middle ]
        test_list_now=train_list[batch_middle+1: batch_end]
        prepared_data = Parallel(n_jobs=PREPARE_JOBS, verbose=0)(delayed(get_tick_data)(o['code'], o['date'], database, All_COLUMNS) for o in train_list_now)
        inputs,targets=getDataAssumble(prepared_data)
        prepared_data = Parallel(n_jobs=PREPARE_JOBS, verbose=0)(delayed(get_tick_data)(o['code'], o['date'], database, All_COLUMNS) for o in test_list_now)
        testInputs,testTargets=getDataAssumble(prepared_data)
        batch_start=batch_start+BATCH_SIZE
        if (len(inputs)==0) | (len(testInputs)==0):
            continue
        #开始训练模型lightbgm
        # 重点来了，通过 init_model 和 keep_training_booster 两个参数实现增量训练
        lgb_train = lgb.Dataset(inputs, targets)
        lgb_eval = lgb.Dataset(testInputs, testTargets, reference=lgb_train)
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=200,
                        valid_sets=lgb_eval,
                        init_model=gbm,  # 如果gbm不为None，那么就是在上次的基础上接着训练
                        # feature_name=x_cols,
                        early_stopping_rounds=20,
                        verbose_eval=True,
                        keep_training_booster=True)
        # 输出模型评估分数
        score_train = dict([(s[1], s[2]) for s in gbm.eval_train()])
        pred_targets = gbm.predict(testInputs)
        r2=r2_score(testTargets,pred_targets)
        print("times: {} ".format(current_loop))  # 当前次数
        print('当前模型在训练集的得分是：mae=%.4f, mse=%.4f, rmse=%.4f'
                % (score_train['l1'], score_train['l2'], score_train['rmse']))  
        print('当前模型在训练集的R2是：R2=%.4f'% (r2)) 
        gbm.save_model(model_save_path)
        current_loop+=1
    pass


days=getTradedays(startDate,endDate)
trainNum=20
stocks=getCodes()
train_list=getDataList(stocks,startDate,endDate)
test_list=getDataList(stocks,testStart,testEnd)
BATCH_SIZE1=(int)(round(0.7*BATCH_SIZE,0))
BATCH_SIZE2=(int)(round(0.3*BATCH_SIZE,0))

for i in range(trainNum,len(days)-1,1):
    start0=days[i-trainNum]
    end0=days[trainNum]
    mytrain(stocks,start0,end0,BATCH_SIZE1,BATCH_SIZE2)
    pass










