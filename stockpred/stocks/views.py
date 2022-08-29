from django.http import HttpResponse
from django.template.loader import get_template
from django.views import View
from xhtml2pdf import pisa
from io import BytesIO
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib import messages
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
import json
from keras.models import Sequential
from keras.layers import Dense, LSTM
# For time stamps
from datetime import datetime


def index(request):


    return render(request, 'stocks/index.html')

def all_companies(request):

    tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
    
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)

    for stock in tech_list:
        globals()[stock] = yf.download(stock, start, end)

    
    company_list = [AAPL, GOOG, MSFT, AMZN]
    company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]
    for company, com_name in zip(company_list, company_name):
        company["company_name"] = com_name
    
    df = pd.concat(company_list, axis=0)
    
    df=df.rename(columns={"Adj Close": "Adj_Close"})
    df1=df.index.strftime('%Y/%m/%d')
    df2=df1.values.tolist()  
    json_records=df.to_json(orient='records')

    arr=[]
    arr=json.loads(json_records)
    data=zip(arr,df2)

    for company, company_name in zip(company_list, tech_list):
        company["company_name"] = company_name

    #Closing Price View
    plt.figure(figsize=(15, 6))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company['Adj Close'].plot()
        plt.ylabel('Adj Close')
        plt.xlabel(None)
        plt.title(f"Closing Price of {tech_list[i - 1]}")
    
    plt.tight_layout()
    plt.savefig('static/images/ClosingPrices.png')
    

    #sales Volume

    plt.figure(figsize=(15, 7))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company['Volume'].plot()
        plt.ylabel('Volume')
        plt.xlabel(None)
        plt.title(f"Sales Volume for {tech_list[i - 1]}")
    
    plt.tight_layout()
    plt.savefig('static/images/SalesVolume.png')


    ma_day = [10, 20, 50]

    for ma in ma_day:
        for company in company_list:
            column_name = f"MA for {ma} days"
            company[column_name] = company['Adj Close'].rolling(ma).mean()
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(8)
    fig.set_figwidth(15)

    AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,0])
    axes[0,0].set_title('APPLE')

    GOOG[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[0,1])
    axes[0,1].set_title('GOOGLE')

    MSFT[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,0])
    axes[1,0].set_title('MICROSOFT')

    AMZN[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[1,1])
    axes[1,1].set_title('AMAZON')

    fig.tight_layout()
    fig.savefig('static/images/xyz.png')
    
    #daily return
    for company in company_list:
        company['Daily Return'] = company['Adj Close'].pct_change()

# Then we'll plot the daily return percentage
    fig1, axes1 = plt.subplots(nrows=2, ncols=2)
    fig1.set_figheight(8)
    fig1.set_figwidth(15)

    AAPL['Daily Return'].plot(ax=axes1[0,0], legend=True, linestyle='--', marker='o')
    axes1[0,0].set_title('APPLE')

    GOOG['Daily Return'].plot(ax=axes1[0,1], legend=True, linestyle='--', marker='o')
    axes1[0,1].set_title('GOOGLE')

    MSFT['Daily Return'].plot(ax=axes1[1,0], legend=True, linestyle='--', marker='o')
    axes1[1,0].set_title('MICROSOFT')

    AMZN['Daily Return'].plot(ax=axes1[1,1], legend=True, linestyle='--', marker='o')
    axes1[1,1].set_title('AMAZON')

    fig1.tight_layout()
    fig1.savefig('static/images/Dailyreturn.png')

    context={
        'json_records':arr,
        'df2':df2,
        'data':data,
    }
    return render(request, 'stocks/all_companies.html', context=context)


def CPS_spec(request):

    if request.method=='POST':

        company_name=request.POST.get('company_name')
        print(company_name)

        if company_name=="MSFT":
            c="MICROSOFT"
        elif company_name=="GOOG":
            c="GOOGLE"
        elif company_name=="AAPL":
            c="APPLE"
        else:
            c="AMAZON"

        yf.pdr_override()
        df = pdr.get_data_yahoo(company_name, data_source='yahoo', start='2012-01-01', end=datetime.now())
        df=df.rename(columns={"Adj Close": "Adj_Close"})
        df1=df.index.strftime('%Y/%m/%d')
        df2=df1.values.tolist()  
        json_records=df.to_json(orient='records')
    
        arr=[]
        arr=json.loads(json_records)
        dataCompany=zip(arr,df2)

        plt.figure(figsize=(16,6))
        plt.title('Close Price History For: '+ c)
        plt.plot(df['Close'])
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.tight_layout()
        plt.savefig('static/images/cph.png')


        data = df.filter(['Close'])
# Convert the dataframe to a numpy array
        dataset = data.values
# Get the number of rows to train the model on
        training_data_len = int(np.ceil( len(dataset) * .95 ))
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        train_data = scaled_data[0:int(training_data_len), :]
        x_train = []
        y_train = []


        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=10, epochs=3)
        


        test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
        x_test = np.array(x_test)

# Reshape the data  
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        rmse=float(rmse)
        

        #Plotting Predicted DATA
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
# Visualize the data
        plt.figure(figsize=(16,6))
        plt.title('Predictions from Model for company: '+ c)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.tight_layout()
        plt.savefig('static/images/pred.png')


        # valid and predicted prices
        df3=valid
        df4=df3.index.strftime('%Y/%m/%d')
        df5=df4.values.tolist()  
        json_records_valid=df3.to_json(orient='records')
        arr3=[]
        arr3=json.loads(json_records_valid)
        dataValid=zip(arr3,df5)

        context={
        'json_records':arr,
        'df2':df2,
        'data':dataCompany,
        'c':c,
        'rmse':rmse,
        'dataValid':dataValid,
         }

        return render (request, 'stocks/CPS_spec.html', context=context)

    return render (request, 'stocks/CPS_spec.html')


def cpscombine(request):

    tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

# Set up End and Start times for data grab
    tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

    yf.pdr_override()
# Get the stock quote
    df = pdr.get_data_yahoo('AAPL', data_source='yahoo', start='2012-01-01', end=datetime.now())
    df=df.rename(columns={"Adj Close": "Adj_Close"})
# Show teh data
    df1=df.index.strftime('%Y/%m/%d')
    df2=df1.values.tolist()  
    json_records=df.to_json(orient='records')
    
    arr=[]
    arr=json.loads(json_records)
    dataApple=zip(arr,df2)
    
    plt.figure(figsize=(16,8))
    plt.title('Close Price History (APPLE)')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.savefig('static/images/cph_Apple.png')

    #FOR MICROSOFT
    df_Mic = pdr.get_data_yahoo('MSFT', data_source='yahoo', start='2012-01-01', end=datetime.now())
    df_Mic=df_Mic.rename(columns={"Adj Close": "Adj_Close"})
# Show teh data
    df_Mic1=df_Mic.index.strftime('%Y/%m/%d')
    df_Mic2=df_Mic1.values.tolist()  
    json_records_Mic=df_Mic.to_json(orient='records')
    
    arr_Mic=[]
    arr_Mic=json.loads(json_records_Mic)
    dataMicrosoft=zip(arr_Mic,df_Mic2)
    
    plt.figure(figsize=(16,8))
    plt.title('Close Price History (Microsoft)')
    plt.plot(df_Mic['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.savefig('static/images/cph_Microsoft.png')

    #FOR Google
    df_Goo = pdr.get_data_yahoo('GOOG', data_source='yahoo', start='2012-01-01', end=datetime.now())
    df_Goo=df_Goo.rename(columns={"Adj Close": "Adj_Close"})
# Show teh data
    df_Goo1=df_Goo.index.strftime('%Y/%m/%d')
    df_Goo2=df_Goo1.values.tolist()  
    json_records_Goo=df_Goo.to_json(orient='records')
    
    arr_Goo=[]
    arr_Goo=json.loads(json_records_Goo)
    dataGoogle=zip(arr_Goo,df_Goo2)
    
    plt.figure(figsize=(16,8))
    plt.title('Close Price History (Google)')
    plt.plot(df_Goo['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.savefig('static/images/cph_Google.png')

    #FOR AMAZON
    df_Ama = pdr.get_data_yahoo('AMZN', data_source='yahoo', start='2012-01-01', end=datetime.now())
    df_Ama=df_Ama.rename(columns={"Adj Close": "Adj_Close"})
# Show teh data
    df_Ama1=df_Ama.index.strftime('%Y/%m/%d')
    df_Ama2=df_Ama1.values.tolist()  
    json_records_Ama=df_Ama.to_json(orient='records')
    
    arr_Ama=[]
    arr_Ama=json.loads(json_records_Ama)
    dataAmazon=zip(arr_Ama,df_Ama2)
    
    plt.figure(figsize=(16,8))
    plt.title('Close Price History (Amazon)')
    plt.plot(df_Ama['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.savefig('static/images/cph_Amazon.png')
    context={
        'json_records':arr,
        'dataApple':dataApple,
        'df2':df2,
        'arr_Mic':arr_Mic,
        'dataMicrosoft':dataMicrosoft,
        'arr_Goo':arr_Goo,
        'dataGoogle':dataGoogle,
        'arr_Ama':arr_Ama,
        'dataAmazon':dataAmazon,


    }
    return render (request,'stocks/cpscombine.html', context=context)


def specmain(request):

    if request.method=="POST":

        company=request.POST.get('company_name')
        data=request.POST.get('data_type')
        end = datetime.now()
        start = datetime(end.year - 1, end.month, end.day)
        


        print(company, data)

        if company=="AAPL":
            tech_list = ['AAPL']
            globals()['AAPL'] = yf.download('AAPL', start, end)
            company_list = [AAPL]
            company_name = ["APPLE"]


            if data=="ALL":
                for company, com_name in zip(company_list, company_name):
                    company["company_name"] = com_name
    
                df = pd.concat(company_list, axis=0)
                df=df.rename(columns={"Adj Close": "Adj_Close"})
# Show teh data
                df1=df.index.strftime('%Y/%m/%d')
                df2=df1.values.tolist()  
                json_records=df.to_json(orient='records')
    
                arr=[]
                arr=json.loads(json_records)
                print(arr)
                dataApple=zip(arr,df2)


                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)



                AAPL['Adj Close'].plot()
                plt.ylabel('Adj Close')
                plt.xlabel(None)
                plt.title(f"Closing Price History of APPLE")
    
                plt.tight_layout()
                plt.savefig('static/images/CCPHH_APPLE.png')


                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                AAPL['Volume'].plot()
                plt.ylabel('Volume')
                plt.xlabel('Date')
                plt.title(f"Sales Volume for APPLE")
    
                plt.tight_layout()
                plt.savefig('static/images/SV_APPLE.png')

                ma_day = [10, 20, 50]

                for ma in ma_day:
                    column_name = f"MA for {ma} days"
                    company[column_name] = company['Adj Close'].rolling(ma).mean()

                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot()
                plt.ylabel('Volume')
                plt.xlabel('Date')
                plt.title(f"Exploratory Analysis for APPLE")
                plt.tight_layout()
                plt.savefig('static/images/MAA_APPLE.png')
    
                for company in company_list:
                    company['Daily Return'] = company['Adj Close'].pct_change()

# Then we'll plot the daily return percentage
                plt.figure(figsize=(15, 7))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                AAPL['Daily Return'].plot()
                plt.ylabel('Daily Return')
                plt.xlabel('Date')
                plt.title(f"Daily Return for APPLE")
    
                plt.tight_layout()
                plt.savefig('static/images/DRR_APPLE.png')

                plt.figure(figsize=(12, 7))
                AAPL['Daily Return'].hist(bins=50)
                plt.ylabel('Daily Return')
                plt.title(f'APPLE Daily Return: HISTOGRAM')
    
                plt.tight_layout()
                plt.savefig('static/images/DRR_HIST_APPLE.png')



                AAPL['Daily Return'].hist(bins=50)
                plt.ylabel('Daily Return')
                plt.title(f'APPLE Daily Return: HISTOGRAM')
    
                plt.tight_layout()
                plt.savefig('static/images/DRR_HIST.png')

                context={
                    'json_records_APPLE_ALL':arr,
                    'dataApple':dataApple,
                }
                return render(request,'stocks/specmain.html', context=context)

            elif data=="CPS":
                type_of_data="CPS"
                for company, com_name in zip(company_list, company_name):
                    company["company_name"] = com_name
    
                df = pd.concat(company_list, axis=0)
                df=df.rename(columns={"Adj Close": "Adj_Close"})
                df=df.filter(['Adj_Close'])
                df1=df.index.strftime('%Y/%m/%d')
                df2=df1.values.tolist()  
                json_records_APPLE_CPS=df.to_json(orient='records')
                
                arr=[]
                arr=json.loads(json_records_APPLE_CPS)
                dataApple_CPS=zip(arr,df2)

                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)
                


                AAPL['Adj Close'].plot()
                plt.ylabel('Adj Close')
                plt.xlabel(None)
                plt.title(f"Closing Price History of APPLE")
    
                plt.tight_layout()
                plt.savefig('static/images/CCPHH_APPLE.png')

                context={
                    'type_of_data': type_of_data,
                    'json_records_APPLE_CPS':arr,
                    'dataApple_CPS':dataApple_CPS,
                }
                return render(request,'stocks/specmain.html', context=context)

            elif data=="SV":
                typeofdatat="SP"
                for company, com_name in zip(company_list, company_name):
                    company["company_name"] = com_name
    
                df = pd.concat(company_list, axis=0)
                
                df=df.filter(['Volume'])
                df1=df.index.strftime('%Y/%m/%d')
                df2=df1.values.tolist()  
                json_records_Vol_Apple=df.to_json(orient='records')
    
                arr=[]
                arr=json.loads(json_records_Vol_Apple)
                dataApple_SV=zip(arr,df2)

                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)



                AAPL['Volume'].plot()
                plt.ylabel('Sales Volume')
                plt.xlabel('Date')
                plt.title(f"Sales Volume of APPLE")
    
                plt.tight_layout()
                plt.savefig('static/images/SV_APPLE.png')

                context={
                    'typeofdatat':typeofdatat,
                    'json_records_Vol_Apple':arr,
                    'dataApple_SV':dataApple_SV,

                }
                return render(request,'stocks/specmain.html', context=context)

            elif data=="MA":
                ddtype="MA"

                for company, com_name in zip(company_list, company_name):
                    company["company_name"] = com_name
    
                df = pd.concat(company_list, axis=0)
                type="MA"

                ma_day = [10, 20, 50]

                for ma in ma_day:
                    column_name = f"MA for {ma} days"
                    company[column_name] = company['Adj Close'].rolling(ma).mean()

                plt.figure(figsize=(15, 7))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot()
                plt.ylabel('Volume')
                plt.xlabel('Date')
                plt.title(f"Sales Volume for APPLE")
    
                plt.tight_layout()
                plt.savefig('static/images/MAA_APPLE.png')

                context={
                    'ddtype':ddtype

                }
                return render(request,'stocks/specmain.html', context=context)

            else:
                dddtype="DR"
                for company in company_list:
                    company['Daily Return'] = company['Adj Close'].pct_change()

# Then we'll plot the daily return percentage
                plt.figure(figsize=(15, 7))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                AAPL['Daily Return'].plot()
                plt.ylabel('Daily Return')
                plt.xlabel('Date')
                plt.title(f"Daily Return for APPLE")
    
                plt.tight_layout()
                plt.savefig('static/images/DRR_APPLE.png')

                plt.figure(figsize=(12, 7))

                context={
                    'dddtype':dddtype

                }
                return render(request,'stocks/specmain.html', context=context)


        elif company=="GOOG":
            tech_list = ['GOOG']
            globals()['GOOG'] = yf.download('GOOG', start, end)
            company_list = [GOOG]
            company_name = ["GOOGLE"]
            
            if data=="ALL":
                for company, com_name in zip(company_list, company_name):
                    company["company_name"] = com_name
    
                df = pd.concat(company_list, axis=0)
                df=df.rename(columns={"Adj Close": "Adj_Close"})
# Show teh data
                df1=df.index.strftime('%Y/%m/%d')
                df2=df1.values.tolist()  
                json_records_GOOGLE=df.to_json(orient='records')
    
                arr=[]
                arr=json.loads(json_records_GOOGLE)
                dataGOOGLE=zip(arr,df2)


                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)



                GOOG['Adj Close'].plot()
                plt.ylabel('Adj Close')
                plt.xlabel('DATE')
                plt.title(f"Closing Price History of GOOGLE")
    
                plt.tight_layout()
                plt.savefig('static/images/CCPHH_GOOGLE.png')


                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                GOOG['Volume'].plot()
                plt.ylabel('Volume')
                plt.xlabel('Date')
                plt.title(f"Sales Volume for GOOGLE")
    
                plt.tight_layout()
                plt.savefig('static/images/SV_GOOGLE.png')

                ma_day = [10, 20, 50]

                for ma in ma_day:
                    column_name = f"MA for {ma} days"
                    company[column_name] = company['Adj Close'].rolling(ma).mean()

                plt.figure(figsize=(15, 7))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                GOOG[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot()
                plt.ylabel('Volume')
                plt.xlabel('Date')
                plt.title(f"EXPLORATORY ANALYSIS FOR GOOGLE")
                plt.tight_layout()
                plt.savefig('static/images/MAA_GOOGLE.png')
    
                for company in company_list:
                    company['Daily Return'] = company['Adj Close'].pct_change()

# Then we'll plot the daily return percentage
                plt.figure(figsize=(15, 7))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                GOOG['Daily Return'].plot()
                plt.ylabel('Daily Return')
                plt.xlabel('Date')
                plt.title(f"Daily Return for GOOGLE")
    
                plt.tight_layout()
                plt.savefig('static/images/DRR_GOOGLE.png')

                plt.figure(figsize=(12, 7))



                GOOG['Daily Return'].hist(bins=50)
                plt.ylabel('Daily Return')
                plt.title(f'GOOGLE Daily Return: HISTOGRAM')
    
                plt.tight_layout()
                plt.savefig('static/images/DRR_HIST_GOOGLE.png')
                context={
                    'json_records_GOOGLE':arr,
                    'dataGOOGLE':dataGOOGLE,

                }
                return render(request,'stocks/specmain.html', context=context)



            elif data=="CPS":
                CPS="CPS"

                for company, com_name in zip(company_list, company_name):
                    company["company_name"] = com_name
    
                df = pd.concat(company_list, axis=0)
                df=df.rename(columns={"Adj Close": "Adj_Close"})
                df=df.filter(['Adj_Close'])
                df1=df.index.strftime('%Y/%m/%d')
                df2=df1.values.tolist()  
                json_records_GCPS=df.to_json(orient='records')
    
                arr=[]
                arr=json.loads(json_records_GCPS)
                dataGOOGLE_CPS=zip(arr,df2)

                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)



                GOOG['Adj Close'].plot()
                plt.ylabel('Adj Close')
                plt.xlabel("Date")
                plt.title(f"Closing Price History of GOOGLE")
    
                plt.tight_layout()
                plt.savefig('static/images/CCPHH_GOOGLE.png')
                context={
                    'CPS_GOOGLE':CPS,
                    'json_records_GCPS':arr,
                    'dataGOOGLE_CPS':dataGOOGLE_CPS,

                }
                return render(request,'stocks/specmain.html', context=context)



            elif data=="SV":
                SV="SV"
                for company, com_name in zip(company_list, company_name):
                    company["company_name"] = com_name
    
                df = pd.concat(company_list, axis=0)
                
                df=df.filter(['Volume'])
                df1=df.index.strftime('%Y/%m/%d')
                df2=df1.values.tolist()  
                json_records_GSV=df.to_json(orient='records')
    
                arr=[]
                arr=json.loads(json_records_GSV)
                dataGOOGLE_SV=zip(arr,df2)

                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)



                GOOG['Volume'].plot()
                plt.ylabel('Sales Volume')
                plt.xlabel('Date')
                plt.title(f"Sales Volume of GOOGLE")
    
                plt.tight_layout()
                plt.savefig('static/images/SV_GOOGLE.png')

                context={
                    'SV_GOOGLE':SV,
                    'json_records_GSV':arr,
                    'dataGOOGLE_SV':dataGOOGLE_SV,

                }
                return render(request,'stocks/specmain.html', context=context)



            elif data=="MA":
                MA="MA"
                for company, com_name in zip(company_list, company_name):
                    company["company_name"] = com_name
    
                df = pd.concat(company_list, axis=0)
                type="MA"

                ma_day = [10, 20, 50]

                for ma in ma_day:
                    column_name = f"MA for {ma} days"
                    company[column_name] = company['Adj Close'].rolling(ma).mean()

                plt.figure(figsize=(15, 7))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                GOOG[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot()
                plt.ylabel('Volume')
                plt.xlabel('Date')
                plt.title(f"Exploratory Analysis for GOOGLE")
    
                plt.tight_layout()
                plt.savefig('static/images/MAA_GOOGLE.png')
                context={
                    'MA_GOOGLE':MA,
                    

                }
                return render(request,'stocks/specmain.html', context=context)

    

            else:
                DR="DR"
                for company in company_list:
                    company['Daily Return'] = company['Adj Close'].pct_change()

# Then we'll plot the daily return percentage
                plt.figure(figsize=(15, 7))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                GOOG['Daily Return'].plot()
                plt.ylabel('Daily Return')
                plt.xlabel('Date')
                plt.title(f"Daily Return for GOOGLE")
    
                plt.tight_layout()
                plt.savefig('static/images/DRR_GOOGLE.png')
                context={
                    'DR_GOOGLE':DR,
                    

                }
                return render(request,'stocks/specmain.html', context=context)

                

        elif company=="MSFT":
            tech_list = ['MSFT']
            globals()['MSFT'] = yf.download('MSFT', start, end)
            company_list = [MSFT]
            company_name = ["MICROSOFT"]
            
            if data=="ALL":
                for company, com_name in zip(company_list, company_name):
                    company["company_name"] = com_name
    
                df = pd.concat(company_list, axis=0)
                df=df.rename(columns={"Adj Close": "Adj_Close"})
# Show teh data
                df1=df.index.strftime('%Y/%m/%d')
                df2=df1.values.tolist()  
                json_records_MICROSOFT=df.to_json(orient='records')
    
                arr=[]
                arr=json.loads(json_records_MICROSOFT)
                dataMICROSOFT=zip(arr,df2)


                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)



                MSFT['Adj Close'].plot()
                plt.ylabel('Adj Close')
                plt.xlabel('DATE')
                plt.title(f"Closing Price History of MICROSOFT")
    
                plt.tight_layout()
                plt.savefig('static/images/CCPHH_MICROSOFT.png')


                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                MSFT['Volume'].plot()
                plt.ylabel('Volume')
                plt.xlabel('Date')
                plt.title(f"Sales Volume for MICROSOFT")
    
                plt.tight_layout()
                plt.savefig('static/images/SV_MICROSOFT.png')

                ma_day = [10, 20, 50]

                for ma in ma_day:
                    column_name = f"MA for {ma} days"
                    company[column_name] = company['Adj Close'].rolling(ma).mean()

                plt.figure(figsize=(15, 7))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                MSFT[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot()
                plt.ylabel('Volume')
                plt.xlabel('Date')
                plt.title(f"EXPLORATORY ANALYSIS FOR MICROSOFT")
                plt.tight_layout()
                plt.savefig('static/images/MAA_MICROSOFT.png')
    
                for company in company_list:
                    company['Daily Return'] = company['Adj Close'].pct_change()

# Then we'll plot the daily return percentage
                plt.figure(figsize=(15, 7))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                MSFT['Daily Return'].plot()
                plt.ylabel('Daily Return')
                plt.xlabel('Date')
                plt.title(f"Daily Return for MICROSOFT")
    
                plt.tight_layout()
                plt.savefig('static/images/DRR_MICROSOFT.png')

                plt.figure(figsize=(12, 7))



                MSFT['Daily Return'].hist(bins=50)
                plt.ylabel('Daily Return')
                plt.title(f'MICROSOFT Daily Return: HISTOGRAM')
    
                plt.tight_layout()
                plt.savefig('static/images/DRR_HIST_MICROSOFT.png')
                context={
                    'json_records_MICROSOFT':arr,
                    'dataMICROSOFT':dataMICROSOFT,

                }
                return render(request,'stocks/specmain.html', context=context)



            elif data=="CPS":
                CPS="CPS"

                for company, com_name in zip(company_list, company_name):
                    company["company_name"] = com_name
    
                df = pd.concat(company_list, axis=0)
                df=df.rename(columns={"Adj Close": "Adj_Close"})
                df=df.filter(['Adj_Close'])
                df1=df.index.strftime('%Y/%m/%d')
                df2=df1.values.tolist()  
                json_records_MCPS=df.to_json(orient='records')
    
                arr=[]
                arr=json.loads(json_records_MCPS)
                dataMICROSOFT_CPS=zip(arr,df2)

                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)



                MSFT['Adj Close'].plot()
                plt.ylabel('Adj Close')
                plt.xlabel("Date")
                plt.title(f"Closing Price History of MICROSOFT")
    
                plt.tight_layout()
                plt.savefig('static/images/CCPHH_MICROSOFT.png')
                context={
                    'CPS_MICROSOFT':CPS,
                    'json_records_MCPS':arr,
                    'dataMICROSOFT_CPS':dataMICROSOFT_CPS,

                }
                return render(request,'stocks/specmain.html', context=context)



            elif data=="SV":
                SV="SV"
                for company, com_name in zip(company_list, company_name):
                    company["company_name"] = com_name
    
                df = pd.concat(company_list, axis=0)
                
                df=df.filter(['Volume'])
                df1=df.index.strftime('%Y/%m/%d')
                df2=df1.values.tolist()  
                json_records_MSV=df.to_json(orient='records')
    
                arr=[]
                arr=json.loads(json_records_MSV)
                dataMICROSOFT_SV=zip(arr,df2)

                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)



                MSFT['Volume'].plot()
                plt.ylabel('Sales Volume')
                plt.xlabel('Date')
                plt.title(f"Sales Volume of MICROSOFT")
    
                plt.tight_layout()
                plt.savefig('static/images/SV_MICROSOFT.png')

                context={
                    'SV_MICROSOFT':SV,
                    'json_records_MSV':arr,
                    'dataMICROSOFT_SV':dataMICROSOFT_SV,

                }
                return render(request,'stocks/specmain.html', context=context)



            elif data=="MA":
                MA="MA"
                for company, com_name in zip(company_list, company_name):
                    company["company_name"] = com_name
    
                df = pd.concat(company_list, axis=0)
                type="MA"

                ma_day = [10, 20, 50]

                for ma in ma_day:
                    column_name = f"MA for {ma} days"
                    company[column_name] = company['Adj Close'].rolling(ma).mean()

                plt.figure(figsize=(15, 7))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                MSFT[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot()
                plt.ylabel('Volume')
                plt.xlabel('Date')
                plt.title(f"Exploratory Analysis for MICROSOFT")
    
                plt.tight_layout()
                plt.savefig('static/images/MAA_MICROSOFT.png')
                context={
                    'MA_MICROSOFT':MA,
                    

                }
                return render(request,'stocks/specmain.html', context=context)

    

            else:
                DR="DR"
                for company in company_list:
                    company['Daily Return'] = company['Adj Close'].pct_change()

# Then we'll plot the daily return percentage
                plt.figure(figsize=(15, 7))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                MSFT['Daily Return'].plot()
                plt.ylabel('Daily Return')
                plt.xlabel('Date')
                plt.title(f"Daily Return for MICROSOFT")
    
                plt.tight_layout()
                plt.savefig('static/images/DRR_MICROSOFT.png')

                
                context={
                    'DR_MICROSOFT':DR,
                    

                }
                return render(request,'stocks/specmain.html', context=context)

        else:
            tech_list = ['AMZN']
            globals()['AMZN'] = yf.download('AMZN', start, end)
            company_list = [AMZN]
            company_name = ["AMAZON"]
            
            if data=="ALL":
                for company, com_name in zip(company_list, company_name):
                    company["company_name"] = com_name
    
                df = pd.concat(company_list, axis=0)
                df=df.rename(columns={"Adj Close": "Adj_Close"})
# Show teh data
                df1=df.index.strftime('%Y/%m/%d')
                df2=df1.values.tolist()  
                json_records_AMAZON=df.to_json(orient='records')
    
                arr=[]
                arr=json.loads(json_records_AMAZON)
                dataAMAZON=zip(arr,df2)


                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)



                AMZN['Adj Close'].plot()
                plt.ylabel('Adj Close')
                plt.xlabel('DATE')
                plt.title(f"Closing Price History of AMAZON")
    
                plt.tight_layout()
                plt.savefig('static/images/CCPHH_AMAZON.png')


                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                AMZN['Volume'].plot()
                plt.ylabel('Volume')
                plt.xlabel('Date')
                plt.title(f"Sales Volume for AMAZON")
    
                plt.tight_layout()
                plt.savefig('static/images/SV_AMAZON.png')

                ma_day = [10, 20, 50]

                for ma in ma_day:
                    column_name = f"MA for {ma} days"
                    company[column_name] = company['Adj Close'].rolling(ma).mean()

                plt.figure(figsize=(15, 7))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                AMZN[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot()
                plt.ylabel('Volume')
                plt.xlabel('Date')
                plt.title(f"EXPLORATORY ANALYSIS FOR AMAZON")
                plt.tight_layout()
                plt.savefig('static/images/MAA_AMAZON.png')
    
                for company in company_list:
                    company['Daily Return'] = company['Adj Close'].pct_change()

# Then we'll plot the daily return percentage
                plt.figure(figsize=(15, 7))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                AMZN['Daily Return'].plot()
                plt.ylabel('Daily Return')
                plt.xlabel('Date')
                plt.title(f"Daily Return for AMAZON")
    
                plt.tight_layout()
                plt.savefig('static/images/DRR_AMAZON.png')

                plt.figure(figsize=(12, 7))



                AMZN['Daily Return'].hist(bins=50)
                plt.ylabel('Daily Return')
                plt.title(f'AMAZON Daily Return: HISTOGRAM')
    
                plt.tight_layout()
                plt.savefig('static/images/DRR_HIST_AMAZON.png')
                context={
                    'json_records_AMAZON':arr,
                    'dataAMAZON':dataAMAZON,

                }
                return render(request,'stocks/specmain.html', context=context)



            elif data=="CPS":
                CPS="CPS"

                for company, com_name in zip(company_list, company_name):
                    company["company_name"] = com_name
    
                df = pd.concat(company_list, axis=0)
                df=df.rename(columns={"Adj Close": "Adj_Close"})
                df=df.filter(['Adj_Close'])
                df1=df.index.strftime('%Y/%m/%d')
                df2=df1.values.tolist()  
                json_records_ACPS=df.to_json(orient='records')
    
                arr=[]
                arr=json.loads(json_records_ACPS)
                dataAMAZON_CPS=zip(arr,df2)

                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)



                AMZN['Adj Close'].plot()
                plt.ylabel('Adj Close')
                plt.xlabel("Date")
                plt.title(f"Closing Price History of AMAZON")
    
                plt.tight_layout()
                plt.savefig('static/images/CCPHH_AMAZON.png')
                context={
                    'CPS_AMAZON':CPS,
                    'json_records_ACPS':arr,
                    'dataAMAZON_CPS':dataAMAZON_CPS,

                }
                return render(request,'stocks/specmain.html', context=context)



            elif data=="SV":
                SV="SV"
                for company, com_name in zip(company_list, company_name):
                    company["company_name"] = com_name
    
                df = pd.concat(company_list, axis=0)
                
                df=df.filter(['Volume'])
                df1=df.index.strftime('%Y/%m/%d')
                df2=df1.values.tolist()  
                json_records_ASV=df.to_json(orient='records')
    
                arr=[]
                arr=json.loads(json_records_ASV)
                dataAMAZON_SV=zip(arr,df2)

                plt.figure(figsize=(15, 6))
                plt.subplots_adjust(top=1.25, bottom=1.2)



                AMZN['Volume'].plot()
                plt.ylabel('Sales Volume')
                plt.xlabel('Date')
                plt.title(f"Sales Volume of AMAZON")
    
                plt.tight_layout()
                plt.savefig('static/images/SV_AMAZON.png')

                context={
                    'SV_AMAZON':SV,
                    'json_records_ASV':arr,
                    'dataAMAZON_SV':dataAMAZON_SV,

                }
                return render(request,'stocks/specmain.html', context=context)



            elif data=="MA":
                MA="MA"
                for company, com_name in zip(company_list, company_name):
                    company["company_name"] = com_name
    
                df = pd.concat(company_list, axis=0)
                type="MA"

                ma_day = [10, 20, 50]

                for ma in ma_day:
                    column_name = f"MA for {ma} days"
                    company[column_name] = company['Adj Close'].rolling(ma).mean()

                plt.figure(figsize=(15, 7))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                AMZN[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot()
                plt.ylabel('Volume')
                plt.xlabel('Date')
                plt.title(f"Exploratory Analysis for AMAZON")
    
                plt.tight_layout()
                plt.savefig('static/images/MAA_AMAZON.png')
                context={
                    'MA_AMAZON':MA,
                    

                }
                return render(request,'stocks/specmain.html', context=context)

    

            else:
                DR="DR"
                for company in company_list:
                    company['Daily Return'] = company['Adj Close'].pct_change()

# Then we'll plot the daily return percentage
                plt.figure(figsize=(15, 7))
                plt.subplots_adjust(top=1.25, bottom=1.2)

                AMZN['Daily Return'].plot()
                plt.ylabel('Daily Return')
                plt.xlabel('Date')
                plt.title(f"Daily Return for AMAZON")
    
                plt.tight_layout()
                plt.savefig('static/images/DRR_AMAZON.png')

                
                context={
                    'DR_AMAZON':DR,
                    

                }
                return render(request,'stocks/specmain.html', context=context)





    return render(request,'stocks/specmain.html')

