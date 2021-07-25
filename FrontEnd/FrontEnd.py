import streamlit as st
import pandas as pd
import numpy as np
import mplfinance as mpf
import datetime
import time
import pandas_ta as ta
from IPython.display import clear_output
import math
import altair as alt
import base64
from io import BytesIO
from tvDatafeed.main import TvDatafeed,Interval
from PIL import Image

#image = Image.open('..\stockPic.jpg')
tv=TvDatafeed(chromedriver_path=None)


#st.beta_set_page_config(page_title='Velocity Trading')
def timePeriodFormatter(timePeriod):
    if (timePeriod=="1m"):
        resolution=Interval.in_1_minute
    elif (timePeriod=="3m"):
        resolution=Interval.in_3_minute
    elif (timePeriod=="5m"):
        resolution=Interval.in_5_minute
    elif (timePeriod=="15m"):
        resolution=Interval.in_15_minute
    elif (timePeriod=="30m"):
        resolution=Interval.in_30_minute
    elif (timePeriod=="45m"):
        resolution=Interval.in_45_minute
    elif (timePeriod=="1H"):
        resolution=Interval.in_1_hour
    elif (timePeriod=="2H"):
        resolution=Interval.in_2_hour
    elif (timePeriod=="3H"):
        resolution=Interval.in_3_hour
    elif (timePeriod=="4H"):
        resolution=Interval.in_4_hour
    elif (timePeriod=="1D"):
        resolution=Interval.in_daily
    elif (timePeriod=="1W"):
        resolution=Interval.in_weekly
    elif (timePeriod=="1M"):
        resolution=Interval.in_monthly
    else:
         resolution=Interval.in_5_minute
    return resolution


def signal(data,mag=100):
    data['signalLong']=np.nan
    data['signalShort']=np.nan
    cond1=data['reversal']==1
    cond2=data['pos']==1
    data.loc[cond1&cond2,'signalLong']=data['low']*0.9995
    data.loc[cond1& ( ~cond2),'signalShort']=data['high']*1.0005
    sigLong = data['signalLong'].tolist()
    sigShort = data['signalShort'].tolist()
    addPlot1= mpf.make_addplot(sigLong[-mag:],type='scatter',markersize=50,marker='^')
    addPlot2= mpf.make_addplot(sigShort[-mag:],type='scatter',markersize=50,marker='v')
    return addPlot1,addPlot2

def currTrade(data):
    data.reset_index(inplace=True,drop=False)
    curr=data.iloc[-1]
    rev=data[data['reversal']==1].iloc[-1]
    currPosition=curr.pos
    entry=rev.open
    price=curr.close
    if currPosition==1:
        net=price-entry
        position="Buy"
    else:
        net=entry-price
        position="Sell"
    net=round(net,2)
    row=data.iloc[4]
    data.set_index("datetime",drop=True,inplace=True)
    data=[row.symbol,position,rev.datetime.date(),rev.datetime.time(),curr.datetime.date(),curr.datetime.time(),entry,price,net]
    tradeOutput=["Symbol","Signal",'Entry Date','Entry Time','Current Date','Current Time',"Entry Price",'Current Price',"Net"]
    df = pd.DataFrame([data], columns = tradeOutput)
    return df

def ma(data, length, type, mag=100):
    if(type=='ema'):
        data['ma']=ta.ema(data['close'],length=length,fillna=0)
    else:
        data['ma']=ta.wma(data['close'],length=length,fillna=0)
    cond1=(data['close'].shift(1))>=(data['ma'].shift(1))
    cond2=(data['close'].shift(1))<(data['ma'].shift(1))
    data.loc[cond1,'pos']=1
    data.loc[cond2,'pos']=-1
    data['reversal']=0
    data.loc[data['pos'].shift(1)!=data['pos'],'reversal']=1
    addPlot1,addPlot2=signal(data,mag)
    addPlot3 = mpf.make_addplot(data['ma'].tail(mag))
    addPlot=[addPlot1,addPlot2,addPlot3]
    data=data[data['ma'].notna()]
    return data,addPlot

def ftsma(data, lenFast, lenMid, lenSlow, mag=100):
    data['slow']=ta.ema(data['close'],length=lenSlow,fillna=0)
    data['mid']=ta.ema(data['close'],length=lenMid,fillna=0)
    data['fast']=ta.ema(data['close'],length=lenFast,fillna=0)

    cond1=(data['fast'].shift(1))>(data['mid'].shift(1))
    cond2=(data['close'].shift(1))>(data['slow'].shift(1))
    data.loc[cond1 & cond2 ,'pos']=1
    data.loc[~cond1 & ~cond2,'pos']=-1
    data=data.ffill()
    data['reversal']=0
    data.loc[data['pos'].shift(1)!=data['pos'],'reversal']=1
    
    addPlot1,addPlot2=signal(data,mag)
    addPlot3 = mpf.make_addplot(data['fast'].tail(mag))
    addPlot4 = mpf.make_addplot(data['mid'].tail(mag))
    addPlot5 = mpf.make_addplot(data['slow'].tail(mag))
    addPlot=[addPlot1,addPlot2,addPlot3,addPlot4,addPlot5]
    return data,addPlot

def supertrend(data, atrlen, superMult,mag=100):
    df=ta.supertrend(data['high'],data['low'],data['close'],atrlen,superMult)
    data['pos']=df.iloc[:,1]
    data.pos=data.pos.shift(1)
    data['reversal']=0
    data.loc[data['pos'].shift(1)!=data['pos'],'reversal']=1
    addPlot1,addPlot2=signal(data,mag)
    addPlot=[addPlot1,addPlot2]
    return data,addPlot

def psar(data, psarAf, psarMaxAf,mag=100):
    df=ta.psar(data['high'],data['low'],data['close'],psarAf,psarMaxAf)
    data['pos']=df.iloc[:,0]
    data=data.fillna(-1)
    data.loc[data['pos']!=-1.0,'pos']=1
    data.pos=data.pos.shift(1)
    data['reversal']=0
    data.loc[data['pos'].shift(1)!=data['pos'],'reversal']=1
    addPlot1,addPlot2=signal(data,mag)
    addPlot=[addPlot1,addPlot2]
    return data,addPlot

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="download.xlsx">Download list of trades as an excel file</a>' # decode b'abc' => abc


def backtest(data,start,end,comm=0.0,flag=1,mult=5):
    trades=data[data['reversal']==1]
    trades.reset_index(inplace=True)
    trades['date']=trades['datetime'].dt.date
    trades=trades[(trades['date']>=start) & (trades['date']<=end)]
    if trades.empty:
        placeholder1.write("No trades executed during this time interval")
        return 0
    trades['Entry Time']=trades['datetime'].shift(1).dt.time
    trades['Exit Time']=trades['datetime'].dt.time
    trades['Entry Date']=trades['datetime'].shift(1).dt.date
    trades['Exit Date']=trades['datetime'].dt.date
    trades['Signal']='Sell'
    trades.loc[trades['pos'].shift(1)==1.0,'Signal']='Buy'
    trades['Entry']=trades['open'].shift(1)
    trades['Exit']=trades['open']
    trades['Net(After Commissions)']=(trades['open']-trades['open'].shift(1))*mult
    trades.loc[trades['Signal']=='Sell','Net(After Commissions)']=-1.0*trades['Net(After Commissions)']
    trades['Net(After Commissions)']=trades['Net(After Commissions)']-(trades['open']*comm*mult/100)
    trades['Total P/L(After Commissions)']=trades['Net(After Commissions)'].cumsum(axis=0)
    trades['Symbol']=trades['symbol']
    trades['Price']=trades['open']
    trades['ROI(%)']=(trades['Net(After Commissions)']/trades['Entry'])*100/mult
    buyHold=trades['close'].iloc[-1]-trades['open'].iloc[0]
    startVal=trades['open'].iloc[0]
    trades=trades[['Symbol','Entry Date','Entry Time','Exit Date','Exit Time','Signal','Entry','Exit','Net(After Commissions)','Total P/L(After Commissions)','Price','ROI(%)','datetime']]
    totalNet=trades.iloc[-1]['Total P/L(After Commissions)']
    totalComm=(comm/100)*mult*trades['Entry'].sum()
    netLong=trades.loc[trades['Signal']=='Buy','Net(After Commissions)'].sum()
    netShort=trades.loc[trades['Signal']=='Sell','Net(After Commissions)'].sum()
    trades.reset_index(inplace=True,drop=True)
    trades.dropna(inplace=True)
    rows=[netLong,netShort,len(trades.index),totalComm,totalNet,(netLong/(startVal*mult))*100,(totalNet/(startVal*mult))*100,buyHold*mult,(buyHold/startVal)*100]
    tradeOutput=["Net Buy(After Commissions)","Net Sell(After Commissions)","Trades","Total Commissions","Net P/L(After Commissions)","Buy ROI(%)","Total Buy/Sell ROI(%)","Buy and Hold P/L","Buy Hold ROI(%)"]
    df = pd.DataFrame([rows], columns = tradeOutput)
    trades['Net']=trades['Net(After Commissions)']
    if flag==1:
        trades['Buy P/L(After Commissions)']=trades[trades['Signal']=='Buy'].Net.cumsum()
        trades=trades.ffill()
        trades.set_index('datetime',inplace=True)
        placeholder1.line_chart(trades['Price'], use_container_width=True)
        placeholder2.line_chart(trades['Buy P/L(After Commissions)'], use_container_width=True)
        placeholder3.line_chart(trades['Total P/L(After Commissions)'], use_container_width=True)
        placeholder4.write(df)
        trades=trades[['Symbol','Signal','Entry Date','Entry Time','Exit Date','Exit Time','Entry','Exit','Net(After Commissions)','Total P/L(After Commissions)','Buy P/L(After Commissions)','ROI(%)']]
        trades.reset_index(drop=True,inplace=True)
        placeholder5.write(trades)
        st.markdown(get_table_download_link(trades), unsafe_allow_html=True)
    else:
        trades=trades[['Symbol','Signal','Entry Date','Entry Time','Exit Date','Exit Time','Entry','Exit','Net','ROI(%)']]
        df=trades.iloc[-10:]
        df=df.reset_index(drop=True)
        placeholder4.write(df,use_container_width=True)
    
    

    

def datafeed(mult,commission,start_date,end_date,mode,stop=False,
resolution="1m",
tick='BTCUSD',
exc='Bitstamp',
indicator=1,
future=0,
mag=100,
ma_len=50,
fastMa=5,
midMa=20,
slowMa=200,
atrlen=10,
superMult=3,
psarAf=0.02,
psarMaxAf=0.2,
waitTime=30):
    data=0
    resolution=timePeriodFormatter(resolution)
    global plots
    while(stop!=True):
        clear_output(wait=True)
        df=data
        message.empty()
        try:
            if(future==1 | future==2):
                data=tv.get_hist(tick, exc,interval=resolution,n_bars=5000,fut_contract=future)
            else:
                data=tv.get_hist(tick, exc,interval=resolution,n_bars=5000)
        except:
            message.error('Connection Error Retrying')
            time.sleep(waitTime)
            data=df
        if(indicator==1):
            data, plots=ma(data,ma_len,"ema",mag)
        elif(indicator==2):
            data, plots=ma(data,ma_len,"wma",mag)
        elif(indicator==3):
            data, plots=ftsma(data,fastMa,midMa,slowMa,mag)
        elif(indicator==4):
            data, plots=supertrend(data,atrlen, superMult,mag)
        elif(indicator==5):
            data, plots=psar(data,psarAf,psarMaxAf)
        else:
            data, plots=ma(data,ma_len,"ema",mag)
        if mode=='Backtesting':
            backtest(data,start_date,end_date,commission,mult=mult)
            break
        
        plots = [x for x in plots if np.isnan(x['data']).all() == False]
        
        placeholder1.pyplot(mpf.plot(data.tail(mag),hlines=dict(hlines=[data['close'].iloc[-1]],colors=['b'],linestyle='-.'),type='candle',style='yahoo',title = tick,tight_layout=True,addplot=plots,figsize=(8, 3)))
        lstRef="Last Chart Refresh - "+str(datetime.datetime.now().time())
        placeholder2.text(lstRef)
        placeholder3.write(currTrade(data),use_container_width=False)
        backtest(data,start_date,end_date,flag=0)
        #time.sleep(waitTime)

def main():
    datafeed(mult,commission,start_date,end_date,mode,stop,timePeriod,tick,exc,indicator,future,mag,ma_len,fastMa,midMa,slowMa,atrlen,superMult,psarAf,psarMaxAf,waitTime)


st.set_page_config(layout="wide")
st.title('Velocite Trading')
mark=st.markdown("""
  Use the menu on the left to select data and set plot parameters, and then click Start
""")
mult=1
stop=False
timePeriod="1"
tick='TCS'
exc='NSE'
future=0
waitTime=1
mode='Live Trades'
indicator=1

# 1   -   EMA
# 2   -   WMA
# 3   -   FTSMA
# 4   -   SuperTrend
# 5   -   Parabolic SAR

mag=100
ma_len=50

fastMa=5
midMa=20
slowMa=200

atrlen=5
superMult=1

psarAf=0.02
psarMaxAf=0.2

commission=0.0
today=tommorow=0
end_date=datetime.date.today()
start_date=end_date-datetime.timedelta(days=10)

mode = st.sidebar.selectbox('Select Mode:', ( "Live Trades","Backtesting"))

tick = st.sidebar.text_input('Ticker:', 'NIFTY')
if st.sidebar.checkbox("Future"):
    option = st.sidebar.selectbox('Select Expiry',('Current Month Expiry', 'Next Month Expiry'))
    if(option=="Current Month Expiry"):
        future=1
    else:
        future=2
exc = st.sidebar.text_input('Exchange:', 'NSE')



indicatorSelect = st.sidebar.radio("Select Indicator :",('E-Velocite', 'W-Velocite', 'F-Velocite','S-Velocite','P-Velocite'))



if indicatorSelect == 'E-Velocite':
    indicator=1
elif indicatorSelect == 'W-Velocite':
    indicator=2
elif indicatorSelect == 'F-Velocite':
    indicator=3
elif indicatorSelect == 'S-Velocite':
    indicator=4
elif indicatorSelect == 'P-Velocite':
    indicator=5
else:
    indicator=1

timePeriod = st.sidebar.selectbox('Select Time Period:', ( "15m","1m","3m","5m","30m","45m","1H","2H","3H","4H","1D","1W","1M"))



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



if mode=="Live Trades":
    mag = st.sidebar.slider('Chart Magnification :', 1, 200, 100)
else:
    testDur = st.sidebar.selectbox('Select Backtest Duration:', ("1W","1D","1M","3M","6M","1Y"))
    if(testDur=='1D'):
        start_date=end_date-datetime.timedelta(days=1)
    elif(testDur=='1W'):
        start_date=end_date-datetime.timedelta(days=7)
    elif(testDur=='1M'):
        start_date=end_date-datetime.timedelta(days=30)
    elif(testDur=='3M'):
        start_date=end_date-datetime.timedelta(days=90)
    elif(testDur=='6M'):
        start_date=end_date-datetime.timedelta(days=180)
    elif(testDur=='1Y'):
        start_date=end_date-datetime.timedelta(days=365)
    else:
        start_date=end_date-datetime.timedelta(days=30)
    start_date = st.sidebar.date_input('Start date', start_date)
    end_date = st.sidebar.date_input('End date', end_date)
    if start_date > end_date:
        st.sidebar.error('Error: End date must fall after start date.')
    commission = st.sidebar.number_input('Enter Commission per trade (% of contract bought)',value=0.125)
    mult = st.sidebar.number_input('Number of Units',value=1,min_value=1)
    commission*=2
   
start_button = st.sidebar.empty()
stop_button = st.sidebar.empty()
#with st.spinner('Wait for it...'):
#time.sleep(5)
#st.success('Done!')

st.set_option('deprecation.showPyplotGlobalUse', False)
message=st.empty()
#placeholder1 = st.image(image)
placeholder1 = st.empty()
placeholder2=st.empty()
placeholder3=st.empty()
placeholder4=st.empty()
placeholder5=st.empty()


if start_button.button('start',key='start'):
    start_button.empty()
    mark.empty()
    if stop_button.button('stop',key='stop'):
        pass
    main()