import streamlit as st
import pandas as pd
import numpy as np
import mplfinance as mpf
from tvDatafeed import TvDatafeed,Interval
import datetime
import time
import pandas_ta as ta
from pynse import *
from IPython.display import clear_output
import math
import altair as alt

username = 'omnipotent203'
password = 'aditya1234'

tv=TvDatafeed(username, password, chromedriver_path=None)

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

def trades(data):
    data.reset_index(inplace=True,drop=False)
    curr=data.iloc[-1]
    rev=data[data['reversal']==1].iloc[-1]
    currPosition=curr.pos
    entry=rev.open
    price=curr.close
    if currPosition==1:
        net=price-entry
        position="Long"
    else:
        net=entry-price
        position="Short"
    net=round(net,2)
    data.set_index("datetime",drop=True,inplace=True)
    tradeOutput="Last Trade : "+str(position)+" Entry Time : "+str(rev.datetime)+" Entry Price : "+str(entry)+' Current Time : '+str(curr.datetime)+' Current Price : '+str(price)+" Net : "+str(net)
    trade.text(tradeOutput)

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
    data['reversal']=0
    data.loc[data['pos'].shift(1)!=data['pos'],'reversal']=1
    addPlot1,addPlot2=signal(data,mag)
    addPlot=[addPlot1,addPlot2]
    return data,addPlot

def datafeed(stop=False,
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
        try:
            data=tv.get_hist(tick, exc,interval=resolution,n_bars=1000)
        except:
            st.error('Connection Error')
            st.stop()
            data=df
        if(future==1 | future==2):
            data=tv.get_hist(tick, exc,interval=resolution,n_bars=1000,fut_contract=future)
        else:
            data=tv.get_hist(tick, exc,interval=resolution,n_bars=1000)
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

        plots = [x for x in plots if np.isnan(x['data']).all() == False]
        
        placeholder.pyplot(mpf.plot(data.tail(mag),hlines=dict(hlines=[data['close'].iloc[-1]],colors=['b'],linestyle='-.'),type='candle',style='yahoo',title = tick,tight_layout=True,addplot=plots,figsize=(8, 3)))
        lstRef="Last Chart Refresh - "+str(datetime.datetime.now().time())
        lastRefresh.text(lstRef)
        trades(data)
        ohlc.write(data[['symbol','open','high','low','close']].tail(mag))
        #time.sleep(waitTime)

def main():
    datafeed(stop,timePeriod,tick,exc,indicator,future,mag,ma_len,fastMa,midMa,slowMa,atrlen,superMult,psarAf,psarMaxAf,waitTime)

    
st.set_page_config(layout="wide")
st.sidebar.title('Control Panel')
st.title('Velocity Trading')
mark=st.markdown("""
  Use the menu on the left to select data and set plot parameters, and then click Start
""")

stop=False
timePeriod="1"
tick='TCS'
exc='NSE'
future=0
waitTime=1

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

atrlen=10
superMult=3

psarAf=0.02
psarMaxAf=0.2

tick = st.sidebar.text_input('Ticker:', 'NIFTY')
if st.sidebar.checkbox("Future"):
    option = st.sidebar.selectbox('Select Expiry',('Current Month Expiry', 'Next Month Expiry'))
    if(option=="Current Month Expiry"):
        future=1
    else:
        future=2
exc = st.sidebar.text_input('Exchange:', 'NSE')



indicatorSelect = st.sidebar.radio("Select Indicator :",('EVelocity', 'WMA', 'FTSMA','SuperTrend','Parabolic SAR'))



if indicatorSelect == 'EVelocity':
    indicator=1
elif indicatorSelect == 'WMA':
    indicator=2
elif indicatorSelect == 'FTSMA':
    indicator=3
elif indicatorSelect == 'SuperTrend':
    indicator=4
elif indicatorSelect == 'Parabolic SAR':
    indicator=5
else:
    indicator=1

timePeriod = st.sidebar.selectbox('Select Time Period:', ( "1m","3m","5m","15m","30m","45m","1H","2H","3H","4H","1D","1W","1M"))



#hide_menu_style = """
#        <style>
#        #MainMenu {visibility: hidden;}
#        </style>
#        """
#st.markdown(hide_menu_style, unsafe_allow_html=True)




mag = st.sidebar.slider('Chart Magnification :', 1, 200, 100)

start_button = st.sidebar.empty()
stop_button = st.sidebar.empty()
#with st.spinner('Wait for it...'):
#time.sleep(5)
#st.success('Done!')

st.set_option('deprecation.showPyplotGlobalUse', False)
placeholder = st.empty()
lastRefresh=st.empty()
trade=st.empty()
ohlc=st.empty()


if start_button.button('start',key='start'):
    start_button.empty()
    mark.empty()
    if stop_button.button('stop',key='stop'):
        pass
    while True:
        main()