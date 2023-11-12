# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:37:32 2023

@author: Source
"""

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret
#==============================================================================
# HEADER
#==============================================================================    

def render_header():
    """
    This function render the header of the dashboard with the following items:
        - Title
        - Dashboard description
        - 3 selection boxes to select: Ticker, Start Date, End Date
        """
    # Add dashboard title and description
    st.title("📈 MY FINANCIAL DASHBOARD 📉") 
    
    # Get the list of stock tickers from S&P500
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'] 
    
    # Set these variables as global, so the functions in all of the tabs can read them 
    global start_date
    global end_date 
    global Date_range 
    global select_intervals 
    global ticker 
    
    #create 3 columns to place the 3 main select boxes: one for the ticker selection, one for the "Update and Download button and one for the date range 
    col1, col2, col3 = st.columns(3)  
    # Add the ticker selection on the sidebar
    with col1:
        ticker = col1.selectbox("Ticker", ticker_list)
        
    # Add the "Update and Download Data" button 
    with col2: 
        col2.write("") 
        if st.button("Update and Download Data"):
            if ticker:
                st.write(f"Updating stock data for {ticker}...")

                # Fetch stock data using yfinance
                stock_data = yf.download(ticker)
                # Display the updated data
                st.write(stock_data)
                # Dowload data as CSV File
                csv_filename = f"{ticker}_data.csv"
                stock_data.to_csv(csv_filename, index=False)
                # Creat a link for downloading of the CSV File
                st.markdown(f'<a href="data:file/csv;base64,{stock_data.to_csv(index=False).encode()}" download="{csv_filename}">Download CSV</a>', unsafe_allow_html=True)
    
    #Add the selection box related to the time range for which we want to have the informations through the five visualizations
    with col3: 
        different_durations = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"]
        Date_range=st.selectbox("Date_range:", different_durations)
        #Define the end date 
        end_date = datetime.now()
        
        #Define the start_date for each interval of time that we considered 
        if Date_range=="1M":
            start_date = end_date - timedelta(days=30)
        if Date_range=='3M':
            start_date= end_date - timedelta(days=90)
        if Date_range=="6M":
            start_date= end_date - timedelta(days=120)
        if Date_range=="YTD":
            start_date=(datetime.now().year,1,1)
        if Date_range=="1Y":
            start_date= end_date - timedelta(days=365)
        if Date_range=="3Y":
            start_date= end_date - timedelta(days=1095)
        if Date_range=="5Y":
            start_date= end_date - timedelta(days=1825) 
        if Date_range=="MAX": 
            start_date = end_date - pd.DateOffset(years = 100)
        
#==============================================================================
# TAB 1
#==============================================================================

def render_tab1(): 
    st.header("SUMMARY")
    """
    This function render the Tab 1 - Company Profile of the dashboard.
    """
    
    # Create two columns: one for the key statistics and one for the area chart in order them to be close like in the YahooFinance website
    col1, col2 = st.columns(2)
    
    # Get the company information
    @st.cache_data
    def GetCompanyInfo(ticker): 
        """
        This function get the company information from Yahoo Finance.
        """
        return YFinance(ticker).info
    
    # Create another another level with two columns: one for the summary on the left and one for the informations about the major shareholders 
    col1, col2 = st.columns(2)    
    
    # in the first column of the first level, show the statistics as a DataFrame and call it: "Key Statistics"
    with col1: 
        info = GetCompanyInfo(ticker)
        st.write('**Key Statistics:**')
        info_keys = {'previousClose':'Previous Close',
                     'open'         :'Open',
                     'bid'          :'Bid',
                     'ask'          :'Ask',
                     'marketCap'    :'Market Cap',
                     'volume'       :'Volume',
                     'averageVolume':'Avg. Volume',
                     'beta'         :'Beta',
                     'trailingEps'  :'EPS (TTM)',
                     'lastDividendDate' :'Ex-Dividend Date',
                     '52WeekChange' :'52 Week Range'}
        
        # Create a dictionary 
        company_stats = {}
        #for each key in the info_keys dictionary, update the company_stats dictionary 
        for key in info_keys:
            company_stats.update({info_keys[key]:info[key]})
        # Convert to DataFrame
        company_stats = pd.DataFrame({'Value':pd.Series(company_stats)})
        st.dataframe(company_stats)
        
    # In the second column of the first level, plot the area chart with the closing prices of the stocks over time    
    with col2: 
            st.write('**Closing Price:**')
            stock_price = yf.Ticker(ticker).history(start=start_date, end=end_date) 
            fig = go.Figure() 
            fig.add_trace(go.Scatter(x=stock_price.index, y=stock_price['Close'], stackgroup='one', name='area_chart'))   
            st.plotly_chart(fig)
            
    #in the first column of the second level, plot the company profile and description as a table        
    with col1: 
        # If the ticker is already selected
        if ticker != '': 
            info = GetCompanyInfo(ticker)
            
            # Show the company description using markdown + HTML
            st.write('**Business Summary:**')
            st.markdown('<div style="text-align: justify;">' + \
                        info['longBusinessSummary'] + \
                        '</div><br>',
                        unsafe_allow_html=True)
                
    # In the second column of the second level, plot the informations abou the major shareholders as a table             
    with col2:        
        def major_shareholders(ticker): 
            if ticker != '':
                # Get and visualize the informations about the major shareholders 
                shareholders_info = yf.Ticker(ticker).get_major_holders()
                st.write(f"**Major Shareholders of {ticker}:**")
                table = st.table(shareholders_info)
                return table 
        major_shareholders(ticker)
        
#==============================================================================
# TAB 2
#==============================================================================

def render_tab2(): 
    
    st.header("CHART")
    
    # Create 2 columns for the selection boxes: one to be able to choose between the Line plot and the Candlestick chart and the other to select the time intervals 
    col1, col2 = st.columns(2)
    
    # Create a selectbox for the type of chart we want to see 
    with col1:
        chart_type = st.selectbox("Select Chart Type:", ['Line Plot', 'Candlestick Chart'])
        
    # Create a selectbox to choose among the time interval of 1 day, 1 week and 1 month
    with col2: 
        select_intervals= st.selectbox("Intervals:", ["1d", "1wk", "1mo"])
        
        if select_intervals == '1D': 
            stock_price = yf.Ticker(ticker).history(start=start_date, end=end_date, interval='1d')
        if select_intervals == '1W': 
            stock_price = yf.Ticker(ticker).history(start=start_date, end=end_date, interval='1wk')
        if select_intervals == '1M':
            stock_price = yf.Ticker(ticker).history(start=start_date, end=end_date, interval='1m')
    
    # if the ticker is selected and if the chart type corresponds to the "Line plot", create and visualize a line plot of the closing prices of the stock 
    if ticker != '': 
        if chart_type == 'Line Plot': 
            st.write('**Line Plot**') 
            stock_price = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=select_intervals) 
            fig = go.Figure(data=[go.Scatter( 
                x=stock_price.index, 
                y=stock_price['Close'], 
                mode='lines', 
                name='Line')]) 
            st.plotly_chart(fig) 
            
    # If the chart type corresponds to Candlestick, then create a candlestick chart representing the open, close, high and low price of the stocks 
    if chart_type == 'Candlestick Chart': 
        st.write('**Candlestick Chart**') 

        stock_price = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=select_intervals)
        fig = go.Figure(data=[go.Candlestick(
            x=stock_price.index, 
            open=stock_price['Open'], 
            high=stock_price['High'], 
            low=stock_price['Low'], 
            close=stock_price['Close'],
            name = "Candlestick chart")])
        
        # Add the trading volume at the bottom of the chart through double Y axis visualization 
        fig.add_trace(go.Bar(
            x=stock_price.index,
            y=stock_price['Volume'],
            name='Volume',
            yaxis='y2', 
            marker=dict(color='rgba(0, 0, 255, 0.5)')
        )) 
        
        # Verify if there are enough data in order to calculate the MA ( Simple moving average)
        if len(stock_price) >= 50: 
            # Add the line of the simple moving average with a window of 50 days
            ma_values = stock_price['Close'].rolling(window=50, min_periods=1).mean()
        
            # Add NaN values for the initial period where MA is not defined
            ma_values[:49] = None
            
            # Add the line of the simple moving average with a window of 50 days
            fig.add_trace(go.Scatter(
                x=stock_price.index,
                # Calculate the MA:
                y=ma_values,
                mode='lines',
                name='MA (50 days)',
                line=dict(color='purple'),
                legendgroup='group'
            )) 


        # update the layout for a better visualization
        fig.update_layout(
            yaxis=dict
            (title='Price'),
            yaxis2=dict(title='Volume', overlaying='y', side='right'),
            height=600,
            width=800,
            showlegend=True, 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        #show the graph on streamlit 
        st.plotly_chart(fig)
 
#==============================================================================
# TAB 3
#==============================================================================

def render_tab3(): 
    st.header(f"Financial informations for {ticker}:")
    
    # Create 3 columns: one to select the financial information type 
    col1, col2, col3 = st.columns(3)
    
    # Create the selectbox to choose among "Income Statement", "Balance Sheet"and "Cashflow" type of information
    with col1: 
        financial_info_type = st.selectbox("Select financial information type:", ("Income Statement", "Balance Sheet", "Cashflow")) 
    
    # Create the selectbox to choose between the period length ( Annual or Quaterly) for which we want to visualize our informations 
    with col2: 
        period = st.selectbox("Select period:", ("Annual", "Quarterly"))
        
    # Download the financial information as dataframes for each type of information and each period length   
    with col3: 
        def financial_statement(ticker, financial_info_type, period): 
            # Ottieni i dati finanziari dal simbolo azionario utilizzando yfinance 
            if (financial_info_type == "Income Statement" and period == "Annual"): 
                st.dataframe(yf.Ticker(ticker).income_stmt)
            if (financial_info_type=='Income Statement' and period=="Quarterly"):   
                st.dataframe(yf.Ticker(ticker).quarterly_income_stmt)
            if (financial_info_type == "Balance Sheet" and period == "Annual"):
                st.dataframe(yf.Ticker(ticker).balance_sheet)
            if (financial_info_type=="Balance Sheet" and period== "Quarterly"):
                st.dataframe(yf.Ticker(ticker).quarterly_balance_sheet)
            if (financial_info_type == "Cashflow" and period == "Annual"):
                st.dataframe(yf.Ticker(ticker).cashflow)
            if (financial_info_type=="Cashflow" and period=="Quarterly"):
                st.dataframe(yf.Ticker(ticker).quarterly_cashflow)
    financial_statement(ticker, financial_info_type, period)
    
#==============================================================================
# TAB 4
#==============================================================================             
            
def render_tab4(): 
    st.header(f"Monte Carlo Simulation for {ticker}:") 
    
    # Create two columns for the selctboxes we want for this tab 
    col1, col2 = st.columns(2)
    
    # Create another level with one column where to plot the Montacarlo Simulation 
    col3 = st.columns(1)
    
    # Create a selectbox to choose among 200, 500 or 1000 simulations 
    with col1: 
        simulations = st.selectbox('Select the number of simulation:', (200, 500, 1000)) 
        
    # Create a selectbox to choose among different time horizons for which we want to visualize the simulation: 30, 60 or 90 
    with col2: 
        time_horizon = st.selectbox('Select the time horizon:', (30, 60, 90)) 
        
    # Plot and visualize the Montecarlo Simulation 
    with col3[0]: 
        def montecarlo_simulation(ticker, simulations, time_horizon): 
            
            # Create a variable that represent the stock prices within a period length of 5 years, the closing prices, the daily return, the daily volatility (the standard deviation of the prices) and the last prices 
            stock_price= yf.Ticker(ticker).history(period = "5y")
            close_price = stock_price['Close']
            daily_return = close_price.pct_change()
            daily_volatility = np.std(daily_return)
            last_price = close_price[-1]
            
            # Generate the stock price of next 30 days 
            next_price = []
        
            for n in range(time_horizon):
               future_return = np.random.normal(0, daily_volatility)
            
               # Generate the random future price
               future_price = last_price * (1 + future_return)
                
               # Save the price and go next
               next_price.append(future_price)
               last_price = future_price
             
            np.random.seed(123)
        
            # Run the simulation
            simulation_df = pd.DataFrame()
        
            for i in range(simulations): 
            
            
                # The list to store the next stock price
                next_price = []
            
                # Create the next stock price
                last_price = close_price[-1]
            
                for j in range(time_horizon):
                    # Generate the random percentage change around the mean (0) and std (daily_volatility)
                    future_return = np.random.normal(0, daily_volatility)
            
                    # Generate the random future price
                    future_price = last_price * (1 + future_return)
            
                    # Save the price and go next
                    next_price.append(future_price)
                    last_price = future_price
                next_price_df = pd.Series(next_price).rename('sim' + str(i))
                simulation_df = pd.concat([simulation_df, next_price_df], axis=1)
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot the prices
            ax.plot(simulation_df)
            ax.axhline(y=close_price[-1], color='red')
            
            # Customize the Montacarlo Simulation plot 
            ax.set_title(f"Monte Carlo simulation for {ticker} stock price in the next {time_horizon} days:")
            ax.set_xlabel('Day')
            ax.set_ylabel('Price')
            ax.legend(['Current stock price is: ' + str(np.round(close_price[-1], 2))])
            ax.get_legend().legend_handles[0].set_color('red')
            
            ending_price = simulation_df.iloc[-1:, :].values[0, ]
            
            # Price at 95% confidence interval
            future_price_95ci = np.percentile(ending_price, 5)
            
            # Estimate and represent the value at Risk at 95% confident intervals 
            # 95% of the time, the losses will not be more than 16.35 USD
            VaR = close_price[-1] - future_price_95ci
            st.subheader('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')
            st.plotly_chart(fig)
                
        montecarlo_simulation(ticker, simulations, time_horizon)
        
#==============================================================================
# TAB 5
#==============================================================================        
        
def render_tab5(): 
    
    # Defining a function in order to get the information about the dividends for each stock from 2014 till 2024 and represent them in a line chart 
    def dividends_information(ticker):
        st.header("DIVIDEND CHARTS")
        dividend_data = yf.Ticker(ticker).dividends 
        dividend_data = dividend_data['2014-01-01':]
        return dividend_data 
    
    # Create the the line chart    
    if ticker != '': 
        dividend_data= dividends_information(ticker)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dividend_data.index, y=dividend_data.values,
                                 mode='lines+markers', name='Dividends informations'))
    
        fig.update_layout(title=f"{ticker} Dividends",
                          showlegend=True) 
        st.plotly_chart(fig)

    # defining the function "fix_for graph" in order to substitute the zero values    
    def fix_for_graph(data): 
        dividend_growth = dividend_data.pct_change() * 100
        # Create a list of all years contained in the data 
        years = dividend_growth.index.year.unique()
        
        # For every year in the list, get the value of the dividend in May and replace all the values of the year with that value
        for year in years:
            val_year_in_may = dividend_growth[(dividend_growth.index.year == year) & (dividend_growth.values != 0) & (dividend_growth.values != np.nan)].values[0]
            dividend_growth.loc[dividend_growth.index.year == year] = val_year_in_may 
        return dividend_growth

    # Defining the function "growth" in order to get the values related to the percentage growth rates of the dividends from 2014 to 2024 and plot them as a line chart     
    def growth(ticker): 
        dividend_data= dividends_information(ticker)
        # Download the information about the Dividends percentage change 
        dividend_growth = dividend_data.pct_change() * 100 
        # call the fix_fo_graph_function in order to not get zero values
        dividend_growth = fix_for_graph(dividend_growth)
        return dividend_growth 
    
    # If the ticker is selected, then create the line chart for the percentage change rate of the dividends for each stock
    if ticker != '': 
            # Call the function
            dividend_growth = growth(ticker)
            
            try:
            # Set a condition if the variable dividend_growth is not empty, then create the line chart  
                if not dividend_growth.empty: 
                    # Convert the index into dates 
                    dividend_growth.index = pd.to_datetime(dividend_growth.index) 
                    # Create the scatter plot unified by a line representing the dividend growth rates 
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dividend_growth.index, y=dividend_growth.values,
                                             mode='lines+markers', name='Percentage change rate'))
                
                    fig.update_layout(title=f"{ticker} Dividend Growth (YoY)",
                                      showlegend=True)
                    st.plotly_chart(fig) 
                    
            # If there are no informations about the dividend change rates, then return the message that there are no data available     
            except AttributeError: 
                st.warning(f"No dividend growth data available for {ticker}.")
            
#==============================================================================
# Main body
#==============================================================================

render_header()
  
# Render the tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Charts","Finance","Monte Carlo Simulation","Dividends"])
with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()
with tab4:
    render_tab4()
with tab5:
    render_tab5()
    
# Customize the dashboard with CSS
st.markdown(
    """
    <style>
        .stApp {
            background: #F0F8FF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
    
###############################################################################
# END
###############################################################################


