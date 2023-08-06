""" Brian Main function for Final project of DS Tools 1"""
""" Program to take in a stock ticker symbol, compare that company to its """
""" closest competition and identify best in class company for further"""
""" review."""

import logging
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import requests
from bs4 import BeautifulSoup as bsoup
from datetime import datetime
import finviz_redo as fv
import re
import os
from finvizfinance.screener.overview import Overview
from finvizfinance.screener.custom import Custom, COLUMNS as fvColumns
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers = []
    
    #Create console handler and set level
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    #Create file handler and set level - file to go to current directory
    file_handler = logging.FileHandler(
        filename=os.path.join(os.getcwd(), r'finvest_log.log'))
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def stck_base_data(sbd_stck, harvest_bool = True):
    """get base stock HTML data from finviz site"""
    #logf.info(f"call finviz basic data to get base stock design")
    sbd_stck_soup = fv.web_scrap(sbd_stck)
    sbd_stck_pretty = sbd_stck_soup.prettify()

    with open ('stck_prty_soup.txt', 'w') as f:
        f.write(sbd_stck_pretty)

    if harvest_bool:
        columns=['Ticker\n\n','Company','Sector','Industry','Country','Exchange']
        sbd_stck_soup = pd.DataFrame(stck_soup_harvest(sbd_stck_soup,columns),index=[0])

    return sbd_stck_soup  

def stck_soup_harvest(h_soup,col_names):
    """return dict of finviz scraped data"""
    logf.info(f"get full company overview of given stock")
    cat_name = ""
    cat_data = ""
    ssh_out_dict={}
    cat_switcher = True
    #ssh_out_list = list()

    ssh_table=h_soup.find("table",class_="fullview-title").find_all('a')

    for idx, a_tag in enumerate(ssh_table):
        ssh_out_dict[col_names[idx]]=a_tag.text.strip()
        #ssh_out_list.append({col_names[idx]: a_tag.text.strip()})

    ssh_table_data = h_soup.find("table",class_="snapshot-table2").find_all('td')

    for rec in ssh_table_data:
        if cat_switcher == True:
            cat_name = rec.text.strip()
            cat_switcher = False
        else:
            cat_data = rec.text.strip()
            cat_switcher = True
            ssh_out_dict[cat_name]=cat_data
    
    return ssh_out_dict

def extract_mrk_cp_lmt(string):
    """function to extract Market Cap string and calc a lower limit"""
    logf.info(f"extract Market Cap data to create lower limit for competitor comps")
    pattern = r'(\d+(\.\d+)?)([BM])'
    matches = re.findall(pattern, string)

    if matches:
        num_str, _, suffix = matches[0]
        multiplier = 1
        if suffix == 'B':
            multiplier = 1e8
        elif suffix == 'M':
            multiplier = 1e5
        return (float(num_str)//10) * multiplier

def screen_stcks(base_df):
    """function takes base stock data and applies filter criteria to screening site for like companies"""
    logf.info(f"call screener data of similar stocks to main starter stock")

    cls_ovrvw = Custom()
    sub_dict = base_df[['Sector','Industry','Country']].to_dict(orient="records")[0]

    cls_ovrvw.set_filter(filters_dict=sub_dict)
    ovr_df=cls_ovrvw.screener_view(columns = list(map(str,fvColumns.keys())))

    lmt_mrkt_cap = extract_mrk_cp_lmt(base_df['Market Cap'][0])

    base_df['Debt/Eq']=base_df['Debt/Eq'].fillna(0)
    lmt_dbt_cap = ((float(base_df['Debt/Eq'][0])//.1)+4)/10


    lmt_ovr_df = ovr_df[(ovr_df['Market Cap'] >= lmt_mrkt_cap) & (ovr_df['Debt/Eq'] < lmt_dbt_cap)].reset_index()

    return lmt_ovr_df

def rank_screener(rs_stck_df, rs_base_stck):
    """rank categories in the given df for a total rank"""
    ranking_columns = ['P/E','Fwd P/E','P/S','P/B','P/C','P/FCF','Debt/Eq','EPS','ROA','ROE','ROI','52W High']
    ascending_rank = [True, True, True, True, True, True, True, False, False, False, False, True]

    for x in range(len(ranking_columns)):
        new_col_name = 'rank ' + ranking_columns[x]
        nan_opt = 'bottom' if ascending_rank[x] else 'top'
        rs_stck_df[new_col_name]=rs_stck_df[ranking_columns[x]].rank(method='average',na_option = nan_opt, ascending=ascending_rank[x])
    
    rank_columns = rs_stck_df.filter(like = 'rank')
    rs_stck_df['total_rank']=rank_columns.sum(axis=1)  
    rs_stck_df['total_rank']=rs_stck_df['total_rank'].rank(method='average', ascending = True)
    rs_stck_df.drop(columns=rank_columns, inplace=True)
    rs_stck_df = rs_stck_df.sort_values(by="total_rank").reset_index(drop=True)
    
    main_stock_idx = rs_stck_df[rs_stck_df['Ticker\n\n']==rs_base_stck].index.item()

    avg_dict = {}
    avg_dict['index']=round(rs_stck_df['index'].mean(axis=0),2)
    avg_dict['Ticker\n\n'] = '_AVG_'

    for col in rs_stck_df.columns[2:]:
        col_bool = is_numeric_dtype(rs_stck_df[col])
        if col_bool:
            temp_value = round(rs_stck_df[col].mean(axis=0),2)
            avg_dict[col] = temp_value
        else:
            temp_value = rs_stck_df[col][main_stock_idx]
            avg_dict[col] = temp_value

    avg_dict_df = pd.DataFrame([avg_dict])

    rs_stck_df = pd.concat([rs_stck_df, avg_dict_df], ignore_index=True)

    rs_stck_df['cat_grp'] = 4
    rs_stck_df['cat_desc'] = 'Others'
    top_stck_cnt = 3

    for idx, rec in rs_stck_df.iterrows():
        if rec['Ticker\n\n'] == rs_base_stck or rec['Ticker\n\n'] == '_AVG_':
            continue
        elif top_stck_cnt >0:
            rs_stck_df.at[idx, 'cat_grp'] = 2
            rs_stck_df.at[idx, 'cat_desc'] = 'Top Comp'
            top_stck_cnt += -1
        else:
            break

    rs_stck_df.loc[rs_stck_df['Ticker\n\n'] == rs_base_stck, 'cat_grp'] = 1
    rs_stck_df.loc[rs_stck_df['Ticker\n\n'] == rs_base_stck, 'cat_desc'] = 'Main'
    rs_stck_df.loc[rs_stck_df['Ticker\n\n'] == '_AVG_', 'cat_grp'] = 3
    rs_stck_df.loc[rs_stck_df['Ticker\n\n'] == '_AVG_', 'cat_desc'] = 'Grp Avg'

    return rs_stck_df

def base_charting(bc_stck_df):
    """create correlation chart and top correlation relationships"""
    map_df = bc_stck_df.select_dtypes(include=np.number)
    map_df = map_df.drop(columns=['index', 'cat_grp','total_rank','Target Price','Volume'])
    corr_col_lst = ['Fwd P/E','P/S','P/B','P/FCF','ROA','ROE','ROI','Debt/Eq','Gross M','Oper M',]
    map_df = map_df[corr_col_lst]
    sns.heatmap(data=map_df.corr(), cmap = 'coolwarm', annot=True, fmt='.2f')
    plt.savefig('base_corr.png')
    plt.close()

    corr_values = map_df.corr().unstack()
    corr_values = corr_values.drop_duplicates()
    corr_values = corr_values[corr_values < 1]
    corr_values = corr_values.sort_values(ascending=False)
    top_corr_cat = corr_values.head(3)
    top_inv_corr_cat = corr_values.tail(3)

    top_corr_df = top_corr_cat.reset_index()
    top_inv_corr_df = top_inv_corr_cat.reset_index()

    col_names = ['Elem1', 'Elem2', 'c_score']
    top_corr_df.columns = col_names
    top_inv_corr_df.columns = col_names

    return top_corr_df, top_inv_corr_df


def main_prog(input_stck="MSFT"):
    
    #logf = setup_logger()
    
    logf.info(f"stock ticker input: {input_stck}")
    base_stck = input_stck
        
    columns=['Ticker\n\n','Company','Sector','Industry','Country','Exchange']
    #stck_df = pd.DataFrame(columns = columns)

    logf.info(f"pull base stock data Overview")
    mp_soup = stck_base_data(base_stck, False) 
    stck_df = pd.DataFrame(stck_soup_harvest(mp_soup,columns),index=[0])
    


    logf.info(f"base stock id: {stck_df.iloc[:,:5]}")

    logf.info(f"use base stock characteristics to screen for similar companies")
    main_stck_df = screen_stcks(stck_df)
    
    logf.info(f"rank given companies and create record for group averages")
    main_stck_df = rank_screener(main_stck_df, base_stck)
    
    top_c_df, bot_c_df = base_charting(main_stck_df)


    return main_stck_df


logf = setup_logger()

if __name__ == "__main__":
    logf = setup_logger()
    #main_prog()