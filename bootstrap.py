# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 11:55:24 2024

@author: balajiramesh
"""

import pandas as pd
import numpy as np
#import geopandas
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import product
import gc
import os
import glob


#read ED data that is already cleaned using the ComparisionAnalysis.py
df = pd.read_csv(r"ED_records_IID.csv",
                 dtype={'RECORD_ID': 'Int64',
                  'STMT_PERIOD_FROM': 'Int64',
                  'PAT_ADDR_CENSUS_BLOCK_GROUP': 'Int64',
                  'PAT_AGE_YEARS': 'float64',
                  'SEX_CODE': 'category',
                  'RACE': 'category',
                  'PAT_STATUS': 'object',
                  'ETHNICITY': 'category',
                  'RACETH':'category',
                  'op': 'bool',
                  'PAT_ADDR_CENSUS_TRACT': 'Int64',
                  'Population': 'int64',
                  'TotalVisits': 'int64',
                  'AGE_cat': 'category',
                  'Time': 'category',
                  'year': 'category',
                  'month': 'category',
                  'weekday': 'category',
                  'Outcome': 'int64'})


#%% Remove 1st post flood period and change post flood 2 to control period
# https://www.environmentalintegrity.org/wp-content/uploads/2018/08/Hurricane-Harvey-Report-Final.pdf
# based on above, the last sewage overflow was only reported till mid september . so its ok to keep post flood 2 in control period
df.loc[df.Time=='PostFlood2','Time'] = 'control'
df= df[df.Time.isin(['control','flood'])]
df['Time']= df.Time.cat.remove_unused_categories()

#%%set proper reference for categorical
df['RACETH'] = pd.Categorical(df['RACETH'], categories=['NH_white', 'Hispanic', 'black', 'other'])
df['AGE_cat'] = pd.Categorical(df['AGE_cat'], categories=['18-45', '13-17',  '46-64', '6-12', 'gt64', 'lte5'])

#%% filter only cts in the study area 
ct_in_sa = pd.read_csv(r"study_area_cts_aer_dfo.csv")
ct_in_sa = ct_in_sa.loc[:, ['GEOID','sel_sa_tracts']].rename(columns={'sel_sa_tracts': 'within_sa'})
hist_mean_aer = pd.read_csv(r"hist_mean_coase_aer\hist_mean_coase_aer.csv")

#%%read flood exposure data and census tract neighbours
flood_data = pd.read_csv(r"censusTracts_AER_DFO_flood.csv")
ct_neighs = pd.read_csv(r"censusTracts_AER_DFO_flood_neighbours.csv")

flood_data['GEOID'] = flood_data.GEOID.astype('Int64')
flood_data = flood_data[['OBJECTID_12','GEOID','within_dfo', 'within_aer','population', 'DFOfRatio', 'AERfRatio', 
                         'DFOfldResRatio', 'AERfldResRatio','fldPopDFO','fldPopAER', 
                         'fldPctDFO', 'fldPctAER',
                         'Shape_Area']]

#rename columns
flood_data.columns = [  'OBJECTID_12','GEOID', 'within_DFO', 'within_AER','population', 'fRatio_DFO', 'fRatio_AER',
                        'fldResRatio_DFO', 'fldResRatio_AER', 'fldPop_DFO', 'fldPop_AER',
                        'fldPopPct_DFO', 'fldPopPct_AER', 
                        'Shape_Area']


flood_data = pd.merge(flood_data, ct_in_sa,on='GEOID', how='left')
flood_data = pd.merge(flood_data, hist_mean_aer,on='GEOID' ,how='left')
flood_data['fRatio_DFOorAER']= flood_data[['fRatio_DFO','fRatio_AER']].fillna(0).mean(axis=1)
flood_data['fldPopPct_DFOorAER']= flood_data[['fldPopPct_DFO','fldPopPct_AER']].fillna(0).mean(axis=1)
#%% setting up parameters 
extent= 'within_sa'#+fldSource #within_AER or within_DFO or within_sa

cntrl_threshold = 0
neighbour_flooded = False
gc.collect()

#create random seeds list 
seed_list=list(range(1, 2010, 2))
boot_indxs_chk = pd.DataFrame(columns= ['c' + str(item) for item in range(1000)])
#%%
#bootstrap loop starts here
for boot_i in range(500,1000):
#for boot_i in range(0,500):
#for boot_i in range(992,997):
    print("\n Boot_i:"+ str(boot_i)+"=========")
    np.random.seed(seed_list[boot_i])
    boot_sam_idx = np.random.choice(np.arange(df.shape[0]), size=df.shape[0], replace=True)
    
    boot_indxs_chk['c'+str(boot_i)]= np.concatenate( [boot_sam_idx[0:100] , boot_sam_idx[-100:] ])
    
#%%    
    reg_table_list= list()
    for fldSource,fldQuant,cntrl_threshold in product(['DFO','AER'],['fldPopPct_','fRatio_' ], [0]):   
        fldColumn=fldQuant+fldSource
        #extent= 'within_'+'DFO' #within_AER or within_DFO
        
        #print(fldColumn)
        fldDataJoin = flood_data[['OBJECTID_12','GEOID', extent, fldColumn]]
        #filter just the needed extent
        fldDataJoin = fldDataJoin.loc[fldDataJoin[extent]==1,:]
        #set null to 0
        fldDataJoin.loc[pd.isna(fldDataJoin[fldColumn]),fldColumn]=0.0
    
        
        #%%cutoff
        cutoff = cntrl_threshold
        
        #for manual cut off or cutoff defined though percentiles
        fldDataJoin['flooded']= fldDataJoin[fldColumn] > cutoff
        
        
        #for using aer hist values as cntrl threshold
        if cntrl_threshold == -888:
            if fldQuant=='fldPopPct_':
                continue
            else:
                fldDataJoin['flooded']= fldDataJoin[fldColumn] > fldDataJoin['hist_mean']
        #%% join and subset records within selected extent
        df_joined=df.merge(fldDataJoin,left_on='PAT_ADDR_CENSUS_TRACT',right_on='GEOID',how='left')
        
        #subset the needed boot idxs than the true df
        df_joined = df_joined.iloc[boot_sam_idx]
        #%% file name ==============
        out_file_name = outcome + fldColumn+'_'+extent+'_'+str(cntrl_threshold)
        print(out_file_name)
       
        #%%running the model
        
        formula='Outcome'+' ~ '+' flooded * Time '+' + year + month + weekday' + '  + RACETH + SEX_CODE + PAT_AGE_YEARS  + op '
        
        model = smf.gee(formula=formula,groups='PAT_ADDR_CENSUS_TRACT', data=df_joined,
                        missing='drop',family=sm.families.Poisson(link=sm.families.links.log()))
        
        results=model.fit()
        #print(results.summary())
        #print(np.exp(results.params))
        # print(np.exp(results.conf_int())) 
        
        
        #%% creating result dataframe tables
        results_as_html = results.summary().tables[1].as_html()
        reg_table=pd.read_html(results_as_html, header=0, index_col=0)[0].reset_index()
        reg_table.loc[:,'coef']=np.exp(reg_table.coef)
        reg_table.loc[:,['[0.025', '0.975]']]=np.exp(reg_table.loc[:,['[0.025', '0.975]']])
        reg_table=reg_table.loc[~(reg_table['index'].str.contains('month') 
                                  | reg_table['index'].str.contains('weekday')
                                  #| reg_table['index'].str.contains('year')
                                  #| reg_table['index'].str.contains('PAT_AGE_YEARS'))
                                  ),]
        reg_table['index']=reg_table['index'].str.replace("\[T.",'_',regex=True).str.replace('\]','',regex=True)
        reg_table['fldColumn']=fldColumn
        reg_table['cntrl_threshold']=cntrl_threshold
        reg_table = reg_table.drop([ 'z', 'P>|z|'], axis=1)
        reg_table_list.append(reg_table)
    #%%write the output
    pd.concat(reg_table_list, ignore_index=True).to_csv('boot_iter_' + str(boot_i)+"_reg"+".csv")
    #reg_table_dev.to_csv(Dis_cat+"_dev"+".csv")
    
#%%
boot_indxs_chk.to_csv('boot_indexes_for_check.csv')


#%% estimate 95% bootstrap interval
#%% bootstrap function
from scipy.stats import norm

#adapted from one of the R packages named coxed::bca
#https://search.r-project.org/CRAN/refmans/coxed/html/bca.html
def bca_confidence_interval(theta, conf_level=0.95):
    low = (1 - conf_level) / 2
    high = 1 - low
    sims = len(theta)
    z_inv = len(theta[theta < np.mean(theta)]) / sims
    z = norm.ppf(z_inv)
    U = (sims - 1) * (np.mean(theta) - theta)
    top = np.sum(U ** 3)
    under = 6 * (np.sum(U ** 2)) ** (3 / 2)
    a = top / under
    lower_inv = norm.cdf(z + (z + norm.ppf(low)) / (1 - a * (z + norm.ppf(low))))
    lower = np.quantile(theta, lower_inv)
    upper_inv = norm.cdf(z + (z + norm.ppf(high)) / (1 - a * (z + norm.ppf(high))))
    upper = np.quantile(theta, upper_inv)
    return np.array([lower, upper])
 
#%% read bootstrapped ouputs to create difference estimates
boot_output_dir = os.path.expanduser(r"IID_boot_output_dir")
boot_files = glob.glob(boot_output_dir+'\*.csv')

boot_dfs=[]
for boot_file in boot_files:
    boot_df = pd.read_csv( boot_file)
    boot_df["cntrl_threshold"]=""
    boot_df["cntrl_threshold"].replace({0:'thres_0',-888:'thres_hist'},inplace=True)
    boot_df.fldColumn = boot_df.fldColumn.str.cat(boot_df.cntrl_threshold,sep='_')
    boot_df = boot_df.loc[boot_df['index']=="flooded_True:Time_flood",["coef","fldColumn"]]
    boot_df = boot_df.T
    boot_df.columns= boot_df.loc['fldColumn',:]
    boot_df = boot_df.loc[boot_df.index=='coef',:].reset_index()
    boot_dfs.append(boot_df)
    
merg_boot_df = pd.concat(boot_dfs, ignore_index=True)

#convert to float
cols = merg_boot_df.columns.difference(['index'])
# convert the relevant columns
merg_boot_df[cols] = merg_boot_df[cols].astype(float)

#take log if needed
#merg_boot_df = merg_boot_df.apply(lambda x: np.log(x) if np.issubdtype(x.dtype, np.number) else x)
#%%
#create coluns that describe the differenc
diff_df = pd.DataFrame()
#comparing DFO vs AER
diff_df["fRatio_DFOVsAER"] = merg_boot_df['fRatio_DFO_thres_0'] - merg_boot_df['fRatio_AER_thres_hist']
diff_df["fldPop_DFOVsAER"] = merg_boot_df['fldPopPct_DFO_thres_0'] - merg_boot_df['fldPopPct_AER_thres_0']
diff_df["fRatio_DFOVsAER_thres0"] = merg_boot_df['fRatio_DFO_thres_0'] - merg_boot_df['fRatio_AER_thres_0']
diff_df["fRatio_DFOVsAER_thres_hist"] = merg_boot_df['fRatio_DFO_thres_hist'] - merg_boot_df['fRatio_AER_thres_hist']

#comparing quantifiaction fration vs fpop
diff_df["AER_fRatioVsfldPop"] = merg_boot_df['fRatio_AER_thres_hist'] - merg_boot_df['fldPopPct_AER_thres_0']
diff_df["DFO_fRatioVsfldPop"] = merg_boot_df['fRatio_DFO_thres_0'] - merg_boot_df['fldPopPct_DFO_thres_0']
diff_df["DFO_fRatioVsfldPop_thres_hist"] = merg_boot_df['fRatio_DFO_thres_hist'] - merg_boot_df['fldPopPct_DFO_thres_0']
diff_df["AER_fRatioVsfldPop_thres_0"] = merg_boot_df['fRatio_AER_thres_0'] - merg_boot_df['fldPopPct_AER_thres_0']

#historic baseline vs zero
diff_df["DFO_histVs0"] = merg_boot_df['fRatio_DFO_thres_hist'] - merg_boot_df['fRatio_DFO_thres_0']
diff_df["AER_histVs0"] = merg_boot_df['fRatio_AER_thres_hist'] - merg_boot_df['fRatio_AER_thres_0']


#%% get 95% interval for each 
print(diff_df.quantile([0.025, 0.975]).T)
print(diff_df.median())

#get bias corrected confidence intervals
bca_confidence_interval(diff_df.AER_fRatioVsfldPop, conf_level=0.90)
print(diff_df.apply(bca_confidence_interval).T)

diff_df.to_csv(boot_output_dir+"../diff_results.csv")