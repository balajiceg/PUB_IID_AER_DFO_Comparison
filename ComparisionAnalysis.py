# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import glob
import sys
from itertools import product
sys.path.insert(1, r'Z:\Balaji\GRAScripts\dhs_scripts')

#%%functions
def get_sp_outcomes(sp,Dis_cat):
    global sp_outcomes
    return sp.merge(sp_outcomes.loc[:,['RECORD_ID','op',Dis_cat]],on=['RECORD_ID','op'],how='left')[Dis_cat].values

#%%read ip op data
INPUT_IPOP_DIR=r'CleanedMergedJoinedED'
#read_op
op=pd.read_pickle(INPUT_IPOP_DIR+'\\op')
op=op.loc[:,['RECORD_ID','STMT_PERIOD_FROM','PAT_ADDR_CENSUS_BLOCK_GROUP','PAT_AGE_YEARS','SEX_CODE','RACE','PAT_STATUS','ETHNICITY','PAT_ZIP','LCODE']]
op['op']=True
#sp=pd.read_pickle(INPUT_IPOP_DIR+r'\op')
#read_ip
ip=pd.read_pickle(INPUT_IPOP_DIR+'\\ip')
ip=ip.loc[:,['RECORD_ID','STMT_PERIOD_FROM','PAT_ADDR_CENSUS_BLOCK_GROUP','PAT_AGE_YEARS','SEX_CODE','RACE','PAT_STATUS','ETHNICITY','PAT_ZIP','LCODE']]
ip['op']=False
#merge Ip and OP
op=pd.concat([op,ip])
sp=op
del op,ip

#read op/ip outcomes df
sp_outcomes=pd.read_csv(INPUT_IPOP_DIR+'\\ed_outcome_classified.csv')


#%%predefine variable 
flood_data_zip=None
washout_period=[20170819,20170825] #including the dates specified
#%%subset records from only the requried counties    
#create tract id from block group id
sp.loc[:,'PAT_ADDR_CENSUS_TRACT']=(sp.PAT_ADDR_CENSUS_BLOCK_GROUP//10)
sp=sp[(sp.PAT_ADDR_CENSUS_TRACT//1000000).isin(county_to_filter)].copy()
#%%cleaing for age, gender and race and create census tract
#age
sp['PAT_AGE_YEARS']=pd.to_numeric(sp.PAT_AGE_YEARS,errors="coerce")
sp['PAT_AGE_YEARS']=sp.loc[:,'PAT_AGE_YEARS'].astype('float')

#sex
sp.loc[~sp.SEX_CODE.isin(["M","F"]),'SEX_CODE']=np.nan
sp["SEX_CODE"]=sp.SEX_CODE.astype('category').cat.reorder_categories(['M','F'],ordered=False)

#ethinicity
sp['ETHNICITY']=pd.to_numeric(sp.ETHNICITY,errors="coerce")
sp.loc[~sp.ETHNICITY.isin([1,2]),'ETHNICITY']=np.nan
sp["ETHNICITY"]=sp.ETHNICITY.astype('category').cat.reorder_categories([2,1],ordered=False)
sp["ETHNICITY"]=sp.ETHNICITY.cat.rename_categories({2:'Non_Hispanic',1:'Hispanic'})

#race
sp['RACE']=pd.to_numeric(sp.RACE,errors="coerce")
sp.loc[(sp.RACE<=0) | (sp.RACE>5),'RACE']=np.nan
sp.loc[sp.RACE<=2,'RACE']=5
sp["RACE"]=sp.RACE.astype('category').cat.reorder_categories([4,3,5],ordered=False)
sp["RACE"]=sp.RACE.cat.rename_categories({3:'black',4:'white',5:'other'})

#create ethnicity/race
sp['RACETH'] = 'other'
sp.loc[(sp.ETHNICITY=='Hispanic'),'RACETH'] = 'Hispanic'
sp.loc[(sp.RACE=='black'),'RACETH'] = 'black'
sp.loc[(sp.RACE=='white') & (sp.ETHNICITY=='Non_Hispanic'),'RACETH'] = 'NH_white'
sp.loc[(sp.ETHNICITY.isna() | sp.RACE.isna()),'RACETH']=np.nan
sp["RACETH"]=sp.RACETH.astype('category')

#age
sp=sp[sp.PAT_AGE_YEARS<119]
#%%keep only the dates we requested for
#remove records before 2016
sp=sp.loc[(~pd.isna(sp.STMT_PERIOD_FROM))&(~pd.isna(sp.PAT_ADDR_CENSUS_BLOCK_GROUP))] 

sp=sp[((sp.STMT_PERIOD_FROM > 20160700) & (sp.STMT_PERIOD_FROM< 20161232))\
    | ((sp.STMT_PERIOD_FROM > 20170400) & (sp.STMT_PERIOD_FROM< 20171232))\
        | ((sp.STMT_PERIOD_FROM > 20180700) & (sp.STMT_PERIOD_FROM< 20181232))]

#%%remove data in washout period
sp= sp[~((sp.STMT_PERIOD_FROM >= washout_period[0]) & (sp.STMT_PERIOD_FROM <= washout_period[1]))]
    
#%%calculating total visits for offset
vists_per_tract=sp.groupby(['PAT_ADDR_CENSUS_TRACT','STMT_PERIOD_FROM'])\
                  .size().reset_index().rename(columns={0:'TotalVisits'})
sp=sp.merge(vists_per_tract,on=['PAT_ADDR_CENSUS_TRACT','STMT_PERIOD_FROM'],how='left')

#%%pat age categoriy 
sp['AGE_cat']=pd.cut(sp.PAT_AGE_YEARS,bins=[-1,5,12,17,45,64,200],labels=['lte5','6-12','13-17','18-45','46-64','gt64']).cat.reorder_categories(['lte5','6-12','13-17','18-45','46-64','gt64'])

#%% categorize flood and post flood periods
interv_dates=[20170825, 20170913, 20171014] #lower bound excluded
interv_dates_cats=['flood','PostFlood1','PostFlood2']

sp.loc[:,'Time']=pd.cut(sp.STMT_PERIOD_FROM,\
                                    bins=[0]+interv_dates+[20190101],\
                                    labels=['control']+[str(i) for i in interv_dates_cats]).cat.as_unordered()
#set after 2018 as control
sp.loc[sp.STMT_PERIOD_FROM>20180100,'Time']="control" 
sp=sp.loc[~pd.isna(sp.Time),]
#%%controling for year month and week of the day
sp['year']=(sp.STMT_PERIOD_FROM.astype('int32')//1e4).astype('category')
sp['month']=(sp.STMT_PERIOD_FROM.astype('int32')//1e2%100).astype('category')
sp['weekday']=pd.to_datetime(sp.STMT_PERIOD_FROM.astype('str'),format='%Y%m%d').dt.dayofweek.astype('category')
sp = sp.drop(columns=['LCODE','PAT_ZIP'])

#%%filter records for specific outcome
df=sp.copy()
Dis_cat = 'Intestinal_infectious_diseases'

df.loc[:,'Outcome']=get_sp_outcomes(df, Dis_cat)

#write cleaned data to file
df.to_csv(r"Z:\Balaji\R session_home_dir (PII)\ED_records_IID.csv", index=False)

#%% -----------================================================---------  
#% -----------============== data cleaning ends here ========---------  
#% -----------================================================---------   


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
https://www.environmentalintegrity.org/wp-content/uploads/2018/08/Hurricane-Harvey-Report-Final.pdf
# based on above, the last sewage overflow was only reported till mid september . so its ok to keep post flood 2 in control period
#df.loc[df.Time=='PostFlood2','Time'] = 'PostFlood1' #for preg complications
df.loc[df.Time=='PostFlood2','Time'] = 'control'  # for IID
#df= df[df.Time.isin(['control','flood'])]
df['Time']= df.Time.cat.remove_unused_categories()


#%%set proper reference for categorical
df['RACETH'] = pd.Categorical(df['RACETH'], categories=['NH_white', 'Hispanic', 'black', 'other'])
df['AGE_cat'] = pd.Categorical(df['AGE_cat'], categories=['18-45', '13-17',  '46-64', '6-12', 'gt64', 'lte5'])

#%% filter only cts in the study area 
ct_in_sa = pd.read_csv(r"study_area_cts_aer_dfo.csv")
ct_in_sa = ct_in_sa.loc[:, ['GEOID','sel_sa_tracts']].rename(columns={'sel_sa_tracts': 'within_sa'})
hist_mean_aer = pd.read_csv(r"hist_mean_coase_aer.csv")

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

#create xor and or categories
flood_data["fRatio_DFOxorAER"]="aa_non_flooded"
flood_data.loc[( (~flood_data.fRatio_AER.isna()) | (~flood_data.fRatio_DFO.isna())),"fRatio_DFOxorAER"]='AERxorDFO'
flood_data.loc[( (~flood_data.fRatio_AER.isna()) & (~flood_data.fRatio_DFO.isna())),"fRatio_DFOxorAER"]='AERandDFO'
flood_data.fRatio_DFOxorAER = flood_data.fRatio_DFOxorAER.astype('category').cat.reorder_categories(['aa_non_flooded','AERandDFO', 'AERxorDFO'], ordered=True)



#%% setting up parameters 
fldSource = 'DFO'  #['DFO','AER','DFOorAER']
fldQuant='fRatio_' # fRatio_ or fldPopPct_ 
extent= 'within_sa'#+fldSource #within_AER or within_DFO or within_sa

#0 or the percntile value as whole nuber or 
# excat value as decimal or -999 for conitnous or 
# -888 for using historic aer flood fraction as cntrl threshold
# -interger for qunties eg.-4 is quartile [-999,-888,0,5,10,25,33,-4,-10] or
cntrl_threshold = 0
neighbour_flooded = False
#%%
for fldSource,extent,fldQuant,cntrl_threshold in product(['DFO','AER','DFOxorAER'], ["within_sa"],
                                                         ['fRatio_','fldPopPct_'], [0, -999,-888,-4]):   
    #skip certarin combination
    if fldSource == 'DFOxorAER' and (fldQuant !='fRatio_' or cntrl_threshold!=-999):
        continue
    #print(fldQuant + fldSource + str(cntrl_threshold))
#%%
    fldColumn = fldQuant + fldSource
    #extent= 'within_'+'DFO' # within_AER or within_DFO
    
    #print(fldColumn)
    fldDataJoin = flood_data[['OBJECTID_12','GEOID','hist_mean', extent, fldColumn, 'Shape_Area']]
    #filter just the needed extent
    fldDataJoin = fldDataJoin.loc[fldDataJoin[extent]==1,:]
    #set null to 0
    if fldSource!='DFOxorAER':
        fldDataJoin.loc[pd.isna(fldDataJoin[fldColumn]),fldColumn]=0.0
    
    #%%cutoff
    cutoff = cntrl_threshold
    
    #for percentile cutoff - only if poitive integer
    if cntrl_threshold>0.99 :
        s=fldDataJoin.loc[fldDataJoin[fldColumn]>0,fldColumn]  
        cutoff=s.quantile(cntrl_threshold/100)
    
    #for manual cut off or cutoff defined though percentiles
    if cntrl_threshold >=0 and cntrl_threshold<1 :
        fldDataJoin['flooded']= fldDataJoin[fldColumn] > cutoff
    
    #for using aer hist values as cntrl threshold
    if cntrl_threshold == -888:
        if fldQuant=='fldPopPct_':
            #continue
            pass
        else:
            fldDataJoin['flooded']= fldDataJoin[fldColumn] > fldDataJoin['hist_mean']  
            
    #set neighbouring census tract to flooded as well
    if neighbour_flooded:
        fld_neighs = ct_neighs.neighbours_OBJECTID_12[ct_neighs.OBJECTID_12.isin(fldDataJoin.OBJECTID_12[fldDataJoin.flooded])]    
        fld_neighs  = fld_neighs.apply(lambda x: x.split(',')).explode().unique().astype('int64')
        fldDataJoin['flooded_temp'] = 'not_flooded'
        fldDataJoin.loc[fldDataJoin.OBJECTID_12.isin(fld_neighs),'flooded_temp'] = 'neigh_flooded'
        fldDataJoin.loc[fldDataJoin.flooded,'flooded_temp']='flooded'
        fldDataJoin['flooded'] = fldDataJoin.flooded_temp.astype('category').cat.reorder_categories(['not_flooded','neigh_flooded','flooded',], ordered=True)
        fldDataJoin = fldDataJoin.drop(columns=['flooded_temp'])
        
    # for treating as linear
    if cntrl_threshold==-999:
        fldDataJoin['flooded']=fldDataJoin[fldColumn]
    
    
    #for quantiles - categorical
    if cntrl_threshold<0 and cntrl_threshold> -100 :
        nquantile= abs(cntrl_threshold)-1
        s=fldDataJoin.loc[fldDataJoin[fldColumn]>0,fldColumn]  
        flood_bins=s.quantile(np.arange(0,1.1,1/nquantile)).to_numpy()
        flood_bins=np.append([0],flood_bins)
        fldDataJoin['flooded']=pd.cut(fldDataJoin[fldColumn],bins=flood_bins,
                                      right=True,include_lowest=True,
                                      labels=list(map(str,range(0,nquantile+1))))
        cutoff= str(np.round(flood_bins,6))
        #print(cutoff)
        
    #%% join and subset records within selected extent
    df_joined=df.merge(fldDataJoin,left_on='PAT_ADDR_CENSUS_TRACT',right_on='GEOID',how='left')
    df_joined= df_joined.loc[df_joined[extent]==1,:]
    
    #%% file name ==============
    out_file_name = outcome + fldColumn+'_'+extent+'_'+str(cntrl_threshold)#+'_rural'
    print(out_file_name)
    #%% save cross tab    
    if cntrl_threshold !=-999:
         #counts_outcome=pd.DataFrame(df.Outcome.value_counts())
        complete_indx = (df_joined.Outcome>0)&(~pd.isna(df_joined.loc[:,['flooded','Time','year','month','weekday' ,'PAT_AGE_YEARS','RACETH']]).any(axis=1))
        outcomes_recs=df_joined.loc[complete_indx,]
        counts_outcome=pd.crosstab(outcomes_recs.flooded,outcomes_recs.Time)
        counts_outcome.to_csv(out_file_name+"_aux"+".csv")
        print(counts_outcome)
        del outcomes_recs
    #%%running the model
    formula='Outcome'+' ~ '+' flooded * Time '+' + year + month + weekday' + '  + RACETH + SEX_CODE + PAT_AGE_YEARS  + op '
    
    model = smf.gee(formula=formula,groups='PAT_ADDR_CENSUS_TRACT', data=df_joined, 
                    missing='drop',family=sm.families.Poisson(link=sm.families.links.log()))
    
    
    results=model.fit()
     
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
    reg_table['AIC'] = results.aic
    reg_table['QIC'] = results.qic(results.scale)[1]
    reg_table['BIC_llf'] = results.bic_llf
    reg_table['extent']=extent
    reg_table['fldColumn']=fldColumn
    reg_table['cntrl_threshold']=cntrl_threshold
    reg_table['cutoff']=cutoff
    
    reg_table_dev=pd.read_html(results.summary().tables[0].as_html())[0]
    
    #reg_table.to_clipboard()
    # counts_outcome.loc["flood_bins",'Outcome']=str(flood_bins)
    #return reg_table
    #%%write the output
    reg_table.to_csv(out_file_name +"_reg"+".csv")
    #reg_table_dev.to_csv(Dis_cat+"_dev"+".csv")