# -*- coding: utf-8 -*-
from sklearn.base import clone, BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

import category_encoders
from sklearn.preprocessing import OneHotEncoder


def inpute_geo(X):

    min_long = 28.633
    max_long = 41.275
    min_lat = -11.740
    max_lat = -0.874

    lon_outl = 'longitude > @max_long or longitude < @min_long'
    lat_outl = 'latitude < @min_lat or latitude > @max_lat'
    geo_outl = lon_outl + ' or ' + lat_outl

    X.loc[X.query(lon_outl).index, 'longitude'] = np.nan
    X.loc[X.query(lat_outl).index, 'latitude'] = np.nan

    ln_lga = X.groupby('lga').longitude.agg(['median', 'count'])
    ln_region = X.groupby('region').longitude.agg(['median', 'count'])
    lt_lga = X.groupby('lga').latitude.agg(['median', 'count'])
    lt_region = X.groupby('region').latitude.agg(['median', 'count'])

    d = X[['region', 'lga', 'longitude', 'latitude']].copy()
    d = pd.merge(d, ln_lga, how='left', left_on='lga', right_index=True).rename(columns={'median': 'ln_lga_med',
                                                                                         'count': 'ln_lga_nb'})
    d = pd.merge(d, ln_region, how='left', left_on='region', right_index=True).rename(columns={'median': 'ln_rg_med',
                                                                                               'count': 'ln_rg_nb'})

    d = pd.merge(d, lt_lga, how='left', left_on='lga', right_index=True).rename(columns={'median': 'lt_lga_med',
                                                                                         'count': 'lt_lga_nb'})

    d = pd.merge(d, lt_region, how='left', left_on='region', right_index=True).rename(columns={'median': 'lt_rg_med',
                                                                                               'count': 'lt_rg_nb'})

    d['ln_fill'] = d.ln_lga_med.where(d.ln_lga_nb > 10, d.ln_rg_med)
    d['lt_fill'] = d.lt_lga_med.where(d.lt_lga_nb > 10, d.lt_rg_med)

    X.longitude = X.longitude.fillna(d.ln_fill)
    X.latitude = X.latitude.fillna(d.lt_fill)

    return X

def fe_funder(df, col):
    # convert all string values to lower case
    df[col] = df[col].str.lower()
    
    # map funder names having typos to the correct value
    df[col].replace(to_replace= 'a/co germany', value='aco/germany',  inplace=True)
    df[col].replace(to_replace= 'churc', value='church',  inplace=True)  
    df[col].replace(to_replace=['cocen', 'cocern', 'conce', 'concen'], value='concern', inplace=True) 
    df[col].replace(to_replace='compa', value='company', inplace=True)  
    df[col].replace(to_replace='commu', value='community', inplace=True)    
    df[col].replace(to_replace=['dasip', 'dasp'], value='dassip', inplace=True) 
    df[col].replace(to_replace='denish', value='danida', inplace=True)      
    df[col].replace(to_replace=['fin water', 'finn water', 'finw', 'finwater'], value='fini water', inplace=True)
    df[col].replace(to_replace='finland government', value='finland', inplace=True)       
    df[col].replace(to_replace='franc', value='france', inplace=True)   
    df[col].replace(to_replace='germany republi', value='germany', inplace=True)    
    df[col].replace(to_replace='greineker', value='greinaker', inplace=True)       
    df[col].replace(to_replace='greineker', value='greinaker', inplace=True) 
    df[col].replace(to_replace='halmashau.*', value='halmashauri', inplace=True, regex=True) # halmashau plus anything
    df[col].replace(to_replace=['hesaw', 'hesawz', 'hewasa'], value='hesawa', inplace=True) 
    df[col].replace(to_replace=['holand', 'holla'], value='holland', inplace=True)     
    df[col].replace(to_replace=['institution', 'insututional'], value='institutional', inplace=True)       
    df[col].replace(to_replace='irish government', value='irish ai', inplace=True)     
    df[col].replace(to_replace='islam', value='islamic', inplace=True) 
    df[col].replace(to_replace='italy government', value='italy', inplace=True) 
    df[col].replace(to_replace=['jaica', 'jeica', 'jika'], value='jica', inplace=True)
    df[col].replace(to_replace='korea', value='koica', inplace=True) 
    df[col].replace(to_replace=['lcdg', 'lgcd', 'lgcbg', 'lgdcg'], value='lgcdg', inplace=True) 
    df[col].replace(to_replace='lions', value='lions club', inplace=True) 
    df[col].replace(to_replace=['luthe', 'lutheran church'], value='lutheran', inplace=True)
    df[col].replace(to_replace=['milenia', 'mileniam project'], value='millenium', inplace=True)
    df[col].replace(to_replace=['missi', 'missio', 'missionaries', 'missionary'], value='mission', inplace=True)
    df[col].replace(to_replace='musilim agency', value='muslims', inplace=True) 
    df[col].replace(to_replace=['nethe', 'netherland'], value='netherlands', inplace=True)
    df[col].replace(to_replace=['ox', 'oxfarm', 'oxfam gb'], value='oxfam', inplace=True)  
    df[col].replace(to_replace='pentecosta church', value='pentecostal church', inplace=True) 
    df[col].replace(to_replace=['people from japan', 'the people of japan'], value='people of japan', inplace=True) 
    df[col].replace(to_replace=['plan int', 'plan internatio'], value='plan international', inplace=True)
    df[col].replace(to_replace='priva.*', value='private', inplace=True, regex=True) # priva plus anything
    df[col].replace(to_replace='quick.*', value='quick wings', inplace=True, regex=True) # priva plus anything
    df[col].replace(to_replace=['quik', 'quwkwin', 'qwickwin', 'qwiqwi'], value='quick wings', inplace=True)
    df[col].replace(to_replace=['rc', 'rc cathoric', 'rc ch', 'rc churc', 'rc church'], value='roman catholic', inplace=True)
    df[col].replace(to_replace=['roman', 'roman ca', 'roman cathoric', 'roman church'], value='roman catholic', inplace=True)
    df[col].replace(to_replace='redcross', value='red cross', inplace=True) 
    df[col].replace(to_replace=['rotaty club', 'lottery', 'lotary club', 'rotary i', 'lottery club'], 
                     value='rotary club', inplace=True)
    df[col].replace(to_replace=['rural water supply', 'rural water supply and sanita'], value='rural water supply and sanitat', inplace=True)
    df[col].replace(to_replace='rw.*', value='rwssp', inplace=True, regex=True) 
    df[col].replace(to_replace='schoo.*', value='school', inplace=True, regex=True) 
    df[col].replace(to_replace='secondary schoo', value='secondary', inplace=True) 
    df[col].replace(to_replace='soliderm', value='solidarm', inplace=True) 
    df[col].replace(to_replace=['sweden', 'swidish'], value='swedish', inplace=True)
    df[col].replace(to_replace='swisland.*', value='swisland', inplace=True, regex=True) 
    df[col].replace(to_replace='tot.*', value='total land care', inplace=True, regex=True) 
    df[col].replace(to_replace=['uniseg', 'uniceg', 'unice'], value='unicef', inplace=True)
    df[col].replace(to_replace='usa embassy', value='us embassy', inplace=True) 
    df[col].replace(to_replace='villa.*', value='village', inplace=True, regex=True) 
  
    # various values with unknown meaning    
    df[col].replace(to_replace=['0', 'not known', 'donor'], value='unknown', inplace=True)
    df[col].fillna(value='unknown', inplace=True)
    
    # group low frequency funders in a single category
    df_count = pd.DataFrame(df[col].value_counts())
    df_count = df_count[df_count[col] <= 2]
    low_frequency_funder = df_count.index.to_list()
    df[col].replace(to_replace=low_frequency_funder, value='low_frequency', inplace=True)

def fe_installer(df, col):
    df[col] = df[col].str.lower()
    
    # replace installer names having typos by proper value and group by installer when possible
    df[col].replace(to_replace='active tank co ltd', value='active tank co', inplace=True) 
    df[col].replace(to_replace='adra.*community', value='adra community', inplace=True, regex=True)   
    df[col].replace(to_replace='adra.*government', value='adra government', inplace=True, regex=True)
    df[col].replace(to_replace='amp contract.*', value='amp contractor', inplace=True, regex=True)
    df[col].replace(to_replace='ang.*', value='anglican', inplace=True, regex=True)    
    df[col].replace(to_replace=['aartisa', 'arisan', 'atisan'], value='artisan', inplace=True)   
    df[col].replace(to_replace='brit.*', value='british', inplace=True, regex=True)       
    df[col].replace(to_replace='building works.*', value='building works', inplace=True, regex=True)     
    df[col].replace(to_replace='care.*', value='care international', inplace=True, regex=True)     
    df[col].replace(to_replace='cartas.*', value='caritas', inplace=True, regex=True)     
    df[col].replace(to_replace=['cebtral government', 'cental government', 'centr', 'centra government', 
                                'centra govt', 'central govt', 'cetral government /rc'], 
                                value='central government', inplace=True)      
    df[col].replace(to_replace='chur', value='church', inplace=True) 
    df[col].replace(to_replace='commu.*', value='community', inplace=True, regex=True)     
    df[col].replace(to_replace='conce.*', value='concern', inplace=True, regex=True)       
    df[col].replace(to_replace='consu.*', value='consulting engineer', inplace=True, regex=True) 
    df[col].replace(to_replace='cosmo.*', value='cosmo', inplace=True, regex=True)  
    df[col].replace(to_replace='coun.*', value='council', inplace=True, regex=True)  
    df[col].replace(to_replace='district.*counci', value='district council', inplace=True, regex=True) 
    df[col].replace(to_replace='district water depar', value='district water department', inplace=True)     
    df[col].replace(to_replace='dwe.*', value='dwe', inplace=True, regex=True)  
    df[col].replace(to_replace='fin.*w.*', value='fini water', inplace=True, regex=True) 
    df[col].replace(to_replace='gold star', value='goldstar', inplace=True) 
    df[col].replace(to_replace='gove.*', value='government', inplace=True, regex=True) 
    df[col].replace(to_replace='individual', value='individuals', inplace=True) 
    df[col].replace(to_replace=['jaica', 'jaica co', 'jeica', 'jika'], value='jica', inplace=True)    
    df[col].replace(to_replace='local.*te.*', value='local technician', inplace=True, regex=True)     
    df[col].replace(to_replace='luthe.*', value='lutheran church', inplace=True, regex=True)     
    df[col].replace(to_replace='milenia.*', value='mileniam project', inplace=True, regex=True)     
    df[col].replace(to_replace='nora.*', value='norad', inplace=True, regex=True)  
    df[col].replace(to_replace='oikos.*', value='oikos africa', inplace=True, regex=True)      
    df[col].replace(to_replace=['oxfarm', 'oxfam gb'], value='oxfam', inplace=True)   
    df[col].replace(to_replace='plan*', value='plan international', inplace=True, regex=True)       
    df[col].replace(to_replace='priva', value='private', inplace=True) 
    df[col].replace(to_replace=['quwkwin', 'qwickwin'], value='quick win project', inplace=True)    
    df[col].replace(to_replace='rc*', value='rc church', inplace=True, regex=True)     
    df[col].replace(to_replace='roman*', value='roman catholic', inplace=True, regex=True)      
    df[col].replace(to_replace='rwe.*community', value='rwe community', inplace=True, regex=True)       
    df[col].replace(to_replace='save the rain usa', value='save the rain', inplace=True) 
    df[col].replace(to_replace='secondary', value='secondary school', inplace=True)     
    df[col].replace(to_replace='tanap', value='tanapa', inplace=True)       
    df[col].replace(to_replace='tanzania.*', value='tanzania government', inplace=True, regex=True)      
    df[col].replace(to_replace='tot.*land.*', value='total land care international', inplace=True, regex=True)    
    df[col].replace(to_replace='unisef', value='unicef', inplace=True)    
    df[col].replace(to_replace='vill.*', value='village', inplace=True, regex=True)      
    df[col].replace(to_replace='water.*sema', value='water aid/sema', inplace=True, regex=True)       
    df[col].replace(to_replace='wo.*bank.*', value='world bank', inplace=True, regex=True)       
    df[col].replace(to_replace='world vission', value='world vision', inplace=True)  
    
    # various values with unknown meaning    
    df[col].replace(to_replace=['0', '-0', 'not known', 'unknown installer'], value='unknown', inplace=True)
    df[col].fillna(value='unknown', inplace=True)    
    
    # group low frequency funders in a single category
    df_count = pd.DataFrame(df[col].value_counts())
    df_count = df_count[df_count[col] <= 2]
    low_frequency_funder = df_count.index.to_list()
    df[col].replace(to_replace=low_frequency_funder, value='low_frequency', inplace=True)    

    
def fe_scheme_name(df, col):
    df[col] = df[col].str.lower()
         
    # various values with unknown meaning    
    df[col].replace(to_replace=['0', '-0', 'not known', 'unknown installer'], value='unknown', inplace=True)
    df[col].fillna(value='unknown', inplace=True)    
    
    # group low frequency funders in a single category
    df_count = pd.DataFrame(df[col].value_counts())
    df_count = df_count[df_count[col] <= 2]
    low_frequency_funder = df_count.index.to_list()
    df[col].replace(to_replace=low_frequency_funder, value='low_frequency', inplace=True)


class ClfEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, base_encoder, base_encoder_cfg, drop_target_category=None):
        self.base_encoder = base_encoder
        self.base_encoder_cfg = base_encoder_cfg
        self.encoders = []
        self.target_encoder = None
        self.drop_target_category = drop_target_category

    def fit(self, X, y, **kwargs):
        drop_list = [self.drop_target_category] if self.drop_target_category else None
        self.target_encoder = OneHotEncoder(drop=drop_list)
        target = self.target_encoder.fit_transform(y.values.reshape(-1, 1)).toarray()

        nb_targets = target.shape[1]
        for i in range(nb_targets):
            enc = getattr(category_encoders, self.base_encoder)(**self.base_encoder_cfg)
            enc.fit(X, target[:, i])
            self.encoders.append(enc)
        return self

    def transform(self, X, y=None):
        target_names = self.target_encoder.get_feature_names()
        l = []
        for i, enc in enumerate(self.encoders):
            X_trf = enc.transform(X, y)
            X_trf = X_trf.add_suffix('_' + target_names[i].split('_')[1])
            l.append(X_trf)

        return pd.concat(l, axis=1)


class HighCardinalityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=4):
        self._frequencies = {}
        self._label_values = []
        self._mapper = {}
        self._alpha = 4
    
    def fit(self, X, y):
        # both X and y must be DF
        # convert y to string to have a fixed and known type  
        y_str = y.astype(str)
        
        # self._label_values = ['functional', 'functional needs repair', 'non functional']
        self._label_values = ['0', '1', '2']        
        
        # compute frequencies over full data
        self._default_frequencies = {}
        for status in self._label_values: 
            self._frequencies[status] = np.count_nonzero(y_str == status) / y_str.shape[0]  
        
        # get list of all columns in X 
        columns_to_map = list(X)
        
        self._mapper = {}
        for col in columns_to_map:
            # create df with the variable to map and the label
            df = pd.concat([X[col],y], axis=1)
            df.rename(columns={'status_group': 'y'}, inplace=True)
            
            # replace y with 3 columns 'y_functional', 'y_functional needs repair', 'y_non functional' with counts 
            df = pd.get_dummies(data=df, columns=['y'])
            df = df.groupby(col).sum()
            
#            df['count'] = df['y_functional'] +  df['y_functional needs repair'] + df['y_non functional']
            df['count'] = df['y_0'] +  df['y_1'] + df['y_2']            
            # calculate the 3 probability estimates
# TODO handle properly removal of one column        
#            df['0'] = (df['y_0'] + self._alpha *self._frequencies['0']) / (df['count'] + self._alpha) 
            df['1'] = (df['y_1'] + self._alpha * self._frequencies['1']) / (df['count'] + self._alpha) 
            df['2'] = (df['y_2'] + self._alpha * self._frequencies['2']) / (df['count'] + self._alpha) 
                    
            # create mapper with format:
            #     {'col1': {'functional': {val1: f1, val2: f2, ...},
            #               'functional needs repair': {val1: f1, val2: f2, ...},
            #               'non functional': {val1: f1, val2: f2, ...}}, 
            #      'col2': ...}
#            self._mapper[col] = df[['0', '1', '2']].to_dict()
            self._mapper[col] = df[['1', '2']].to_dict()    
    
        return self
    
    def transform(self, X, y=None):
        res = X.copy()
        
        for col_name in list(X):
            for y_name in ['1', '2']:
#            for y_name in self._label_values:
                mapper = self._mapper[col_name][y_name]
                new_col = col_name + '_f_' + y_name
                res[new_col] = X[col_name].map(mapper)
                
                # fill NA value with frequency of corresponding status - TODO not tested 
                res[new_col] = res[new_col].fillna(value=self._frequencies[y_name])
             
        #drop initial columns
        res.drop(list(X), axis=1, inplace=True)
        return res
    
    def get_frequencies(self):        
        return self._frequencies
    
    def get_mapper(self):
        return self._mapper