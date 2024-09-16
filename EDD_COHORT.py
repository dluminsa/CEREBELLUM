import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
from IPython.display import display
import streamlit as st

cola, colb = st.columns([1,3])
colb.markdown("<h4><b>PMTCT CEREBELLUM</b></h4>", unsafe_allow_html=True)
cola, colb = st.columns([1,4])
colb.markdown("<p><b><i>Know where all mothers are at any time t</i></b></p>", unsafe_allow_html=True)

#try:
#cola,colb= st.columns(2)
st.write('**SHOWING DATA FROM ANC DATABASE**')
conn = st.connection('gsheets', type=GSheetsConnection)
#if 'exist_de' not in st.session_state:/
exist = conn.read(worksheet= 'PMTCT', usecols=list(range(26)),ttl=5)
exist = exist.dropna(how = 'all')

back = conn.read(worksheet= 'BACK1', usecols=list(range(26)),ttl=5)
back = back.dropna(how = 'all')
cola, colb = st.columns(2)         
A = back.shape[0]
cola.write(A)
B = exist.shape[0]
colb.write(B)
df = pd.concat([back, exist])
df['IS THIS HER PARENT FACILITY?'] = df['IS THIS HER PARENT FACILITY?'].astype(str)
dfa = df[df['IS THIS HER PARENT FACILITY?']=='YES'].copy()
dfb = df[df['IS THIS HER PARENT FACILITY?']=='NO'].copy()

dfs=[]
faci = dfa['HEALTH FACILITY'].unique()
for facility in faci:
     dfa['HEALTH FACILITY'] = dfa['HEALTH FACILITY'].astype(str)
     dfx = df[df['HEALTH FACILITY']==facility].copy()
     #dfx['ART No.'] = dfx['ART No.'].astype(str)
     dfx['ART No.'] = pd.to_numeric(dfx['ART No.'], errors = 'coerce')#.astype(int)
     dfx = dfx.drop_duplicates(subset = ['ART No.'], keep='first')
     dfs.append(dfx)
dfa = pd.concat(dfs)

dfas=[]
facy = dfb['HEALTH FACILITY'].unique()
for facility in facy:
     dfb['HEALTH FACILITY'] = dfb['HEALTH FACILITY'].astype(str)
     dfx = df[df['HEALTH FACILITY']==facility].copy()
     #dfx['UNIQUE ID'] = dfx['UNIQUE ID'].astype(str)
     dfx['UNIQUE ID'] = pd.to_numeric(dfx['UNIQUE ID'], errors = 'coerce')#.astype(int)
     dfx = dfx.drop_duplicates(subset = ['UNIQUE ID'], keep='first')
     dfas.append(dfx)
dfb = pd.concat(dfas)
df = pd.concat([dfa, dfb])

facy = df['HEALTH FACILITY'].unique()

dfc = []
for facility in facy:
     df['HEALTH FACILITY'] = df['HEALTH FACILITY'].astype(str)
     dfx = df[df['HEALTH FACILITY']==facility].copy()
     dfx['NAME'] = dfx['NAME'].astype(str)
     dfx = dfx.drop_duplicates(subset = ['NAME'], keep='first')           
     #dfx = dfx.drop_duplicates(subset = ['UNIQUE ID'], keep='first')
     dfc.append(dfx)
pm = pd.concat(dfc)
#except:
 #     st.write("POOR NETWORK, COULDN'T CONNECT TO ANC DATABASE")
  #    st.stop()
