import pandas as pd 
from datetime import datetime
from streamlit_gsheets import GSheetsConnection
import streamlit as st
import time
import datetime as dt

#st.title("PMTCT DASHBOARD DATA ENTRY FORM")
st.markdown("<h4><b>MONITORING ENTRIES DONE SO FAR</b></h4>", unsafe_allow_html=True)
#st.stop()


#CHOICE = 'ANC'
CHOICE = st.radio('**WHICH DATABASE DO YOU WANT TO MONITOR?**', options= ['ANC', 'PCR', 'DELIVERY'], horizontal=True, index = None)
if not CHOICE:
     st.stop()
elif CHOICE == 'ANC':
    try:
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
          df = pd.concat(dfc)
          dow = df.copy()
               
                #st.write('SEEN')
          #else:
           #     df = st.session_state['exist_de']
          df = df.rename(columns={'ANC DATE': 'DATEY', 'FACILITY DISTRICT':'DISTRICT', 'HEALTH FACILITY':'FACILITY'})
          #st.session_state['exist_de'] = df 
          #df = st.session_state['exist_de'] 
    except Exception as e:
         st.write(f"An error occurred: {e}")
         st.write("POOR NETWORK, COULDN'T CONNECT TO DATABASE")
         st.stop()

elif CHOICE == 'DELIVERY':
    try:
        #cola,colb= st.columns(2)
        st.write('**SHOWING DATA FROM DELIVERY DATABASE**')
        conn = st.connection('gsheets', type=GSheetsConnection)
        exist = conn.read(worksheet= 'DELIVERY', usecols=list(range(25)),ttl=5)
        df = exist.dropna(how='all')
        df = df.rename(columns={'DATE OF DELIVERY': 'DATEY'})
    except:
         st.write("POOR NETWORK, COUDN'T CONNECT TO DELIVERY DATABASE")
         st.stop()
elif CHOICE == 'PCR':
    try:
        #cola,colb= st.columns(2)
        st.write('**SHOWING DATA FROM PCR DATABASE**')
        conn = st.connection('gsheets', type=GSheetsConnection)
        exist = conn.read(worksheet= 'PCR', usecols=list(range(25)),ttl=5)
        df = exist.dropna(how='all')
        df = df.rename(columns={'DATE OF PCR': 'DATEY'})
    except:
         st.write("POOR NETWORK, COUDN'T CONNECT TO PCR DATABASE")
         st.stop()
#st.write(df.shape[0])

#file =r"C:\Users\Desire Lumisa\Desktop\APP\PMTCT (6).xlsx"
file2 = r'BACKLOG.csv'
#df = pd.read_excel(file)
#df = df.rename(columns={'ANC DATE': 'DATEY', 'FACILITY DISTRICT':'DISTRICT', 'HEALTH FACILITY':'FACILITY'})
dfa = pd.read_csv(file2)

#df = df[['DISTRICT', 'FACILITY', 'DATEY']].copy()
df['DATEY'] = pd.to_datetime(df['DATEY'], errors='coerce')
df['MONTH'] = df['DATEY'].dt.strftime('%B')
df['YEAR'] = df['DATEY'].dt.year


st.sidebar.subheader('Filter from here ')
district = st.sidebar.multiselect('Pick a DISTRICT', dfa['DISTRICT'].unique())

if not district:
    df2 = df.copy()
    dfa2 = dfa.copy()
else:
    df2 = df[df['DISTRICT'].isin(district)]
    dfa2 = dfa[dfa['DISTRICT'].isin(district)]

#create for facility
facility = st.sidebar.multiselect('**Select a facility**', dfa2['FACILITY'].unique())
if not facility:
    df3 = df2.copy()
    dfa3 = dfa2.copy()
else:
    df3 = df2[df2['FACILITY'].isin(facility)]
    dfa3 = dfa2[dfa2['FACILITY'].isin(facility)]
 
#for year
year = st.sidebar.multiselect('**Select a year**', dfa3['YEAR'].unique())

if not year:
    df4 = df3.copy()
    dfa4 = dfa3.copy()
else:
    df4 = df3[df3['YEAR'].isin(year)]
    dfa4 = dfa3[dfa3['YEAR'].isin(year)]

#for month
month = st.sidebar.multiselect('**Select a month**', dfa4['MONTH'].unique())

if not month:
    df5 = df4.copy()
    dfa5 = dfa4.copy()
else:
    df5 = df4[df4['MONTH'].isin(month)]
    dfa5 = dfa4[dfa4['MONTH'].isin(month)]

##################################################################################################
# Base DataFrame to filter
fdf = df5.copy()
fdfa = dfa5.copy()

# Apply filters based on selected criteria
if district:
    fdf = fdf[fdf['DISTRICT'].isin(district)]
    fdfa = fdfa[fdfa['DISTRICT'].isin(district)]

if facility:
    fdf = fdf[fdf['FACILITY'].isin(facility)]
    fdfa = fdfa[fdfa['FACILITY'].isin(facility)]

if year:
    fdf = fdf[fdf['YEAR'].isin(year)]
    fdfa = fdfa[fdfa['YEAR'].isin(year)]

if month:
    fdf = fdf[fdf['MONTH'].isin(month)]
    fdfa = fdfa[fdfa['MONTH'].isin(month)]

##################################################################################
dis = fdf['DISTRICT'].unique()
mon = fdf['MONTH'].unique()
yea = fdf['YEAR'].unique()
fac = fdf['FACILITY'].unique()

fdfa['TOTALS'] = fdfa['TOTALS'].astype(int)

exp = fdfa['TOTALS'].sum()

fdf['DISTRICT'] = fdf['DISTRICT'].astype(str)
disa = ', '.join(fdf['DISTRICT'].unique())

fdf['MONTH'] = fdf['MONTH'].astype(str)
mona = ', '.join(fdf['MONTH'].unique())

fdf['YEAR'] = fdf['YEAR'].astype(str)
yeaa = ', '.join(fdf['YEAR'].unique())

faca = ', '.join(fdf['FACILITY'].unique())

total = fdf.shape[0]
if exp==0:
    pro = int(total)*100
else:
    pro = round(int(total)/int(exp) * 100)
    
pro = str(pro) + '%'
bal = int(exp) - int(total)


cola,colb,colc = st.columns(3)

colb.write('**SUMMARIES**')

cola,colb = st.columns(2)
if len(dis)==1:
    cola.markdown(f'**DISTRICT : {disa}**')

if len(fac)==1:
    colb.markdown(f'**FACILITY: {faca}**')

if len(yea)==1:  
    cola.markdown(f'**YEAR : {yeaa}**')
if len(mon)==1:
    colb.markdown(f'**MONTH : {mona}**')

if CHOICE=='ANC':
    cola, colb,colc,cold = st.columns(4)
    cola.metric(label ='**EXPECTED**', value= exp)
    colb.metric(label = '**ENTERED**', value= total)
    colc.metric(label='**PROGRESS**', value=pro)
    cold.metric(label='**BALANCE**', value = bal)
    if bal <0:
        st.warning('**MORE THAN EXPECTED WAS ENTERED, DOUBLE CHECK**')
    elif bal ==0:
        st.success('ALL HAVE BEEN ENTERED')
        st.balloons()
        time.sleep(1)
        st.balloons()
        time.sleep(1)
        st.balloons()
    else:
        pass
else:
    st.markdown(f'**TOTAL ENTERED : {total}**')

if CHOICE == 'ANC':
    if not district:
        if not facility:
            if not year:
                if not month:
                    #entered
                    SEMBA = fdf[fdf['DISTRICT']=='SEMBABULE'].shape[0]
                    #expected
                    SEMBt = fdfa[fdfa['DISTRICT']=='SEMBABULE'].copy()
                    SEMB = SEMBt['TOTALS'].sum()
                    SEM = int(SEMB)-int(SEMBA)
                    SE  = round((int(SEMBA)/int(SEMB))*100)
                    SE = str(SE) + '%'

                    BUMBA = fdf[fdf['DISTRICT']=='BUKOMANSIMBI'].shape[0]
                    #expected
                    BUMBt = fdfa[fdfa['DISTRICT']=='BUKOMANSIMBI'].copy()
                    BUMB = BUMBt['TOTALS'].sum()
                    BUM = int(BUMB)-int(BUMBA)
                    BU  = round((int(BUMBA)/int(BUMB))*100)
                    BU = str(BU) + '%'
                     
                    KALMBA = fdf[fdf['DISTRICT']=='KALUNGU'].shape[0]
                    #expected
                    KALMBt = fdfa[fdfa['DISTRICT']=='KALUNGU'].copy()
                    KALMB = KALMBt['TOTALS'].sum()
                    KALM = int(KALMB)-int(KALMBA)
                    KAL  = round((int(KALMBA)/int(KALMB))*100)
                    KAL = str(KAL) + '%'

                    WAMBA = fdf[fdf['DISTRICT']=='WAKISO'].shape[0]
                    #expected
                    WAMBt = fdfa[fdfa['DISTRICT']=='WAKISO'].copy()
                    WAMB = WAMBt['TOTALS'].sum()
                    WAM = int(WAMB)-int(WAMBA)
                    WA  = round((int(WAMBA)/int(WAMB))*100)
                    WA = str(WA) + '%'

                    MPMBA = fdf[fdf['DISTRICT']=='MPIGI'].shape[0]
                    #expected
                    MPMBt = fdfa[fdfa['DISTRICT']=='MPIGI'].copy()
                    MPMB = MPMBt['TOTALS'].sum()
                    MPM = int(MPMB)-int(MPMBA)
                    MP  = round((int(MPMBA)/int(MPMB))*100)
                    MP = str(MP) + '%'

                    BUTMBA = fdf[fdf['DISTRICT']=='BUTAMBALA'].shape[0]
                    #expected
                    BUTMBt = fdfa[fdfa['DISTRICT']=='BUTAMBALA'].copy()
                    BUTMB = BUTMBt['TOTALS'].sum()
                    BUTM = int(BUTMB)-int(BUTMBA)
                    BUT  = round((int(BUTMBA)/int(BUTMB))*100)
                    BUT = str(BUT) + '%'

                    GOMMBA = fdf[fdf['DISTRICT']=='GOMBA'].shape[0]
                    #expected
                    GOMMBt = fdfa[fdfa['DISTRICT']=='GOMBA'].copy()
                    GOMMB = GOMMBt['TOTALS'].sum()
                    GOMM = int(GOMMB)-int(GOMMBA)
                    GOM  = round((int(GOMMBA)/int(GOMMB))*100)
                    GOM = str(GOM) + '%'

                    MASMBA = fdf[fdf['DISTRICT']=='MASAKA DISTRICT'].shape[0]
                    #expected
                    MASMBt = fdfa[fdfa['DISTRICT']=='MASAKA DISTRICT'].copy()
                    MASMB = MASMBt['TOTALS'].sum()
                    MASM = int(MASMB)-int(MASMBA)
                    MAS  = round((int(MASMBA)/int(MASMB))*100)
                    MAS = str(MAS) + '%'

                    MSKMBA = fdf[fdf['DISTRICT']=='MASAKA CITY'].shape[0]
                    #expected
                    MSKMBt = fdfa[fdfa['DISTRICT']=='MASAKA CITY'].copy()
                    MSKMB = MSKMBt['TOTALS'].sum()
                    MSKM = int(MSKMB)-int(MSKMBA)
                    MSK  = round((int(MSKMBA)/int(MSKMB))*100)
                    MSK = str(MSK) + '%'

                    KLGMBA = fdf[fdf['DISTRICT']=='KALANGALA'].shape[0]
                    #expected
                    KLGMBt = fdfa[fdfa['DISTRICT']=='KALANGALA'].copy()
                    KLGMB = KLGMBt['TOTALS'].sum()
                    KLGM = int(KLGMB)-int(KLGMBA)
                    KLG  = round((int(KLGMBA)/int(KLGMB))*100)
                    KLG = str(KLG) + '%'

                    LWEMBA = fdf[fdf['DISTRICT']=='LWENGO'].shape[0]
                    #expected
                    LWEMBt = fdfa[fdfa['DISTRICT']=='LWENGO'].copy()
                    LWEMB = LWEMBt['TOTALS'].sum()
                    LWEM = int(LWEMB)-int(LWEMBA)
                    LWE  = round((int(LWEMBA)/int(LWEMB))*100)
                    LWE = str(LWE) + '%'

                    LYAMBA = fdf[fdf['DISTRICT']=='LYANTONDE'].shape[0]
                    #expected
                    LYAMBt = fdfa[fdfa['DISTRICT']=='LYANTONDE'].copy()
                    LYAMB = LYAMBt['TOTALS'].sum()
                    LYAM = int(LYAMB)-int(LYAMBA)
                    LYA  = round((int(LYAMBA)/int(LYAMB))*100)
                    LYA = str(LYA) + '%'

                    RAKMBA = fdf[fdf['DISTRICT']=='RAKAI'].shape[0]
                    #expected
                    RAKMBt = fdfa[fdfa['DISTRICT']=='RAKAI'].copy()
                    RAKMB = RAKMBt['TOTALS'].sum()
                    RAKM = int(RAKMB)-int(RAKMBA)
                    RAK  = round((int(RAKMBA)/int(RAKMB))*100)
                    RAK = str(RAK) + '%'

                    KYOMBA = fdf[fdf['DISTRICT']=='KYOTERA'].shape[0]
                    #expected
                    KYOMBt = fdfa[fdfa['DISTRICT']=='KYOTERA'].copy()
                    KYOMB = KYOMBt['TOTALS'].sum()
                    KYOM = int(KYOMB)-int(KYOMBA)
                    KYO  = round((int(KYOMBA)/int(KYOMB))*100)
                    KYO = str(KYO) + '%'


if CHOICE == 'ANC':
    if not district:
        if not facility:
            if not year:
                if not month:
                    st.divider()
                    cola,colb,colc,cold,cole,colf = st.columns([2,1,1,1,1,1])
                    cola.write('**DISTRICT**')
                    colb.write('**EXPECTED**')
                    colc.write('**ENTERED**')
                    cold.write('**PROGRESS**')
                    cole.write('**BALANCE**')

                    cola.write('**SEMBABULE**')
                    colb.write(str(SEMB))
                    colc.write(str(SEMBA))
                    cold.write(str(SE))
                    cole.write(str(SEM))

                    cola.write('**BUKOMANSIMBI**')
                    colb.write(str(BUMB))
                    colc.write(str(BUMBA))
                    cold.write(str(BU))
                    cole.write(str(BUM))

                    cola.write('**WAKISO**')
                    colb.write(str(WAMB))
                    colc.write(str(WAMBA))
                    cold.write(str(WA))
                    cole.write(str(WAM))

                    cola.write('**MPIGI**')
                    colb.write(str(MPMB))
                    colc.write(str(MPMBA))
                    cold.write(str(MP))
                    cole.write(str(MPM))

                    cola.write('**BUTAMBALA**')
                    colb.write(str(BUTMB))
                    colc.write(str(BUTMBA))
                    cold.write(str(BUT))
                    cole.write(str(BUTM))

                    cola.write('**GOMBA**')
                    colb.write(str(GOMMB))
                    colc.write(str(GOMMBA))
                    cold.write(str(GOM))
                    cole.write(str(GOMM))

                    cola.write('**MASAKA DISTRICT**')
                    colb.write(str(MASMB))
                    colc.write(str(MASMBA))
                    cold.write(str(MAS))
                    cole.write(str(MASM))

                    cola.write('**MASAKA CITY**')
                    colb.write(str(MSKMB))
                    colc.write(str(MSKMBA))
                    cold.write(str(MSK))
                    cole.write(str(MSKM))

                    cola.write('**KALUNGU**')
                    colb.write(str(KALMB))
                    colc.write(str(KALMBA))
                    cold.write(str(KAL))
                    cole.write(str(KALM))

                    cola.write('**KALANGALA**')
                    colb.write(str(KLGMB))
                    colc.write(str(KLGMBA))
                    cold.write(str(KLG))
                    cole.write(str(KLGM))

                    cola.write('**LWENGO**')
                    colb.write(str(LWEMB))
                    colc.write(str(LWEMBA))
                    cold.write(str(LWE))
                    cole.write(str(LWEM))

                    cola.write('**LYANTONDE**')
                    colb.write(str(LYAMB))
                    colc.write(str(LYAMBA))
                    cold.write(str(LYA))
                    cole.write(str(LYAM))

                    cola.write('**RAKAI**')
                    colb.write(str(RAKMB))
                    colc.write(str(RAKMBA))
                    cold.write(str(RAK))
                    cole.write(str(RAKM))

                    cola.write('**KYOTERA**')
                    colb.write(str(KYOMB))
                    colc.write(str(KYOMBA))
                    cold.write(str(KYO))
                    cole.write(str(KYOM))
                    #st.write(dow)
                    st.stop()
#st.divider()
#st.writ
fdfa = fdf.copy()
st.write('**VIEW DATA SET HERE**')
fdf['DATEY'] = fdf['DATEY'].astype(str)
fdf['DATEY'] = fdf['DATEY'].str.replace('00:00:00', '', regex=False)


if CHOICE == 'ANC':
     fdf = fdf.rename(columns= {'DATEY': 'ANC DATE'})
     #fdf = fdf[['DISTRICT', 'FACILITY', 'ANC DATE', 'MONTH', 'YEAR', 'CODE']]
     fdf = fdf[['DISTRICT', 'FACILITY','ART No.','UNIQUE ID', 'ANC DATE', 'MONTH', 'YEAR', 'CODE']]
elif CHOICE == 'PCR':
     fdf = fdf.rename(columns= {'DATEY': 'PCR DATE'})
     fdf = fdf[['DISTRICT', 'FACILITY', 'PCR DATE', 'MONTH', 'YEAR']]
elif CHOICE == 'DELIVERY':
     fdf = fdf.rename(columns= {'DATEY': 'DELIVERY DATE'})
     fdf = fdf[['DISTRICT', 'FACILITY', 'DELIVERY DATE', 'MONTH', 'YEAR', 'OUTCOME']]
fdf = fdf.set_index('DISTRICT')
st.write(fdf)
#######
#DOWNLOAD
siz = fdfa['FACILITY'].unique()
fdfa = fdfa.reset_index()
fdfa['DATEY'] = pd.to_datetime(fdfa['DATEY'], errors = 'coerce')
fdfa['DAY'] = fdfa['DATEY'].dt.day
fdfa = fdfa.sort_values(by = ['DATEY'])
if CHOICE == 'ANC':
     pass
     #fdfa = fdfa[['DISTRICT', 'FACILITY', 'DAY','MONTH', 'YEAR', 'CODE']]
elif CHOICE == 'PCR':
     fdfa = fdfa[['DISTRICT', 'FACILITY', 'DAY', 'MONTH', 'YEAR']]
elif CHOICE == 'DELIVERY':
     fdfa = fdfa[['DISTRICT', 'FACILITY', 'DAY', 'MONTH', 'YEAR', 'OUTCOME']]
name = ','.join(siz)
if len(siz)==1:    
     dat = fdfa.to_csv(index=False)
     st.download_button(
          label = '**DOWNLOAD THIS DATA**',
          data = dat,
          file_name = f'{name}_{CHOICE}_DATA.csv',
          mime="text/csv")
