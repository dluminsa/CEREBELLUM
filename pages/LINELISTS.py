import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection
import datetime as dt
from IPython.display import display
import streamlit as st
from datetime import datetime
import bcrypt
cola,colb,colc = st.columns([1,1,3])
colc.info('**Created by LUMINSA DESIRE**')
# Example hashed password for "password123" using bcrypt
hashed_password = bcrypt.hashpw("pmtct8910".encode('utf-8'), bcrypt.gensalt())

USER_CREDENTIALS = {
    'admin': hashed_password,  # Store hashed password
}

def validate_login(username, password):
    if username in USER_CREDENTIALS:
        # Compare the hashed password
        if bcrypt.checkpw(password.encode('utf-8'), USER_CREDENTIALS[username]):
            return True
    return False

# Initialize session state for login status
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['username'] = ''

# If the user is logged in, show a welcome message and logout button
if st.session_state['logged_in']:
    st.success("WELCOME, YOU CAN NOW ACCESS THE LINE-LISTS")
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()  # Refresh the app state to show the login form again

# If the user is not logged in, show the login form
if not st.session_state['logged_in']:
    st.header("Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label="Login")

    # Validate login on form submission
    if submit_button:
        if validate_login(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Login successful!")
            st.rerun()  # Refresh the app to remove the form after login
        else:
            st.error("Invalid username or password")
if st.session_state['username'] and st.session_state['username']:
    pass
else:
    st.stop()
cola, colb = st.columns([1,3])
colb.markdown("<h4><b>LINELISTS</b></h4>", unsafe_allow_html=True)
#cola, colb = st.columns([1,4])
#colb.markdown("<p><b><i>Know where all mothers are at any time t</i></b></p>", unsafe_allow_html=True)

if 'pm_df' not in st.session_state:
     try:
          #cola,colb= st.columns(2)
          st.write('**SHOWING THE CURRENT DATA FROM ANC, DELIVERY AND PCR DATABASES**')
          conn = st.connection('gsheets', type=GSheetsConnection)
          #if 'exist_de' not in st.session_state:/
          exist = conn.read(worksheet= 'PMTCT', usecols=list(range(26)),ttl=5)
          exist = exist.dropna(how = 'all')
          
          back = conn.read(worksheet= 'BACK1', usecols=list(range(26)),ttl=5)
          back = back.dropna(how = 'all')
          cola, colb = st.columns(2)         
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
          st.session_state.pm_df = df
          pm = st.session_state.pm_df
          
     except:
          st.write("POOR NETWORK, COULDN'T CONNECT TO ANC DATABASE")
          st.stop()
pm = st.session_state.pm_df.copy()

if 'de_df' not in st.session_state:     
     try:
        #cola,colb= st.columns(2)
        conn = st.connection('gsheets', type=GSheetsConnection)
        exist = conn.read(worksheet= 'DELIVERY', usecols=list(range(26)),ttl=5)
        df = exist.dropna(how='all')
        delvr = df.copy()
        st.session_state.de_df = delvr
     except:
         st.write("POOR NETWORK, COULDN'T CONNECT TO DELIVERY DATABASE")
         st.stop()
delvr = st.session_state.de_df.copy()
          
if 'pc_df' not in st.session_state:
     try:
        conn = st.connection('gsheets', type=GSheetsConnection)
        exist = conn.read(worksheet= 'PCR', usecols=list(range(25)),ttl=5)
        pcr = exist.dropna(how='all')
        st.session_state.pc_df = pcr
     except:
         st.write("POOR NETWORK, COULDN'T CONNECT TO PCR DATABASE")
         st.stop()
pcr = st.session_state.pc_df.copy()
        
pm['ANC DATE'] = pd.to_datetime(pm['ANC DATE'], errors = 'coerce')
pm['MONTH'] = pm['ANC DATE'].dt.strftime('%B')

mapper = {'December':'M', 'May': 'E', 'October':'K', 'January': 'A', 'February':'B', 'July': 'G', 'November': 'L', 'August': 'H', 'September':'J',
       'March': 'C', 'April': 'D', 'June': 'F'}
pm['LETTER'] = pm['MONTH'].map(mapper)
pm = pm.sort_values(by = ['LETTER'])
pm['YEAR'] = pm['ANC DATE'].dt.strftime('%Y')
#TOTAL MOTHERS IN COHORT
total = pm.shape[0]
pm['IS THIS HER PARENT FACILITY?'] = pm['IS THIS HER PARENT FACILITY?'].astype(str) #to use this to separte those from the facility and the rest

#LIST OF FACILITIES IN COHORT
pm['HEALTH FACILITY'] = pm['HEALTH FACILITY'].astype(str)
facilities = pm['HEALTH FACILITY'].unique()

#in the database, how many were from the facility
pma = pm[pm['IS THIS HER PARENT FACILITY?']=='YES'].copy()
#in the ANC database, how many were from other facilities
pmb = pm[pm['IS THIS HER PARENT FACILITY?']=='NO'].copy()

###THE DELIVERY DASHBOARD HAS TWO SECTIONS.. THOSE THAT WERE IN COHORT AND THOSE REGISTERED AFTER COHORT
#SPLIT THEM AND THE JOIN THEM AGAIN
delvr['IN COHORT?'] = delvr['IN COHORT?'].astype(str)

delv = delvr[delvr['IN COHORT?']=='YES'].copy()
extrad = delvr[delvr['IN COHORT?']=='NO'].copy()

delv['SEARCHED ART NO.'] = delv['SEARCHED ART NO.'].astype(str) #use this to determine mothers in pm who are in del, by art nos, first to str to remove None
#those that have been delivered, already in cohort by art nos
delva = delv[delv['SEARCHED ART NO.']!='NONE'].copy()
delv['SEARCHED ID'] = delv['SEARCHED ID'].astype(str) #use this to determine mothers in pm who are in del, by unique id, first to str to remove None
#in the delivery database, how many were not from other facilities
delvb = delv[delv['SEARCHED ID']!='NONE'].copy()
##to compare pma with delva, pmb with delvb
facd = [] #from facility, have delivered, pma vs delva
for facility in facilities:
    pma['HEALTH FACILITY'] = pma['HEALTH FACILITY'].astype(str)
    dfx = pma[pma['HEALTH FACILITY']== facility].copy()  # each facility should be compared with each facility
    
    delva['FACILITY'] = delva['FACILITY'].astype(str)
    dfy = delva[delva['FACILITY']== facility].copy()
    dfy = dfy[['SEARCHED ART NO.', 'OUTCOME', 'DATE OF DELIVERY']].copy() #ART NOs FOR ART NOs
    dfy = dfy.rename(columns = {'SEARCHED ART NO.':'ART No.'})

    dfx['ART No.'] = pd.to_numeric(dfx['ART No.'], errors='coerce')
    dfy['ART No.'] = pd.to_numeric(dfy['ART No.'], errors='coerce')
    dfy = dfy.drop_duplicates(subset=['ART No.'], keep ='last')
      
    dfz = pd.merge(dfx,dfy, on = 'ART No.', how = 'left')
    facd.append(dfz)
#those that have delivered
pma = pd.concat(facd)
### visitors pmb vs delvb, that ave delivered, search by unique ids
vfacd = []

for facility in facilities:
    pmb['HEALTH FACILITY'] = pmb['HEALTH FACILITY'].astype(str)
    dfx = pmb[pmb['HEALTH FACILITY']== facility].copy()  # each facility should be compared with each facility
    
    delvb['FACILITY'] = delvb['FACILITY'].astype(str)
    dfy = delvb[delvb['FACILITY']== facility].copy()
    dfy = dfy[['SEARCHED ID', 'OUTCOME', 'DATE OF DELIVERY']].copy() #UNIQU ID FOR UNIQUE ID
    dfy = dfy.rename(columns = {'SEARCHED ID':'UNIQUE ID'})

    dfx['UNIQUE ID'] = pd.to_numeric(dfx['UNIQUE ID'], errors='coerce')
    dfy['UNIQUE ID'] = pd.to_numeric(dfy['UNIQUE ID'], errors='coerce')
    dfy = dfy.drop_duplicates(subset=['UNIQUE ID'], keep ='last')
      
    dfz = pd.merge(dfx,dfy, on = 'UNIQUE ID', how = 'left')
    vfacd.append(dfz)
pmb = pd.concat(vfacd)
df = pd.concat([pma,pmb])
extrad = extrad.rename(columns={'DISTRICT':'FACILITY DISTRICT','FACILITY':'HEALTH FACILITY', 'FROM THIS FACILITY?':'IS THIS HER PARENT FACILITY?', 
'NEW ART NO.':'ART No.','FROM IDI SUPPORTED DISTRICT':'MWP IDI DISTRICT?', 'IDI DISTRICT':'IDI SUPPORTED DISTRICT','FROM IDI FACILITY': 'FROM IDI FACILITY?',
    'PARENT FACILITY':'IDI PARENT FACILITY?','PHONE':'TELEPHONE'})
extrad = extrad[['DATE OF SUBMISSION', 'CLUSTER' ,'FACILITY DISTRICT', 'HEALTH FACILITY','IN COHORT?',
                     'IS THIS HER PARENT FACILITY?', 'ART No.', 'MWP IDI DISTRICT?',
                            'IDI SUPPORTED DISTRICT', 'FROM IDI FACILITY?', 'IDI PARENT FACILITY?','UNIQUE ID',
                            'OTHER PARENT FACILITY','OTHER DISTRICT','OUTSIDE FACILITY', 'NAME', 'AGE', 'HER DISTRICT','VILLAGE', 'TELEPHONE','OUTCOME',
                           'DATE OF DELIVERY']]
extrad['EDD'] = extrad['DATE OF DELIVERY']
df = pd.concat([extrad, df])

df['EDD'] = pd.to_datetime(df['EDD'], errors='coerce')#, format = '%Y -%m-%d') #CONVERT edd to date time
df['DMONTH'] = df['EDD'].dt.month # EDD MONTH
df['DYEAR'] = df['EDD'].dt.year #EDD YEAR

today = dt.datetime.now() # DATE TODAY
dmonth = int(today.strftime('%m')) #CURRENT MONTH
dyear = int(today.strftime('%Y'))  #CURRENT YEAR
def DUE(a,b):
    if a > dyear:
         return 'NOT DUE'
    elif a == dyear:
        if b > dmonth:
            return 'NOT DUE'
        else:
            return 'DUE'
    else:
        return 'DUE'
df[['DMONTH', 'DYEAR']] = df[['DMONTH', 'DYEAR']].apply(pd.to_numeric, errors='coerce')#astype(int)
df['DUE'] = df.apply(lambda wee: DUE(wee['DYEAR'],wee['DMONTH']), axis=1) #APP;Y ABOVE FORMULA TO DETERMINE WHO IS DUE
due = df[df['DUE'] == 'NOT DUE'].copy()
###### PCR SECTION
df['IS THIS HER PARENT FACILITY?'] = df['IS THIS HER PARENT FACILITY?'].astype(str)
#LIST OF FACILITIES IN COHORT
df['HEALTH FACILITY'] = df['HEALTH FACILITY'].astype(str)
facilities = df['HEALTH FACILITY'].unique()

#in the database, how many were from the facility
dfa = df[df['IS THIS HER PARENT FACILITY?']=='YES'].copy() 
af = dfa.copy()
#in the database, how many were from other facilities
dfb = df[df['IS THIS HER PARENT FACILITY?']=='NO'].copy() 

pcr['SEARCHED ART NO.'] = pcr['SEARCHED ART NO.'].astype(str) #use this to determine mothers in df who are in pcr, by art nos, first to str to remove None
#those that have been bled, already in cohort by art nos
pcra = pcr[pcr['SEARCHED ART NO.']!='NONE'].copy()
pcr['SEARCHED ID'] = pcr['SEARCHED ID'].astype(str) #use this to determine mothers in df who are in pcr, by unique id, first to str to remove None
#in the pcr database, how many were not from other facilities
pcrb = pcr[pcr['SEARCHED ID']!='NONE'].copy()

facp = [] 

for facility in facilities:
    dfa['HEALTH FACILITY'] = dfa['HEALTH FACILITY'].astype(str)
    dfx = dfa[dfa['HEALTH FACILITY']== facility].copy()  # each facility should be compared with each facility
    
    pcra['FACILITY'] = pcra['FACILITY'].astype(str)
    dfy = pcra[pcra['FACILITY']== facility].copy()
    dfy = dfy[['SEARCHED ART NO.', 'AGE AT PCR', 'DATE OF PCR']].copy() 
    dfy = dfy.rename(columns = {'SEARCHED ART NO.':'ART No.'})

    dfx['ART No.'] = pd.to_numeric(dfx['ART No.'], errors='coerce')
    dfy['ART No.'] = pd.to_numeric(dfy['ART No.'], errors='coerce')
    dfy = dfy.drop_duplicates(subset=['ART No.'], keep ='last')
    
    dfz = pd.merge(dfx,dfy, on = 'ART No.', how = 'left')
    facp.append(dfz)
#those that have been bled        #################################
dfa = pd.concat(facp)

ag =dfa.copy()
vfacn = []

for facility in facilities:
    dfb['HEALTH FACILITY'] = dfb['HEALTH FACILITY'].astype(str)
    dfx = dfb[dfb['HEALTH FACILITY']== facility].copy()  # each facility should be compared with each facility
    
    pcrb['FACILITY'] = pcrb['FACILITY'].astype(str)  #UNIQUE IDS FOR UNIQUE IDS
    dfy = pcrb[pcrb['FACILITY']== facility].copy()
    dfy = dfy[['SEARCHED ID', 'AGE AT PCR', 'DATE OF PCR']].copy()
    dfy = dfy.rename(columns = {'SEARCHED ID':'UNIQUE ID'})

    dfx['UNIQUE ID'] = pd.to_numeric(dfx['UNIQUE ID'], errors='coerce')
    dfy['UNIQUE ID'] = pd.to_numeric(dfy['UNIQUE ID'], errors='coerce')
    dfy = dfy.drop_duplicates(subset=['UNIQUE ID'], keep ='last')  
    
    dfz = pd.merge(dfx,dfy, on = 'UNIQUE ID', how = 'left')
    vfacn.append(dfz)
     
dfb = pd.concat(vfacn) 
df = pd.concat([dfa,dfb])
df['DATE OF DELIVERY'] = pd.to_datetime(df['DATE OF DELIVERY'],errors = 'coerce')
df['EMONTH'] = df['DATE OF DELIVERY'].dt.strftime('%B')
df['EYEAR'] = df['DATE OF DELIVERY'].dt.year

df['EDD'] = pd.to_datetime(df['EDD'],errors = 'coerce')
df['EDMONTH'] = df['DATE OF DELIVERY'].dt.strftime('%B')

dfdel = df.copy()
###########################FILTERS

file2 = r'BACKLOG.csv'
dfj = pd.read_csv(file2)

st.sidebar.subheader('Filter from here ')
district = st.sidebar.multiselect('Pick a DISTRICT', dfj['DISTRICT'].unique())

if not district:
    df2 = df.copy()
    dfj2 = dfj.copy()
    dfdel2 = dfdel.copy()
else:
    df2 = df[df['FACILITY DISTRICT'].isin(district)].copy()
    dfj2 = dfj[dfj['DISTRICT'].isin(district)].copy()
    dfdel2 = dfdel[dfdel['FACILITY DISTRICT'].isin(district)].copy()

#create for facility
facility = st.sidebar.multiselect('**Select a facility**', dfj2['FACILITY'].unique())
if not facility:
    df3 = df2.copy()
    dfj3 = dfj2.copy()
    dfdel3 = dfdel2.copy()
else:
    df3 = df2[df2['HEALTH FACILITY'].isin(facility)].copy()
    dfj3 = dfj2[dfj2['FACILITY'].isin(facility)].copy()
    dfdel3 = dfdel2[dfdel2['HEALTH FACILITY'].isin(facility)].copy()
 
#for year
year = st.sidebar.multiselect('**Select a year**', dfj3['YEAR'].unique())

if not year:
    df4 = df3.copy()
    dfj4 = dfj3.copy()
    dfdel4 = dfdel3.copy()
else:
    df4 = df3[df3['DYEAR'].isin(year)].copy()
    dfj4 = dfj3[dfj3['YEAR'].isin(year)].copy()
    dfdel4 = dfdel2[dfdel2['EYEAR'].isin(year)].copy()

#for month
month = st.sidebar.multiselect('**Select a month**', dfj4['MONTH'].unique())

if not month:
    df5 = df4.copy()
    dfj5 = dfj4.copy()
    dfdel5 = dfdel4.copy()
else:
    df5 = df4[df4['EDMONTH'].isin(month)].copy()
    dfj5 = dfj4[dfj4['MONTH'].isin(month)].copy()
    dfdel5 = dfdel4[dfdel4['EMONTH'].isin(month)].copy()
#########################FILTERED DF
# Base DataFrame to filter
fdf = df5.copy()
fdfd = dfdel5.copy()

# Apply filters based on selected criteria
if district:
    fdf = fdf[fdf['FACILITY DISTRICT'].isin(district)]
    fdfd = fdfd[fdfd['FACILITY DISTRICT'].isin(district)]

if facility:
    fdf = fdf[fdf['HEALTH FACILITY'].isin(facility)]
    fdfd = fdfd[fdfd['HEALTH FACILITY'].isin(facility)]
if year:
    fdf = fdf[fdf['DYEAR'].isin(year)]
    fdfd = fdfd[fdfd['EYEAR'].isin(year)]

if month:
    fdf = fdf[fdf['EDMONTH'].isin(month)]
    fdfd = fdfd[fdfd['EMONTH'].isin(month)]
df = fdf.copy()
dfdel = fdfd.copy()

################################GRAPHING

df['IN COHORT?'] = df['IN COHORT?'].astype(str)
incohort = df[df['IN COHORT?']!='NO'].copy()
inc = int(incohort.shape[0])
notcohort = df[df['IN COHORT?']=='NO'].copy()
notc = int(notcohort.shape[0])
total = int(df.shape[0])
df['OUTCOME'] = df['OUTCOME'].astype(str)
df['OUTCOME'] = df['OUTCOME'].fillna('NOT')
df['OUTCOME'] = df['OUTCOME'].str.replace('nan','NOT', regex=False)
delivered = df[df['OUTCOME']!='NOT'].copy()
delv = int(delivered.shape[0])
notdelivered = df[df['OUTCOME']=='NOT'].copy()
notdelv = int(notdelivered.shape[0])
notdelivered['DUE'] = notdelivered['DUE'].astype(str)
due = notdelivered[notdelivered['DUE']=='DUE'].copy()
duec = int(due.shape[0])
notdue = notdelivered[notdelivered['DUE']=='NOT DUE'].copy()
notduec = int(notdue.shape[0])
labels = ["IN COHORT", "NOT", "TOTAL", 'DELIVERED',"NOT DUE", "DUE"]
values = [inc, notc, total, -delv, -notduec, -duec]
measure = ["absolute", "relative", "total", "relative", "relative", "total"]

# Create the waterfall chart

# Show the plot
st.markdown(f'**{duec} mothers are now due and need to be tracked**')

dues = due.drop(columns = ['DATE OF SUBMISSION','CLUSTER', 'DMONTH', 'DYEAR', 'EYEAR', 'EMONTH', 'EDMONTH'])
dues['ANC DATE'] = pd.to_datetime(dues['ANC DATE'], errors='coerce')
dues['ANC DATE'] = dues['ANC DATE'].dt.date#str.replace('00:00:00', regex=False)
with st.expander ('Click here to see and download mothers that are due'):
    dues = dues.set_index('FACILITY DISTRICT')
    st.write(dues.head(3))
    data = dues.to_csv(index=False)
    st.download_button(
                       label='DOWNLOAD_DUE_MOTHERS',
                       data= data,
                       file_name="DUE.csv",
                       mime="text/csv")
    
st.divider()

#VISITORS
df['IS THIS HER PARENT FACILITY?'] = df['IS THIS HER PARENT FACILITY?'].astype(str)
df['IS THIS HER PARENT FACILITY?'].unique()
visitors = df[df['IS THIS HER PARENT FACILITY?']== 'NO'].copy()
nonvisitors = df[df['IS THIS HER PARENT FACILITY?']== 'YES'].copy()
mapper = {'YES':'OURS', 'NO': 'VISITORS'}
df['VISITORS'] = df['IS THIS HER PARENT FACILITY?'].map(mapper)


# Count occurrences of each category
counts = df['VISITORS'].value_counts()
ourd = int(df[df['VISITORS']=='OURS'].shape[0])
theird = int(df[df['VISITORS']=='VISITORS'].shape[0])

# Prepare labels with counts


#OF THE VISITORS, HOW MANY ARE FROM THE REGION
visitors['MWP IDI DISTRICT?'].value_counts()
mapper2 = {'YES':'REGION', 'NO': 'OUTSIDE'}
visitors['REGIONAL'] = visitors['MWP IDI DISTRICT?'].map(mapper2)

# Count occurrences of each category
counts = visitors['REGIONAL'].value_counts()
outd =int( visitors[visitors['REGIONAL']=='REGION'].shape[0])


#OF THOSE THAT ARE DUE, HOW MANY ARE OURS, HOW MANY ARE VISITOR
due = notdelivered[notdelivered['DUE']=='DUE'].copy()
notdelivered['IS THIS HER PARENT FACILITY?'] =notdelivered['IS THIS HER PARENT FACILITY?'].astype(str)
duevisitors = notdelivered[notdelivered['IS THIS HER PARENT FACILITY?']=='NO'].copy()
duevs = duevisitors
duev = duevisitors.shape[0]

colb.write('**OF MOTHERS DUE, HOW MANY ARE VISITORS**')
st.markdown(f'**There are {duev} mothers who are due and are visitors**')
with st.expander ('Click here to see and download due visitors'):
    #duevs = duevs.copy()#.set_index('FACILITY DISTRICT')
    st.write(duevs.head(3))
    data = duevs.to_csv(index=False)
    st.download_button(
                       label='DOWNLOAD_DUE_VISITORS',
                       data= data,
                       file_name="DUE_VISITORS.csv",
                       mime="text/csv")

st.divider()
#OF THOSE THAT HAVE DELIVERED, HOW MANY HAVE HAD A PCR DONE FOR THEIR BABIES
live = dfdel[dfdel['OUTCOME'] == 'LIVE BIRTH'].copy()
totallive = live.shape[0]
nopcr = live[live['AGE AT PCR'].isnull()].copy()
totalnopcr = nopcr.shape[0]
withpcr = live[~live['AGE AT PCR'].isnull()].copy()
totalpcr = withpcr.shape[0]

today = datetime.today()

# Subtract DATE OF DELIVERY from today's date and convert to days
nopcr['DATE OF DELIVERY'] = pd.to_datetime(nopcr['DATE OF DELIVERY'], errors='coerce')
nopcr['TME'] = (today - nopcr['DATE OF DELIVERY']).dt.days
#of those with no pcr, how many are due
def pcr(a):
    if a<29:
        return 'NOT DUE'
    elif a<61:
        return 'DUE'
    else:
         return 'OVER DUE'
         
nopcr['TME'] = nopcr['TME'].astype(int)
nopcr['PCR DUE'] = nopcr['TME'].apply(pcr)
nopcr['PCR DUE'] = nopcr['PCR DUE'].astype(str)
pcrdue = nopcr[nopcr['PCR DUE'] == 'DUE']
totalpcrdue = pcrdue.shape[0]

pcrnotdue = nopcr[nopcr['PCR DUE'] == 'NOT DUE'].copy()
pcroverdue = int(nopcr[nopcr['PCR DUE'] == 'OVER DUE'].shape[0])
totalpcrnotdue = pcrnotdue.shape[0]
pcroverdues = nopcr[nopcr['PCR DUE'] == 'OVER DUE']


cola,colb =st.columns(2)
with cola:
    st.markdown(f'**{totalpcrdue} mothers, their babies are due for a timely first PCR**')#, {pcroverdue} are over due**')
    with st.expander ('**BABIES DUE FOR PCR**'):
            pcrss = pcrdue.copy()#.set_index('FACILITY DISTRICT')
            st.write(pcrss.head(4))
            data = pcrss.to_csv(index=False)
            st.download_button(
                               label='DOWNLOAD_PCR_DUE',
                               data= data,
                               file_name="DUE_PCR.csv",
                               mime="text/csv")

with colb:
    st.markdown(f'**{pcroverdue} mothers, their babies are over due for the first PCR**')#, {pcroverdue} are over due**')
    with st.expander ('**BABIES OVER DUE FOR PCR**'):
            pcrss = pcroverdues.copy()#.set_index('FACILITY DISTRICT')
            st.write(pcrss.head(4))
            data = pcrss.to_csv(index=False)
            st.download_button(
                               label='DOWNLOAD_PCR_OVER_DUE',
                               data= data,
                               file_name="OVER_DUE_PCR.csv",
                               mime="text/csv")
#st.divider()
