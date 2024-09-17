import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection
import datetime as dt
from IPython.display import display
import streamlit as st

cola, colb = st.columns([1,3])
colb.markdown("<h4><b>PMTCT CEREBELLUM</b></h4>", unsafe_allow_html=True)
cola, colb = st.columns([1,4])
colb.markdown("<p><b><i>Know where all mothers are at any time t</i></b></p>", unsafe_allow_html=True)

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
     pm = pd.concat(dfc)
except:
     st.write("POOR NETWORK, COULDN'T CONNECT TO ANC DATABASE")
     st.stop()
if 'pm_df' not in st.session_state:
     st.session_state.pm_df = pm
     pm = st.session_state.pm_df
#st.write(pm.shape[0])
try:
   #cola,colb= st.columns(2)
  # st.write('**SHOWING DATA FROM DELIVERY DATABASE**')
   conn = st.connection('gsheets', type=GSheetsConnection)
   exist = conn.read(worksheet= 'DELIVERY', usecols=list(range(26)),ttl=5)
   df = exist.dropna(how='all')
   delvr = df.copy()
   #delvr = df.rename(columns={'DATE OF DELIVERY': 'DATEY'})
except:
    st.write("POOR NETWORK, COULDN'T CONNECT TO DELIVERY DATABASE")
    st.stop()
     
if 'de_df' not in st.session_state:
     st.session_state.de_df = delvr
     delvr = st.session_state.de_df
#st.write(delvr)
try:
   #cola,colb= st.columns(2)
   #st.write('**SHOWING DATA FROM PCR DATABASE**')
   conn = st.connection('gsheets', type=GSheetsConnection)
   exist = conn.read(worksheet= 'PCR', usecols=list(range(25)),ttl=5)
   pcr = exist.dropna(how='all')
   #pcr = df.rename(columns={'DATE OF PCR': 'DATEY'})
except:
    st.write("POOR NETWORK, COULDN'T CONNECT TO PCR DATABASE")
    st.stop()
     
if 'pc_df' not in st.session_state:
     st.session_state.pc_df = pcr
     pcr = st.session_state.pc_df
     
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
#st.write(delvr.columns)
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
#st.write(delva.columns)
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

#st.write(extrad.columns)
#st.write(extrad['DATE OF SUBMISSION'])
#extrad = extrad[['DATE OF SUBMISSION']]
extrad = extrad[['DATE OF SUBMISSION', 'CLUSTER' ,'FACILITY DISTRICT', 'HEALTH FACILITY','IN COHORT?',
                     'IS THIS HER PARENT FACILITY?', 'ART No.', 'MWP IDI DISTRICT?',
                            'IDI SUPPORTED DISTRICT', 'FROM IDI FACILITY?', 'IDI PARENT FACILITY?','UNIQUE ID',
                            'OTHER PARENT FACILITY','OTHER DISTRICT','OUTSIDE FACILITY', 'NAME', 'AGE', 'HER DISTRICT','VILLAGE', 'TELEPHONE','OUTCOME',
                           'DATE OF DELIVERY']]
df = pd.concat([extrad, df])
#st.write(extrad)
df['EDD'] = pd.to_datetime(df['EDD'], errors='coerce', format = '%d -%m-%Y') #CONVERT edd to date time
df['DMONTH'] = df['EDD'].dt.month # EDD MONTH
df['DYEAR'] = df['EDD'].dt.year #EDD YEAR
st.write(df['DYEAR'].value_counts())

today = dt.datetime.now() # DATE TODAY
dmonth = int(today.strftime('%m')) #CURRENT MONTH
dyear = int(today.strftime('%Y'))  #CURRENT YEAR
st.write(f'the year is {dyear} and the month is {dmonth}')

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
df['DUE'] = df.apply(lambda wee: DUE(wee['DYEAR'],wee['DMONTH']), axis=1) #APP;Y ABOVE FORMULA TO DETERMINE WHO IS DUE
due = df[df['DUE'] == 'NOT DUE'].copy()
st.write(due)
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
#GRAPHING
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
fig = go.Figure(go.Waterfall(
    name="Waterfall",
    orientation="v",
    measure=measure,
    x=labels,
    textposition="outside",
    text=[f"{v}" for v in values],
    y=values
))

# Add titles and labels and adjust layout properties
fig.update_layout(
    title="WATERFALL ANALYSIS OF THE COHORT",
    xaxis_title="Categories",
    yaxis_title="Values",
    showlegend=True,
    height=425,  # Adjust height to ensure the chart fits well
    margin=dict(l=20, r=20, t=60, b=20),  # Adjust margins to prevent clipping
    yaxis=dict(automargin=True)
)
# Show the plot
#fig.show()
#st.title("Waterfall Chart in Streamlit")
#st.plotly_chart(fig)
# st.divider()

#VISITORS
df['IS THIS HER PARENT FACILITY?'] = df['IS THIS HER PARENT FACILITY?'].astype(str)
df['IS THIS HER PARENT FACILITY?'].unique()
visitors = df[df['IS THIS HER PARENT FACILITY?']== 'NO'].copy()
nonvisitors = df[df['IS THIS HER PARENT FACILITY?']== 'YES'].copy()
mapper = {'YES':'OURS', 'NO': 'VISITORS'}
df['VISITORS'] = df['IS THIS HER PARENT FACILITY?'].map(mapper)


# Count occurrences of each category
counts = df['VISITORS'].value_counts()

# Prepare labels with counts
labels = [f"{label}: {count}" for label, count in counts.items()]

# Create the donut chart
fig = go.Figure(data=[go.Pie(
    labels=counts.index,
    values=counts.values,
    hole=0.4,  # This creates the donut shape
    marker=dict(
        colors=['blue', 'red']  # Colors for 'YES' and 'NO'
    ),
    text=labels,  # Use labels with counts
    textinfo='text+percent',  # Display text and percentage
    insidetextorientation='radial'  # Text orientation
)])

# Update layout
fig.update_layout(
    title_text='PROPORTION OF VISITORS',
    showlegend=True
)

# Display the chart
#fig.show()

















































         











     














































