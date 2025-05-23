import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_gsheets import GSheetsConnection
import datetime as dt
from IPython.display import display
import streamlit as st
from datetime import datetime

cola, colb = st.columns([1,3])
colb.markdown("<h4><b>PMTCT CEREBELLUM</b></h4>", unsafe_allow_html=True)
cola, colb = st.columns([1,4])
colb.markdown("<p><b><i>Know where all mothers are at any time t</i></b></p>", unsafe_allow_html=True)

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
        exist = conn.read(worksheet= 'DELIVERYA', usecols=list(range(26)),ttl=5)
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
        exist = conn.read(worksheet= 'PCRA', usecols=list(range(25)),ttl=5)
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
df['TELEPHONE'] =df['TELEPHONE'].astype(str)
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
fig1 = go.Figure(go.Waterfall(
    name="Waterfall",
    orientation="v",
    measure=measure,
    x=labels,
    textposition="outside",
    text=[f"{v}" for v in values],
    y=values
))

# Add titles and labels and adjust layout properties
fig1.update_layout(
    title="",
    xaxis_title="CATEGORIES",
    yaxis_title="No. OF MOTHERS",
    showlegend=True,
    height=425,  # Adjust height to ensure the chart fits well
    margin=dict(l=20, r=20, t=60, b=20),  # Adjust margins to prevent clipping
    yaxis=dict(automargin=True)
)
# Show the plot
cola,colb = st.columns([1,4])
colb.write('**WATER FALL ANALYSIS OF THE COHORT**')
st.write(f'**TOTAL: {total} MOTHERS ARE IN COHORT**')
st.markdown(f'**Of these {inc} were registered during ANC, {notc} were registered during delivery**')#, giving a total of {total} mothers**')
st.markdown(f'**Of these, {delv} have delivered, {notduec} are not yet due,  leaving a total of {duec} mothers to track**')
st.plotly_chart(fig1)
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
labels = [f"{label}: {count}" for label, count in counts.items()]

# Create the donut chart
fig2 = go.Figure(data=[go.Pie(
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
fig2.update_layout(
    title_text='',
    showlegend=True
)
cola,colb = st.columns([1,4])
colb.write('**PROPORTION OF VISITORS**')
st.markdown(f'**Of the {inc + notc} mothers in cohort, {ourd} are from the same reporting facility, {theird} are visitors**')
st.plotly_chart(fig2)
st.divider()


#OF THE VISITORS, HOW MANY ARE FROM THE REGION
visitors['MWP IDI DISTRICT?'].value_counts()
mapper2 = {'YES':'REGION', 'NO': 'OUTSIDE'}
visitors['REGIONAL'] = visitors['MWP IDI DISTRICT?'].map(mapper2)

# Count occurrences of each category
counts = visitors['REGIONAL'].value_counts()
outd =int( visitors[visitors['REGIONAL']=='REGION'].shape[0])
# Prepare labels with counts
labels = [f"{label}: {count}" for label, count in counts.items()]

# Create the donut chart
fig3 = go.Figure(data=[go.Pie(
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
fig3.update_layout(
    title_text='',
    showlegend=True
)

# Display the chart
cola,colb = st.columns([1,4])
colb.write('**VISITORS FROM WITHIN REGION VS FROM OUTSIDE**')
st.markdown(f'**Of the {theird} visitors in cohort, {outd} are from within the region, {theird-outd} are from outside the region**')
st.plotly_chart(fig3)
st.divider()

#OF THOSE THAT ARE DUE, HOW MANY ARE OURS, HOW MANY ARE VISITOR

due = notdelivered[notdelivered['DUE']=='DUE'].copy()
due['IS THIS HER PARENT FACILITY?'] =due['IS THIS HER PARENT FACILITY?'].astype(str)
duevisitors = due[due['IS THIS HER PARENT FACILITY?']=='NO'].copy()
duev = duevisitors.shape[0]
dueours = due[due['IS THIS HER PARENT FACILITY?']=='YES'].copy()
dueo = dueours.shape[0]
duec = int(due.shape[0])

# Creating the grouped bar chart
fig4 = go.Figure(data=[
    go.Bar(name='TOTAL DUE', x=['TOTAL DUE'], y=[duec], marker=dict(color='blue')),
    go.Bar(name='OURS', x=['OURS'], y=[dueo], marker=dict(color='green')),
    go.Bar(name='VISITORS', x=['VISITORS'], y=[duev], marker=dict(color='red'))
])

# Setting the layout to have no gap between bars
fig4.update_layout(barmode='group', bargap=0, bargroupgap=0)

# Display the chart
cola,colb = st.columns([1,4])
colb.write('**OF MOTHERS DUE, HOW MANY ARE VISITORS**')
st.markdown(f'**Of the {duec} Mothers due, {dueo} are from within the same reporting facility, {duev} are visitors**')
st.plotly_chart(fig4)
st.divider()

#CONTRIBUTION OF DISTRICTS TO VISITORS
df['MWP IDI DISTRICT?'] = df['MWP IDI DISTRICT?'].astype(str)
df['MWP IDI DISTRICT?'].value_counts()
ours = df[df['MWP IDI DISTRICT?']=='YES'].copy()

theirs = df[df['MWP IDI DISTRICT?']=='NO'].copy()
dcount = ours['IDI SUPPORTED DISTRICT'].value_counts()
district_counts =  ours['IDI SUPPORTED DISTRICT'].value_counts().reset_index()
district_counts.columns = ['District', 'Count']
district_counts = district_counts.sort_values(by='Count')#, ascending=False)

# Create horizontal bar graph
fig5 = px.bar(district_counts, 
             x='Count', 
             y='District', 
             orientation='h', 
             title='',
             labels={'Count': 'Number of visitors', 'District': 'DISTRICT'},
             color_discrete_sequence=['#1f77b4'])  # Use a single color

# Show plot
cola,colb = st.columns([1,4])
colb.write('**FOR DISTRICTS WITHIN, WHERE DO MOST VISITORS COME FROM**')
st.plotly_chart(fig5)
st.divider()

# ourscount =ours['IDI SUPPORTED DISTRICT'].value_counts()
# ourscount = ourscount.reset_index()
# ourscount.columns = ['DISTRICT', 'VISITORS']
# ourscount = ourscount.sort_values(by= ['VISITORS'])
# # Assuming `dcount_df` contains your data
# fig6 = px.bar(
#     ourscount,
#     x='VISITORS',
#     y='DISTRICT',
#     orientation='h',
#     title='REGIONAL DISTRICTS WHERE MOST VISITORS COME FROM',
#     labels={'DISTRICT': 'TXML Value', 'VISITORS': 'Facility'}
# )
# st.plotly_chart(fig6)
# st.divider()

theirscount = theirs['OTHER DISTRICT'].value_counts()
theirscount  = theirscount .reset_index()
theirscount.columns = ['DISTRICT', 'VISITORS']
theirscounta = theirscount[theirscount['VISITORS']>2].copy()
theirscountb = theirscount[theirscount['VISITORS']<3].copy()
totalothers = theirscount['VISITORS'].sum()
addothers = pd.DataFrame({'DISTRICT': 'OTHERS',
                          'VISITORS': [totalothers]})
theirscount = pd.concat([theirscounta,addothers])
theirscount = theirscount.sort_values(by= ['VISITORS'])
# Assuming `dcount_df` contains your data
fig7 = px.bar(
    theirscount,
    x='VISITORS',
    y='DISTRICT',
    orientation='h',
    title='OUTSIDE DISTRICTS WHERE MOST VISITORS COME FROM',
    labels={'DISTRICT': 'DISTRICT', 'VISITORS': 'No. of Visitors'}
)

# To display the figure
st.plotly_chart(fig7)
st.divider()

#OF THOSE THAT COME FROM OUR REGION, HOW MANY COME FROM OUR FACILITIES
ours['FROM IDI FACILITY?'] = ours['FROM IDI FACILITY?'].astype(str)

ourt = int(ours.shape[0])
ourc = int(ours[ours['FROM IDI FACILITY?']=='YES'].shape[0])

map2 = {'YES': 'OUR FACILITIES', 'NO': 'OTHER FACILITIES'}
ours['FROM'] = ours['FROM IDI FACILITY?'].map(map2)
# Count occurrences of each category
counts = ours['FROM'].value_counts()

# Prepare labels with counts
labels = [f"{label}: {count}" for label, count in counts.items()]

# Create the donut chart
fig8 = go.Figure(data=[go.Pie(
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
fig8.update_layout(
    #title_text='',
    showlegend=True
)

# Display the chart
cola,colb = st.columns([1,4])
colb.write('**OF VISITORS FROM OUR REGION, HOW MANY ARE FROM OUR FACILITIES**')
st.markdown(f'**Of the {ourt} visitors from within our region, {ourc} are from within IDI supported facilities, {ourt-ourc} are from non IDI facilities**')
st.plotly_chart(fig8)
st.divider()


#of our facilities, where do most come from
ourfacilities = ours[ours['FROM IDI FACILITY?']=='YES'].copy()
ourfacilcount =ourfacilities['IDI PARENT FACILITY?'].value_counts()
ourfacilcount = ourfacilcount.reset_index()
ourfacilcount.columns = ['OUR FACILITY', 'VISITORS']
ourfacilcounta = ourfacilcount[ourfacilcount['VISITORS']>7].copy()
ourfacilcountb = ourfacilcount[ourfacilcount['VISITORS']<8].copy()
ourfacil_others = ourfacilcountb['VISITORS'].sum()
ourfacilcountb = pd.DataFrame({'OUR FACILITY': 'OTHERS', 'VISITORS': [ourfacil_others]})
allourfacility = pd.concat([ourfacilcounta,ourfacilcountb])
allourfacility = allourfacility.sort_values(by = ['VISITORS'])
# Assuming `dcount_df` contains your data
fig9 = px.bar(
    allourfacility,
    x='VISITORS',
    y='OUR FACILITY',
    orientation='h',
    title='FOR VISITORS FROM OUR REGION, FROM WHICH FACILITIES DO THE VISITORS COME FROM',
    labels={'OUR FACILITY': 'FACILITY', 'VISITORS': 'Visitors'}
)

# To display the figure
# Display the chart
st.plotly_chart(fig9)
st.divider()

###DELIVERY
#OF THOSE THAT HAVE DELIVERED, HOW MANY ARE DUE
cola,colb = st.columns([1,4])
colb.write('**DELIVERY SECTION**')

dfdel['OUTCOME'] = dfdel['OUTCOME'].astype(str)
delivered = dfdel[dfdel['OUTCOME']!='NOT'].copy()
delivered = delivered[delivered['OUTCOME']!='nan'].copy()
delived = delivered['OUTCOME'].value_counts().reset_index()

devd = int(delivered.shape[0])
lived = int(delivered[delivered['OUTCOME']=='LIVE BIRTH'].shape[0])

delivered = delivered[['OUTCOME']].copy()

# Count occurrences of each category
counts = delivered['OUTCOME'].value_counts()

# Prepare labels with counts
labels = [f"{label}: {count}" for label, count in counts.items()]

# Create the donut chart
fig10 = go.Figure(data=[go.Pie(
    labels=counts.index,
    values=counts.values,
    hole=0.3,  # This creates the donut shape
    marker=dict(
        colors=['blue', 'red', 'green', 'purple', 'yellow', 'pink']  # Colors for 'YES' and 'NO'
    ),
    text=labels,  # Use labels with counts
    textinfo='text+percent',  # Display text and percentage
    insidetextorientation='radial'  # Text orientation
)])

# Update layout
fig10.update_layout(
    title_text='',
    showlegend=True
)

# Display the chart
cola,colb = st.columns([1,4])
colb.write('**OF THE DELIVERIES, HOW MANY WERE LIVE BIRTHS**')
st.markdown(f'**Of the {devd} Deliveries, {lived} were live births, {devd -lived} were lost**')
st.plotly_chart(fig10)
st.divider()

#OF THOSE THAT HAVE DELIVERED, HOW MANY HAVE HAD A PCR DONE FOR THEIR BABIES
live = dfdel[dfdel['OUTCOME'] == 'LIVE BIRTH'].copy()
totallive = live.shape[0]
nopcr = live[live['AGE AT PCR'].isnull()].copy()
totalnopcr = nopcr.shape[0]
withpcr = live[~live['AGE AT PCR'].isnull()].copy()
totalpcr = withpcr.shape[0]
labels = ["TOTAL LIVE BIRTHS", "TOTAL WITH PCR", "NOT DUE", 'OVER DUE',"DUE"]
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
pcrnotdue = nopcr[nopcr['PCR DUE'] == 'NOT DUE']
pcroverdue = int(nopcr[nopcr['PCR DUE'] == 'OVER DUE'].shape[0])
totalpcrnotdue = pcrnotdue.shape[0]

values = [totallive, -totalpcr, -totalpcrnotdue, pcroverdue,totalpcrdue]
measure = ["absolute", "relative","relative", 'absolute',"relative"]

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
    title="",
    xaxis_title="Categories",
    yaxis_title="Values",
    showlegend=True,
    height=425,  # Adjust height to ensure the chart fits well
    margin=dict(l=20, r=20, t=60, b=20),  # Adjust margins to prevent clipping
    yaxis=dict(automargin=True)
)
# Show the plot
cola,colb = st.columns([1,4])
colb.write('**WATER FALL ANALYSIS OF DELIVERIES VS PCR DONE**')
st.write(f'**TOTAL: {totallive} MOTHERS HAVE HAD LIVE BIRTHS**')
st.markdown(f'**Of these, {totalpcr} have had their babies bled for first PCR, {totalpcrnotdue} are not yet due**')#, giving a total of {total} mothers**')
st.markdown(f'**{totalpcrdue} are due for a timely PCR, {pcroverdue} are over due**')
st.plotly_chart(fig)
st.divider()
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
cola,colb,colc = st.columns([2,1,1])
cola.info('**Created by LUMINSA DESIRE**')








