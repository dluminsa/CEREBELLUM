import pandas as pd 
import streamlit as st 
import os
import gspread
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import traceback
import time
from google.oauth2.service_account import Credentials
from oauth2client.service_account import ServiceAccountCredentials
#from streamlit_gsheets import GSheetsConnection
from datetime import datetime

st.set_page_config(
    page_title = 'PROGRAM GROWTH',
    page_icon =":bar_chart"
    )

#st.header('CODE UNDER MAINTENANCE, TRY AGAIN TOMORROW')
#st.stop()
cola,colb,colc = st.columns([1,3,1])
colb.subheader('PROGRAM GROWTH')

today = datetime.now()
todayd = today.strftime("%Y-%m-%d")# %H:%M")
wk = today.strftime("%V")
week = int(wk)-39
cola,colb = st.columns(2)
cola.write(f"**DATE TODAY:    {todayd}**")
colb.write(f"**CURRENT WEEK:    {week}**")
k = int(wk)

# secrets = st.secrets["connections"]["gsheets"]

#     # Prepare the credentials dictionary
# credentials_info = {
#         "type": secrets["type"],
#         "project_id": secrets["project_id"],
#         "private_key_id": secrets["private_key_id"],
#         "private_key": secrets["private_key"],
#         "client_email": secrets["client_email"],
#         "client_id": secrets["client_id"],
#         "auth_uri": secrets["auth_uri"],
#         "token_uri": secrets["token_uri"],
#         "auth_provider_x509_cert_url": secrets["auth_provider_x509_cert_url"],
#         "client_x509_cert_url": secrets["client_x509_cert_url"]
#     }

# try:
#     # Define the scopes needed for your application
#     scopes = ["https://www.googleapis.com/auth/spreadsheets",
#             "https://www.googleapis.com/auth/drive"]
    
        
#     credentials = Credentials.from_service_account_info(credentials_info, scopes=scopes)
        
#         # Authorize and access Google Sheets
#     client = gspread.authorize(credentials)
        
#         # Open the Google Sheet by URL
#     spreadsheetu = "https://docs.google.com/spreadsheets/d/1twNlv9MNQWWsM73_dA19juHkp_Hua-k-fJA1qNVwQl0"
#     spreadsheet = client.open_by_url(spreadsheetu)
# except Exception as e:
#         # Log the error message
#     st.write(f"CHECK: {e}")
#     st.write(traceback.format_exc())
#     st.write("COULDN'T CONNECT TO GOOGLE SHEET, TRY AGAIN")
#     st.stop()

# st.cache_data
# def sheeta():
#     worksheet = spreadsheet.worksheet('TX')

#     # Get all data from the worksheet
#     data = worksheet.get_all_values()

#     # Convert the data into a Pandas DataFrame
#     df = pd.DataFrame(data[1:], columns=data[0])  # First row is assumed to be headers

#     return df

# st.cache_data
# def sheetb():
#     worksheet = spreadsheet.worksheet('YEARS')

#     # Get all data from the worksheet
#     data = worksheet.get_all_values()

#     # Convert the data into a Pandas DataFrame
#     df = pd.DataFrame(data[1:], columns=data[0])  # First row is assumed to be headers

#     return df

# st.cache_data
# def sheetc():
#     worksheet = spreadsheet.worksheet('THREEO')

#     # Get all data from the worksheet
#     data = worksheet.get_all_values()

#     # Convert the data into a Pandas DataFrame
#     df = pd.DataFrame(data[1:], columns=data[0])  # First row is assumed to be headers

#     return df

# dftx = sheeta()
# water = sheeta()
# dfyr = sheetb()
# dfearly = sheetc()

dftx = pd.read_csv(r'C:\Users\Desire Lumisa\Desktop\TXALL-main\PROGRAM GROWTH - TX.csv')
dfyr = pd.read_csv(r'C:\Users\Desire Lumisa\Desktop\TXALL-main\PROGRAM GROWTH - YEARS.csv')
dfearly = pd.read_csv(r'C:\Users\Desire Lumisa\Desktop\TXALL-main\PROGRAM GROWTH - THREEO.csv')

#REPORTING RATES
@st.cache_data
def report():
    df = pd.read_csv('CLUSTERS.csv')
    df['Q4 CUR'] = pd.to_numeric(df['Q4 CUR'], errors='coerce')
    df = df[df['Q4 CUR']>0].copy()
    return df  

dfrep = report()
dfa = dfrep[['DISTRICT', 'FACILITY']].copy() ## EXPECTED DISTRICTS
dftx['SURGE'] = pd.to_numeric(dftx['SURGE'], errors='coerce')
dfb = dftx[dftx['SURGE'] == 2].copy()  #FACILITIES FROM TX SHEET
dfb = dfb[['DISTRICT' , 'FACILITY']]
dfb = dfb.drop_duplicates(subset='FACILITY', keep='last')
dfa['FACILITY'] = dfa['FACILITY'].astype(str)
dfb['FACILITY'] = dfb['FACILITY'].astype(str)
none = dfa[~dfa['FACILITY'].isin(dfb['FACILITY'])].copy()
# merged = dfa.merge(dfb, on=['DISTRICT', 'FACILITY'], how='left', indicator=True)
# none = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
# none = none.reset_index()
# none = none.drop(columns='index')
all = none.shape[0]
buk = none[none['DISTRICT']=='BUKOMANSIMBI'].copy()
semb = none[none['DISTRICT']=='SEMBABULE'].copy()
dist = none[none['DISTRICT']=='MASAKA DISTRICT'].copy()
kal = none[none['DISTRICT']=='KALUNGU'].copy()
city = none[none['DISTRICT']=='MASAKA CITY'].copy()
lwe = none[none['DISTRICT']=='LWENGO'].copy()
lya = none[none['DISTRICT']=='LYANTONDE'].copy()
kala = none[none['DISTRICT']=='KALANGALA'].copy()
mpi = none[none['DISTRICT']=='MPIGI'].copy()
goa = none[none['DISTRICT']=='GOMBA'].copy()
but = none[none['DISTRICT']=='BUTAMBALA'].copy()
wak = none[none['DISTRICT']=='WAKISO'].copy()
rak = none[none['DISTRICT']=='RAKAI'].copy()
kyo = none[none['DISTRICT']=='KYOTERA'].copy()


bu = buk.shape[0]
se = semb.shape[0]
di = dist.shape[0]
ka = kal.shape[0]
ci = city.shape[0]
ky = kyo.shape[0]
rk = rak.shape[0]
waki = wak.shape[0]
bt = but.shape[0]
g =  goa.shape[0]
mp = mpi.shape[0]
kal = kala.shape[0]
ly = lya.shape[0]
lw = lwe.shape[0]


if kal ==0: ####KALAGALA
   kg = 'all facilities have reported'
else:
   kg = f'{kal}'

if waki ==0: ####WAKISO
   wak = 'all facilities have reported'
else:
   wak = f'{waki}'

if mp ==0: ####MPIGI
   mpi = 'all facilities have reported'
else:
   mpi = f'{mp}'

if g ==0: ####GOMBA
   gom = 'all facilities have reported'
else:
   gom = f'{g}'

if bt ==0: ####BUTAMBALA
   but = 'all facilities have reported'
else:
   but = f'{bt}'


if lw ==0: ####LWENGO
   lwe = 'all facilities have reported'
else:
   lwe = f'{lw}'

if ly ==0: ####LYANTONDE
    lya= 'all facilities have reported'
else:
   lya = f'{ly}'


if rk ==0: ####RAKAI
   r = 'all facilities have reported'
else:
   r = f'{rk}'

if ky ==0: ####KYOTERA
   y = 'all facilities have reported'
else:
   y = f'{ky}'
 
if bu ==0:
   b = 'all facilities have reported'
else:
   b = f'{bu}'
    
if se ==0:
   s = 'all facilities reported'
else:
   s = f'{se}'
    
if di ==0:
   d = 'all facilities reported'
else:
   d = f'{di}'
    
if ka ==0:
   k = 'all facilities reported'
else:
   k = f'{ka}'
    
if ci ==0:
   c = 'all facilities reported'
else:
   c = f'{ci}'
if all ==0:
    st.write('** ALL FACILITIES HAVE REPORTED**')
else:
    st.divider()
    st.markdown(f"**{all} FACILITIES HAVEN'T REPORTED THIS WEEK**")
    st.markdown(f'**KALANGALA {kg}, WAKISO {wak}, BUKOMANSIMBI {b}, SEMBABULE {s}, KALUNGU {k}, MKA CITY {c}, MKA DISTRICT {d}, MPIGI {mpi}, BUTAMBALA {but}, GOMBA {gom},LYANTONDE {lya}, LWENGO {lwe}, KYOTERA {r}, RAKAI {r}**')
    with st.expander('ClICK TO SEE PENDING FACILITIES'):
        st.dataframe(none)
#######################FILTERS
clusters = dfrep['CLUSTER'].unique()
weeks = dftx['SURGE'].unique()

fac = dfyr['FACILITY'].unique()
#TO USE WHERE WEEKS ARE NOT NEEDED FOR TX
dfy = []
for every in fac:
    dff = dftx[dftx['FACILITY']== every]
    dff = dff.drop_duplicates(subset=['FACILITY'], keep = 'last')
    dfy.append(dff)
water = pd.concat(dfy)

#TO USE WHERE WEEKS ARE NOT NEEDED FOR 1 YR
dfy = []
for every in fac:
    dff = dfyr[dfyr['FACILITY']== every]
    dff = dff.drop_duplicates(subset=['FACILITY'], keep = 'last')
    dfy.append(dff)
wateryr = pd.concat(dfy)

dfy = []
for every in fac:
    dff = dfearly[dfearly['FACILITY']== every]
    dff = dff.drop_duplicates(subset=['FACILITY'], keep = 'last')
    dfy.append(dff)
waterly = pd.concat(dfy)

#REMOVE DUPLICATES FROM TX SHEET # HOLD THIS IN SESSION LATER
dfs=[]   
for each in weeks:
    dftx['SURGE'] = pd.to_numeric(dftx['SURGE'], errors='coerce')
    dfa = dftx[dftx['SURGE']==each]
    dfa = dfa.drop_duplicates(subset=['FACILITY'], keep = 'last')
    dfs.append(dfa)
dftx = pd.concat(dfs)

#REMOVE DUPLICATES FROM YEAR SHEET # HOLD THIS IN SESSION LATER
dfs=[]   
for each in weeks:
    dfyr['SURGE'] = pd.to_numeric(dfyr['SURGE'], errors='coerce')
    dfa = dfyr[dfyr['SURGE']==each]
    dfa = dfa.drop_duplicates(subset=['FACILITY'], keep = 'last')
    dfs.append(dfa)    
dfyr= pd.concat(dfs)


#REMOVE DUPLICATES FROM EARLY SHEET # HOLD THIS IN SESSION LATER
dfs=[]   
for each in weeks:
    dfearly['SURGE'] = pd.to_numeric(dfearly['SURGE'], errors='coerce')
    dfa = dfearly[dfearly['SURGE']==each]
    dfa = dfa.drop_duplicates(subset=['FACILITY'], keep = 'last')
    dfs.append(dfa)
dfearly = pd.concat(dfs)

#FILTERS
st.sidebar.subheader('**Filter from here**')
CLUSTER = st.sidebar.multiselect('CHOOSE A CLUSTER', clusters, key='a')

#create for the state
if not CLUSTER:
    dfyr2 = dfyr.copy()
    dfearly2 = dfearly.copy()
    dfrep2 = dfrep.copy()
    dftx2 = dftx.copy()
    water2 = water.copy()
    wateryr2 = wateryr.copy()
    waterly2 = waterly.copy()

else:
    dfyr['CLUSTER'] = dfyr['CLUSTER'].astype(str)
    dfyr2 = dfyr[dfyr['CLUSTER'].isin(CLUSTER)]

    dfearly['CLUSTER'] = dfearly['CLUSTER'].astype(str)
    dfearly2 = dfearly[dfearly['CLUSTER'].isin(CLUSTER)]

    dfrep['CLUSTER'] = dfrep['CLUSTER'].astype(str)
    dfrep2 = dfrep[dfrep['CLUSTER'].isin(CLUSTER)]

    dftx['CLUSTER'] = dftx['CLUSTER'].astype(str)
    dftx2 = dftx[dftx['CLUSTER'].isin(CLUSTER)]
    
    water['CLUSTER'] = water['CLUSTER'].astype(str)
    water2 = water[water['CLUSTER'].isin(CLUSTER)]
    wateryr2 = wateryr[wateryr['CLUSTER'].isin(CLUSTER)]
    waterly2 = waterly[waterly['CLUSTER'].isin(CLUSTER)]

district = st.sidebar.multiselect('**CHOOSE A DISTRICT**', dfrep2['DISTRICT'].unique(), key='b')
if not district:
    dfyr3 = dfyr2.copy()
    dfearly3 = dfearly2.copy()
    dfrep3 = dfrep2.copy()
    dftx3 = dftx2.copy()
    water3 = water2.copy()
    wateryr3 = wateryr2.copy()
    waterly3 = waterly.copy()
else:
    dfyr2['DISTRICT'] = dfyr2['DISTRICT'].astype(str)
    dfyr3 = dfyr2[dfyr2['DISTRICT'].isin(district)]

    dfearly2['DISTRICT'] = dfearly2['DISTRICT'].astype(str)
    dfearly3 = dfearly2[dfearly2['DISTRICT'].isin(district)]

    dfrep2['DISTRICT'] = dfrep2['DISTRICT'].astype(str)
    dfrep3 = dfrep2[dfrep2['DISTRICT'].isin(district)]

    dftx2['DISTRICT'] = dftx2['DISTRICT'].astype(str)
    dftx3 = dftx2[dftx2['DISTRICT'].isin(district)]
    
    water2['DISTRICT'] = water2['DISTRICT'].astype(str)
    wateryr3 = wateryr2[wateryr2['DISTRICT'].isin(district)]
    waterly3 = waterly2[waterly2['DISTRICT'].isin(district)]

facility = st.sidebar.multiselect('**CHOOSE A FACILITY**', dfrep3['FACILITY'].unique(), key='c')
if not facility:
    dfyr4 = dfyr3.copy()
    dfearly4 = dfearly3.copy()
    dfrep4 = dfrep3.copy()
    dftx4 = dftx3.copy()
    water4 = water3.copy()
    wateryr4 = wateryr3.copy()
    waterly4 = waterly3.copy()
else:
    dfyr4 = dfyr3[dfyr3['FACILITY'].isin(facility)]
    dfearly4 = dfearly3[dfearly3['FACILITY'].isin(facility)]
    dfrep4 = dfrep3[dfrep3['FACILITY'].isin(facility)]
    dftx4 = dftx3[dftx3['FACILITY'].isin(facility)]
    water4 = water3[water3['FACILITY'].isin(facility)]
    wateryr4 = waterly3[waterly3['FACILITY'].isin(facility)]
    wateryly4 = waterly3[waterly3['FACILITY'].isin(facility)]

# Base DataFrame to filter
dfyr = dfyr4.copy()
dfdearly = dfearly4.copy()
dfrep = dfrep4.copy()
dftx = dftx4.copy()
water = water4.copy()
wateryr4 = wateryr4.copy()
waterly4 = waterly4.copy()

# Apply filters based on selected criteria
if CLUSTER:
    dfyr = dfyr[dfyr['CLUSTER'].isin(CLUSTER)]
    water = water[water['CLUSTER'].isin(CLUSTER)]
    dfearly = dfearly[dfearly['CLUSTER'].isin(CLUSTER)]
    dfrep = dfrep[dfrep['CLUSTER'].isin(CLUSTER)]
    dftx = dftx[dftx['CLUSTER'].isin(CLUSTER)]
    wateryr = wateryr[wateryr['CLUSTER'].isin(CLUSTER)]
    waterly = waterly[waterly['CLUSTER'].isin(CLUSTER)]

if district:
    dfyr = dfyr[dfyr['DISTRICT'].isin(district)]
    water = water[water['DISTRICT'].isin(district)]
    dfearly = dfearly[dfearly['DISTRICT'].isin(district)]
    dfrep = dfrep[dfrep['DISTRICT'].isin(district)]
    dftx = dftx[dftx['DISTRICT'].isin(district)]
    wateryr = wateryr[wateryr['DISTRICT'].isin(district)]
    waterly = waterly[waterly['DISTRICT'].isin(district)]

if facility:
    dfyr = dfyr[dfyr['FACILITY'].isin(facility)]
    water = water[water['FACILITY'].isin(facility)]
    dfearly = dfearly[dfearly['FACILITY'].isin(facility)]
    dfrep = dfrep[dfrep['FACILITY'].isin(facility)]
    dftx = dftx[dftx['FACILITY'].isin(facility)]
    wateryr = wateryr[wateryr['FACILITY'].isin(facility)]
    waterly = waterly[waterly['FACILITY'].isin(facility)]

st.divider()
#############################################################################################
#filtered_df = filtered_df[filtered_df['WEEK']==k].copy()
pot = water['POTENTIAL'].sum()
Q4 = water['Q4'].sum()
ti = water['TI'].sum()
new = water['TXNEW'].sum()
uk = int(pot) - int(ti)- int(Q4) - int(new)
rt = water['RTT'].sum()  
los = water['TWO'].sum()
to  = water['TO'].sum()
dd = water['DEAD'].sum()
Q1 = water['ACTIVE'].sum()

labels = ["Q4 Curr",   "TI",     "TX NEW",     'RTT' ,  'Unkown',  "Potential",  "MISSED",  "DEAD",     "TO",     "ACTIVE"]
values = [Q4,           ti,        new,         rt,       uk,        pot,        -los,       -dd,        -to,       Q1]
measure = ["absolute", "relative","relative", "relative","relative","total",    "relative", "relative","realative","total"]
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
    title="Waterfall Analysis",
    xaxis_title="Categories",
    yaxis_title="Values",
    showlegend=True,
    height=425,  # Adjust height to ensure the chart fits well
    margin=dict(l=20, r=20, t=60, b=20),  # Adjust margins to prevent clipping
    yaxis=dict(automargin=True)
)

# Show the plot
st.plotly_chart(fig)

#############################################################################################
#LINE GRAPHS
st.divider()
#TREND OF MISSED APOINTMENTS
st.write('**TRENDS IN CLIENTS WHO HAVE MISSED APPOINTMENTS FOR MORE THAN 2, 3 AND 4 WEEKS**')

grouped = dftx.groupby('SURGE').sum(numeric_only=True).reset_index()

melted = grouped.melt(id_vars=['SURGE'], value_vars=['TWO', 'THREE', 'FOUR'],
                            var_name='INTERVAL', value_name='Total')

# melted = grouped.melt(id_vars=['SURGE'], value_vars=['TWO', 'THREE', 'FOUR'],
#                             var_name='INTERVAL', value_name='Total')

melted2 = grouped.melt(id_vars=['SURGE'], value_vars=['RTT', 'TO','DEAD'],
                            var_name='INTERVAL', value_name='Total')
melted['SURGE'] = melted['SURGE'].astype(int)
melted['SURGE'] = melted['SURGE'].astype(str)
melted2['SURGE'] = melted2['SURGE'].astype(int)
melted2['SURGE'] = melted2['SURGE'].astype(str)

fig2 = px.line(melted, x='SURGE', y='Total', color='INTERVAL', markers=True,
              title='MISSED APPOINTMENTS', labels={'SURGE':'WEEK', 'Total': 'No. of clients', 'INTERVAL': 'INTERVALS'})

fig3 = px.line(melted2, x='SURGE', y='Total', color='INTERVAL', markers=True, color_discrete_sequence=['blue','red'],
              title='RTT VS TO VS DEAD', labels={'SURGE':'WEEK', 'Total': 'No. of clients', 'INTERVAL': 'INTERVALS'})

fig2.update_layout(
    width=800,  # Set the width of the plot
    height=400,  # Set the height of the plot
    xaxis=dict(showline=True, linewidth=1, linecolor='black'),  # Show x-axis line
    yaxis=dict(showline=True, linewidth=1, linecolor='black')   # Show y-axis line
)
fig2.update_xaxes(type='category')
fig3.update_layout(
    width=800,  # Set the width of the plot
    height = 400,  # Set the height of the plot
    xaxis=dict(showline=True, linewidth=1, linecolor='black'),  # Show x-axis line
    yaxis=dict(showline=True, linewidth=1, linecolor='black')   # Show y-axis line
)
fig3.update_xaxes(type='category')
colx,coly = st.columns([2,1])
with colx:
    st.plotly_chart(fig2, use_container_width= True)

with coly:
    st.plotly_chart(fig3, use_container_width= True)
    #st.plotly_chart(fig3, use_container_width=True)
#############################################################################################
# #HIGHEST TXML 
st.divider()
highest = water[water['TWO']>100]

highest = highest.sort_values(by=['TWO'])#, ascending=False)
highesta = highest.shape[0]

highesty = water[water['TWO']<101]
highesty = highesty[highesty['TWO']>49]
highestb = highesty.sort_values(by=['TWO'], ascending=False)
highestb = highestb.shape[0]
# highestb = highest[highest['WEEK']==m]

coly, colu = st.columns(2)
with coly:
    if highesta ==0:
        st.write('**FACILITY SELECTED IS NOT AMONG**')
        pass
    else:
        figa = px.bar(
        highest,
        x='TWO',
        y='FACILITY',
        orientation='h',
        title='FACILITIES WITH >100 MISSED APPTS',
        labels={'TWO': 'CLIENTS MISSED', 'FACILITY': 'Facility'}
            )
        figa.update_traces(marker_color='#be7869')
        st.plotly_chart(figa, use_container_width=True)
with colu:
    if highestb ==0:
        st.write('**FACILITY SELECTED IS NOT AMONG**')
        pass
    else:
        figa = px.bar(
        highesty,
        x='TWO',
        y='FACILITY',
        orientation='h',
        title='FACILITIES WITH 50-100 MISSED APPTS',
        labels={'TWO': 'CLIENTS MISSED', 'FACILITY': 'Facility'}
        )
        figa.update_traces(marker_color='green')
        st.plotly_chart(figa, use_container_width=True)
st.divider()
#MMD PERFORMANCE
#OF THOSE THAT ARE DUE, HOW MANY ARE OURS, HOW MANY ARE VISITOR

dftx[['M2','M3', 'M6']] = dftx[['M2','M3', 'M6']].apply(pd.to_numeric, errors='coerce')
M2 = water['M2'].sum()
M3 = water['M3'].sum()
M6 = water['M6'].sum()


# Creating the grouped bar chart
fig4 = go.Figure(data=[
    go.Bar(name='<3 MTHS', x=['<3 MTHS'], y=[M2], marker=dict(color='red')),
    go.Bar(name='3-5 MTHS', x=['3-5 MTHS'], y=[M3], marker=dict(color='green')),
    go.Bar(name='6+ MTHS', x=['6+ MTHS'], y=[M6], marker=dict(color='blue'))
])

# Setting the layout to have no gap between bars
fig4.update_layout(barmode='group', bargap=0, bargroupgap=0)

NO_MMD = M2
MMD = int(M3) + int(M6)

labels = ['NO MMD', 'MMD']
values = [NO_MMD, MMD]
# Specify custom colors
colors = ['DarkRed', 'purple']  # Colors for NO_MMD and MMD
# Create the 3D pie chart
figp = go.Figure(data=[go.Pie(
    labels=labels,
    values=values,
    hole=0.3,  # Creates a donut chart (0 for a full pie)
    textinfo='label+percent',  # Show labels and percentages
    pull=[0.1, 0],  # Slightly pull both slices for emphasis
    marker=dict(colors=colors)
)])

# Update layout for 3D effect
figp.update_traces(textposition='inside', textinfo='percent+label')


# Display the chart
st.markdown(f'**{M2} Clients were given < 3 Months, {M3} received between 4 to 5 five months, {M6} received 6+ MTHS**')
cola,colb = st.columns(2)
with cola:
    st.plotly_chart(fig4, use_container_width=True)

with colb:
    st.markdown('')
    st.markdown('')
    st.plotly_chart(figp, use_container_width=True)

st.divider()
##########################################################################
#######ONE YEAR COHORT
#filtered_df = filtered_df[filtered_df['WEEK']==k].copy()
total = wateryr['TOTAL'].sum()
newti = wateryr['TI'].sum()
newlydx = wateryr['ORIG'].sum()

newlos = wateryr['LOST'].sum()
newto  = wateryr['TO'].sum()
newdd = wateryr['DEAD'].sum()
active = wateryr['ACTIVE'].sum()

labels = ["NEWLY DX", "TIs", "TOTAL", 'LTFU',"TOs","DEAD", "ACTIVE"]
values = [newlydx, newti, total, -newlos, -newto,-newdd, active]
measure = ["absolute", "relative", "total", "relative", "relative", "relative","total"]
# Create the waterfall chart
figy = go.Figure(go.Waterfall(
    name="Waterfall",
    orientation="v",
    measure=measure,
    x=labels,
    textposition="outside",
    text=[f"{v}" for v in values],
    y=values
))

# Add titles and labels and adjust layout properties
figy.update_layout(
    title="ONE YEAR COHORT ANALYSIS",
    xaxis_title="Categories",
    yaxis_title="Values",
    showlegend=True,
    height=425,  # Adjust height to ensure the chart fits well
    margin=dict(l=20, r=20, t=60, b=20),  # Adjust margins to prevent clipping
    yaxis=dict(automargin=True)
)

st.plotly_chart(figy)
st.divider()
########################################################################################
#ONE YEAR PIE CHART
col1, col2 = st.columns(2)
pied = wateryr.copy()#[filtered_df['WEEK']==k]
#pied['LOST NEW'] = pied['ORIGINAL COHORT']- pied['ONE YEAR ACTIVE'] 
pied = pied[['LOST', 'ACTIVE']]
melted = pied.melt(var_name='Category', value_name='values')
fig = px.pie(melted, values= 'values', title='ONE YEAR RETENTION RATE', names='Category', hole=0.3,color='Category',  
             color_discrete_map={'LOST': 'red', 'ACTIVE': 'blue'} )
    #fig.update_traces(text = 'RETENTION', text_position='Outside')
grouped = dfyr2.groupby('SURGE').sum(numeric_only=True).reset_index()

melted = grouped.melt(id_vars=['SURGE'], value_vars=['ACTIVE', 'LOST'],
                            var_name='OUTCOME', value_name='Total')
melted['SURGE'] = melted['SURGE'].astype(int)
melted['SURGE'] = melted['SURGE'].astype(str)
colors = ['DarkRed', 'purple']

fig2 = px.line(melted, x='SURGE', y='Total', color='OUTCOME', markers=True,
              title='MISSED APPOINTMENTS', labels={'SURGE':'WEEK', 'Total': 'No. of clients', 'OUTCOME': 'OUTCOME'})

fig2.update_layout(
    width=800,  # Set the width of the plot
    height=400,  # Set the height of the plot
    xaxis=dict(showline=True, linewidth=1, linecolor='black'),  # Show x-axis line
    yaxis=dict(showline=True, linewidth=1, linecolor='black')   # Show y-axis line
    #marker=dict(colors=colors)
)


fig2.update_xaxes(type='category')
if pied.shape[0]==0:
    pass
else:
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

###############################
st.divider()
st.write('EARLY RETENTION')
pied = wateryr.copy()#[filtered_df['WEEK']==k]
#pied['LOST NEW'] = pied['ORIGINAL COHORT']- pied['ONE YEAR ACTIVE'] 
pied = pied[['LOSTS', 'ACTIVES']]
pied = pied.rename(columns={'LOSTT':'LOST', 'ACTIVET':'ACTIVE'})
melted = pied.melt(var_name='Category', value_name='values')
fig6 = px.pie(melted, values= 'values', title='6 MTHS', names='Category', hole=0.3,color='Category',  
             color_discrete_map={'LOST': 'red', 'ACTIVE': 'blue'} )

pied = waterly.copy()#[filtered_df['WEEK']==k]
#pied['LOST NEW'] = pied['ORIGINAL COHORT']- pied['ONE YEAR ACTIVE'] 
pied = pied[['LOSTT', 'ACTIVET']]
pied = pied.rename(columns={'LOSTT':'LOST', 'ACTIVET':'ACTIVE'})
melted = pied.melt(var_name='Category', value_name='values')
fig3 = px.pie(melted, values= 'values', title='3 MTHS', names='Category', hole=0.3,color='Category',  
             color_discrete_map={'LOST': 'red', 'ACTIVE': 'yellow'} )
#pied['LOST NEW'] = pied['ORIGINAL COHORT']- pied['ONE YEAR ACTIVE']
pied = waterly.copy() 
pied = pied[['LOSTO', 'ACTIVEO']]
pied = pied.rename(columns={'LOSTO':'LOST', 'ACTIVEO':'ACTIVE'})
melted = pied.melt(var_name='Category', value_name='values')
fig1 = px.pie(melted, values= 'values', title='TX NEWS', names='Category', hole=0.3,color='Category',  
             color_discrete_map={'LOST': 'red', 'ACTIVE': 'green'} )

cola, colb, colc = st.columns(3)
with cola:
    st.plotly_chart(fig6, use_container_width=True)
with colb:
    st.plotly_chart(fig3, use_container_width=True)
with colc:
    st.plotly_chart(fig1, use_container_width=True)