# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:47:18 2024

@author: Kathryn Hopkins
"""
#%%Preliminaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import style
import matplotlib.ticker as mtick
from matplotlib.ticker import StrMethodFormatter
mpl.rcParams.update(mpl.rcParamsDefault)
#%% Function to calculate percentages for pie charts
def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%".format(pct, absolute)
#First download data from UNHCR website https://www.unhcr.org/refugee-statistics/
#%%Import Applications data
UNapps=pd.read_csv('UNHCRApplications2000ToJun2023.csv')
UNapps
UNapps.describe()
UNapps.shape
UNapps.axes
UNapps.info()
UNapps['Country of asylum (ISO)'].value_counts(dropna=False)
UNapps['Country of asylum (ISO)'].unique()
#%%Check UK application  volumes for 2023 and 2022
UK = UNapps.loc[(UNapps['Country of asylum (ISO)']=='GBR') & (UNapps['Year']==2023)]
UK['applied'].sum()
UK['applied']
UK = UNapps.loc[(UNapps['Country of asylum (ISO)']=='GBR') & (UNapps['Year']==2022)]
UK['applied'].sum()
#%%Import Decisions Data
UNdecs=pd.read_csv('asylum-decisions.csv')
UNdecs.head()
#Sum the decision data as there are a few data anomalies
UNdecs['Total decisions'] = UNdecs.loc[:,['Recognized decisions' ,'Complementary protection','Rejected decisions','Otherwise closed']].sum(axis = 1)
#%% Check UK figures
UK = UNdecs.loc[(UNdecs['Country of asylum (ISO)']=='GBR') & (UNdecs['Year']==2023)]
UK['Total decisions'].sum()
UK = UNdecs.loc[(UNdecs['Country of asylum (ISO)']=='GBR') & (UNdecs['Year']==2022)]
UK['Total decisions'].sum()
#%%Group Apps and Decs By Year
UNapps=UNapps.groupby(['Year', 'Country of origin (ISO)', 'Country of asylum (ISO)']).agg({'applied':np.sum})
UNdecs=UNdecs.groupby(['Year', 'Country of origin (ISO)', 'Country of asylum (ISO)']).agg({'Recognized decisions':np.sum, 'Complementary protection':np.sum, 'Rejected decisions':np.sum, 'Otherwise closed':np.sum, 'Total decisions':np.sum})
#%%Calculate grant rates
UNdecs['Granted']=UNdecs['Recognized decisions']/UNdecs['Total decisions']
UNdecs['Other protection']=UNdecs['Complementary protection']/UNdecs['Total decisions']
UNdecs['Refused']=UNdecs['Rejected decisions']/UNdecs['Total decisions']
UNdecs['Otherwise Closed']=UNdecs['Otherwise closed']/UNdecs['Total decisions']
#Check values
UNdecs['Total%'] = UNdecs.loc[:,['Granted' ,'Other protection','Refused',
      'Otherwise Closed']].sum(axis = 1)
# Drop after checking
UNdecs.drop('Total%', axis = 1, inplace = True)
#%%Join Apps and Decs Datasets together
UNTotalAppsDecs=pd.merge(UNapps, UNdecs, how = 'left', left_index=True, right_index=True)
UNTotalAppsDecs.head()
#and reset the index
UNTotalAppsDecs.reset_index(inplace=True)
# Reset UNapps index so it can be used later
UNapps.reset_index(inplace = True)
#Reset index for later
UNdecs.reset_index(inplace = True)
#Check the application and decision volumes again
UK = UNTotalAppsDecs.loc[(UNTotalAppsDecs['Country of asylum (ISO)']=='GBR') & (UNTotalAppsDecs['Year']==2023)]
UK['applied'].sum()
UK['Total decisions'].sum()
UK = UNTotalAppsDecs.loc[(UNTotalAppsDecs['Country of asylum (ISO)']=='GBR') & (UNTotalAppsDecs['Year']==2022)]
UK['applied'].sum()
UK['Total decisions'].sum()
#%% Create Receiving Country dataset (removes country of origin)
UNTotalAppsDecs = UNTotalAppsDecs.groupby(['Year', 'Country of asylum (ISO)']).agg({'applied':np.sum, 'Recognized decisions':np.sum, 'Complementary protection':np.sum, 'Rejected decisions':np.sum, 'Otherwise closed':np.sum, 'Total decisions':np.sum})
#%%Create new columns in the Receiving Country dataset
UNTotalAppsDecs['Backlog']=UNTotalAppsDecs['Total decisions']-UNTotalAppsDecs['applied']
UNTotalAppsDecs['FVP']=UNTotalAppsDecs['Total decisions']/UNTotalAppsDecs['applied']
UNTotalAppsDecs.reset_index(inplace=True)
#%% Create 2017 Onwards dataset (to get application volumes recently)
UNTotalAppsDecs2017On=UNTotalAppsDecs[(UNTotalAppsDecs['Year']==2017)|(UNTotalAppsDecs['Year']==2018)|
        (UNTotalAppsDecs['Year']==2019)|
        (UNTotalAppsDecs['Year']==2020)|
        (UNTotalAppsDecs['Year']==2021)|
        (UNTotalAppsDecs['Year']==2022)|
        (UNTotalAppsDecs['Year']==2023)]
# Create a list of all countries in the dataset
TotalCountries = UNTotalAppsDecs2017On['Country of asylum (ISO)'].unique().tolist()
#%% =============================================================================
#Calculate the Top 20 application volumes from 2017 onwards
# =============================================================================
UNTotalAppsDecs2017On = UNTotalAppsDecs2017On.groupby(['Country of asylum (ISO)']).agg({'applied':np.sum})
UNTotalAppsDecs2017On=UNTotalAppsDecs2017On.sort_values(by=['applied'], ascending = False)
UNTotalAppsDecs2017On.head(20)#This is the list of top 20 by volume
UNTotalAppsDecs2017On.reset_index(inplace = True)
# Calculate a new column that formats the applied values as strings with comma-separated thousand values
UNTotalAppsDecs2017On['FormattedApps'] = UNTotalAppsDecs2017On['applied'].apply(lambda x: '{:,.0f}'.format(x))
# Summary Statistic - total volumes of applications received worldwide
TotalApps = UNTotalAppsDecs2017On['applied'].sum()
#%% Create a new dataset for the Top 20 countries only from 2017 onwards
TotalAppsTop20 = UNTotalAppsDecs2017On[(UNTotalAppsDecs2017On['Country of asylum (ISO)']=='USA')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='DEU')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='FRA')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='ESP')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='PER')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='MEX')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='ITA')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='GRC')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='CAN')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='GBR')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='CRI')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='TUR')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='BRA')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='UGA')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='AUT')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='AUS')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='SWE')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='EGY')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='NLD')|
        (UNTotalAppsDecs2017On['Country of asylum (ISO)']=='BEL')]
#Calculate the total number of applications to these countries
TotalAppsTop20Applied = TotalAppsTop20['applied'].sum()
TotalAppsTop20Applied
#Calculate the number of applications to other countries
OtherCountries = TotalApps-TotalAppsTop20Applied
OtherCountries
#%%Pie chart of the proportion asylum applications received between the Top 20 and all other countries
# Creating dataset
Countries = ['Top 20 Countries', 'All Other Countries']
data = [TotalAppsTop20Applied, OtherCountries]
colors = ('#002664', '#FC9D9A')
fig = plt.figure(figsize = (10,15))
plt.pie(data, labels=Countries,
        autopct=lambda pct: func(pct, data),
        startangle = 165,
        textprops=dict(color='#FFFFFF', fontsize = 24),
        colors = colors,
        wedgeprops = {"edgecolor" : "w", 
                      'linewidth': 2})
plt.title('\nProportion of Asylum Applications Received: \n\nTop 20 Countries and All Other Countries, 2017 to June 2023', fontsize = 20, color = '#000000', pad = 20)
plt.legend(loc = 'lower left', fontsize = 14)
#plt.tight_layout
plt.savefig("AppVolumesPieChart.png")
#%%Plot the 2017On data
# Create the data for the x-axis, and their integer position on the axis
countries = UNTotalAppsDecs2017On['Country of asylum (ISO)'].head(20)
pos = np.arange(len(countries))
# Create the data for the y-axis
applications=UNTotalAppsDecs2017On['applied'].head(20)
#%% Create line plot
#Plot a line plot of application volumes
fig, ax = plt.subplots(figsize=(50, 20))
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
plt.xticks(pos, countries, size = 20)
plt.yticks(size = 20)
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add title and labels
plt.title('Asylum Applications: Top 20 Countries: 2017 to June 2023',color = '#000000',fontsize=24, pad = 20)
plt.xlabel('Country of Asylum (ISO)',color = '#000000', fontsize = 20, labelpad = 30)
plt.ylabel('Applications', color = '#000000',fontsize = 20, labelpad = 30)
plt.yticks([])
# Add annotations
#plt.annotate(Top202023Apps['FormattedApps'][0], (pos[0],applications[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20,bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=2))
#plt.annotate(Top202023Apps['FormattedApps'][7], (pos[7],applications[7]), textcoords = 'offset points', xytext = (0,20), ha = 'center', size = 20)
#plt.annotate(Top202023Apps['FormattedApps'][18], (pos[18],applications[18]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20)
# To label the entire line
## zip joins x and y coordinates in pairs
for x,y in zip(pos,applications):
    label = "{:,.0f}".format(y)
    if x != pos[9] and y!= applications[9]:
        plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center', size =20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
    if x == pos[9] and y == applications[9]:
        plt.annotate(label, (x,y),textcoords="offset points", xytext=(0,10), ha='center', size =20,color = 'w', bbox=dict(boxstyle="round,pad=0.3",fc="#732282", ec="#732282", lw=2))
# Save the plot
plt.savefig("Applications2017To23.png", bbox_inches = 'tight')
#%% And create bar plot
bars = plt.bar(pos, applications, align = 'center', linewidth=0, color = '#00000090')
# Set the UK bar at a different color from the rest
bars[9].set_color('#732282')
# Set the position and tick labels of the x-axis
plt.xticks(pos, countries, color = '#000000',fontsize=20)
# Set the formatting of the ytick labels
plt.yticks(color = '#000000',fontsize=24)
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
# Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Set axis labels and titles
plt.xlabel('Country of Asylum (ISO)',color = '#000000', fontsize = 20, labelpad = 50)
plt.ylabel('Number of Applications', color = '#000000',fontsize = 20, labelpad = 50)
plt.title('Top 20 Countries by Asylum Application Volumes 2017 to Jun 2023', color = '#000000',fontsize=30, pad = 100)
# Remove all the spines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
#Add text annotation to the UK bar
UNTotalAppsDecs2017On.at[9, 'applied']
plt.text(8.5, 450000, UNTotalAppsDecs2017On['FormattedApps'][9], fontsize = 20, color = '#000000')
plt.savefig("Top202017To2023AppVolumes.png")
#%%##############################################################################
### Create 2022 Datasets
#################################################################################
UNTotalAppsDecs2022=UNTotalAppsDecs[UNTotalAppsDecs['Year']==2022]
# Calculate summary statistics for whole world in 2022
UNTotalAppsDecs2022['applied'].sum()
UNTotalAppsDecs2022['Total decisions'].sum()
World2022Backlog = UNTotalAppsDecs2022['Total decisions'].sum()- UNTotalAppsDecs2022['applied'].sum()
#%% Now create Top 20 2022 dataset
Top202022Apps = UNTotalAppsDecs2022[(UNTotalAppsDecs2022['Country of asylum (ISO)']=='USA')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='DEU')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='FRA')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='ESP')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='PER')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='MEX')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='ITA')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='GRC')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='CAN')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='GBR')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='CRI')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='TUR')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='BRA')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='UGA')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='AUT')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='AUS')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='SWE')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='EGY')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='NLD')|
        (UNTotalAppsDecs2022['Country of asylum (ISO)']=='BEL')]
Top202022Apps.sort_values(by=['applied'], ascending=False, inplace=True)
Top202022Apps.reset_index(inplace = True)
Top202022Apps = Top202022Apps.drop('index', axis = 1)
# Calculate a new column that formats the applied values as strings with comma-separated thousand values
Top202022Apps['FormattedApps'] = Top202022Apps['applied'].apply(lambda x: '{:,.0f}'.format(x))
# Calculate summary statistics for 2022 data
TotalTop202022Apps = Top202022Apps['applied'].sum()
TotalTop202022Decs = Top202022Apps['Total decisions'].sum()
Top202022Shortfall = TotalTop202022Apps-TotalTop202022Decs
#%% Plot the 2022 data
# Create the data for the x-axis, and their integer position on the axis
countries = Top202022Apps['Country of asylum (ISO)']
pos = np.arange(len(countries))
# Create the data for the y-axis
applications=Top202022Apps['applied']
# Create line plot
fig, ax = plt.subplots(figsize=(25, 15))
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
plt.xticks(pos, countries, size = 20)
plt.yticks(size = 20)
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add title and labels
plt.title('Asylum Applications: Top 20 Countries: 2022',color = '#000000',fontsize=24, pad = 20)
plt.xlabel('Country of Asylum (ISO)',color = '#000000', fontsize = 20, labelpad = 30)
plt.ylabel('Applications', color = '#000000',fontsize = 20, labelpad = 30)
plt.yticks([])
# To label the entire line
## zip joins x and y coordinates in pairs
plt.annotate(Top202022Apps['FormattedApps'][3], (pos[3],applications[3]), textcoords = 'offset points', xytext = (0,-40), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(Top202022Apps['FormattedApps'][4], (pos[4],applications[4]), textcoords = 'offset points', xytext = (0,50), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(Top202022Apps['FormattedApps'][6], (pos[6],applications[6]), textcoords = 'offset points', xytext = (0,-40), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(Top202022Apps['FormattedApps'][9], (pos[9],applications[9]), textcoords = 'offset points', xytext = (0,30), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(Top202022Apps['FormattedApps'][11], (pos[11],applications[11]), textcoords = 'offset points', xytext = (0,50), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(Top202022Apps['FormattedApps'][13], (pos[13],applications[13]), textcoords = 'offset points', xytext = (0,60), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(Top202022Apps['FormattedApps'][15], (pos[15],applications[15]), textcoords = 'offset points', xytext = (0,60), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(Top202022Apps['FormattedApps'][17], (pos[17],applications[17]), textcoords = 'offset points', xytext = (0,60), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(Top202022Apps['FormattedApps'][19], (pos[19],applications[19]), textcoords = 'offset points', xytext = (0,70), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
for x,y in zip(pos,applications):
    label = "{:,.0f}".format(y)
    if (x != pos[8] and y!= applications[8]) and (x != pos[3] and y!=applications[3])and (x != pos[4] and y!=applications[4])and (x != pos[6] and y!=applications[6])and (x != pos[9] and y!=applications[9])and \
    (x != pos[11] and y!=applications[11]) and (x != pos[13] and y!=applications[13])and (x != pos[15] and y!=applications[15]) and (x != pos[17] and y!=applications[17])and (x != pos[19] and y!=applications[19]):
        plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center', size =20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
    if x == pos[8] and y == applications[8]:
        plt.annotate(label, (x,y),textcoords="offset points", xytext=(0,-40), ha='center', color = 'w', size =20, bbox=dict(boxstyle="round,pad=0.3",fc="#732282", ec="#732282", lw=2))
plt.savefig('Top20Applications2022LINE.png', bbox_inches = 'tight')
#%% And create bar plot
plt.figure(figsize=(30, 20))
bars = plt.bar(pos, applications, align = 'center', linewidth=0, color = '#00000090')
# Set the UK bar at a different color from the rest
bars[8].set_color('#732282')
# Set the position and tick labels of the x-axis
plt.xticks(pos, countries,fontsize=24, color = '#000000')
# Set the formatting of the ytick labels
plt.yticks(color = '#000000',fontsize=24)
# Annotate with text
plt.annotate(Top202022Apps['FormattedApps'][0], (pos[0],applications[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202022Apps['FormattedApps'][8], (pos[8],applications[8]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202022Apps['FormattedApps'][19], (pos[19],applications[19]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
# Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Remove all the spines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
# Set the labels and title
plt.xlabel('Country of Asylum (ISO)',color = '#000000', fontsize = 24, labelpad = 20)
plt.ylabel('Number of Applications', color = '#000000',fontsize = 24, labelpad = 20)
plt.title('Top 20 Countries: Total Application Volumes 2022', color = '#000000',fontsize=24, pad = 100)
# And save the figure
plt.savefig('Top20Applications2022.png')
#%% Calculate 2022 backlog data
backlog2022 = Top202022Apps[['Backlog', 'Country of asylum (ISO)']]
# Create new formatted column
backlog2022['Backlogformatted']=backlog2022['Backlog'].apply(lambda x: '{:,.0f}'.format(x))
# Sort and reset
backlog2022 =backlog2022.sort_values(by='Backlog', ascending = True)
backlog2022.reset_index(inplace = True)
backlog2022 = backlog2022.drop('index', axis = 1)
#%% Plot 2022 backlog
# First create the data for plotting
country = backlog2022['Country of asylum (ISO)']
backlog = backlog2022['Backlog']
colors = ['#00000090', '#00000090','#732282','#00000090','#00000090','#00000090','#00000090','#00000090','#00000090','#00000090','#00000090','#00000090','#00000090','#00000090','#00000090','#00000090','#00000090','#00000090','#00000090','#00000090']
# Create figure and plot
fig, ax = plt.subplots(figsize = (32,20))

ax.barh(country,backlog, color = colors)
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# invert the y-axis and set the tick labels and axis limits
ax.invert_yaxis()
#plt.yticks(country, size = 24)
#plt.ylim(21, -1)
# Set x-tick labels
#Format the x-axis so it has thousand-value commas
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Set the position and tick labels of the x-axis
#pos = np.arange(len(backlog))
#plt.xticks(pos, backlog, fontsize=24, color = '#000000')
plt.xticks(size = 24)
xmax = max(backlog)
# Annotate the text
plt.text(-520000, 0.17, backlog2022['Backlogformatted'][0], fontsize=16, color = 'w')
plt.text(-58000,2.13, backlog2022['Backlogformatted'][2], fontsize = 16, color = 'w')
plt.text(3000, 19.15, backlog2022['Backlogformatted'][19], fontsize=16, color = 'w')
# Add title and labels
plt.title('Asylum Decision Shortfalls or Surplus: Top 20 Countries: 2022',color = '#000000',fontsize=24, pad = 20)
plt.xlabel('Decisions',color = '#000000', fontsize = 20, labelpad = 30)
plt.ylabel('Country of Asylum (ISO)', color = '#000000',fontsize = 20, labelpad = 30)
# Save the plot
plt.savefig("Backlog2022.png")
#%%###############################################################################
## Create 2023 Datasets
##################################################################################
#%% Worldwide dataset
UNTotalAppsDecs2023=UNTotalAppsDecs[UNTotalAppsDecs['Year']==2023]
UNTotalAppsDecs2023.sort_values(by=['applied'], ascending=False, inplace=True)
# Calculate summary statistics for worldwide 2023 dataset
FirstSixMonths2023Apps = UNTotalAppsDecs2023['applied'].sum()
FirstSixMonths2023Decs = UNTotalAppsDecs2023['Total decisions'].sum()
FirstSixMonths2023DecisionShortfall = FirstSixMonths2023Decs-FirstSixMonths2023Apps
#%% Create Top 20 Countries 2023 dataset
Top202023Apps = UNTotalAppsDecs2023[(UNTotalAppsDecs2023['Country of asylum (ISO)']=='USA')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='DEU')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='FRA')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='ESP')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='PER')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='MEX')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='ITA')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='GRC')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='CAN')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='GBR')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='CRI')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='TUR')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='BRA')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='UGA')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='AUT')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='AUS')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='SWE')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='EGY')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='NLD')|
        (UNTotalAppsDecs2023['Country of asylum (ISO)']=='BEL')]
Top202023Apps.reset_index(inplace = True)
# Create formatted application column
Top202023Apps['FormattedApps'] = Top202023Apps['applied'].apply(lambda x: '{:,.0f}'.format(x))
formattedapplications = Top202023Apps['FormattedApps']
# Create formatted decisions column
Top202023Apps['FormattedDecs'] = Top202023Apps['Total decisions'].apply(lambda x: '{:,.0f}'.format(x))
#Calculate grant rates
Top202023Apps['Granted']=Top202023Apps['Recognized decisions']/Top202023Apps['Total decisions']
Top202023Apps['Granted %'] = Top202023Apps['Granted'].apply(lambda x: '{:,.0%}'.format(x))
Top202023Apps['Other protection']=Top202023Apps['Complementary protection']/Top202023Apps['Total decisions']
Top202023Apps['Other protection %'] = Top202023Apps['Other protection'].apply(lambda x: '{:,.0%}'.format(x))
Top202023Apps['Refused']=Top202023Apps['Rejected decisions']/Top202023Apps['Total decisions']
Top202023Apps['Refused %'] = Top202023Apps['Refused'].apply(lambda x: '{:,.0%}'.format(x))
Top202023Apps['Otherwise Closed']=Top202023Apps['Otherwise closed']/Top202023Apps['Total decisions']
Top202023Apps['Otherwise Closed %'] = Top202023Apps['Otherwise Closed'].apply(lambda x: '{:,.0%}'.format(x))
## Look at 2023 FVP for each country
Top202023AppsFormattedFVP = Top202023Apps['FVP'].apply(lambda x: '{:,.0%}'.format(x))
#Calculate total grant rates for 2023
Top202023TotalDecisions = Top202023Apps['Total decisions'].sum()
Top202023Granted = Top202023Apps['Recognized decisions'].sum()
Top202023HP = Top202023Apps['Complementary protection'].sum()
Top202023Refused = Top202023Apps['Rejected decisions'].sum()
Top202023Closed = Top202023Apps['Otherwise closed'].sum()
Top202023GrantRate = Top202023Granted/Top202023TotalDecisions
Top202023HPRate = Top202023HP/Top202023TotalDecisions
Top202023RefusedRate = Top202023Refused/Top202023TotalDecisions
Top202023ClosedRate = Top202023Closed/Top202023TotalDecisions
Top202023GrantRate
Top202023HPRate
Top202023RefusedRate
Top202023ClosedRate
#%%Plot on pie chart
fig = plt.figure(figsize=(15, 15))
colors = ('#002664', '#317EFF', '#FE4365', '#FC9D9A')
data = [Top202023GrantRate, Top202023HPRate, Top202023RefusedRate, Top202023ClosedRate]
labels = ['Granted', 'Humanitarian Protection', 'Refused','Admin Closed']
plt.pie(data, labels = labels,
        autopct=lambda pct: func(pct, data),
        startangle = 90,
        counterclock = False,
        textprops=dict(color='#FFFFFF', fontsize = 24),
                       wedgeprops = {"edgecolor" : "w", 
                      'linewidth': 2},
                        colors = colors)
plt.title('Asylum Grant and Refusal Rates: \n\nTop 20 Countries January to June 2023', color = '#000000', fontsize = 20, pad = 20)
plt.legend(loc = 'lower left', fontsize = 16)
plt.savefig("GrantRatesPieChart2023.png")
#%% Create plotting data
applications = Top202023Apps['applied'].values
countries = Top202023Apps['Country of asylum (ISO)'].values
pos = np.arange(len(countries))
#Plot a line plot of application volumes
fig, ax = plt.subplots(figsize=(40, 20))
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
plt.xticks(pos, countries, size = 20)
plt.yticks(size = 20)
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add title and labels
plt.title('Asylum Applications: Top 19 Countries: Jan - Jun 2023',color = '#000000',fontsize=24, pad = 20)
plt.xlabel('Country of Asylum (ISO)',color = '#000000', fontsize = 20, labelpad = 30)
plt.ylabel('Applications', color = '#000000',fontsize = 20, labelpad = 30)
plt.yticks([])
# Add annotations
#plt.annotate(Top202023Apps['FormattedApps'][0], (pos[0],applications[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20,bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=2))
#plt.annotate(Top202023Apps['FormattedApps'][7], (pos[7],applications[7]), textcoords = 'offset points', xytext = (0,20), ha = 'center', size = 20)
#plt.annotate(Top202023Apps['FormattedApps'][18], (pos[18],applications[18]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20)
# To label the entire line
## zip joins x and y coordinates in pairs
for x,y in zip(pos,applications):

    label = "{:,.0f}".format(y)
    if (x != pos[7] and y!= applications[7]):
        plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center',
                 size =20,
                 bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1)) # horizontal alignment can be left, right or center
    if x == pos[7] and y == applications[7]:
        plt.annotate(label, (x,y),textcoords="offset points", xytext=(0,-40), ha='center', color = 'w', size =20, bbox=dict(boxstyle="round,pad=0.3",fc="#732282", ec="#732282", lw=2))
# Save the plot
plt.savefig("ApplicationsFirst6Months2023.png")
#%%Now bar plot of decision volumes by type
Top202023Apps.sort_values(by ='Total decisions', ascending = False, inplace = True)
Top202023Apps.reset_index(inplace = True)
granted = Top202023Apps['Recognized decisions'].values
hp = Top202023Apps['Complementary protection'].values
refused = Top202023Apps['Rejected decisions'].values
adminclosed = Top202023Apps['Otherwise closed'].values
countries = Top202023Apps['Country of asylum (ISO)'].values

#%% and plot
width = 0.8
fig, ax = plt.subplots(figsize=(30, 20))
ax.bar(countries, granted, width, color='#002664', label= 'Granted')
ax.bar(countries, hp, width, bottom=granted, color='#317EFF', label = 'Humanitarian Protection')
ax.bar(countries, refused, width, bottom=hp + granted, color='#FE4365', label = 'Refused')
ax.bar(countries, adminclosed, width, bottom=refused + hp+granted, color='#FC9D9A', label = 'Admin Closed')
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations

# Granted annotations
ax.annotate(Top202023Apps['Granted %'][0], (countries[0],Top202023Apps['Granted'][0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = 'w')
ax.annotate(Top202023Apps['Granted %'][1], (countries[1],Top202023Apps['Granted'][1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = 'w')
ax.annotate(Top202023Apps['Granted %'][2], (countries[2],Top202023Apps['Granted'][2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = 'w')
ax.annotate(Top202023Apps['Granted %'][3], (countries[3],Top202023Apps['Granted'][3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(Top202023Apps['Granted %'][6], (countries[6],Top202023Apps['Granted'][6]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = 'w')
ax.annotate(Top202023Apps['Granted %'][8], (countries[8],Top202023Apps['Granted'][8]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(Top202023Apps['Granted %'][9], (countries[9],Top202023Apps['Granted'][9]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(Top202023Apps['Granted %'][10], (countries[10],Top202023Apps['Granted'][10]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(Top202023Apps['Granted %'][15], (countries[15],Top202023Apps['Granted'][15]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(Top202023Apps['Granted %'][17], (countries[17],Top202023Apps['Granted'][17]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = 'w')
#HP annotations
ax.annotate(Top202023Apps['Other protection %'][0], (countries[0],hp[0]-granted[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = 'w')
ax.annotate(Top202023Apps['Other protection %'][4], (countries[4],hp[4]-(7*granted[4])), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = 'w')
# Decisions annotations
ax.annotate(Top202023Apps['FormattedDecs'][0], (countries[0],Top202023Apps['Total decisions'][0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202023Apps['FormattedDecs'][1], (countries[1],Top202023Apps['Total decisions'][1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202023Apps['FormattedDecs'][2], (countries[2],Top202023Apps['Total decisions'][2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202023Apps['FormattedDecs'][3], (countries[3],Top202023Apps['Total decisions'][3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202023Apps['FormattedDecs'][4], (countries[4],Top202023Apps['Total decisions'][4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202023Apps['FormattedDecs'][5], (countries[5],Top202023Apps['Total decisions'][5]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202023Apps['FormattedDecs'][6], (countries[6],Top202023Apps['Total decisions'][6]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202023Apps['FormattedDecs'][7], (countries[7],Top202023Apps['Total decisions'][7]), textcoords = 'offset points', xytext = (0,10), ha = 'center', color = 'black', size =20)
plt.annotate(Top202023Apps['FormattedDecs'][8], (countries[8],Top202023Apps['Total decisions'][8]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="w", ec="#732282", lw=2))
plt.annotate(Top202023Apps['FormattedDecs'][9], (countries[9],Top202023Apps['Total decisions'][9]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202023Apps['FormattedDecs'][10], (countries[10],Top202023Apps['Total decisions'][10]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202023Apps['FormattedDecs'][11], (countries[11],Top202023Apps['Total decisions'][11]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202023Apps['FormattedDecs'][12], (countries[12],Top202023Apps['Total decisions'][12]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202023Apps['FormattedDecs'][13], (countries[13],Top202023Apps['Total decisions'][13]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202023Apps['FormattedDecs'][14], (countries[14],Top202023Apps['Total decisions'][14]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202023Apps['FormattedDecs'][15], (countries[15],Top202023Apps['Total decisions'][15]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202023Apps['FormattedDecs'][16], (countries[16],Top202023Apps['Total decisions'][16]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202023Apps['FormattedDecs'][17], (countries[17],Top202023Apps['Total decisions'][17]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(Top202023Apps['FormattedDecs'][18], (countries[18],Top202023Apps['Total decisions'][18]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#Closed annotations
ax.annotate(Top202023Apps['Otherwise Closed %'][0], (countries[0],Top202023Apps['Total decisions'][0]), textcoords = 'offset points', xytext = (0,-20), ha = 'center', size = 20, color = 'w')
plt.annotate(Top202023Apps['Otherwise Closed %'][1], (countries[1],Top202023Apps['Total decisions'][1]), textcoords = 'offset points', xytext = (0,-20), ha = 'center', size = 20, color = 'w')
plt.annotate(Top202023Apps['Otherwise Closed %'][2], (countries[2],Top202023Apps['Total decisions'][2]), textcoords = 'offset points', xytext = (0,-20), ha = 'center', size = 20, color = 'w')
plt.annotate(Top202023Apps['Otherwise Closed %'][5], (countries[5],Top202023Apps['Total decisions'][5]), textcoords = 'offset points', xytext = (0,-20), ha = 'center', size = 20, color = 'w')
plt.annotate(Top202023Apps['Otherwise Closed %'][8], (countries[8],Top202023Apps['Total decisions'][8]), textcoords = 'offset points', xytext = (0,-20), ha = 'center', size = 20, color = 'w')
plt.annotate(Top202023Apps['Otherwise Closed %'][9], (countries[9],Top202023Apps['Total decisions'][9]), textcoords = 'offset points', xytext = (0,-20), ha = 'center', size = 20, color = 'w')
plt.annotate(Top202023Apps['Otherwise Closed %'][10], (countries[10],Top202023Apps['Total decisions'][10]), textcoords = 'offset points', xytext = (0,-20), ha = 'center', size = 20, color = 'w')
plt.annotate(Top202023Apps['Otherwise Closed %'][11], (countries[11],Top202023Apps['Total decisions'][11]), textcoords = 'offset points', xytext = (0,-20), ha = 'center', size = 20, color = 'w')
plt.annotate(Top202023Apps['Otherwise Closed %'][14], (countries[14],Top202023Apps['Total decisions'][14]), textcoords = 'offset points', xytext = (0,-20), ha = 'center', size = 20, color = 'w')
# Add title and labels
plt.xlabel('Country of Asylum (ISO)',color = '#000000', fontsize = 20, labelpad = 30)
plt.ylabel('Decisions', color = '#000000',fontsize = 20)
ax.set_title("Decision Volumes by Type: Top 19 Countries Jan - Jun 2023", color = '#000000', fontsize = 24, pad = 20)
ax.legend(loc="upper center", fontsize = 14)
# Save the plot
plt.savefig("DecisionsFirst6Months2023.png")
#%% Now plot FVP
Top202023Apps.sort_values(by = 'FVP', ascending = False, inplace = True)
Top202023Apps.reset_index(inplace = True)
countries =Top202023Apps['Country of asylum (ISO)']
pos = range(len(countries))
#%% Plot
fig, ax = plt.subplots(figsize=(30, 10))
plt.bar(countries, Top202023Apps['FVP'], width = .8, color = '#000000', label = 'Face-Value Productivity')
plt.xticks(pos, countries, size = 20)
plt.yticks([])
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
for x,y in zip(pos,Top202023Apps['FVP']):

    label = "{:,.0%}".format(y)

    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center',
                 size =20,
                 bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#000000", lw=1)) # horizontal alignment can be left, right or center
    if x == pos[8] and y == Top202023Apps['FVP'][8]:
        plt.annotate(label, (x,y),textcoords="offset points", xytext=(0,10), ha='center', color = 'black', size =20, bbox=dict(boxstyle="round,pad=0.3",fc="w", ec="#732282", lw=2))
# Add hline
plt.axhline(y=1, color='#88234590', linestyle='--')
plt.text(-1.01, 1.05,'100% FVP', fontsize=16, color = '#88234590')
         # Add title and labels
plt.xlabel('Country of Asylum (ISO)',color = '#000000', fontsize = 20, labelpad = 20)
plt.ylabel('FVP', color = '#000000',fontsize = 20, labelpad = 20)
plt.title("Face-Value Productivity: Top 19 Countries January to June 2023", color = '#000000', fontsize = 24, pad = 30)
plt.legend(loc="center", fontsize = 12)
plt.savefig("FVPFirst6Months2023.png")
#%%###############################################################################
## Calculate Average Annual Applications 2000 to 2022
##################################################################################
#%%Create aggregated dataset with annual averages for each Top 20 country
AverageApps2000To2022=UNTotalAppsDecs[UNTotalAppsDecs['Year']!=2023]
AverageApps2000To2022 = AverageApps2000To2022.groupby('Country of asylum (ISO)').agg({'applied':np.nanmean})
AverageApps2000To2022.reset_index(inplace=True)
AverageApps = AverageApps2000To2022[(AverageApps2000To2022['Country of asylum (ISO)']=='USA')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='DEU')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='FRA')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='ESP')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='PER')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='MEX')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='ITA')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='GRC')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='CAN')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='GBR')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='CRI')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='TUR')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='BRA')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='UGA')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='AUT')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='AUS')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='SWE')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='EGY')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='NLD')|
        (AverageApps2000To2022['Country of asylum (ISO)']=='BEL')]
AverageApps.sort_values(by=['applied'], ascending=False, inplace=True)
AverageApps.reset_index(inplace = True)
AverageApps = AverageApps.drop('index', axis = 1)
AverageApps['FormattedApps'] = AverageApps['applied'].apply(lambda x: '{:,.0f}'.format(x))
#%% Create plotting data
countries = AverageApps['Country of asylum (ISO)']
pos = np.arange(len(countries))
applications=AverageApps['applied']
# Create line plot
fig, ax = plt.subplots(figsize=(20, 10))
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
plt.xticks(pos, countries, size = 20)
plt.yticks(size = 20)
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add title and labels
plt.title('Asylum Applications: Top 20 Countries: Annual Averages 2000 to 2022',color = '#000000',fontsize=24, pad = 20)
plt.xlabel('Country of Asylum (ISO)',color = '#000000', fontsize = 20, labelpad = 30)
plt.ylabel('Applications', color = '#000000',fontsize = 20, labelpad = 30)
plt.yticks([])
# Annotate

plt.annotate(AverageApps['FormattedApps'][0], (pos[0],applications[0]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][1], (pos[1],applications[1]), textcoords = 'offset points', xytext = (0,-20), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][2], (pos[2],applications[2]), textcoords = 'offset points', xytext = (0,30), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][3], (pos[3],applications[3]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =20, color = 'w', bbox=dict(boxstyle="round,pad=0.3",fc="#732282", ec="#732282", lw=2))
plt.annotate(AverageApps['FormattedApps'][4], (pos[4],applications[4]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][5], (pos[5],applications[5]), textcoords = 'offset points', xytext = (0,-20), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][6], (pos[6],applications[6]), textcoords = 'offset points', xytext = (0,30), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][7], (pos[7],applications[7]), textcoords = 'offset points', xytext = (0,-50), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][8], (pos[8],applications[8]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][9], (pos[9],applications[9]), textcoords = 'offset points', xytext = (0,60), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][10], (pos[10],applications[10]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][11], (pos[11],applications[11]), textcoords = 'offset points', xytext = (0,60), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][12], (pos[12],applications[12]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][13], (pos[13],applications[13]), textcoords = 'offset points', xytext = (0,60), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][14], (pos[14],applications[14]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][15], (pos[15],applications[15]), textcoords = 'offset points', xytext = (0,60), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][16], (pos[16],applications[16]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][17], (pos[17],applications[17]), textcoords = 'offset points', xytext = (0,60), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][18], (pos[18],applications[18]), textcoords = 'offset points',  xytext=(0,10), ha='center', size =20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(AverageApps['FormattedApps'][19], (pos[19],applications[19]), textcoords = 'offset points', xytext = (0,60), ha = 'center', size = 20, color = '#000000',bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.savefig("Top20AppsAnnualAveragesLINE.png", bbox_inches='tight')
#%%And plot
plt.figure(figsize=(30, 20))
bars = plt.bar(pos, applications, align = 'center', linewidth=0, color = '#00000090')
bars[3].set_color('#732282')
plt.xticks(pos, countries, alpha = 0.8,fontsize=20, color = '#000000')
plt.yticks(fontsize=20, color = '#000000')
plt.annotate(AverageApps['FormattedApps'][0], (pos[0],applications[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(AverageApps['FormattedApps'][3], (pos[3],applications[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(AverageApps['FormattedApps'][19], (pos[19],applications[19]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
#Set tick sizes
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.xlabel('Country of Asylum (ISO)',color = '#000000', fontsize = 20, labelpad = 20)
plt.ylabel('Average Application Volumes', color = '#000000',fontsize = 20, labelpad = 20)
plt.savefig("Top20AppsAnnualAverages.png", bbox_inches='tight')
#%%############################################################################
## Top 20 Countries - Average Grant Rates
###############################################################################
#%%Create data for plotting
Top20=UNTotalAppsDecs[(UNTotalAppsDecs['Country of asylum (ISO)']=='USA')|(UNTotalAppsDecs['Country of asylum (ISO)']=='DEU')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='FRA')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='ESP')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='PER')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='MEX')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='ITA')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='GRC')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='CAN')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='GBR')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='CRI')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='TUR')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='BRA')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='UGA')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='AUT')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='AUS')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='SWE')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='EGY')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='NLD')|
        (UNTotalAppsDecs['Country of asylum (ISO)']=='BEL')]
Top20.reset_index(inplace=True)
Top20=Top20.drop('index', axis = 1)
#%% Calculate overall grant & refusal rates
Top20TotalDecisions = Top20['Total decisions'].sum()
Top20Granted = Top20['Recognized decisions'].sum()
Top20HP = Top20['Complementary protection'].sum()
Top20Refused = Top20['Rejected decisions'].sum()
Top20Closed = Top20['Otherwise closed'].sum()
Top20GrantRate = Top20Granted/Top20TotalDecisions
Top20HPRate = Top20HP/Top20TotalDecisions
Top20RefusedRate = Top20Refused/Top20TotalDecisions
Top20ClosedRate = Top20Closed/Top20TotalDecisions
Top20GrantRate
Top20HPRate
Top20RefusedRate
Top20ClosedRate
#%%Plot on pie chart
# Creating plot
fig = plt.figure(figsize=(15, 15))
colors = ('#002664', '#317EFF', '#FE4365', '#FC9D9A')
data = [Top20GrantRate, Top20HPRate, Top20RefusedRate, Top20ClosedRate]
labels = ['Granted', 'Humanitarian Protection', 'Refused','Admin Closed']
plt.pie(data, labels = labels,
        autopct=lambda pct: func(pct, data),
        startangle = 90,
        counterclock = False,
        textprops=dict(color='#FFFFFF', fontsize = 30),
                       wedgeprops = {"edgecolor" : "w", 
                      'linewidth': 2},
                        colors = colors)
plt.title('Asylum Grant and Refusal Rates: \n\nTop 20 Countries 2000 to June 2023', color = '#000000', fontsize = 20, pad = 20)
plt.legend(loc = 'lower left', fontsize = 16)
plt.savefig("GrantRatesPieChart2000To2023.png")
#%%
#Calculate grant rates
Top20['Granted']=Top20['Recognized decisions']/Top20['Total decisions']
Top20['Other protection']=Top20['Complementary protection']/Top20['Total decisions']
Top20['Refused']=Top20['Rejected decisions']/Top20['Total decisions']
Top20['Otherwise Closed']=Top20['Otherwise closed']/Top20['Total decisions']
#%%
AverageGrantRates2000To2023=Top20.groupby(['Country of asylum (ISO)']).agg({'Granted':np.nanmean, 'Other protection':np.nanmean,'Refused':np.nanmean,'Otherwise Closed':np.nanmean})
AverageGrantRates2000To2023 = AverageGrantRates2000To2023.sort_values(by=['Granted'], ascending = False)
AverageGrantRates2000To2023.reset_index(inplace = True)
#Create formatted and annotation values
AverageGrantRates2000To2023['Granted%']=AverageGrantRates2000To2023['Granted'].apply(lambda x: '{:,.0%}'.format(x))
formattedgranted = AverageGrantRates2000To2023['Granted%']
AverageGrantRates2000To2023['HP%']=AverageGrantRates2000To2023['Other protection'].apply(lambda x: '{:,.0%}'.format(x))
formattedhp = AverageGrantRates2000To2023['HP%']
AverageGrantRates2000To2023['closed%']=AverageGrantRates2000To2023['Otherwise Closed'].apply(lambda x: '{:,.0%}'.format(x))
formattedclosed = AverageGrantRates2000To2023['closed%']
granted = AverageGrantRates2000To2023['Granted']
closed = AverageGrantRates2000To2023['closed%']
#%% Now plot
AverageGrantRates2000To2023.plot(x='Country of asylum (ISO)', kind='bar', stacked=True, figsize=(20,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, width = 0.85)
#Now annotate
plt.annotate(formattedgranted[0], (0, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[1], (1, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[2], (2, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[3], (3, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[4], (4, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[5], (5, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[6], (6, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[7], (7, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[8], (8, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF', weight = 'bold')
plt.annotate(formattedgranted[9], (9, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[10], (10, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[11], (11, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[12], (12, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[13], (13, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[14], (14, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[15], (15, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[16], (16, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[17], (17, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[18], (18, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[19], (19, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
             #HP annotations
plt.annotate(formattedhp[10], (10, granted[10]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedhp[13], (13, granted[13]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedhp[14], (14, granted[14]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedhp[17], (17, granted[17]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedhp[18], (18, granted[18]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedhp[19], (19, granted[19]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
             # Closed annotations
plt.annotate(closed[0], (0, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[1], (1, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[2], (2, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[3], (3, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[4], (4, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[5], (5, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[6], (6, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[7], (7, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[8], (8, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF', weight ='bold')
plt.annotate(closed[9], (9, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[1], (10, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[11], (11, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[12], (12, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[13], (13, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[14], (14, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[15], (15, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[16], (16, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[17], (17, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[18], (18, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(closed[19], (19, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
#Format y-axis
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0%}'))
#Remove ticks and spines
plt.tick_params(
    axis='y',        
    which='both',     
    left=False,      
    right=False,         
    labelbottom=True) 
plt.tick_params(
    axis='x',         
    which='both',     
    top=False,      
    bottom=False,        
    labelbottom=True) 
for spine in plt.gca().spines.values():
    spine.set_visible(False)
#Set tick sizes
plt.xticks(size = 20)
plt.yticks(size = 20)
#Add legends and title and labels
plt.legend(['Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'center', frameon = True, prop={'size': 14})
plt.xlabel('Country of Asylum (ISO)', fontsize = 24, labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate', fontsize = 24, labelpad = 20)
plt.title('Top 20: Average Grant Rates 2000 to June 2023', color = '#000000', fontsize = 24, pad = 10)
plt.savefig("Top20GrantRates.png", bbox_inches='tight')
#%% ###########################################################################
## Create 2022 Grant Rates for Comparison with Average
###############################################################################
#%%Create data
Top202022=Top20[Top20['Year']==2022]
Top202022['Granted%']=Top202022['Granted'].apply(lambda x: '{:,.0%}'.format(x))
Top202022['HP%']=Top202022['Other protection'].apply(lambda x: '{:,.0%}'.format(x))
Top202022['Refused%']=Top202022['Refused'].apply(lambda x: '{:,.0%}'.format(x))
Top202022['closed%']=Top202022['Otherwise Closed'].apply(lambda x: '{:,.0%}'.format(x))
Top202022 = Top202022.sort_values(by=['Granted'], ascending = False)
Top202022.reset_index(inplace=True)
Top202022=Top202022.drop('index', axis = 1)
Top202022.drop(['Year', 'applied', 'Recognized decisions', 'Complementary protection', 'Rejected decisions','Otherwise closed', 'Total decisions', 'Backlog', 'FVP'], axis = 1, inplace = True)
#Create formatted and annotation values
granted = Top202022['Granted']
formattedgranted = Top202022['Granted%']
formattedhp = Top202022['HP%']
formattedrefused = Top202022['Refused%']
formattedclosed = Top202022['closed%']
#%% Now plot 2022 grant rates
Top202022.plot(x='Country of asylum (ISO)', kind='bar', stacked=True, figsize=(20,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, width = 0.85)
#Now annotate
plt.annotate(formattedgranted[0], (0, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[1], (1, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[2], (2, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[3], (3, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF', weight ='bold')
plt.annotate(formattedgranted[4], (4, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[5], (5, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[6], (6, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[7], (7, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[8], (8, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[9], (9, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[10], (10, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[11], (11, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[12], (12, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[13], (13, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[14], (14, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[15], (15, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[16], (16, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[17], (17, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[18], (18, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedgranted[19], (19, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
             #HP annotations
plt.annotate(formattedhp[9], (9, granted[9]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedhp[13], (13, granted[13]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedhp[14], (14, granted[14]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedhp[16], (16, granted[16]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedhp[19], (19, granted[19]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 18, color = '#FFFFFF')
             # Closed annotations
plt.annotate(formattedclosed[2], (2, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedclosed[3], (3, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF', weight ='bold')
plt.annotate(formattedclosed[4], (4, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedclosed[5], (5, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedclosed[6], (6, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedclosed[7], (7, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedclosed[8], (8, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedclosed[9], (9, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedclosed[1], (10, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedclosed[11], (11, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedclosed[12], (12, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedclosed[13], (13, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedclosed[14], (14, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedclosed[15], (15, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedclosed[17], (17, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedclosed[18], (18, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
plt.annotate(formattedclosed[19], (19, .98), textcoords = 'offset points', xytext = (0,-10), ha = 'center', size = 18, color = '#FFFFFF')
#Remove the label by setting it to an empty list and add titles
#plt.yticks([])
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0%}')) # No decimal places
plt.tick_params(
    axis='y',          
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False) # labels along the bottom edge are off
#Remove the label by setting it to an empty list
#plt.yticks([])
plt.xlabel('Country of Asylum (ISO)',color = '#000000', fontsize = 20, labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate', color = '#000000',fontsize = 20, labelpad = 20)
plt.title('Top 20 Countries by Application Volumes 2017 to June 2023: Grant Rates in 2022',fontsize = 24, color = '#000000', pad = 50)
plt.legend(['Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'center', frameon = True, prop={'size': 14})
#Set tick sizes
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.savefig("2022GrantRates.png", bbox_inches='tight')
#%%#######################################################################
## Now create pie chart with 2022 Total Grant Rates
############################################################################
#%%
#%% Calculate 2022 grant & refusal rates
Grants2022 = Top20[Top20['Year']==2022]
Grants2022TotalDecisions = Grants2022['Total decisions'].sum()
Grants2022Granted = Grants2022['Recognized decisions'].sum()
Grants2022HP = Grants2022['Complementary protection'].sum()
Grants2022Refused = Grants2022['Rejected decisions'].sum()
Grants2022Closed = Grants2022['Otherwise closed'].sum()
Grants2022GrantRate = Grants2022Granted/Grants2022TotalDecisions
Grants2022HPRate = Grants2022HP/Grants2022TotalDecisions
Grants2022RefusedRate = Grants2022Refused/Grants2022TotalDecisions
Grants2022ClosedRate = Grants2022Closed/Grants2022TotalDecisions
Grants2022GrantRate
Grants2022HPRate
Grants2022RefusedRate
Grants2022ClosedRate
colors = ('#002664', '#317EFF', '#FE4365', '#FC9D9A')
#%%Plot on pie chart
# Creating plot
fig = plt.figure(figsize=(15, 15))
data = [Grants2022GrantRate, Grants2022HPRate, Grants2022RefusedRate, Grants2022ClosedRate]
labels = ['Granted', 'Humanitarian Protection', 'Refused','Admin Closed']
plt.pie(data, labels = labels,
        autopct=lambda pct: func(pct, data),
        startangle = 90,
        counterclock = False,
        textprops=dict(color='#FFFFFF', fontsize = 30),
        colors = colors,
        wedgeprops = {"edgecolor" : "w", 
                      'linewidth': 2} )
plt.title('Asylum Grant and Refusal Rates: \n\nTop 20 Countries 2022', color = '#000000', fontsize = 20, pad = 20)
plt.legend(loc = 'lower left', fontsize = 16)
plt.savefig("2022GrantRatesPieChart.png")
#%%%##########################################################################
## Worldwide Grant Rates since 2000 for All Countries in dataset
###############################################################################
AnnualGrantRates = UNTotalAppsDecs.groupby('Year').agg({'applied':np.sum, 'Recognized decisions':np.sum, 'Complementary protection':np.sum, 'Rejected decisions':np.sum, 'Otherwise closed':np.sum, 'Total decisions':np.sum})
AnnualGrantRates.reset_index(inplace=True)
AnnualGrantRates['Granted']=AnnualGrantRates['Recognized decisions']/AnnualGrantRates['Total decisions']
AnnualGrantRates['Other protection']=AnnualGrantRates['Complementary protection']/AnnualGrantRates['Total decisions']
AnnualGrantRates['Refused']=AnnualGrantRates['Rejected decisions']/AnnualGrantRates['Total decisions']
AnnualGrantRates['Otherwise Closed']=AnnualGrantRates['Otherwise closed']/AnnualGrantRates['Total decisions']
AnnualGrantRates.drop(['applied', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed', 'Total decisions'], axis = 1, inplace = True)
#%%###########################################################################
## FVP 2017, 2018, 2019, 2020, 2021, 2022 for Top 20 Countries
##############################################################################
FVP2017To2022 =Top20[(Top20['Year']==2017)|
        (Top20['Year']==2018)|
        (Top20['Year']==2019)|
        (Top20['Year']==2020)|
        (Top20['Year']==2021)|
        (Top20['Year']==2022)]
FVP2017To2022.sort_values(by = ['Country of asylum (ISO)'], ascending = True, inplace = True)
FVP2017To2022.reset_index(inplace=True)
FVP2017To2022=FVP2017To2022.drop('index', axis = 1)
FVP2017To2022.columns
FVP2017To2022 = FVP2017To2022.drop(['applied', 'Recognized decisions'],axis = 1)
FVP2017To2022 = FVP2017To2022.drop(['Complementary protection', 'Rejected decisions'],axis = 1)
FVP2017To2022 = FVP2017To2022.drop(['Otherwise closed', 'Total decisions'],axis = 1)
FVP2017To2022 = FVP2017To2022.drop(['Backlog', 'Granted'],axis = 1)
FVP2017To2022 = FVP2017To2022.drop(['Other protection', 'Refused', 'Otherwise Closed'],axis = 1)
FVP2017To2022=FVP2017To2022.pivot(index='Year', columns='Country of asylum (ISO)', values='FVP')
FVP2017To2022.columns
FVP2017To2022.min()
FVP2017To2022.max()
#%% Calculate Top 20 Average FVP
AUSAve=FVP2017To2022['AUS'].mean()
AUTAve=FVP2017To2022['AUT'].mean()
BELAve=FVP2017To2022['BEL'].mean()
BRAAve=FVP2017To2022['BRA'].mean()
CANAve=FVP2017To2022['CAN'].mean()
CRIAve=FVP2017To2022['CRI'].mean()
DEUAve=FVP2017To2022['DEU'].mean()
EGYAve=FVP2017To2022['EGY'].mean()
ESPAve=FVP2017To2022['ESP'].mean()
FRAAve=FVP2017To2022['FRA'].mean()
GBRAve=FVP2017To2022['GBR'].mean()
GRCAve=FVP2017To2022['GRC'].mean()
ITAAve=FVP2017To2022['ITA'].mean()
MEXAve=FVP2017To2022['MEX'].mean()
NLDAve=FVP2017To2022['NLD'].mean()
PERAve=FVP2017To2022['PER'].mean()
SWEAve=FVP2017To2022['SWE'].mean()
TURAve=FVP2017To2022['TUR'].mean()
UGAAve=FVP2017To2022['UGA'].mean()
USAAve=FVP2017To2022['USA'].mean()
#%%
DEUAve
AUTAve
ITAAve
SWEAve
TURAve
UGAAve
GRCAve
EGYAve
FRAAve
BELAve
CANAve
AUSAve
NLDAve
ESPAve
GBRAve
BRAAve
USAAve
MEXAve
CRIAve
PERAve
#%%
#%% FVP Plot
#%%Plot a series of small multiples
#FVP2017To2022.reset_index(inplace = True)
#FVP2017To2022['Year']=map(int, FVP2017To2022['Year'])

fig, ((ax7,ax2,ax13, ax17, ax18), (ax19,ax12,ax8, ax10, ax3), (ax5,ax1,ax15, ax9,ax11), (ax4,ax20,ax14,ax6, ax16))=plt.subplots(4,5, sharex=True, sharey=True,figsize=(32,17))
axs = [ax1,ax2,ax3,ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13,
       ax14, ax15, ax16, ax17, ax18, ax19, ax20]

###
ax1.plot(FVP2017To2022['AUS'], 'o-',color='#000000', linewidth = 1)
ax1.axhline(y=1, color='#88234590', linestyle='--')
ax1.axhline(y = AUSAve, linestyle = '-.', color = 'r', linewidth = 3)
ax1.annotate(format(AUSAve, '.0%'), (2017, AUSAve), textcoords = 'offset points', xytext = (0,-30), ha = 'center', size = 20, color = '#000000')
#ax1.text(0,0, format(AUSAve, '.0%'), horizontalalignment = 'center', verticalalignment='center', transform = ax1.transAxes, color = 'r')
##
ax2.plot(FVP2017To2022['AUT'], 'o-', color='#000000')
ax2.axhline(y=1, color='#88234590', linestyle='--')
ax2.axhline(y = AUTAve, linestyle = '--', color = 'g', linewidth = 3)
ax2.text(2017-.2,AUTAve+0.2, format(AUTAve, '.0%'), ha='right', va='center', fontsize = 20)
###
ax3.plot(FVP2017To2022['BEL'], 'o-', color='#000000')
ax3.axhline(y=1, color='#88234590', linestyle='--')
ax3.axhline(y = BELAve, linestyle = '-.', color = 'r', linewidth = 3)
ax3.text(2017,BELAve-0.2, format(BELAve, '.0%'), ha='right', va='center', fontsize = 20)

###
ax4.plot(FVP2017To2022['BRA'], 'o-', color='#000000')
ax4.axhline(y=1, color='#88234590', linestyle='--')
ax4.axhline(y = BRAAve, linestyle = '-.', color = 'r', linewidth = 3)
ax4.text(2017-.2,BRAAve-0.2, format(BRAAve, '.0%'), ha='right', va='center', fontsize = 20)
ax4.text(2017,1.2, '100% FVP', horizontalalignment='right', verticalalignment='center', color='#88234590', fontsize = 8)
###
ax5.plot(FVP2017To2022['CAN'], 'o-', color='#000000')
ax5.axhline(y=1, color='#88234590', linestyle='--')
ax5.axhline(y = CANAve, linestyle = '-.', color = 'r', linewidth = 3)
ax5.annotate(format(CANAve, '.0%'), (2017, CANAve), textcoords = 'offset points', xytext = (0,-15), ha = 'center', size = 20, color = '#000000')
ax5.text(2017-.2,1.2, '100% FVP', horizontalalignment='right', verticalalignment='center', color='#88234590', fontsize = 8)
###
ax6.plot(FVP2017To2022['CRI'], 'o-', color='#000000')
ax6.axhline(y=1, color='#88234590', linestyle='--')
ax6.axhline(y = CRIAve, linestyle = '-.', color = 'r', linewidth = 3)
ax6.text(2017-.2,CRIAve-0.2, format(CRIAve, '.0%'), ha='right', va='center', fontsize = 20)
###
ax7.plot(FVP2017To2022['DEU'], 'o-', color='#000000')
ax7.axhline(y=1, color='#88234590', linestyle='--')
ax7.axhline(y = DEUAve, linestyle = '--', color = 'g', linewidth = 3)
ax7.text(2017,DEUAve+0.2, format(DEUAve, '.0%'), ha='right', va='center', fontsize = 20)
ax7.text(2017,DEUAve+.55, 'Ave FVP:', ha='right', va='center', fontsize = 20)
ax7.text(2017,1.2, '100% FVP', horizontalalignment='right', verticalalignment='center', color='#88234590', fontsize = 8)
###
ax8.plot(FVP2017To2022['EGY'], 'o-', color='#000000')
ax8.axhline(y=1, color='#88234590', linestyle='--')
ax8.axhline(y = EGYAve, linestyle = ':', color = 'orange', linewidth = 3)
ax8.text(2017-.2,EGYAve-0.2, format(EGYAve, '.0%'), ha='right', va='center', fontsize = 20)
###
ax9.plot(FVP2017To2022['ESP'], 'o-', color='#000000')
ax9.axhline(y=1, color='#88234590', linestyle='--')
ax9.axhline(y = ESPAve, linestyle = '-.', color = 'r', linewidth = 3)
ax9.annotate(format(ESPAve, '.0%'), (2017, ESPAve), textcoords = 'offset points', xytext = (0,-30), ha = 'center', size = 20, color = '#000000')
###
ax10.plot(FVP2017To2022['FRA'], 'o-', color='#000000')
ax10.axhline(y=1, color='#88234590', linestyle='--')
ax10.axhline(y = FRAAve, linestyle = ':', color = 'orange', linewidth = 3)
ax10.text(2017-.2,FRAAve-0.2, format(FRAAve, '.0%'), ha='right', va='center', fontsize = 20)
###
ax11.plot(FVP2017To2022['GBR'], 'o-', color = '#000000')
ax11.axhline(y=1, color='#88234590', linestyle='--')
ax11.axhline(y = GBRAve, linestyle = '-.', color = 'r', linewidth = 3)
ax11.text(2017,GBRAve-0.2, format(GBRAve, '.0%'), ha='right', va='center', fontsize = 20)
###
ax12.plot(FVP2017To2022['GRC'], 'o-', color='#000000')
ax12.axhline(y=1, color='#88234590', linestyle='--')
ax12.axhline(y = GRCAve, linestyle = '--', color = 'g', linewidth = 3)
ax12.text(2017,GRCAve+0.2, format(GRCAve, '.0%'), ha='right', va='center', fontsize = 20)
###
ax13.plot(FVP2017To2022['ITA'], 'o-', color='#000000')
ax13.axhline(y=1, color='#88234590', linestyle='--')
ax13.axhline(y = ITAAve, linestyle = '--', color = 'g', linewidth = 3)
ax13.text(2017,ITAAve+0.2, format(ITAAve, '.0%'), ha='right', va='center', fontsize = 20)
###
ax14.plot(FVP2017To2022['MEX'], 'o-', color='#000000')
ax14.axhline(y=1, color='#88234590', linestyle='--')
ax14.axhline(y = MEXAve, linestyle = '-.', color = 'r', linewidth = 3)
ax14.text(2017-.2,MEXAve-0.2, format(MEXAve, '.0%'), ha='right', va='center', fontsize = 20)
###
ax15.plot(FVP2017To2022['NLD'], 'o-', color='#000000')
ax15.axhline(y=1, color='#88234590', linestyle='--')
ax15.axhline(y = NLDAve, linestyle = '-.', color = 'r', linewidth = 3)
ax15.annotate(format(NLDAve, '.0%'), (2017, NLDAve), textcoords = 'offset points', xytext = (0,-25), ha = 'center', size = 20, color = '#000000')
###
ax16.plot(FVP2017To2022['PER'], 'o-', color='#000000')
ax16.axhline(y=1, color='#88234590', linestyle='--')
ax16.axhline(y = PERAve, linestyle = '-.', color = 'r', linewidth = 3)
ax16.text(2017-.2,PERAve-0.2, format(PERAve, '.0%'), ha='right', va='center', fontsize = 20)
#ax16.text(-.2,.35, '100% FVP', horizontalalignment='center', verticalalignment='center', transform=ax16.transAxes, color='#88234590')
###
ax17.plot(FVP2017To2022['SWE'], 'o-', color='#000000')
ax17.axhline(y=1, color='#88234590', linestyle='--')
ax17.axhline(y = SWEAve, linestyle = '--', color = 'g', linewidth = 3)
ax17.text(2017-.2,SWEAve+0.2, format(SWEAve, '.0%'), ha='right', va='center', fontsize = 20)
###
ax18.plot(FVP2017To2022['TUR'], 'o-', color='#000000')
ax18.axhline(y=1, color='#88234590', linestyle='--')
ax18.axhline(y = TURAve, linestyle = '--', color = 'g', linewidth = 3)
ax18.text(2017,TURAve+0.2, format(TURAve, '.0%'), ha='right', va='center', fontsize = 20)
###
ax19.plot(FVP2017To2022['UGA'], 'o-', color='#000000')
ax19.axhline(y=1, color='#88234590', linestyle='--')
ax19.axhline(y = UGAAve, linestyle = '--', color = 'g', linewidth = 3)
ax19.text(2017,UGAAve+0.2, format(UGAAve, '.0%'), ha='right', va='center', fontsize = 20)
ax19.text(2017,.85, '100% FVP', horizontalalignment='right', verticalalignment='center', color='#88234590', fontsize = 8)
###
ax20.plot(FVP2017To2022['USA'], 'o-', color='#000000')
ax20.axhline(y=1, color='#88234590', linestyle='--')
ax20.axhline(y = USAAve, linestyle = '-.', color = 'r', linewidth = 3)
ax20.text(2017-.2,USAAve-0.2, format(USAAve, '.0%'), horizontalalignment = 'right', verticalalignment='center', fontsize = 20)
##Set Ticks
##
ax1.set_title('12. Australia', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax1.set_axis_off()
##
ax2.set_title('2. Austria', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax2.set_axis_off()
##
ax3.set_title('10. Belgium', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax3.set_axis_off()
##
ax4.set_title('16. Brazil', fontsize = 20, y=.8,pad = -10, color = '#000000')
xticks = np.arange(2017,2023,1)
ax4.get_yaxis().set_visible(False)
ax4.spines['top'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.tick_params(axis='both', which='both', length=0)
ax4.set_xticks(xticks)
ax4.set_xticklabels(xticks, rotation=90, size=20)
ax4.set_xlabel('Year', size = 20, labelpad = 10)
##
ax5.set_title('11. Canada', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax5.set_axis_off()
##
ax6.set_title('19. Costa Rica', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax6.get_xaxis().set_visible(True)
ax6.get_yaxis().set_visible(False)
ax6.spines['top'].set_visible(False)
ax6.spines['left'].set_visible(False)
ax6.spines['bottom'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.tick_params(axis='both', which='both', length=0)
ax6.set_xticks(xticks)
ax6.set_xticklabels(xticks, rotation=90, fontsize=20)
ax6.set_xlabel('Year', size = 20, labelpad = 10)
##
ax7.set_title('1. Germany', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax7.set_axis_off()
##
ax8.set_title('8. Egypt', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax8.set_axis_off()
##
ax9.set_title('14. Spain', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax9.set_axis_off()
##
ax10.set_title('9. France', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax10.set_axis_off()
##
ax11.set_title('15. UK', fontsize = 20, y=.8,pad = -10, color = 'w', bbox=dict(boxstyle="round,pad=0.3",fc="#732282", ec="#732282", lw=2))
ax11.set_axis_off()
##
ax12.set_title('7. Greece', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax12.set_axis_off()
##
ax13.set_title('3. Italy', fontsize = 20, y=.8,pad = -10, color = '#000000', loc = 'right')
ax13.set_axis_off()
##
ax14.set_title('18. Mexico', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax14.get_xaxis().set_visible(True)
ax14.spines['top'].set_visible(False)
ax14.spines['left'].set_visible(False)
ax14.spines['bottom'].set_visible(False)
ax14.spines['right'].set_visible(False)
ax14.tick_params(axis='both', which='both', length=0)
ax14.set_xticks(xticks)
ax14.set_xticklabels(xticks, rotation=90, fontsize=20)
ax14.set_xlabel('Year', size = 20, labelpad = 10)
##
ax15.set_title('13. The Netherlands', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax15.set_axis_off()
#
ax16.set_title('20. Peru', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax16.get_xaxis().set_visible(True)
ax16.spines['top'].set_visible(False)
ax16.spines['left'].set_visible(False)
ax16.spines['bottom'].set_visible(False)
ax16.spines['right'].set_visible(False)
ax16.tick_params(axis='both', which='both', length=0)
ax16.set_xticks(xticks)
ax16.set_xticklabels(xticks, rotation=90, fontsize=20)
ax16.set_xlabel('Year', size = 20, labelpad = 10)
#
ax17.set_title('4. Sweden', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax17.set_axis_off()
##
ax18.set_title('5. Turkey', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax18.set_axis_off()
##
ax19.set_title('6. Uganda', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax19.set_axis_off()
##
ax20.set_title('17. USA', fontsize = 20, y=.8,pad = -10, color = '#000000')
ax20.get_xaxis().set_visible(True)
ax20.spines['top'].set_visible(False)
ax20.spines['left'].set_visible(False)
ax20.spines['bottom'].set_visible(False)
ax20.spines['right'].set_visible(False)
ax20.tick_params(axis='both', which='both', length=0)
ax20.set_xticks(xticks)
ax20.set_xticklabels(xticks, rotation=90, fontsize=20)
ax20.set_xlabel('Year', size = 20, labelpad = 10)
###
fig.suptitle('Average Face-Value Productivity: Application to Decision Ratio 2017 To 2022', fontsize = 24, color = '#000000')
#
plt.savefig('FVP2017To2022.png', bbox_inches='tight')
#%%############################################################################
## World Application & Decision volumes & grant types 2000 to 2022
##############################################################################
#%% Create datasets
UNTotalAppsDecsTo2022 = UNTotalAppsDecs[UNTotalAppsDecs['Year']!=2023]
#Create volumes of applications dataset
UNTotalAppsDecsTo2022 = UNTotalAppsDecsTo2022.groupby('Year').agg({'applied':np.sum, 'Recognized decisions':np.sum, 'Complementary protection':np.sum, 'Rejected decisions':np.sum, 'Otherwise closed':np.sum, 'Total decisions':np.sum})
#Calculate grant rates and formatted values
UNTotalAppsDecsTo2022['Granted']=UNTotalAppsDecsTo2022['Recognized decisions']/UNTotalAppsDecsTo2022['Total decisions']
UNTotalAppsDecsTo2022['Granted%']=UNTotalAppsDecsTo2022['Granted'].apply(lambda x: '{:,.0%}'.format(x))
UNTotalAppsDecsTo2022['Humanitarian Protection']=UNTotalAppsDecsTo2022['Complementary protection']/UNTotalAppsDecsTo2022['Total decisions']
UNTotalAppsDecsTo2022['HP%']=UNTotalAppsDecsTo2022['Humanitarian Protection'].apply(lambda x: '{:,.0%}'.format(x))
UNTotalAppsDecsTo2022['Formatted Applications']=UNTotalAppsDecsTo2022['applied'].apply(lambda x: '{:,.0f}'.format(x))
UNTotalAppsDecsTo2022['Formatted Decisions']=UNTotalAppsDecsTo2022['Total decisions'].apply(lambda x: '{:,.0f}'.format(x))
UNTotalAppsDecsTo2022.reset_index(inplace = True)
TotalGrant=UNTotalAppsDecsTo2022['Granted%']
applications = UNTotalAppsDecsTo2022['applied']
decisions = UNTotalAppsDecsTo2022['Total decisions']
formatteddecisions = UNTotalAppsDecsTo2022['Formatted Decisions']
formattedapplications = UNTotalAppsDecsTo2022['Formatted Applications']
formattedHP = UNTotalAppsDecsTo2022['HP%']
UNTotalAppsDecsTo2022.drop(['Formatted Applications','Formatted Decisions', 'applied', 'Total decisions', 'Granted', 'Granted%', 'Humanitarian Protection', 'HP%'], axis = 1, inplace = True)
Year = UNTotalAppsDecsTo2022['Year']
shortfall2015 = decisions[15]-applications[15]
surplus2016 = decisions[16]-applications[16]
#%% Decision Shortfalls and Surpluses
a2001 = decisions[1]-applications[1]
a2002 = decisions[2]-applications[2]
a2003 = decisions[3]-applications[3]
a2004 = decisions[4]-applications[4]
a2005 = decisions[5]-applications[5]
a2006 = decisions[6]-applications[6]
a2007 = decisions[7]-applications[7]
a2008 = decisions[8]-applications[8]
a2009 = decisions[9]-applications[9]
a20010 = decisions[10]-applications[10]
a20011 = decisions[11]-applications[11]
a20012 = decisions[12]-applications[12]
a20013 = decisions[13]-applications[13]
a20014 = decisions[14]-applications[14]
a20015 = decisions[15]-applications[15]
a20016 = decisions[16]-applications[16]
a20017 = decisions[17]-applications[17]
a20018 = decisions[18]-applications[18]
a20019 = decisions[19]-applications[19]
a20020 = decisions[20]-applications[20]
a20021 = decisions[21]-applications[21]
a20022 = decisions[22]-applications[22]
newlist = [a2001,a2002, a2003, a2004, a2005, a2006, a2007, a2008, a2009, a20010, a20011, a20012, a20013, a20014, a20015, a20016, a20017, a20018, a20019, a20020, a20021, a20022]
waterfall = pd.DataFrame(newlist)
waterfall.to_csv('waterfall.csv')

#%% Plot
UNTotalAppsDecsTo2022.plot(x='Year', kind='bar', stacked=True, figsize=(30,15), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, width = 0.85)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')

#Add data labels
plt.annotate(formattedapplications[0], (0, applications[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedapplications[6], (6, applications[6]), textcoords = 'offset points', xytext = (5,30), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedapplications[15], (15, applications[15]), textcoords = 'offset points', xytext = (-15,8), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedapplications[22], (22, applications[22]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formatteddecisions[15], (15, decisions[15]), textcoords = 'offset points', xytext = (-10,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[16], (16, decisions[16]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[22], (22, decisions[22]), textcoords = 'offset points', xytext = (10,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedHP[14], (14, UNTotalAppsDecsTo2022['Recognized decisions'][14]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(formattedHP[15], (15, UNTotalAppsDecsTo2022['Recognized decisions'][15]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(formattedHP[16], (16, UNTotalAppsDecsTo2022['Recognized decisions'][16]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(formattedHP[17], (17, UNTotalAppsDecsTo2022['Recognized decisions'][17]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(formattedHP[22], (22, UNTotalAppsDecsTo2022['Recognized decisions'][22]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
#Add grant rates
plt.annotate(TotalGrant[0], (0, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[1], (1, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[2], (2, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[3], (3, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[4], (4, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[5], (5, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[6], (6, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[7], (7, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[8], (8, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[9], (9, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[1], (10, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[11], (11, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[12], (12, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[13], (13, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[14], (14, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[15], (15, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[16], (16, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[17], (17, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[18], (18, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[19], (19, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[20], (20, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[21], (21, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')
plt.annotate(TotalGrant[21], (22, 0), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#FFFFFF')

#Remove axes and ticks
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.tick_params(
    axis='y',          
    which='both',     
    left=False,     
    right=False,        
    labelbottom=True) 
plt.tick_params(
    axis='x',          
    which='both',      
    top=False,     
    bottom=False,        
    labelbottom=True) 
tick = range(len(Year))
plt.xticks(tick, Year, size = 14)
plt.yticks(size = 14)
#Format y-axis
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Add legend, labels, tielt
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper left', frameon = True, fontsize = 14)
plt.xlabel('Year', fontsize = 16, labelpad = 20)
plt.ylabel('Application & Decision Volumes (by type)', fontsize = 20, labelpad = 20)
plt.title('World - Application and Decision Volumes (by decision type) and Asylum Grant Rates, 2000 to 2022',fontsize = 20, color = 'black', pad = 100)
plt.savefig('WorldAppsDecs.png')
#%%###########################################################################
## Dataset 2017 to 2022 Only to look at individual countries
##############################################################################
#%%Create datasets
UNTotalAppsDecs2017To2022=UNTotalAppsDecs[(UNTotalAppsDecs['Year']==2017)|(UNTotalAppsDecs['Year']==2018)|
        (UNTotalAppsDecs['Year']==2019)|
        (UNTotalAppsDecs['Year']==2020)|
        (UNTotalAppsDecs['Year']==2021)|
        (UNTotalAppsDecs['Year']==2022)]


#Calculate grant rates and formatted values
UNTotalAppsDecs2017To2022['Granted']=UNTotalAppsDecs2017To2022['Recognized decisions']/UNTotalAppsDecs2017To2022['Total decisions']
UNTotalAppsDecs2017To2022['Granted%']=UNTotalAppsDecs2017To2022['Granted'].apply(lambda x: '{:,.0%}'.format(x))
UNTotalAppsDecs2017To2022['Humanitarian Protection']=UNTotalAppsDecs2017To2022['Complementary protection']/UNTotalAppsDecs2017To2022['Total decisions']
UNTotalAppsDecs2017To2022['HP%']=UNTotalAppsDecs2017To2022['Humanitarian Protection'].apply(lambda x: '{:,.0%}'.format(x))
UNTotalAppsDecs2017To2022['Refused']=UNTotalAppsDecs2017To2022['Rejected decisions']/UNTotalAppsDecs2017To2022['Total decisions']
UNTotalAppsDecs2017To2022['Refused%']=UNTotalAppsDecs2017To2022['Refused'].apply(lambda x: '{:,.0%}'.format(x))
UNTotalAppsDecs2017To2022['Closed']=UNTotalAppsDecs2017To2022['Otherwise closed']/UNTotalAppsDecs2017To2022['Total decisions']
UNTotalAppsDecs2017To2022['Closed%']=UNTotalAppsDecs2017To2022['Closed'].apply(lambda x: '{:,.0%}'.format(x))
UNTotalAppsDecs2017To2022['Formatted Applications']=UNTotalAppsDecs2017To2022['applied'].apply(lambda x: '{:,.0f}'.format(x))
UNTotalAppsDecs2017To2022['Formatted Decisions']=UNTotalAppsDecs2017To2022['Total decisions'].apply(lambda x: '{:,.0f}'.format(x))
#%%############################################################################
## Individual countries application volumes and grant rates 2017 to 2022
###############################################################################
###############################################################################
#%%############################################################################
## USA
###############################################################################
#%% A. USA Apps & Deps
#%%Create data for plotting
UNUSA=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'USA']
UNUSA.reset_index(inplace = True)
UNUSA.drop(['index'], axis = 1, inplace = True)
UNUSA.columns
# Create df for bar plot
UNUSABar = UNUSA[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNUSA['applied']
decisions = UNUSA['Total decisions']
Year = UNUSA['Year'].values
formattedapplications = UNUSA['Formatted Applications']
formatteddecisions = UNUSA['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNUSA['Granted']
grantedformatted = UNUSA['Granted%']
hp = UNUSA['Complementary protection']
hpformatted = UNUSA['HP%']
refused = UNUSA['Refused']
refusedformatted = UNUSA['Refused%']
closed = UNUSA['Closed']
closedformatted = UNUSA['Closed%']
#%%Now bar plot of decision volumes by type
UNUSABar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,40), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(0,20), size = 20, ha = 'center', color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#HP annotations
#plt.annotate(hpformatted[0], (pos[0], hp[0]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[1], (pos[1], hp[1]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[2], (pos[2], hp[2]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[3], (pos[3], hp[3]), textcoords = 'offset points', xytext=(0,40), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[4], (pos[4], hp[4]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[5], (pos[5], hp[5]), textcoords = 'offset points', xytext=(0,0), ha = 'center', size = 20, color = 'w')
#Closed annotations
#plt.annotate(closedformatted[0], (pos[0], decisions[0]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[1], (pos[1], decisions[1]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[2], (pos[2], decisions[2]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[3], (pos[3], decisions[3]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[4], (pos[4], decisions[4]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[5], (pos[5], decisions[5]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('USA - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper seft', frameon = True, fontsize = 14)
plt.savefig('USAAppsDecs.png')
#%% USA Applicants
#%% Create data for plotting
USAApps = UNapps[UNapps['Country of asylum (ISO)']=='USA']
USAApps2017To2022 = USAApps[(USAApps['Year']== 2017)|
        (USAApps['Year']==2018)|
        (USAApps['Year']==2019)|
        (USAApps['Year']==2020)|
        (USAApps['Year']==2021)|
        (USAApps['Year']==2022)]
#2017
USATopApps2017=USAApps[USAApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
USATopApps2017.set_index('Year', inplace = True)
USATopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
USATopApps2018=USAApps[USAApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
USATopApps2018.set_index('Year', inplace = True)
USATopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
USATopApps2019=USAApps[USAApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
USATopApps2019.set_index('Year', inplace = True)
USATopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
USATopApps2020=USAApps[USAApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
USATopApps2020.set_index('Year', inplace = True)
USATopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
USATopApps2021=USAApps[USAApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
USATopApps2021.set_index('Year', inplace = True)
USATopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
USATopApps2022=USAApps[USAApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
USATopApps2022.set_index('Year', inplace = True)
USATopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = USATopApps2017['applied'].tolist()
bars01 = USATopApps2017['Country of origin (ISO)'].tolist()
height02 = USATopApps2018['applied']
bars02=USATopApps2018['Country of origin (ISO)']
height03 = USATopApps2019['applied']
bars03 = USATopApps2019['Country of origin (ISO)']
height04 = USATopApps2020['applied'].tolist()
bars04 = USATopApps2020['Country of origin (ISO)'].tolist()
height05 = USATopApps2021['applied']
bars05=USATopApps2021['Country of origin (ISO)']
height06 = USATopApps2022['applied']
bars06 = USATopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('USA - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('USATop3.png')
#%%############################################################################
## Germany
###############################################################################
#%% A. Germany Apps & Deps
#%%Create data for plotting
UNGermany=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'DEU']
UNGermany.reset_index(inplace = True)
UNGermany.drop(['index'], axis = 1, inplace = True)
UNGermany.columns
# Create df for bar plot
UNGermanyBar = UNGermany[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNGermany['applied']
decisions = UNGermany['Total decisions']
Year = UNGermany['Year'].values
formattedapplications = UNGermany['Formatted Applications']
formatteddecisions = UNGermany['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNGermany['Granted']
grantedformatted = UNGermany['Granted%']
hp = UNGermany['Complementary protection']
hpformatted = UNGermany['HP%']
refused = UNGermany['Refused']
refusedformatted = UNGermany['Refused%']
closed = UNGermany['Closed']
closedformatted = UNGermany['Closed%']
#%%Now bar plot of decision volumes by type
UNGermanyBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,40), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(0,20), size = 20, ha = 'center', color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#HP annotations
plt.annotate(hpformatted[0], (pos[0], hp[0]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[1], (pos[1], hp[1]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[2], (pos[2], hp[2]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[3], (pos[3], hp[3]), textcoords = 'offset points', xytext=(0,40), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[4], (pos[4], hp[4]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = 'w')
plt.annotate(hpformatted[5], (pos[5], hp[5]), textcoords = 'offset points', xytext=(0,0), ha = 'center', size = 20, color = 'w')
#Closed annotations
#plt.annotate(closedformatted[0], (pos[0], decisions[0]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[1], (pos[1], decisions[1]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[2], (pos[2], decisions[2]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[3], (pos[3], decisions[3]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[4], (pos[4], decisions[4]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[5], (pos[5], decisions[5]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Germany - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper seft', frameon = True, fontsize = 14)
plt.savefig('GermanyAppsDecs.png')
#%% Germany Applicants
#%% Create data for plotting
GermanyApps = UNapps[UNapps['Country of asylum (ISO)']=='DEU']
GermanyApps2017To2022 = GermanyApps[(GermanyApps['Year']== 2017)|
        (GermanyApps['Year']==2018)|
        (GermanyApps['Year']==2019)|
        (GermanyApps['Year']==2020)|
        (GermanyApps['Year']==2021)|
        (GermanyApps['Year']==2022)]
#2017
GermanyTopApps2017=GermanyApps[GermanyApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
GermanyTopApps2017.set_index('Year', inplace = True)
GermanyTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
GermanyTopApps2018=GermanyApps[GermanyApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
GermanyTopApps2018.set_index('Year', inplace = True)
GermanyTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
GermanyTopApps2019=GermanyApps[GermanyApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
GermanyTopApps2019.set_index('Year', inplace = True)
GermanyTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
GermanyTopApps2020=GermanyApps[GermanyApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
GermanyTopApps2020.set_index('Year', inplace = True)
GermanyTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
GermanyTopApps2021=GermanyApps[GermanyApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
GermanyTopApps2021.set_index('Year', inplace = True)
GermanyTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
GermanyTopApps2022=GermanyApps[GermanyApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
GermanyTopApps2022.set_index('Year', inplace = True)
GermanyTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = GermanyTopApps2017['applied'].tolist()
bars01 = GermanyTopApps2017['Country of origin (ISO)'].tolist()
height02 = GermanyTopApps2018['applied']
bars02=GermanyTopApps2018['Country of origin (ISO)']
height03 = GermanyTopApps2019['applied']
bars03 = GermanyTopApps2019['Country of origin (ISO)']
height04 = GermanyTopApps2020['applied'].tolist()
bars04 = GermanyTopApps2020['Country of origin (ISO)'].tolist()
height05 = GermanyTopApps2021['applied']
bars05=GermanyTopApps2021['Country of origin (ISO)']
height06 = GermanyTopApps2022['applied']
bars06 = GermanyTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Germany - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('GermanyTop3.png')
#%%############################################################################
## France
###############################################################################
#%% A. France Apps & Deps
#%%Create data for plotting
UNFrance=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'FRA']
UNFrance.reset_index(inplace = True)
UNFrance.drop(['index'], axis = 1, inplace = True)
UNFrance.columns
# Create df for bar plot
UNFranceBar = UNFrance[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNFrance['applied']
decisions = UNFrance['Total decisions']
Year = UNFrance['Year'].values
formattedapplications = UNFrance['Formatted Applications']
formatteddecisions = UNFrance['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNFrance['Granted']
grantedformatted = UNFrance['Granted%']
hp = UNFrance['Complementary protection']
hpformatted = UNFrance['HP%']
refused = UNFrance['Refused']
refusedformatted = UNFrance['Refused%']
closed = UNFrance['Closed']
closedformatted = UNFrance['Closed%']
#%%Now bar plot of decision volumes by type
UNFranceBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,40), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(0,20), size = 20, ha = 'center', color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#HP annotations
plt.annotate(hpformatted[0], (pos[0], hp[0]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = 'w')
plt.annotate(hpformatted[1], (pos[1], hp[1]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = 'w')
plt.annotate(hpformatted[2], (pos[2], hp[2]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = 'w')
plt.annotate(hpformatted[3], (pos[3], hp[3]), textcoords = 'offset points', xytext=(0,40), ha = 'center', size = 20, color = 'w')
plt.annotate(hpformatted[4], (pos[4], hp[4]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = 'w')
plt.annotate(hpformatted[5], (pos[5], hp[5]), textcoords = 'offset points', xytext=(0,100), ha = 'center', size = 20, color = 'w')
#Closed annotations
#plt.annotate(closedformatted[0], (pos[0], decisions[0]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[1], (pos[1], decisions[1]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[2], (pos[2], decisions[2]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[3], (pos[3], decisions[3]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[4], (pos[4], decisions[4]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[5], (pos[5], decisions[5]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('France - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper seft', frameon = True, fontsize = 14)
plt.savefig('FranceAppsDecs.png')
#%% France Applicants
#%% Create data for plotting
FranceApps = UNapps[UNapps['Country of asylum (ISO)']=='FRA']
FranceApps2017To2022 = FranceApps[(FranceApps['Year']== 2017)|
        (FranceApps['Year']==2018)|
        (FranceApps['Year']==2019)|
        (FranceApps['Year']==2020)|
        (FranceApps['Year']==2021)|
        (FranceApps['Year']==2022)]
#2017
FranceTopApps2017=FranceApps[FranceApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
FranceTopApps2017.set_index('Year', inplace = True)
FranceTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
FranceTopApps2018=FranceApps[FranceApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
FranceTopApps2018.set_index('Year', inplace = True)
FranceTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
FranceTopApps2019=FranceApps[FranceApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
FranceTopApps2019.set_index('Year', inplace = True)
FranceTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
FranceTopApps2020=FranceApps[FranceApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
FranceTopApps2020.set_index('Year', inplace = True)
FranceTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
FranceTopApps2021=FranceApps[FranceApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
FranceTopApps2021.set_index('Year', inplace = True)
FranceTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
FranceTopApps2022=FranceApps[FranceApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
FranceTopApps2022.set_index('Year', inplace = True)
FranceTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = FranceTopApps2017['applied'].tolist()
bars01 = FranceTopApps2017['Country of origin (ISO)'].tolist()
height02 = FranceTopApps2018['applied']
bars02=FranceTopApps2018['Country of origin (ISO)']
height03 = FranceTopApps2019['applied']
bars03 = FranceTopApps2019['Country of origin (ISO)']
height04 = FranceTopApps2020['applied'].tolist()
bars04 = FranceTopApps2020['Country of origin (ISO)'].tolist()
height05 = FranceTopApps2021['applied']
bars05=FranceTopApps2021['Country of origin (ISO)']
height06 = FranceTopApps2022['applied']
bars06 = FranceTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('France - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('FranceTop3.png')
#%%############################################################################
## Spain
###############################################################################
#%% A. Spain Apps & Deps
#%%Create data for plotting
UNSpain=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'ESP']
UNSpain.reset_index(inplace = True)
UNSpain.drop(['index'], axis = 1, inplace = True)
UNSpain.columns
# Create df for bar plot
UNSpainBar = UNSpain[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNSpain['applied']
decisions = UNSpain['Total decisions']
Year = UNSpain['Year'].values
formattedapplications = UNSpain['Formatted Applications']
formatteddecisions = UNSpain['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNSpain['Granted']
grantedformatted = UNSpain['Granted%']
hp = UNSpain['Complementary protection']
hpformatted = UNSpain['HP%']
refused = UNSpain['Refused']
refusedformatted = UNSpain['Refused%']
closed = UNSpain['Closed']
closedformatted = UNSpain['Closed%']
#%%Now bar plot of decision volumes by type
UNSpainBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(0,20), size = 20, ha = 'center', color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (0,5), ha = 'center', size = 20, color = '#000000')
#Granted annotations
#plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#HP annotations
#plt.annotate(hpformatted[0], (pos[0], hp[0]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[1], (pos[1], hp[1]), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 20, color = 'w')
plt.annotate(hpformatted[2], (pos[2], hp[2]), textcoords = 'offset points', xytext=(0,-10), ha = 'center', size = 20, color = 'w')
plt.annotate(hpformatted[3], (pos[3], hp[3]), textcoords = 'offset points', xytext=(0,0), ha = 'center', size = 20, color = 'w')
plt.annotate(hpformatted[4], (pos[4], hp[4]), textcoords = 'offset points', xytext=(0,0), ha = 'center', size = 20, color = 'w')
plt.annotate(hpformatted[5], (pos[5], hp[5]), textcoords = 'offset points', xytext=(0,0), ha = 'center', size = 20, color = 'w')
#Closed annotations
#plt.annotate(closedformatted[0], (pos[0], decisions[0]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[1], (pos[1], decisions[1]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[2], (pos[2], decisions[2]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[3], (pos[3], decisions[3]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[4], (pos[4], decisions[4]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[5], (pos[5], decisions[5]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Spain - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper seft', frameon = True, fontsize = 14)
plt.savefig('SpainAppsDecs.png')
#%% Spain Applicants
#%% Create data for plotting
SpainApps = UNapps[UNapps['Country of asylum (ISO)']=='ESP']
SpainApps2017To2022 = SpainApps[(SpainApps['Year']== 2017)|
        (SpainApps['Year']==2018)|
        (SpainApps['Year']==2019)|
        (SpainApps['Year']==2020)|
        (SpainApps['Year']==2021)|
        (SpainApps['Year']==2022)]
#2017
SpainTopApps2017=SpainApps[SpainApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
SpainTopApps2017.set_index('Year', inplace = True)
SpainTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
SpainTopApps2018=SpainApps[SpainApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
SpainTopApps2018.set_index('Year', inplace = True)
SpainTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
SpainTopApps2019=SpainApps[SpainApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
SpainTopApps2019.set_index('Year', inplace = True)
SpainTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
SpainTopApps2020=SpainApps[SpainApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
SpainTopApps2020.set_index('Year', inplace = True)
SpainTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
SpainTopApps2021=SpainApps[SpainApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
SpainTopApps2021.set_index('Year', inplace = True)
SpainTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
SpainTopApps2022=SpainApps[SpainApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
SpainTopApps2022.set_index('Year', inplace = True)
SpainTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = SpainTopApps2017['applied'].tolist()
bars01 = SpainTopApps2017['Country of origin (ISO)'].tolist()
height02 = SpainTopApps2018['applied']
bars02=SpainTopApps2018['Country of origin (ISO)']
height03 = SpainTopApps2019['applied']
bars03 = SpainTopApps2019['Country of origin (ISO)']
height04 = SpainTopApps2020['applied'].tolist()
bars04 = SpainTopApps2020['Country of origin (ISO)'].tolist()
height05 = SpainTopApps2021['applied']
bars05=SpainTopApps2021['Country of origin (ISO)']
height06 = SpainTopApps2022['applied']
bars06 = SpainTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Spain - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('SpainTop3.png')
#%%############################################################################
## Peru
###############################################################################
#%% A. Peru Apps & Deps
#%%Create data for plotting
UNPeru=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'PER']
UNPeru.reset_index(inplace = True)
UNPeru.drop(['index'], axis = 1, inplace = True)
UNPeru.columns
# Create df for bar plot
UNPeruBar = UNPeru[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNPeru['applied']
decisions = UNPeru['Total decisions']
Year = UNPeru['Year'].values
formattedapplications = UNPeru['Formatted Applications']
formatteddecisions = UNPeru['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNPeru['Granted']
grantedformatted = UNPeru['Granted%']
hp = UNPeru['Complementary protection']
hpformatted = UNPeru['HP%']
refused = UNPeru['Refused']
refusedformatted = UNPeru['Refused%']
closed = UNPeru['Closed']
closedformatted = UNPeru['Closed%']
#%%Now bar plot of decision volumes by type
UNPeruBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(0,20), size = 20, ha = 'center', color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (0,5), ha = 'center', size = 20, color = '#000000')
#Granted annotations
#plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#HP annotations
#plt.annotate(hpformatted[0], (pos[0], hp[0]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[1], (pos[1], hp[1]), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[2], (pos[2], hp[2]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[3], (pos[3], hp[3]), textcoords = 'offset points', xytext=(0,0), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[4], (pos[4], hp[4]), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[5], (pos[5], hp[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#Closed annotations
#plt.annotate(closedformatted[0], (pos[0], decisions[0]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[1], (pos[1], decisions[1]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[2], (pos[2], decisions[2]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[3], (pos[3], decisions[3]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[4], (pos[4], decisions[4]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[5], (pos[5], decisions[5]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Peru - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper seft', frameon = True, fontsize = 14)
plt.savefig('PeruAppsDecs.png')
#%% Peru Applicants
#%% Create data for plotting
PeruApps = UNapps[UNapps['Country of asylum (ISO)']=='PER']
PeruApps2017To2022 = PeruApps[(PeruApps['Year']== 2017)|
        (PeruApps['Year']==2018)|
        (PeruApps['Year']==2019)|
        (PeruApps['Year']==2020)|
        (PeruApps['Year']==2021)|
        (PeruApps['Year']==2022)]
#2017
PeruTopApps2017=PeruApps[PeruApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
PeruTopApps2017.set_index('Year', inplace = True)
PeruTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
PeruTopApps2018=PeruApps[PeruApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
PeruTopApps2018.set_index('Year', inplace = True)
PeruTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
PeruTopApps2019=PeruApps[PeruApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
PeruTopApps2019.set_index('Year', inplace = True)
PeruTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
PeruTopApps2020=PeruApps[PeruApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
PeruTopApps2020.set_index('Year', inplace = True)
PeruTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
PeruTopApps2021=PeruApps[PeruApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
PeruTopApps2021.set_index('Year', inplace = True)
PeruTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
PeruTopApps2022=PeruApps[PeruApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
PeruTopApps2022.set_index('Year', inplace = True)
PeruTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = PeruTopApps2017['applied'].tolist()
bars01 = PeruTopApps2017['Country of origin (ISO)'].tolist()
height02 = PeruTopApps2018['applied']
bars02=PeruTopApps2018['Country of origin (ISO)']
height03 = PeruTopApps2019['applied']
bars03 = PeruTopApps2019['Country of origin (ISO)']
height04 = PeruTopApps2020['applied'].tolist()
bars04 = PeruTopApps2020['Country of origin (ISO)'].tolist()
height05 = PeruTopApps2021['applied']
bars05=PeruTopApps2021['Country of origin (ISO)']
height06 = PeruTopApps2022['applied']
bars06 = PeruTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Peru - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('PeruTop3.png')
#%%############################################################################
## Mexico
###############################################################################
#%% A. Mexico Apps & Deps
#%%Create data for plotting
UNMexico=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'MEX']
UNMexico.reset_index(inplace = True)
UNMexico.drop(['index'], axis = 1, inplace = True)
UNMexico.columns
# Create df for bar plot
UNMexicoBar = UNMexico[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNMexico['applied']
decisions = UNMexico['Total decisions']
Year = UNMexico['Year'].values
formattedapplications = UNMexico['Formatted Applications']
formatteddecisions = UNMexico['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNMexico['Granted']
grantedformatted = UNMexico['Granted%']
hp = UNMexico['Complementary protection']
hpformatted = UNMexico['HP%']
refused = UNMexico['Refused']
refusedformatted = UNMexico['Refused%']
closed = UNMexico['Closed']
closedformatted = UNMexico['Closed%']
#%%Now bar plot of decision volumes by type
UNMexicoBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (0,5), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#HP annotations
#plt.annotate(hpformatted[0], (pos[0], hp[0]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[1], (pos[1], hp[1]), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[2], (pos[2], hp[2]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[3], (pos[3], hp[3]), textcoords = 'offset points', xytext=(0,0), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[4], (pos[4], hp[4]), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[5], (pos[5], hp[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#Closed annotations
#plt.annotate(closedformatted[0], (pos[0], decisions[0]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[1], (pos[1], decisions[1]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[2], (pos[2], decisions[2]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[3], (pos[3], decisions[3]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[4], (pos[4], decisions[4]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[5], (pos[5], decisions[5]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Mexico - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper seft', frameon = True, fontsize = 14)
plt.savefig('MexicoAppsDecs.png')
#%% Mexico Applicants
#%% Create data for plotting
MexicoApps = UNapps[UNapps['Country of asylum (ISO)']=='MEX']
MexicoApps2017To2022 = MexicoApps[(MexicoApps['Year']== 2017)|
        (MexicoApps['Year']==2018)|
        (MexicoApps['Year']==2019)|
        (MexicoApps['Year']==2020)|
        (MexicoApps['Year']==2021)|
        (MexicoApps['Year']==2022)]
#2017
MexicoTopApps2017=MexicoApps[MexicoApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
MexicoTopApps2017.set_index('Year', inplace = True)
MexicoTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
MexicoTopApps2018=MexicoApps[MexicoApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
MexicoTopApps2018.set_index('Year', inplace = True)
MexicoTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
MexicoTopApps2019=MexicoApps[MexicoApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
MexicoTopApps2019.set_index('Year', inplace = True)
MexicoTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
MexicoTopApps2020=MexicoApps[MexicoApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
MexicoTopApps2020.set_index('Year', inplace = True)
MexicoTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
MexicoTopApps2021=MexicoApps[MexicoApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
MexicoTopApps2021.set_index('Year', inplace = True)
MexicoTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
MexicoTopApps2022=MexicoApps[MexicoApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
MexicoTopApps2022.set_index('Year', inplace = True)
MexicoTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = MexicoTopApps2017['applied'].tolist()
bars01 = MexicoTopApps2017['Country of origin (ISO)'].tolist()
height02 = MexicoTopApps2018['applied']
bars02=MexicoTopApps2018['Country of origin (ISO)']
height03 = MexicoTopApps2019['applied']
bars03 = MexicoTopApps2019['Country of origin (ISO)']
height04 = MexicoTopApps2020['applied'].tolist()
bars04 = MexicoTopApps2020['Country of origin (ISO)'].tolist()
height05 = MexicoTopApps2021['applied']
bars05=MexicoTopApps2021['Country of origin (ISO)']
height06 = MexicoTopApps2022['applied']
bars06 = MexicoTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Mexico - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('MexicoTop3.png')
#%%############################################################################
## Sweden
###############################################################################
#%% A. Sweden Apps & Deps
#%%Create data for plotting
UNSweden=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'SWE']
UNSweden.reset_index(inplace = True)
UNSweden.drop(['index'], axis = 1, inplace = True)
UNSweden.columns
# Create df for bar plot
UNSwedenBar = UNSweden[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNSweden['applied']
decisions = UNSweden['Total decisions']
Year = UNSweden['Year'].values
formattedapplications = UNSweden['Formatted Applications']
formatteddecisions = UNSweden['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNSweden['Granted']
grantedformatted = UNSweden['Granted%']
hp = UNSweden['Complementary protection']
hpformatted = UNSweden['HP%']
refused = UNSweden['Refused']
refusedformatted = UNSweden['Refused%']
closed = UNSweden['Closed']
closedformatted = UNSweden['Closed%']
#%%Now bar plot of decision volumes by type
UNSwedenBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (100,5), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#HP annotations
plt.annotate(hpformatted[0], (pos[0], hp[0]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[1], (pos[1], hp[1]), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[2], (pos[2], hp[2]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[3], (pos[3], hp[3]), textcoords = 'offset points', xytext=(0,0), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[4], (pos[4], hp[4]), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 20, color = 'w')
#plt.annotate(hpformatted[5], (pos[5], hp[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#Closed annotations
#plt.annotate(closedformatted[0], (pos[0], decisions[0]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[1], (pos[1], decisions[1]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[2], (pos[2], decisions[2]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[3], (pos[3], decisions[3]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[4], (pos[4], decisions[4]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[0], (pos[0], decisions[0]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Sweden - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper right', frameon = True, fontsize = 14)
plt.savefig('SwedenAppsDecs.png')
#%% Sweden Applicants
#%% Create data for plotting
SwedenApps = UNapps[UNapps['Country of asylum (ISO)']=='SWE']
SwedenApps2017To2022 = SwedenApps[(SwedenApps['Year']== 2017)|
        (SwedenApps['Year']==2018)|
        (SwedenApps['Year']==2019)|
        (SwedenApps['Year']==2020)|
        (SwedenApps['Year']==2021)|
        (SwedenApps['Year']==2022)]
#2017
SwedenTopApps2017=SwedenApps[SwedenApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
SwedenTopApps2017.set_index('Year', inplace = True)
SwedenTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
SwedenTopApps2018=SwedenApps[SwedenApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
SwedenTopApps2018.set_index('Year', inplace = True)
SwedenTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
SwedenTopApps2019=SwedenApps[SwedenApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
SwedenTopApps2019.set_index('Year', inplace = True)
SwedenTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
SwedenTopApps2020=SwedenApps[SwedenApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
SwedenTopApps2020.set_index('Year', inplace = True)
SwedenTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
SwedenTopApps2021=SwedenApps[SwedenApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
SwedenTopApps2021.set_index('Year', inplace = True)
SwedenTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
SwedenTopApps2022=SwedenApps[SwedenApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
SwedenTopApps2022.set_index('Year', inplace = True)
SwedenTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = SwedenTopApps2017['applied'].tolist()
bars01 = SwedenTopApps2017['Country of origin (ISO)'].tolist()
height02 = SwedenTopApps2018['applied']
bars02=SwedenTopApps2018['Country of origin (ISO)']
height03 = SwedenTopApps2019['applied']
bars03 = SwedenTopApps2019['Country of origin (ISO)']
height04 = SwedenTopApps2020['applied'].tolist()
bars04 = SwedenTopApps2020['Country of origin (ISO)'].tolist()
height05 = SwedenTopApps2021['applied']
bars05=SwedenTopApps2021['Country of origin (ISO)']
height06 = SwedenTopApps2022['applied']
bars06 = SwedenTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Sweden - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('SwedenTop3.png')
#%%############################################################################
## Italy
###############################################################################
#%% A. Italy Apps & Deps
#%%Create data for plotting
UNItaly=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'ITA']
UNItaly.reset_index(inplace = True)
UNItaly.drop(['index'], axis = 1, inplace = True)
UNItaly.columns
# Create df for bar plot
UNItalyBar = UNItaly[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNItaly['applied']
decisions = UNItaly['Total decisions']
Year = UNItaly['Year'].values
formattedapplications = UNItaly['Formatted Applications']
formatteddecisions = UNItaly['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNItaly['Granted']
grantedformatted = UNItaly['Granted%']
hp = UNItaly['Complementary protection']
hpformatted = UNItaly['HP%']
refused = UNItaly['Refused']
refusedformatted = UNItaly['Refused%']
closed = UNItaly['Closed']
closedformatted = UNItaly['Closed%']
#%%Now bar plot of decision volumes by type
UNItalyBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (5,5), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#HP annotations
plt.annotate(hpformatted[0], (pos[0], hp[0]), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 20, color = 'w')
plt.annotate(hpformatted[1], (pos[1], hp[1]), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 20, color = 'w')
plt.annotate(hpformatted[2], (pos[2], hp[2]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
plt.annotate(hpformatted[3], (pos[3], hp[3]), textcoords = 'offset points', xytext=(0,0), ha = 'center', size = 20, color = 'w')
plt.annotate(hpformatted[4], (pos[4], hp[4]), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 20, color = 'w')
plt.annotate(hpformatted[5], (pos[5], hp[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#Closed annotations
#plt.annotate(closedformatted[0], (pos[0], decisions[0]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[1], (pos[1], decisions[1]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[2], (pos[2], decisions[2]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[3], (pos[3], decisions[3]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[4], (pos[4], decisions[4]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[0], (pos[0], decisions[0]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Italy - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper right', frameon = True, fontsize = 14)
plt.savefig('ItalyAppsDecs.png')
#%% Italy Applicants
#%% Create data for plotting
ItalyApps = UNapps[UNapps['Country of asylum (ISO)']=='ITA']
ItalyApps2017To2022 = ItalyApps[(ItalyApps['Year']== 2017)|
        (ItalyApps['Year']==2018)|
        (ItalyApps['Year']==2019)|
        (ItalyApps['Year']==2020)|
        (ItalyApps['Year']==2021)|
        (ItalyApps['Year']==2022)]
#2017
ItalyTopApps2017=ItalyApps[ItalyApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
ItalyTopApps2017.set_index('Year', inplace = True)
ItalyTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
ItalyTopApps2018=ItalyApps[ItalyApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
ItalyTopApps2018.set_index('Year', inplace = True)
ItalyTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
ItalyTopApps2019=ItalyApps[ItalyApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
ItalyTopApps2019.set_index('Year', inplace = True)
ItalyTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
ItalyTopApps2020=ItalyApps[ItalyApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
ItalyTopApps2020.set_index('Year', inplace = True)
ItalyTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
ItalyTopApps2021=ItalyApps[ItalyApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
ItalyTopApps2021.set_index('Year', inplace = True)
ItalyTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
ItalyTopApps2022=ItalyApps[ItalyApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
ItalyTopApps2022.set_index('Year', inplace = True)
ItalyTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = ItalyTopApps2017['applied'].tolist()
bars01 = ItalyTopApps2017['Country of origin (ISO)'].tolist()
height02 = ItalyTopApps2018['applied']
bars02=ItalyTopApps2018['Country of origin (ISO)']
height03 = ItalyTopApps2019['applied']
bars03 = ItalyTopApps2019['Country of origin (ISO)']
height04 = ItalyTopApps2020['applied'].tolist()
bars04 = ItalyTopApps2020['Country of origin (ISO)'].tolist()
height05 = ItalyTopApps2021['applied']
bars05=ItalyTopApps2021['Country of origin (ISO)']
height06 = ItalyTopApps2022['applied']
bars06 = ItalyTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Italy - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('ItalyTop3.png')
#%%############################################################################
## Greece
###############################################################################
#%% A. Greece Apps & Deps
#%%Create data for plotting
UNGreece=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'GRC']
UNGreece.reset_index(inplace = True)
UNGreece.drop(['index'], axis = 1, inplace = True)
UNGreece.columns
# Create df for bar plot
UNGreeceBar = UNGreece[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNGreece['applied']
decisions = UNGreece['Total decisions']
Year = UNGreece['Year'].values
formattedapplications = UNGreece['Formatted Applications']
formatteddecisions = UNGreece['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNGreece['Granted']
grantedformatted = UNGreece['Granted%']
hp = UNGreece['Complementary protection']
hpformatted = UNGreece['HP%']
refused = UNGreece['Refused']
refusedformatted = UNGreece['Refused%']
closed = UNGreece['Closed']
closedformatted = UNGreece['Closed%']
#%%Now bar plot of decision volumes by type
UNGreeceBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (5,5), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#HP annotations
plt.annotate(hpformatted[3], (pos[3], granted[3]+hp[3]), textcoords = 'offset points', xytext=(0,150), ha = 'center', size = 20, color = 'w')
#Closed annotations
#plt.annotate(closedformatted[0], (pos[0], decisions[0]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[1], (pos[1], decisions[1]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[2], (pos[2], decisions[2]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[3], (pos[3], decisions[3]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[4], (pos[4], decisions[4]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[5], (pos[5], decisions[5]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Greece - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper left', frameon = True, fontsize = 14)
plt.savefig('GreeceAppsDecs.png')
#%% Greece Applicants
#%% Create data for plotting
GreeceApps = UNapps[UNapps['Country of asylum (ISO)']=='GRC']
GreeceApps2017To2022 = GreeceApps[(GreeceApps['Year']== 2017)|
        (GreeceApps['Year']==2018)|
        (GreeceApps['Year']==2019)|
        (GreeceApps['Year']==2020)|
        (GreeceApps['Year']==2021)|
        (GreeceApps['Year']==2022)]
#2017
GreeceTopApps2017=GreeceApps[GreeceApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
GreeceTopApps2017.set_index('Year', inplace = True)
GreeceTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
GreeceTopApps2018=GreeceApps[GreeceApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
GreeceTopApps2018.set_index('Year', inplace = True)
GreeceTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
GreeceTopApps2019=GreeceApps[GreeceApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
GreeceTopApps2019.set_index('Year', inplace = True)
GreeceTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
GreeceTopApps2020=GreeceApps[GreeceApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
GreeceTopApps2020.set_index('Year', inplace = True)
GreeceTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
GreeceTopApps2021=GreeceApps[GreeceApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
GreeceTopApps2021.set_index('Year', inplace = True)
GreeceTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
GreeceTopApps2022=GreeceApps[GreeceApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
GreeceTopApps2022.set_index('Year', inplace = True)
GreeceTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = GreeceTopApps2017['applied'].tolist()
bars01 = GreeceTopApps2017['Country of origin (ISO)'].tolist()
height02 = GreeceTopApps2018['applied']
bars02=GreeceTopApps2018['Country of origin (ISO)']
height03 = GreeceTopApps2019['applied']
bars03 = GreeceTopApps2019['Country of origin (ISO)']
height04 = GreeceTopApps2020['applied'].tolist()
bars04 = GreeceTopApps2020['Country of origin (ISO)'].tolist()
height05 = GreeceTopApps2021['applied']
bars05=GreeceTopApps2021['Country of origin (ISO)']
height06 = GreeceTopApps2022['applied']
bars06 = GreeceTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Greece - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('GreeceTop3.png')
#%%############################################################################
## Canada
###############################################################################
#%% A. Canada Apps & Deps
#%%Create data for plotting
UNCanada=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'CAN']
UNCanada.reset_index(inplace = True)
UNCanada.drop(['index'], axis = 1, inplace = True)
UNCanada.columns
# Create df for bar plot
UNCanadaBar = UNCanada[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNCanada['applied']
decisions = UNCanada['Total decisions']
Year = UNCanada['Year'].values
formattedapplications = UNCanada['Formatted Applications']
formatteddecisions = UNCanada['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNCanada['Granted']
grantedformatted = UNCanada['Granted%']
hp = UNCanada['Complementary protection']
hpformatted = UNCanada['HP%']
refused = UNCanada['Refused']
refusedformatted = UNCanada['Refused%']
closed = UNCanada['Closed']
closedformatted = UNCanada['Closed%']
#%%Now bar plot of decision volumes by type
UNCanadaBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (5,5), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#Closed annotations
#plt.annotate(closedformatted[0], (pos[0], decisions[0]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[1], (pos[1], decisions[1]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[2], (pos[2], decisions[2]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[3], (pos[3], decisions[3]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[4], (pos[4], decisions[4]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[5], (pos[5], decisions[5]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Canada - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper left', frameon = True, fontsize = 14)
plt.savefig('CanadaAppsDecs.png')
#%% Canada Applicants
#%% Create data for plotting
CanadaApps = UNapps[UNapps['Country of asylum (ISO)']=='CAN']
CanadaApps2017To2022 = CanadaApps[(CanadaApps['Year']== 2017)|
        (CanadaApps['Year']==2018)|
        (CanadaApps['Year']==2019)|
        (CanadaApps['Year']==2020)|
        (CanadaApps['Year']==2021)|
        (CanadaApps['Year']==2022)]
#2017
CanadaTopApps2017=CanadaApps[CanadaApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
CanadaTopApps2017.set_index('Year', inplace = True)
CanadaTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
CanadaTopApps2018=CanadaApps[CanadaApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
CanadaTopApps2018.set_index('Year', inplace = True)
CanadaTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
CanadaTopApps2019=CanadaApps[CanadaApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
CanadaTopApps2019.set_index('Year', inplace = True)
CanadaTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
CanadaTopApps2020=CanadaApps[CanadaApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
CanadaTopApps2020.set_index('Year', inplace = True)
CanadaTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
CanadaTopApps2021=CanadaApps[CanadaApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
CanadaTopApps2021.set_index('Year', inplace = True)
CanadaTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
CanadaTopApps2022=CanadaApps[CanadaApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
CanadaTopApps2022.set_index('Year', inplace = True)
CanadaTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = CanadaTopApps2017['applied'].tolist()
bars01 = CanadaTopApps2017['Country of origin (ISO)'].tolist()
height02 = CanadaTopApps2018['applied']
bars02=CanadaTopApps2018['Country of origin (ISO)']
height03 = CanadaTopApps2019['applied']
bars03 = CanadaTopApps2019['Country of origin (ISO)']
height04 = CanadaTopApps2020['applied'].tolist()
bars04 = CanadaTopApps2020['Country of origin (ISO)'].tolist()
height05 = CanadaTopApps2021['applied']
bars05=CanadaTopApps2021['Country of origin (ISO)']
height06 = CanadaTopApps2022['applied']
bars06 = CanadaTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('The Canada - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('CanadaTop3.png')
#%%############################################################################
## UK
###############################################################################
#%% A. UK Apps & Deps
#%%Create data for plotting
UNUK=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'GBR']
UNUK.reset_index(inplace = True)
UNUK.drop(['index'], axis = 1, inplace = True)
UNUK.columns
# Create df for bar plot
UNUKBar = UNUK[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNUK['applied']
decisions = UNUK['Total decisions']
Year = UNUK['Year'].values
formattedapplications = UNUK['Formatted Applications']
formatteddecisions = UNUK['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNUK['Granted']
grantedformatted = UNUK['Granted%']
hp = UNUK['Complementary protection']
hpformatted = UNUK['HP%']
refused = UNUK['Refused']
refusedformatted = UNUK['Refused%']
closed = UNUK['Closed']
closedformatted = UNUK['Closed%']
#%% UK FVP

fig, ax = plt.subplots(figsize=(15, 10))
pos = np.arange(len(Year))
plt.bar(Year, UNUK['FVP'].values, width = .8, color = '#00747A', label = 'Face-Value Productivity')
UNUK['FVP%']=UNUK['FVP'].apply(lambda x: '{:,.0%}'.format(x))
plt.xticks(Year, size = 20)
plt.yticks([])
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
for x,y in zip(Year,UNUK['FVP']):

    label = "{:,.0%}".format(y)

    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center',
                 size =20,
                 bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#00747A", lw=1)) # horizontal alignment can be left, right or center
plt.axhline(y=1, color='#88234590', linestyle='--')
plt.text(2022, .95,'100% FVP', fontsize=16, color = '#88234590')
         # Add title and labels
plt.xlabel('Year',color = '#000000', fontsize = 20, labelpad = 20)
plt.ylabel('FVP', color = '#000000',fontsize = 20, labelpad = 20)
plt.title("Face-Value Productivity: UK 2017 to 2022", color = '#000000', fontsize = 24, pad = 30)
plt.legend(loc=1, bbox_to_anchor=(0.95,0.6), fontsize = 12)
plt.savefig("UKFVP2017to2022.png")
#%% 2023 FVP
Top202023Apps['FVP'][7]
fig, ax = plt.subplots(figsize=(5, 15))
pos = np.arange(len(Year))
plt.bar(Top202023Apps['Country of asylum (ISO)'][7], Top202023Apps['FVP'][7], width = 0.05, color = '#000000', label = 'Face-Value Productivity')
plt.show()
Top202023Apps['FVP%']=Top202023Apps['FVP'].apply(lambda x: '{:,.0%}'.format(x))
plt.xticks(size = 20)
plt.yticks([])
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
plt.annotate(Top202023Apps['FVP%'][7],(0,Top202023Apps['FVP'][7]), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center',
                 size =20,
                 bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#000000", lw=1)) # horizontal alignment can be left, right or center
plt.axhline(y=1, color='#88234590', linestyle='--')
plt.text(0, .95,'100% FVP', fontsize=16, color = '#88234590')
         # Add title and labels
plt.xlabel('Country of Asylum (ISO)',color = '#000000', fontsize = 20, labelpad = 20)
plt.ylabel('FVP', color = '#000000',fontsize = 20, labelpad = 20)
plt.title("Face-Value Productivity:\nUK Jan - Jun 2023", color = '#000000', fontsize = 24, pad = 30)
plt.legend(loc=1, bbox_to_anchor=(0.85,0.6), fontsize = 12)
plt.savefig("UKFVP2023.png")
#%% UK decision surplus/shortfall
#%% Decision Shortfalls and Surpluses
uk20017 = decisions[0]-applications[0]
uk20018 = decisions[1]-applications[1]
uk20019 = decisions[2]-applications[2]
uk20020 = decisions[3]-applications[3]
uk20021 = decisions[4]-applications[4]
uk20022 = decisions[5]-applications[5]
uklist = [uk20017, uk20018, uk20019, uk20020, uk20021, uk20022]
ukwaterfall = pd.DataFrame(uklist)
ukwaterfall.to_csv('waterfall.csv')
#2023 data
UK2023 = Top202023Apps[Top202023Apps['Country of asylum (ISO)']=='GBR']
UK2023Shortfall = UK2023['Total decisions']-UK2023['applied']
#%%Now bar plot of decision volumes by type
UNUKBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,50), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (5,5), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#Closed annotations
#plt.annotate(closedformatted[0], (pos[0], decisions[0]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[1], (pos[1], decisions[1]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[2], (pos[2], decisions[2]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[3], (pos[3], decisions[3]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[4], (pos[4], decisions[4]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[5], (pos[5], decisions[5]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('The UK - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper left', frameon = True, fontsize = 14)
plt.savefig('UKAppsDecs.png')
#%% UK Applicants
#%% Create data for plotting
UKApps = UNapps[UNapps['Country of asylum (ISO)']=='GBR']
UKApps2017To2022 = UKApps[(UKApps['Year']== 2017)|
        (UKApps['Year']==2018)|
        (UKApps['Year']==2019)|
        (UKApps['Year']==2020)|
        (UKApps['Year']==2021)|
        (UKApps['Year']==2022)]
#2017
UKTopApps2017=UKApps[UKApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
UKTopApps2017.set_index('Year', inplace = True)
UKTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
UKTopApps2018=UKApps[UKApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
UKTopApps2018.set_index('Year', inplace = True)
UKTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
UKTopApps2019=UKApps[UKApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
UKTopApps2019.set_index('Year', inplace = True)
UKTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
UKTopApps2020=UKApps[UKApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
UKTopApps2020.set_index('Year', inplace = True)
UKTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
UKTopApps2021=UKApps[UKApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
UKTopApps2021.set_index('Year', inplace = True)
UKTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
UKTopApps2022=UKApps[UKApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
UKTopApps2022.set_index('Year', inplace = True)
UKTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = UKTopApps2017['applied'].tolist()
bars01 = UKTopApps2017['Country of origin (ISO)'].tolist()
height02 = UKTopApps2018['applied']
bars02=UKTopApps2018['Country of origin (ISO)']
height03 = UKTopApps2019['applied']
bars03 = UKTopApps2019['Country of origin (ISO)']
height04 = UKTopApps2020['applied'].tolist()
bars04 = UKTopApps2020['Country of origin (ISO)'].tolist()
height05 = UKTopApps2021['applied']
bars05=UKTopApps2021['Country of origin (ISO)']
height06 = UKTopApps2022['applied']
bars06 = UKTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('The UK - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('UKTop3.png')
#%%############################################################################
## CostaRica
###############################################################################
#%% A. CostaRica Apps & Deps
#%%Create data for plotting
UNCostaRica=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'CRI']
UNCostaRica.reset_index(inplace = True)
UNCostaRica.drop(['index'], axis = 1, inplace = True)
UNCostaRica.columns
# Create df for bar plot
UNCostaRicaBar = UNCostaRica[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNCostaRica['applied']
decisions = UNCostaRica['Total decisions']
Year = UNCostaRica['Year']
formattedapplications = UNCostaRica['Formatted Applications']
formatteddecisions = UNCostaRica['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNCostaRica['Granted']
grantedformatted = UNCostaRica['Granted%']
hp = UNCostaRica['Complementary protection']
hpformatted = UNCostaRica['HP%']
refused = UNCostaRica['Refused']
refusedformatted = UNCostaRica['Refused%']
closed = UNCostaRica['Closed']
closedformatted = UNCostaRica['Closed%']
#%%Now bar plot of decision volumes by type
UNCostaRicaBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (50,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (5,5), ha = 'center', size = 20, color = '#000000')
#Granted annotations
#plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#Closed annotations
#plt.annotate(closedformatted[0], (pos[0], decisions[0]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[1], (pos[1], decisions[1]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[2], (pos[2], decisions[2]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[3], (pos[3], decisions[3]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(closedformatted[4], (pos[4], decisions[4]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[5], (pos[5], decisions[5]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Costa Rica - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper left', frameon = True, fontsize = 14)
plt.savefig('CostaRicaAppsDecs.png')
#%% CostaRica Applicants
#%% Create data for plotting
CostaRicaApps = UNapps[UNapps['Country of asylum (ISO)']=='CRI']
CostaRicaApps2017To2022 = CostaRicaApps[(CostaRicaApps['Year']== 2017)|
        (CostaRicaApps['Year']==2018)|
        (CostaRicaApps['Year']==2019)|
        (CostaRicaApps['Year']==2020)|
        (CostaRicaApps['Year']==2021)|
        (CostaRicaApps['Year']==2022)]
#2017
CostaRicaTopApps2017=CostaRicaApps[CostaRicaApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
CostaRicaTopApps2017.set_index('Year', inplace = True)
CostaRicaTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
CostaRicaTopApps2018=CostaRicaApps[CostaRicaApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
CostaRicaTopApps2018.set_index('Year', inplace = True)
CostaRicaTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
CostaRicaTopApps2019=CostaRicaApps[CostaRicaApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
CostaRicaTopApps2019.set_index('Year', inplace = True)
CostaRicaTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
CostaRicaTopApps2020=CostaRicaApps[CostaRicaApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
CostaRicaTopApps2020.set_index('Year', inplace = True)
CostaRicaTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
CostaRicaTopApps2021=CostaRicaApps[CostaRicaApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
CostaRicaTopApps2021.set_index('Year', inplace = True)
CostaRicaTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
CostaRicaTopApps2022=CostaRicaApps[CostaRicaApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
CostaRicaTopApps2022.set_index('Year', inplace = True)
CostaRicaTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = CostaRicaTopApps2017['applied'].tolist()
bars01 = CostaRicaTopApps2017['Country of origin (ISO)'].tolist()
height02 = CostaRicaTopApps2018['applied']
bars02=CostaRicaTopApps2018['Country of origin (ISO)']
height03 = CostaRicaTopApps2019['applied']
bars03 = CostaRicaTopApps2019['Country of origin (ISO)']
height04 = CostaRicaTopApps2020['applied'].tolist()
bars04 = CostaRicaTopApps2020['Country of origin (ISO)'].tolist()
height05 = CostaRicaTopApps2021['applied']
bars05=CostaRicaTopApps2021['Country of origin (ISO)']
height06 = CostaRicaTopApps2022['applied']
bars06 = CostaRicaTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Costa Rica - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('CostaRicaTop3.png')
#%%############################################################################
## Turkey
###############################################################################
#%% A. Turkey Apps & Deps
#%%Create data for plotting
UNTurkey=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'TUR']
UNTurkey.reset_index(inplace = True)
UNTurkey.drop(['index'], axis = 1, inplace = True)
UNTurkey.columns
# Create df for bar plot
UNTurkeyBar = UNTurkey[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNTurkey['applied']
decisions = UNTurkey['Total decisions']
Year = UNTurkey['Year']
formattedapplications = UNTurkey['Formatted Applications']
formatteddecisions = UNTurkey['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNTurkey['Granted']
grantedformatted = UNTurkey['Granted%']
hp = UNTurkey['Complementary protection']
hpformatted = UNTurkey['HP%']
refused = UNTurkey['Refused']
refusedformatted = UNTurkey['Refused%']
closed = UNTurkey['Closed']
closedformatted = UNTurkey['Closed%']
#%%Now bar plot of decision volumes by type
UNTurkeyBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (5,5), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#Closed annotations
plt.annotate(closedformatted[0], (pos[0], decisions[0]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[1], (pos[1], decisions[1]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[2], (pos[2], decisions[2]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[3], (pos[3], decisions[3]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[4], (pos[4], decisions[4]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(closedformatted[5], (pos[5], decisions[5]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Turkey - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper right', frameon = True, fontsize = 14)
plt.savefig('TurkeyAppsDecs.png')
#%% Turkey Applicants
#%% Create data for plotting
TurkeyApps = UNapps[UNapps['Country of asylum (ISO)']=='TUR']
TurkeyApps2017To2022 = TurkeyApps[(TurkeyApps['Year']== 2017)|
        (TurkeyApps['Year']==2018)|
        (TurkeyApps['Year']==2019)|
        (TurkeyApps['Year']==2020)|
        (TurkeyApps['Year']==2021)|
        (TurkeyApps['Year']==2022)]
#2017
TurkeyTopApps2017=TurkeyApps[TurkeyApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
TurkeyTopApps2017.set_index('Year', inplace = True)
TurkeyTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
TurkeyTopApps2018=TurkeyApps[TurkeyApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
TurkeyTopApps2018.set_index('Year', inplace = True)
TurkeyTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
TurkeyTopApps2019=TurkeyApps[TurkeyApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
TurkeyTopApps2019.set_index('Year', inplace = True)
TurkeyTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
TurkeyTopApps2020=TurkeyApps[TurkeyApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
TurkeyTopApps2020.set_index('Year', inplace = True)
TurkeyTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
TurkeyTopApps2021=TurkeyApps[TurkeyApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
TurkeyTopApps2021.set_index('Year', inplace = True)
TurkeyTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
TurkeyTopApps2022=TurkeyApps[TurkeyApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
TurkeyTopApps2022.set_index('Year', inplace = True)
TurkeyTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = TurkeyTopApps2017['applied'].tolist()
bars01 = TurkeyTopApps2017['Country of origin (ISO)'].tolist()
height02 = TurkeyTopApps2018['applied']
bars02=TurkeyTopApps2018['Country of origin (ISO)']
height03 = TurkeyTopApps2019['applied']
bars03 = TurkeyTopApps2019['Country of origin (ISO)']
height04 = TurkeyTopApps2020['applied'].tolist()
bars04 = TurkeyTopApps2020['Country of origin (ISO)'].tolist()
height05 = TurkeyTopApps2021['applied']
bars05=TurkeyTopApps2021['Country of origin (ISO)']
height06 = TurkeyTopApps2022['applied']
bars06 = TurkeyTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Turkey - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('TurkeyTop3.png')
#%%############################################################################
## Brazil
###############################################################################
#%% A. Brazil Apps & Deps
#%%Create data for plotting
UNBrazil=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'BRA']
UNBrazil.reset_index(inplace = True)
UNBrazil.drop(['index'], axis = 1, inplace = True)
UNBrazil.columns
# Create df for bar plot
UNBrazilBar = UNBrazil[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNBrazil['applied']
decisions = UNBrazil['Total decisions']
Year = UNBrazil['Year']
formattedapplications = UNBrazil['Formatted Applications']
formatteddecisions = UNBrazil['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNBrazil['Granted']
grantedformatted = UNBrazil['Granted%']
hp = UNBrazil['Complementary protection']
hpformatted = UNBrazil['HP%']
refused = UNBrazil['Refused']
refusedformatted = UNBrazil['Refused%']
closed = UNBrazil['Closed']
closedformatted = UNBrazil['Closed%']
#%%Now bar plot of decision volumes by type
UNBrazilBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (5,5), ha = 'center', size = 20, color = '#000000')
#Granted annotations
#plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#Closed annotations
plt.annotate(closedformatted[5], (pos[5], decisions[5]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Brazil - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper right', frameon = True, fontsize = 14)
plt.savefig('BrazilAppsDecs.png')
#%% Brazil Applicants
#%% Create data for plotting
BrazilApps = UNapps[UNapps['Country of asylum (ISO)']=='BRA']
BrazilApps2017To2022 = BrazilApps[(BrazilApps['Year']== 2017)|
        (BrazilApps['Year']==2018)|
        (BrazilApps['Year']==2019)|
        (BrazilApps['Year']==2020)|
        (BrazilApps['Year']==2021)|
        (BrazilApps['Year']==2022)]
#2017
BrazilTopApps2017=BrazilApps[BrazilApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
BrazilTopApps2017.set_index('Year', inplace = True)
BrazilTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
BrazilTopApps2018=BrazilApps[BrazilApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
BrazilTopApps2018.set_index('Year', inplace = True)
BrazilTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
BrazilTopApps2019=BrazilApps[BrazilApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
BrazilTopApps2019.set_index('Year', inplace = True)
BrazilTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
BrazilTopApps2020=BrazilApps[BrazilApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
BrazilTopApps2020.set_index('Year', inplace = True)
BrazilTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
BrazilTopApps2021=BrazilApps[BrazilApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
BrazilTopApps2021.set_index('Year', inplace = True)
BrazilTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
BrazilTopApps2022=BrazilApps[BrazilApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
BrazilTopApps2022.set_index('Year', inplace = True)
BrazilTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = BrazilTopApps2017['applied'].tolist()
bars01 = BrazilTopApps2017['Country of origin (ISO)'].tolist()
height02 = BrazilTopApps2018['applied']
bars02=BrazilTopApps2018['Country of origin (ISO)']
height03 = BrazilTopApps2019['applied']
bars03 = BrazilTopApps2019['Country of origin (ISO)']
height04 = BrazilTopApps2020['applied'].tolist()
bars04 = BrazilTopApps2020['Country of origin (ISO)'].tolist()
height05 = BrazilTopApps2021['applied']
bars05=BrazilTopApps2021['Country of origin (ISO)']
height06 = BrazilTopApps2022['applied']
bars06 = BrazilTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Brazil - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('BrazilTop3.png')
#%%############################################################################
## Uganda
###############################################################################
#%% A. Uganda Apps & Deps
#%%Create data for plotting
UNUganda=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'UGA']
UNUganda.reset_index(inplace = True)
UNUganda.drop(['index'], axis = 1, inplace = True)
UNUganda.columns
# Create df for bar plot
UNUgandaBar = UNUganda[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNUganda['applied']
decisions = UNUganda['Total decisions']
Year = UNUganda['Year']
formattedapplications = UNUganda['Formatted Applications']
formatteddecisions = UNUganda['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNUganda['Granted']
grantedformatted = UNUganda['Granted%']
hp = UNUganda['Complementary protection']
hpformatted = UNUganda['HP%']
refused = UNUganda['Refused']
refusedformatted = UNUganda['Refused%']
closed = UNUganda['Closed']
closedformatted = UNUganda['Closed%']
#%%Now bar plot of decision volumes by type
UNUgandaBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (50,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (5,5), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#Closed annotations

plt.annotate(closedformatted[1], (pos[1], decisions[1]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Uganda - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper right', frameon = True, fontsize = 14)
plt.savefig('UgandaAppsDecs.png')
#%% Uganda Applicants
#%% Create data for plotting
UgandaApps = UNapps[UNapps['Country of asylum (ISO)']=='UGA']
UgandaApps2017To2022 = UgandaApps[(UgandaApps['Year']== 2017)|
        (UgandaApps['Year']==2018)|
        (UgandaApps['Year']==2019)|
        (UgandaApps['Year']==2020)|
        (UgandaApps['Year']==2021)|
        (UgandaApps['Year']==2022)]
#2017
UgandaTopApps2017=UgandaApps[UgandaApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
UgandaTopApps2017.set_index('Year', inplace = True)
UgandaTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
UgandaTopApps2018=UgandaApps[UgandaApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
UgandaTopApps2018.set_index('Year', inplace = True)
UgandaTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
UgandaTopApps2019=UgandaApps[UgandaApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
UgandaTopApps2019.set_index('Year', inplace = True)
UgandaTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
UgandaTopApps2020=UgandaApps[UgandaApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
UgandaTopApps2020.set_index('Year', inplace = True)
UgandaTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
UgandaTopApps2021=UgandaApps[UgandaApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
UgandaTopApps2021.set_index('Year', inplace = True)
UgandaTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
UgandaTopApps2022=UgandaApps[UgandaApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
UgandaTopApps2022.set_index('Year', inplace = True)
UgandaTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = UgandaTopApps2017['applied'].tolist()
bars01 = UgandaTopApps2017['Country of origin (ISO)'].tolist()
height02 = UgandaTopApps2018['applied']
bars02=UgandaTopApps2018['Country of origin (ISO)']
height03 = UgandaTopApps2019['applied']
bars03 = UgandaTopApps2019['Country of origin (ISO)']
height04 = UgandaTopApps2020['applied'].tolist()
bars04 = UgandaTopApps2020['Country of origin (ISO)'].tolist()
height05 = UgandaTopApps2021['applied']
bars05=UgandaTopApps2021['Country of origin (ISO)']
height06 = UgandaTopApps2022['applied']
bars06 = UgandaTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Uganda - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('UgandaTop3.png')
#%%############################################################################
## Austria
###############################################################################
#%% A. Austria Apps & Deps
#%%Create data for plotting
UNAustria=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'AUT']
UNAustria.reset_index(inplace = True)
UNAustria.drop(['index'], axis = 1, inplace = True)
UNAustria.columns
# Create df for bar plot
UNAustriaBar = UNAustria[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNAustria['applied']
decisions = UNAustria['Total decisions']
Year = UNAustria['Year']
formattedapplications = UNAustria['Formatted Applications']
formatteddecisions = UNAustria['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNAustria['Granted']
grantedformatted = UNAustria['Granted%']
hp = UNAustria['Complementary protection']
hpformatted = UNAustria['HP%']
refused = UNAustria['Refused']
refusedformatted = UNAustria['Refused%']
closed = UNAustria['Closed']
closedformatted = UNAustria['Closed%']
#%%Now bar plot of decision volumes by type
UNAustriaBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (5,5), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#Closed annotations
plt.annotate(closedformatted[5], (pos[5], decisions[5]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Austria - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper left', frameon = True, fontsize = 14)
plt.savefig('AustriaAppsDecs.png')
#%% Austria Applicants
#%% Create data for plotting
AustriaApps = UNapps[UNapps['Country of asylum (ISO)']=='AUT']
AustriaApps2017To2022 = AustriaApps[(AustriaApps['Year']== 2017)|
        (AustriaApps['Year']==2018)|
        (AustriaApps['Year']==2019)|
        (AustriaApps['Year']==2020)|
        (AustriaApps['Year']==2021)|
        (AustriaApps['Year']==2022)]
#2017
AustriaTopApps2017=AustriaApps[AustriaApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
AustriaTopApps2017.set_index('Year', inplace = True)
AustriaTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
AustriaTopApps2018=AustriaApps[AustriaApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
AustriaTopApps2018.set_index('Year', inplace = True)
AustriaTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
AustriaTopApps2019=AustriaApps[AustriaApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
AustriaTopApps2019.set_index('Year', inplace = True)
AustriaTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
AustriaTopApps2020=AustriaApps[AustriaApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
AustriaTopApps2020.set_index('Year', inplace = True)
AustriaTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
AustriaTopApps2021=AustriaApps[AustriaApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
AustriaTopApps2021.set_index('Year', inplace = True)
AustriaTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
AustriaTopApps2022=AustriaApps[AustriaApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
AustriaTopApps2022.set_index('Year', inplace = True)
AustriaTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = AustriaTopApps2017['applied'].tolist()
bars01 = AustriaTopApps2017['Country of origin (ISO)'].tolist()
height02 = AustriaTopApps2018['applied']
bars02=AustriaTopApps2018['Country of origin (ISO)']
height03 = AustriaTopApps2019['applied']
bars03 = AustriaTopApps2019['Country of origin (ISO)']
height04 = AustriaTopApps2020['applied'].tolist()
bars04 = AustriaTopApps2020['Country of origin (ISO)'].tolist()
height05 = AustriaTopApps2021['applied']
bars05=AustriaTopApps2021['Country of origin (ISO)']
height06 = AustriaTopApps2022['applied']
bars06 = AustriaTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Austria - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('AustriaTop3.png')
#%%############################################################################
## Australia
###############################################################################
#%% A. Australia Apps & Deps
#%%Create data for plotting
UNAustralia=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'AUS']
UNAustralia.reset_index(inplace = True)
UNAustralia.drop(['index'], axis = 1, inplace = True)
UNAustralia.columns
# Create df for bar plot
UNAustraliaBar = UNAustralia[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNAustralia['applied']
decisions = UNAustralia['Total decisions']
Year = UNAustralia['Year']
formattedapplications = UNAustralia['Formatted Applications']
formatteddecisions = UNAustralia['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNAustralia['Granted']
grantedformatted = UNAustralia['Granted%']
hp = UNAustralia['Complementary protection']
hpformatted = UNAustralia['HP%']
refused = UNAustralia['Refused']
refusedformatted = UNAustralia['Refused%']
closed = UNAustralia['Closed']
closedformatted = UNAustralia['Closed%']
#%%Now bar plot of decision volumes by type
UNAustraliaBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (5,5), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#Closed annotations
plt.annotate(closedformatted[5], (pos[5], decisions[5]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Australia - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper right', frameon = True, fontsize = 14)
plt.savefig('AustraliaAppsDecs.png')
#%% Australia Applicants
#%% Create data for plotting
AustraliaApps = UNapps[UNapps['Country of asylum (ISO)']=='AUS']
AustraliaApps2017To2022 = AustraliaApps[(AustraliaApps['Year']== 2017)|
        (AustraliaApps['Year']==2018)|
        (AustraliaApps['Year']==2019)|
        (AustraliaApps['Year']==2020)|
        (AustraliaApps['Year']==2021)|
        (AustraliaApps['Year']==2022)]
#2017
AustraliaTopApps2017=AustraliaApps[AustraliaApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
AustraliaTopApps2017.set_index('Year', inplace = True)
AustraliaTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
AustraliaTopApps2018=AustraliaApps[AustraliaApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
AustraliaTopApps2018.set_index('Year', inplace = True)
AustraliaTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
AustraliaTopApps2019=AustraliaApps[AustraliaApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
AustraliaTopApps2019.set_index('Year', inplace = True)
AustraliaTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
AustraliaTopApps2020=AustraliaApps[AustraliaApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
AustraliaTopApps2020.set_index('Year', inplace = True)
AustraliaTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
AustraliaTopApps2021=AustraliaApps[AustraliaApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
AustraliaTopApps2021.set_index('Year', inplace = True)
AustraliaTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
AustraliaTopApps2022=AustraliaApps[AustraliaApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
AustraliaTopApps2022.set_index('Year', inplace = True)
AustraliaTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = AustraliaTopApps2017['applied'].tolist()
bars01 = AustraliaTopApps2017['Country of origin (ISO)'].tolist()
height02 = AustraliaTopApps2018['applied']
bars02=AustraliaTopApps2018['Country of origin (ISO)']
height03 = AustraliaTopApps2019['applied']
bars03 = AustraliaTopApps2019['Country of origin (ISO)']
height04 = AustraliaTopApps2020['applied'].tolist()
bars04 = AustraliaTopApps2020['Country of origin (ISO)'].tolist()
height05 = AustraliaTopApps2021['applied']
bars05=AustraliaTopApps2021['Country of origin (ISO)']
height06 = AustraliaTopApps2022['applied']
bars06 = AustraliaTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Australia - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('AustraliaTop3.png')
#%%############################################################################
## Sweden
###############################################################################
#%% A. Sweden Apps & Deps
#%%Create data for plotting
UNSweden=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'SWE']
UNSweden.reset_index(inplace = True)
UNSweden.drop(['index'], axis = 1, inplace = True)
UNSweden.columns
# Create df for bar plot
UNSwedenBar = UNSweden[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNSweden['applied']
decisions = UNSweden['Total decisions']
Year = UNSweden['Year']
formattedapplications = UNSweden['Formatted Applications']
formatteddecisions = UNSweden['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNSweden['Granted']
grantedformatted = UNSweden['Granted%']
hp = UNSweden['Complementary protection']
hpformatted = UNSweden['HP%']
refused = UNSweden['Refused']
refusedformatted = UNSweden['Refused%']
closed = UNSweden['Closed']
closedformatted = UNSweden['Closed%']
#%%Now bar plot of decision volumes by type
UNSwedenBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (5,5), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#Closed annotations
plt.annotate(closedformatted[5], (pos[5], decisions[5]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Sweden - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper center', frameon = True, fontsize = 14)
plt.savefig('SwedenAppsDecs.png')
#%% Sweden Applicants
#%% Create data for plotting
SwedenApps = UNapps[UNapps['Country of asylum (ISO)']=='SWE']
SwedenApps2017To2022 = SwedenApps[(SwedenApps['Year']== 2017)|
        (SwedenApps['Year']==2018)|
        (SwedenApps['Year']==2019)|
        (SwedenApps['Year']==2020)|
        (SwedenApps['Year']==2021)|
        (SwedenApps['Year']==2022)]
#2017
SwedenTopApps2017=SwedenApps[SwedenApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
SwedenTopApps2017.set_index('Year', inplace = True)
SwedenTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
SwedenTopApps2018=SwedenApps[SwedenApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
SwedenTopApps2018.set_index('Year', inplace = True)
SwedenTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
SwedenTopApps2019=SwedenApps[SwedenApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
SwedenTopApps2019.set_index('Year', inplace = True)
SwedenTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
SwedenTopApps2020=SwedenApps[SwedenApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
SwedenTopApps2020.set_index('Year', inplace = True)
SwedenTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
SwedenTopApps2021=SwedenApps[SwedenApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
SwedenTopApps2021.set_index('Year', inplace = True)
SwedenTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
SwedenTopApps2022=SwedenApps[SwedenApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
SwedenTopApps2022.set_index('Year', inplace = True)
SwedenTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = SwedenTopApps2017['applied'].tolist()
bars01 = SwedenTopApps2017['Country of origin (ISO)'].tolist()
height02 = SwedenTopApps2018['applied']
bars02=SwedenTopApps2018['Country of origin (ISO)']
height03 = SwedenTopApps2019['applied']
bars03 = SwedenTopApps2019['Country of origin (ISO)']
height04 = SwedenTopApps2020['applied'].tolist()
bars04 = SwedenTopApps2020['Country of origin (ISO)'].tolist()
height05 = SwedenTopApps2021['applied']
bars05=SwedenTopApps2021['Country of origin (ISO)']
height06 = SwedenTopApps2022['applied']
bars06 = SwedenTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Sweden - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('SwedenTop3.png')
#%%############################################################################
## Egypt
###############################################################################
#%% A. Egypt Apps & Deps
#%%Create data for plotting
UNEgypt=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'EGY']
UNEgypt.reset_index(inplace = True)
UNEgypt.drop(['index'], axis = 1, inplace = True)
UNEgypt.columns
# Create df for bar plot
UNEgyptBar = UNEgypt[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNEgypt['applied']
decisions = UNEgypt['Total decisions']
Year = UNEgypt['Year']
formattedapplications = UNEgypt['Formatted Applications']
formatteddecisions = UNEgypt['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNEgypt['Granted']
grantedformatted = UNEgypt['Granted%']
hp = UNEgypt['Complementary protection']
hpformatted = UNEgypt['HP%']
refused = UNEgypt['Refused']
refusedformatted = UNEgypt['Refused%']
closed = UNEgypt['Closed']
closedformatted = UNEgypt['Closed%']
#%%Now bar plot of decision volumes by type
UNEgyptBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = '#000000', bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (5,5), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#Closed annotations
plt.annotate(closedformatted[5], (pos[5], decisions[5]), textcoords = 'offset points', xytext=(0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Egypt - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper center', frameon = True, fontsize = 14)
plt.savefig('EgyptAppsDecs.png')
#%% Egypt Applicants
#%% Create data for plotting
EgyptApps = UNapps[UNapps['Country of asylum (ISO)']=='EGY']
EgyptApps2017To2022 = EgyptApps[(EgyptApps['Year']== 2017)|
        (EgyptApps['Year']==2018)|
        (EgyptApps['Year']==2019)|
        (EgyptApps['Year']==2020)|
        (EgyptApps['Year']==2021)|
        (EgyptApps['Year']==2022)]
#2017
EgyptTopApps2017=EgyptApps[EgyptApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
EgyptTopApps2017.set_index('Year', inplace = True)
EgyptTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
EgyptTopApps2018=EgyptApps[EgyptApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
EgyptTopApps2018.set_index('Year', inplace = True)
EgyptTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
EgyptTopApps2019=EgyptApps[EgyptApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
EgyptTopApps2019.set_index('Year', inplace = True)
EgyptTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
EgyptTopApps2020=EgyptApps[EgyptApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
EgyptTopApps2020.set_index('Year', inplace = True)
EgyptTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
EgyptTopApps2021=EgyptApps[EgyptApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
EgyptTopApps2021.set_index('Year', inplace = True)
EgyptTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
EgyptTopApps2022=EgyptApps[EgyptApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
EgyptTopApps2022.set_index('Year', inplace = True)
EgyptTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = EgyptTopApps2017['applied'].tolist()
bars01 = EgyptTopApps2017['Country of origin (ISO)'].tolist()
height02 = EgyptTopApps2018['applied']
bars02=EgyptTopApps2018['Country of origin (ISO)']
height03 = EgyptTopApps2019['applied']
bars03 = EgyptTopApps2019['Country of origin (ISO)']
height04 = EgyptTopApps2020['applied'].tolist()
bars04 = EgyptTopApps2020['Country of origin (ISO)'].tolist()
height05 = EgyptTopApps2021['applied']
bars05=EgyptTopApps2021['Country of origin (ISO)']
height06 = EgyptTopApps2022['applied']
bars06 = EgyptTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Egypt - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('EgyptTop3.png')
#%%############################################################################
## Netherlands
###############################################################################
#%% A. Netherlands Apps & Deps
#%%Create data for plotting
UNNetherlands=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'NLD']
UNNetherlands.reset_index(inplace = True)
UNNetherlands.drop(['index'], axis = 1, inplace = True)
UNNetherlands.columns
# Create df for bar plot
UNNetherlandsBar = UNNetherlands[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNNetherlands['applied']
decisions = UNNetherlands['Total decisions']
Year = UNNetherlands['Year']
formattedapplications = UNNetherlands['Formatted Applications']
formatteddecisions = UNNetherlands['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNNetherlands['Granted']
grantedformatted = UNNetherlands['Granted%']
hp = UNNetherlands['Complementary protection']
hpformatted = UNNetherlands['HP%']
refused = UNNetherlands['Refused']
refusedformatted = UNNetherlands['Refused%']
closed = UNNetherlands['Closed']
closedformatted = UNNetherlands['Closed%']
#%%Now bar plot of decision volumes by type
UNNetherlandsBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
#plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,30), ha = 'center', size = 20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#Closed annotations
plt.annotate(closedformatted[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,-30), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Netherlands - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper left', frameon = True, fontsize = 14)
plt.savefig('NetherlandsAppsDecs.png')
#%% Netherlands Applicants
#%% Create data for plotting
NetherlandsApps = UNapps[UNapps['Country of asylum (ISO)']=='NLD']
NetherlandsApps2017To2022 = NetherlandsApps[(NetherlandsApps['Year']== 2017)|
        (NetherlandsApps['Year']==2018)|
        (NetherlandsApps['Year']==2019)|
        (NetherlandsApps['Year']==2020)|
        (NetherlandsApps['Year']==2021)|
        (NetherlandsApps['Year']==2022)]
#2017
NetherlandsTopApps2017=NetherlandsApps[NetherlandsApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
NetherlandsTopApps2017.set_index('Year', inplace = True)
NetherlandsTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
NetherlandsTopApps2018=NetherlandsApps[NetherlandsApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
NetherlandsTopApps2018.set_index('Year', inplace = True)
NetherlandsTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
NetherlandsTopApps2019=NetherlandsApps[NetherlandsApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
NetherlandsTopApps2019.set_index('Year', inplace = True)
NetherlandsTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
NetherlandsTopApps2020=NetherlandsApps[NetherlandsApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
NetherlandsTopApps2020.set_index('Year', inplace = True)
NetherlandsTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
NetherlandsTopApps2021=NetherlandsApps[NetherlandsApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
NetherlandsTopApps2021.set_index('Year', inplace = True)
NetherlandsTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
NetherlandsTopApps2022=NetherlandsApps[NetherlandsApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
NetherlandsTopApps2022.set_index('Year', inplace = True)
NetherlandsTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = NetherlandsTopApps2017['applied'].tolist()
bars01 = NetherlandsTopApps2017['Country of origin (ISO)'].tolist()
height02 = NetherlandsTopApps2018['applied']
bars02=NetherlandsTopApps2018['Country of origin (ISO)']
height03 = NetherlandsTopApps2019['applied']
bars03 = NetherlandsTopApps2019['Country of origin (ISO)']
height04 = NetherlandsTopApps2020['applied'].tolist()
bars04 = NetherlandsTopApps2020['Country of origin (ISO)'].tolist()
height05 = NetherlandsTopApps2021['applied']
bars05=NetherlandsTopApps2021['Country of origin (ISO)']
height06 = NetherlandsTopApps2022['applied']
bars06 = NetherlandsTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Netherlands - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('NetherlandsTop3.png')
#%%############################################################################
## Belgium
###############################################################################
#%% A. Belgium Apps & Deps
#%%Create data for plotting
UNBelgium=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'BEL']
UNBelgium.reset_index(inplace = True)
UNBelgium.drop(['index'], axis = 1, inplace = True)
UNBelgium.columns
# Create df for bar plot
UNBelgiumBar = UNBelgium[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNBelgium['applied']
decisions = UNBelgium['Total decisions']
Year = UNBelgium['Year']
formattedapplications = UNBelgium['Formatted Applications']
formatteddecisions = UNBelgium['Formatted Decisions']
pos = np.arange(len(Year))
granted = UNBelgium['Granted']
grantedformatted = UNBelgium['Granted%']
hp = UNBelgium['Complementary protection']
hpformatted = UNBelgium['HP%']
refused = UNBelgium['Refused']
refusedformatted = UNBelgium['Refused%']
closed = UNBelgium['Closed']
closedformatted = UNBelgium['Closed%']
#%%Now bar plot of decision volumes by type
UNBelgiumBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(.1,20), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(-.1,20), ha = 'center', size = 20, color = '#000000')
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decision annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[4], (pos[4],decisions[4]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[5], (pos[5],decisions[5]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#Granted annotations
plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
#Add sundries
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Belgium - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper left', frameon = True, fontsize = 14)
plt.savefig('BelgiumAppsDecs.png')
#%% Belgium Applicants
#%% Create data for plotting
BelgiumApps = UNapps[UNapps['Country of asylum (ISO)']=='BEL']
BelgiumApps2017To2022 = BelgiumApps[(BelgiumApps['Year']== 2017)|
        (BelgiumApps['Year']==2018)|
        (BelgiumApps['Year']==2019)|
        (BelgiumApps['Year']==2020)|
        (BelgiumApps['Year']==2021)|
        (BelgiumApps['Year']==2022)]
#2017
BelgiumTopApps2017=BelgiumApps[BelgiumApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
BelgiumTopApps2017.set_index('Year', inplace = True)
BelgiumTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
BelgiumTopApps2018=BelgiumApps[BelgiumApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
BelgiumTopApps2018.set_index('Year', inplace = True)
BelgiumTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
BelgiumTopApps2019=BelgiumApps[BelgiumApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
BelgiumTopApps2019.set_index('Year', inplace = True)
BelgiumTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
BelgiumTopApps2020=BelgiumApps[BelgiumApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
BelgiumTopApps2020.set_index('Year', inplace = True)
BelgiumTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
BelgiumTopApps2021=BelgiumApps[BelgiumApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
BelgiumTopApps2021.set_index('Year', inplace = True)
BelgiumTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
BelgiumTopApps2022=BelgiumApps[BelgiumApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
BelgiumTopApps2022.set_index('Year', inplace = True)
BelgiumTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = BelgiumTopApps2017['applied'].tolist()
bars01 = BelgiumTopApps2017['Country of origin (ISO)'].tolist()
height02 = BelgiumTopApps2018['applied']
bars02=BelgiumTopApps2018['Country of origin (ISO)']
height03 = BelgiumTopApps2019['applied']
bars03 = BelgiumTopApps2019['Country of origin (ISO)']
height04 = BelgiumTopApps2020['applied'].tolist()
bars04 = BelgiumTopApps2020['Country of origin (ISO)'].tolist()
height05 = BelgiumTopApps2021['applied']
bars05=BelgiumTopApps2021['Country of origin (ISO)']
height06 = BelgiumTopApps2022['applied']
bars06 = BelgiumTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Belgium - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('BelgiumTop3.png')
#%%############################################################################
## HUNGARY
###############################################################################
#%% A. Hungary Apps & Deps
#%%Create data for plotting
UNHungary=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'HUN']
UNHungary.reset_index(inplace = True)
UNHungary.drop(['index'], axis = 1, inplace = True)
UNHungary.columns
# Create df for bar plot
UNHungaryBar = UNHungary[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNHungary['applied']
decisions = UNHungary['Total decisions']
Year = UNHungary['Year']
formattedapplications = UNHungary['Formatted Applications']
formatteddecisions = UNHungary['Formatted Decisions']
formattedclosed = UNHungary['Closed%']
pos = np.arange(len(Year))
#%%Now bar plot of decision volumes by type
UNHungaryBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decisin annotations
plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#Closed annotations
plt.annotate(formattedclosed[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,-30), ha = 'center', size = 20, color = 'w')
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Hungary - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'center', frameon = True, fontsize = 14)
plt.savefig('HungaryAppsDecs.png')
#%% Hungary Applicants
#%% Create data for plotting
HungaryApps = UNapps[UNapps['Country of asylum (ISO)']=='HUN']
HungaryApps2017To2022 = HungaryApps[(HungaryApps['Year']== 2017)|
        (HungaryApps['Year']==2018)|
        (HungaryApps['Year']==2019)|
        (HungaryApps['Year']==2020)|
        (HungaryApps['Year']==2021)|
        (HungaryApps['Year']==2022)]
#2017
HungaryTopApps2017=HungaryApps[HungaryApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
HungaryTopApps2017.set_index('Year', inplace = True)
HungaryTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
HungaryTopApps2018=HungaryApps[HungaryApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
HungaryTopApps2018.set_index('Year', inplace = True)
HungaryTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
HungaryTopApps2019=HungaryApps[HungaryApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
HungaryTopApps2019.set_index('Year', inplace = True)
HungaryTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
HungaryTopApps2020=HungaryApps[HungaryApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
HungaryTopApps2020.set_index('Year', inplace = True)
HungaryTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
HungaryTopApps2021=HungaryApps[HungaryApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
HungaryTopApps2021.set_index('Year', inplace = True)
HungaryTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
HungaryTopApps2022=HungaryApps[HungaryApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
HungaryTopApps2022.set_index('Year', inplace = True)
HungaryTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = HungaryTopApps2017['applied'].tolist()
bars01 = HungaryTopApps2017['Country of origin (ISO)'].tolist()
height02 = HungaryTopApps2018['applied']
bars02=HungaryTopApps2018['Country of origin (ISO)']
height03 = HungaryTopApps2019['applied']
bars03 = HungaryTopApps2019['Country of origin (ISO)']
height04 = HungaryTopApps2020['applied'].tolist()
bars04 = HungaryTopApps2020['Country of origin (ISO)'].tolist()
height05 = HungaryTopApps2021['applied']
bars05=HungaryTopApps2021['Country of origin (ISO)']
height06 = HungaryTopApps2022['applied']
bars06 = HungaryTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Hungary - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('HungaryTop3.png')
###############################################################################
#%% A. Rwanda Apps & Deps
#%%Create data for plotting
UNRwanda=UNTotalAppsDecs2017To2022[UNTotalAppsDecs2017To2022['Country of asylum (ISO)']== 'RWA']
UNRwanda.reset_index(inplace = True)
UNRwanda.drop(['index'], axis = 1, inplace = True)
UNRwanda.columns
# Create df for bar plot
UNRwandaBar = UNRwanda[['Year', 'Recognized decisions', 'Complementary protection', 'Rejected decisions', 'Otherwise closed']]
#Remove applications to add to bar plot
applications = UNRwanda['applied']
decisions = UNRwanda['Total decisions']
Year = UNRwanda['Year']
formattedapplications = UNRwanda['Formatted Applications']
formatteddecisions = UNRwanda['Formatted Decisions']
formattedclosed = UNRwanda['Closed%']
granted = UNRwanda['Granted']
grantedformatted = UNRwanda['Granted%']
hp = UNRwanda['Complementary protection']
hpformatted = UNRwanda['HP%']
pos = np.arange(len(Year))
#%%Now bar plot of decision volumes by type
UNRwandaBar.plot(x='Year', kind='bar', stacked=True,
        figsize=(22,13), color = ('#002664', '#317EFF', '#FE4365', '#FC9D9A'), legend = None, fontsize=16)
plt.plot(applications, '-o', linewidth = 5, color = '#002664')
 #To label the entire line
# zip joins x and y coordinates in pairs
#for x,y in zip(pos,applications):
#    label = "{:,.0f}".format(y)
#    plt.annotate(label,(x,y), textcoords="offset points", xytext=(75,5),ha='center',
#                 size = 18, color = 'black') # horizontal alignment can be left, right or center
plt.xticks(size = 20)
plt.yticks([])
#Format the y-axis so it has thousand-value commas
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# Remove axes splines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    # Remove the tick marks (- and | from both axes)
plt.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False)
plt.tick_params(
    axis='x',
    which='both',
    top=False,
    bottom=False)
# Add annotations
#Application annotations
plt.annotate(formattedapplications[0], (pos[0], applications[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[1], (pos[1], applications[1]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedapplications[2], (pos[2], applications[2]), textcoords = 'offset points', xytext=(0,5), ha = 'center', size = 20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#plt.annotate(formattedapplications[3], (pos[3], applications[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedapplications[4], (pos[4], applications[4]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedapplications[5], (pos[5], applications[5]), textcoords = 'offset points', xytext=(0,20), ha = 'center', size = 20, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
#Decisin annotations
#plt.annotate(formatteddecisions[0], (pos[0],decisions[0]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#plt.annotate(formatteddecisions[3], (pos[3],decisions[3]), textcoords = 'offset points', xytext = (0,10), ha = 'center', size = 20, color = '#000000')
#Closed annotations
plt.annotate(formattedclosed[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(formattedclosed[2], (pos[2],decisions[2]), textcoords = 'offset points', xytext = (0,-30), ha = 'center', size = 20, color = 'w')
plt.annotate(formattedclosed[3], (pos[3],decisions[1]), textcoords = 'offset points', xytext = (0,-30), ha = 'center', size = 20, color = 'w')
#plt.annotate(formattedclosed[1], (pos[1],decisions[1]), textcoords = 'offset points', xytext = (0,-30), ha = 'center', size = 20, color = 'w')
#Granted annotations
#plt.annotate(grantedformatted[0], (pos[0], granted[0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[1], (pos[1], granted[1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[2], (pos[2], granted[2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[3], (pos[3], granted[3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[4], (pos[4], granted[4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.annotate(grantedformatted[5], (pos[5], granted[5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 20, color = 'w')
plt.xlabel('Year of Application', fontsize = 20, color = 'black', labelpad = 20)
plt.ylabel('Asylum Grant or Refusal Rate / Application Volumes', fontsize = 20, labelpad = 20)
plt.title('Rwanda - Application and Decision Volumes (by decision type) and Asylum Grant Rates 2017 to 2022',fontsize = 20, color = 'black', pad = 30)
plt.legend(['Applications','Granted', 'Humanitarian Protection', 'Refused', 'Admin Closed'], loc = 'upper left', frameon = True, fontsize = 14)
plt.savefig('RwandaAppsDecs.png')
#%% Rwanda Applicants
#%% Create data for plotting
RwandaApps = UNapps[UNapps['Country of asylum (ISO)']=='RWA']
RwandaApps2017To2022 = RwandaApps[(RwandaApps['Year']== 2017)|
        (RwandaApps['Year']==2018)|
        (RwandaApps['Year']==2019)|
        (RwandaApps['Year']==2020)|
        (RwandaApps['Year']==2021)|
        (RwandaApps['Year']==2022)]
#2017
RwandaTopApps2017=RwandaApps[RwandaApps['Year']==2017].sort_values(by='applied', ascending = False).head(3)
RwandaTopApps2017.set_index('Year', inplace = True)
RwandaTopApps2017.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2018
RwandaTopApps2018=RwandaApps[RwandaApps['Year']==2018].sort_values(by='applied', ascending = False).head(3)
RwandaTopApps2018.set_index('Year', inplace = True)
RwandaTopApps2018.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2019
RwandaTopApps2019=RwandaApps[RwandaApps['Year']==2019].sort_values(by='applied', ascending = False).head(3)
RwandaTopApps2019.set_index('Year', inplace = True)
RwandaTopApps2019.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2020
RwandaTopApps2020=RwandaApps[RwandaApps['Year']==2020].sort_values(by='applied', ascending = False).head(3)
RwandaTopApps2020.set_index('Year', inplace = True)
RwandaTopApps2020.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2021
RwandaTopApps2021=RwandaApps[RwandaApps['Year']==2021].sort_values(by='applied', ascending = False).head(3)
RwandaTopApps2021.set_index('Year', inplace = True)
RwandaTopApps2021.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#2022
RwandaTopApps2022=RwandaApps[RwandaApps['Year']==2022].sort_values(by='applied', ascending = False).head(3)
RwandaTopApps2022.set_index('Year', inplace = True)
RwandaTopApps2022.drop('Country of asylum (ISO)', axis = 1, inplace = True)
#%% Now plot
height01 = RwandaTopApps2017['applied'].tolist()
bars01 = RwandaTopApps2017['Country of origin (ISO)'].tolist()
height02 = RwandaTopApps2018['applied']
bars02=RwandaTopApps2018['Country of origin (ISO)']
height03 = RwandaTopApps2019['applied']
bars03 = RwandaTopApps2019['Country of origin (ISO)']
height04 = RwandaTopApps2020['applied'].tolist()
bars04 = RwandaTopApps2020['Country of origin (ISO)'].tolist()
height05 = RwandaTopApps2021['applied']
bars05=RwandaTopApps2021['Country of origin (ISO)']
height06 = RwandaTopApps2022['applied']
bars06 = RwandaTopApps2022['Country of origin (ISO)']
Year = ['2017', '2018', '2019', '2020', '2021', '2022']
###
fig, axs = plt.subplots(1, 6, figsize=(10, 4), sharey=True)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('xtick', labelsize=9) 
axs[0].bar(bars01, height01, width = .9, color = '#00000090')
axs[1].bar(bars02, height02, width = .9, color = '#00000090')
axs[2].bar(bars03, height03, width = .9, color = '#00000090')
axs[3].bar(bars04, height04, width = .9, color = '#00000090')
axs[4].bar(bars05, height05, width = .9, color = '#00000090')
axs[5].bar(bars06, height06, width = .9, color = '#00000090')
fig.suptitle('Rwanda - Top 3 Country of Origin by Volume of Applications', fontsize = 10, y = .98)
sizes = {'fontsize':8}
##
axs[0].set_title('2017', fontdict = sizes,y=.98, pad=-10)
axs[1].set_title('2018', fontdict = sizes, y=.98,pad = -10)
axs[2].set_title('2019', fontdict = sizes, y=.98,pad = -10)
axs[3].set_title('2020', fontdict = sizes, y=.98,pad = -10)
axs[4].set_title('2021', fontdict = sizes, y=.98,pad = -10)
axs[5].set_title('2022', fontdict = sizes, y=.98,pad = -10)
for i in range(6):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    if i==0:
        axs[i].get_yaxis().set_visible(True)
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
## Remove ticks
for i in range(6):
    axs[i].tick_params(
    axis='y',          
    which='both',      
    left=False,     
    right=False,        
    labelbottom=True) 
    axs[i].tick_params(
    axis='x',         
    which='both',      
    top=False,      
    bottom=False,        
    labelbottom=True) 
#Label the first plot
axs[0].set_xlabel('Country of Origin (ISO)')
axs[0].set_ylabel('Applications')
axs[0].legend(['Applications'], loc = (0,1), frameon = True)
plt.savefig('RwandaTop3.png')
#%%####################################################################
## Applicant journeys
#######################################################################
# =============================================================================
# #%% Albania Applications
# =============================================================================
AlbaniaApps = UNapps[UNapps['Country of origin (ISO)']=='ALB']
AlbaniaApps['Formatted Applied']= UNapps['applied'].apply(lambda x: '{:,.0f}'.format(x))
#AlbaniaApps.sort_values(by = 'applied', ascending = False, inplace = True)
AlbaniaApps2017To2022 = AlbaniaApps[(AlbaniaApps['Year']>=2017) & (AlbaniaApps['Year'] < 2023)]
TotalAlbApps = AlbaniaApps['applied'].sum()
TotalAlbApps
#%% Fra and GBR only
#Plot
FraApplied = AlbaniaApps2017To2022[(AlbaniaApps2017To2022['Country of asylum (ISO)']=='FRA')]
FraApplied.reset_index(inplace = True)
UKApplied = AlbaniaApps2017To2022[AlbaniaApps2017To2022['Country of asylum (ISO)']=='GBR']
UKApplied.reset_index(inplace = True)
AlbYear = UKApplied['Year'].tolist()
x_indexes = np.arange(len(AlbYear))
width = 0.3
formattedUKapplications = UKApplied['Formatted Applied']
formattedFraapplications = FraApplied['Formatted Applied']
#UNTotalAppsDecsTo2022['HP%']=UNTotalAppsDecsTo2022['Humanitarian Protection'].apply(lambda x: '{:,.0%}'.format(x))
#UNTotalAppsDecsTo2022['Formatted Applications']=UNTotalAppsDecsTo2022['applied'].apply(lambda x: '{:,.0f}'.format(x))
#%% Plot
fig = plt.figure(figsize = (10,10))
plt.bar(x_indexes - width, UKApplied['applied'], width = width, color = '#00747A', label = 'UK Applications')
plt.bar(x_indexes, FraApplied['applied'], width = width, color = '#002664', label = 'France Applications')
plt.legend()
plt.title('Albania Applications to the UK and France, 2017 to 2022', pad = 50)
plt.xlabel('Year', labelpad = 20)
plt.ylabel('Applications', labelpad = 20)
#Remove spines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    top=False,      # ticks along the bottom edge are off
    bottom=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.yticks([])
#Application annotations
plt.annotate(formattedUKapplications[0], (-.32, UKApplied['applied'][0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 10, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#00747A", lw=1))
plt.annotate(formattedFraapplications[0], (0, FraApplied['applied'][0]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 10, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedUKapplications[1], (1-.32, UKApplied['applied'][1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 10, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#00747A", lw=1))
plt.annotate(formattedFraapplications[1], (1, FraApplied['applied'][1]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 10, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedUKapplications[2], (2-.32, UKApplied['applied'][2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 10, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#00747A", lw=1))
plt.annotate(formattedFraapplications[2], (2, FraApplied['applied'][2]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 10, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedUKapplications[3], (3-.32, UKApplied['applied'][3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 10, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#00747A", lw=1))
plt.annotate(formattedFraapplications[3], (3, FraApplied['applied'][3]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 10, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedUKapplications[4], (4-.32, UKApplied['applied'][4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 10, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#00747A", lw=1))
plt.annotate(formattedFraapplications[4], (4, FraApplied['applied'][4]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 10, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.annotate(formattedUKapplications[5], (5-.32, UKApplied['applied'][5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 10, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#00747A", lw=1))
plt.annotate(formattedFraapplications[5], (5, FraApplied['applied'][5]), textcoords = 'offset points', xytext=(0,10), ha = 'center', size = 10, bbox=dict(boxstyle="round,pad=0.3",fc="white", ec="#002664", lw=1))
plt.xticks(ticks = x_indexes, labels = ('2017', '2018', '2019', '2020', '2021', '2022'))
plt.savefig('AlbApps.png')
plt.show()
#%% Plot grant rate
AlbaniaDecs = UNdecs[(UNdecs['Country of origin (ISO)']=='ALB')]
AlbaniaDecsFRGBR = AlbaniaDecs[(AlbaniaDecs['Country of asylum (ISO)']=='GBR') | (AlbaniaDecs['Country of asylum (ISO)']=='FRA')]
AlbaniaDecsFRGBR2017To2022 = AlbaniaDecsFRGBR[(AlbaniaDecsFRGBR['Year']== 2017)|
        (AlbaniaDecsFRGBR['Year']== 2018)|
        (AlbaniaDecsFRGBR['Year']== 2019)|
        (AlbaniaDecsFRGBR['Year']== 2020)|
        (AlbaniaDecsFRGBR['Year']== 2021)|
        (AlbaniaDecsFRGBR['Year']== 2022)]

FraGranted = AlbaniaDecsFRGBR2017To2022[(AlbaniaDecsFRGBR2017To2022['Country of asylum (ISO)']=='FRA')]
FraGranted.reset_index(inplace = True)
FraGranted['Formatted Grants'] = FraGranted['Granted'].apply(lambda x: '{:,.1%}'.format(x))
UKGranted = AlbaniaDecsFRGBR2017To2022[AlbaniaDecsFRGBR2017To2022['Country of asylum (ISO)']=='GBR']
formattedFraGrants = FraGranted['Formatted Grants'].values
UKGranted['Formatted Grants'] = UKGranted['Granted'].apply(lambda x: '{:,.0%}'.format(x))
formattedUKGrants = UKGranted['Formatted Grants'].values
UKGranted.reset_index(inplace = True)
Year = FraGranted['Year'].tolist()
x_indexes = np.arange(len(Year))
width = 0.3
#%% Plot
fig = plt.figure(figsize = (10,10))
plt.bar(x_indexes - width, UKGranted['Granted'], width = width, color = '#732282', label = 'UK Grant Rate')
plt.bar(x_indexes, FraGranted['Granted'], width = width, color = '#00000090', label = 'France Grant Rate')
plt.legend()
plt.title('Albania Asylum Grant Rates in the UK and France, 2017 to 2022', pad = 50)
plt.xlabel('Year', labelpad = 20)
plt.ylabel('Asylum Grant Rate', labelpad = 20)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0%}')) # No decimal places
#Remove spines
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    top=False,      # ticks along the bottom edge are off
    bottom=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
#Application annotations
plt.annotate(formattedUKGrants[0], (-.3, .11), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 12,color = 'w')
plt.annotate(formattedFraGrants[0], (0.02, FraGranted['Granted'][0]), textcoords = 'offset points', xytext=(0,15), ha = 'center', size = 12, color = 'black')
plt.annotate(formattedUKGrants[1], (1-.3,.1), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 12, color = 'w')
plt.annotate(formattedFraGrants[1], (1.02, FraGranted['Granted'][1]), textcoords = 'offset points', xytext=(0,15), ha = 'center', size = 12, color = 'black')
plt.annotate(formattedUKGrants[2], (2-.3, .21), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 12, color = 'w')
plt.annotate(formattedFraGrants[2], (2.02, FraGranted['Granted'][2]), textcoords = 'offset points', xytext=(0,15), ha = 'center', size = 12, color = 'black')
plt.annotate(formattedUKGrants[3], (3-.3, .29), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 12, color = 'w')
plt.annotate(formattedFraGrants[3], (3.02, FraGranted['Granted'][3]), textcoords = 'offset points', xytext=(0,15), ha = 'center', size = 12, color = 'black')
plt.annotate(formattedUKGrants[4], (4-.3, .38), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 12, color = 'w')
plt.annotate(formattedFraGrants[4], (4.02, FraGranted['Granted'][4]), textcoords = 'offset points', xytext=(0,15), ha = 'center', size = 12, color = 'black')
plt.annotate(formattedUKGrants[5], (5-.3, .27), textcoords = 'offset points', xytext=(0,-20), ha = 'center', size = 12, color = 'w')
plt.annotate(formattedFraGrants[5], (5.02, FraGranted['Granted'][5]), textcoords = 'offset points', xytext=(0,15), ha = 'center', size = 12, color = 'black')
plt.xticks(ticks = x_indexes, labels = ('2017', '2018', '2019', '2020', '2021', '2022'))
plt.ylim((0,.4))
plt.yticks([])
plt.savefig('AlbGrants.png')
plt.show()
#%% All Applicants
Apps2017To2022 = UNapps[(UNapps['Year']==2017)|
        (UNapps['Year']==2018)|
        (UNapps['Year']==2019)|
        (UNapps['Year']==2020)|
        (UNapps['Year']==2021)|
        (UNapps['Year']==2022)]
Apps2017To2022 = Apps2017To2022.groupby(['Country of origin (ISO)']).agg({'applied':np.sum})
Apps2017To2022.reset_index(inplace = True)
Apps2017To2022=Apps2017To2022.sort_values(by='applied', ascending = False)
Apps2017To2022.reset_index(inplace = True)
TotalApplicants2017To2022=Apps2017To2022['applied'].sum()
Top20Applicants2017To2022 = Apps2017To2022.head(20)
Top20Volumes = Top20Applicants2017To2022['applied'].sum()
Top20Volumes/TotalApplicants2017To2022
Top20Applicants2017To2022.drop('index', axis = 1, inplace = True)
Top20Applicants2017To2022['applied%'] = Top20Applicants2017To2022['applied'].apply(lambda x: '{:,.0f}'.format(x))
#%% now plot
ax = Top20Applicants2017To2022[['Country of origin (ISO)','applied']].plot(kind='barh', figsize=(15, 10), legend=True, fontsize=12, color = '#00000090', width = 0.8)

# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
ax.invert_yaxis()
#plt.yticks([])
ax.set_title('Top 20 Countries of Origin 2017 To 2022',color = '#000000',fontsize=20, pad = 20)
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
#ax.get_xaxis().set_major_formatter(
#    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax.tick_params(axis='both', which='both', length=0)

labels = ['Venezuela', 'Afghanistan', 'Syria', 'Iraq', 'Nicaragua', 'Congo (DRC)', 'Honduras', 'Cuba', 'Nigeria', 'Pakistan', 'El Salvador', 'Colombia', 'Haiti', 'Ukraine',
          'Guatemala', 'Turkey', 'Eritrea', 'Somalia', 'Iran', 'Bangladesh']
plt.annotate(Top20Applicants2017To2022['applied%'][0], (Top20Applicants2017To2022['applied'][0],0.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][1], (Top20Applicants2017To2022['applied'][1],1.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][2], (Top20Applicants2017To2022['applied'][2],2.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][3], (Top20Applicants2017To2022['applied'][3],3.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][4], (Top20Applicants2017To2022['applied'][4],4.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][5], (Top20Applicants2017To2022['applied'][5],5.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][6], (Top20Applicants2017To2022['applied'][6],6.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][7], (Top20Applicants2017To2022['applied'][7],7.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][8], (Top20Applicants2017To2022['applied'][8],8.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][9], (Top20Applicants2017To2022['applied'][9],9.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][10], (Top20Applicants2017To2022['applied'][10],10.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][11], (Top20Applicants2017To2022['applied'][11],11.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][12], (Top20Applicants2017To2022['applied'][12],12.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][13], (Top20Applicants2017To2022['applied'][13],13.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][14], (Top20Applicants2017To2022['applied'][14],14.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][15], (Top20Applicants2017To2022['applied'][15],15.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][16], (Top20Applicants2017To2022['applied'][16],16.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][17], (Top20Applicants2017To2022['applied'][17],17.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][18], (Top20Applicants2017To2022['applied'][18],18.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
plt.annotate(Top20Applicants2017To2022['applied%'][19], (Top20Applicants2017To2022['applied'][19],19.15), textcoords = 'offset points', xytext=(-50,0), ha = 'center', size = 12,color = 'w')
ax.set_yticklabels(labels, fontsize=12)
ax.set_xticks([])

ax.legend(['Applied'], loc = 'lower right')

ax.set_xlabel('Applications',color = '#000000', fontsize = 14, labelpad = 20)
ax.set_ylabel('Country of Origin (ISO)', color = '#000000',fontsize = 14, labelpad = 20)
ax.figure.savefig('Top20Applicants2017To2022.png')
#%%
#

