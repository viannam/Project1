#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Title: Visualization Crop Production and Climate Change
#Source data: https://www.kaggle.com/datasets/thedevastator/the-relationship-between-crop-production-and-cli?resource=download
#API command: kaggle datasets download -d thedevastator/the-relationship-between-crop-production-and-cli

#Import data
import os
import pandas as pd

os.chdir('/Users/paul.adunola/Desktop')
crop = pd.read_csv(r'crop_production.csv')

#print(crop.head())
#crop.shape

# Drop non-informative columns:
crop = crop.drop(["index", "INDICATOR", "FREQUENCY","Flag Codes"], axis=1)
# Drop non-informative countries:
crop = crop[crop["LOCATION"].str.contains("SSA|OECD|BRICS|WLD") == False]

print(crop.head())

#Check unique crops and countries
print(crop.SUBJECT.unique()) #'RICE', 'WHEAT', 'MAIZE', 'SOYBEAN'
print(crop.LOCATION.unique())


# In[ ]:


#Subset data for rice

crop_rice = crop[crop["SUBJECT"]=="RICE"]

#Yield data were in different measurements.
#Subset by measurement
crop_rice_1k_ton = crop_rice[crop_rice["MEASURE"]=="THND_TONNE"]

#Subset data for wheat

crop_wheat = crop[crop["SUBJECT"]=="WHEAT"]

#Yield data were in different measurements.
#Subset by measurement
crop_wheat_1k_ton = crop_wheat[crop_wheat["MEASURE"]=="THND_TONNE"]

#Subset data for maize

crop_maize = crop[crop["SUBJECT"]=="MAIZE"]

#Yield data were in different measurements.
#Subset by measurement
crop_maize_1k_ton = crop_maize[crop_maize["MEASURE"]=="THND_TONNE"]

#Subset data for soybean

crop_soybean = crop[crop["SUBJECT"]=="SOYBEAN"]

#Yield data were in different measurements.
#Subset by measurement
crop_soybean_1k_ton = crop_soybean[crop_soybean["MEASURE"]=="THND_TONNE"]


# In[ ]:


#Plot rice yield by country and year

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(15,8.27)})
ax = sns.lineplot(data=crop_rice_1k_ton, x="TIME", y="Value", hue="LOCATION", style="LOCATION")
sns.move_legend(
    ax, "lower center", #lower center
    bbox_to_anchor=(.9, 1),
    ncol=3, title=None, frameon=False,
)
plt.xlabel("Year")
plt.ylabel("Yield (1000/Ton)")
plt.title("Rice Yield in 48 countries from 1990 to 2021", size=18)


# In[ ]:


#Plot wheat yield by country and year

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(15,8.27)})
ax = sns.lineplot(data=crop_wheat_1k_ton, x="TIME", y="Value", hue="LOCATION", style="LOCATION")
sns.move_legend(
    ax, "lower center", #lower center
    bbox_to_anchor=(.9, 1),
    ncol=3, title=None, frameon=False,
)
plt.xlabel("Year")
plt.ylabel("Yield (1000/Ton)")
plt.title("Wheat Yield in 48 countries from 1990 to 2021", size=18)


# In[ ]:


#Plot maize yield by country and year

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(15,8.27)})
ax = sns.lineplot(data=crop_maize_1k_ton, x="TIME", y="Value", hue="LOCATION", style="LOCATION")
sns.move_legend(
    ax, "lower center", #lower center
    bbox_to_anchor=(.9, 1),
    ncol=3, title=None, frameon=False,
)
plt.xlabel("Year")
plt.ylabel("Yield (1000/Ton)")
plt.title("Maize Yield in 48 countries from 1990 to 2021", size=18)


# In[ ]:


#Plot soybean yield by country and year

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(15,8.27)})
ax = sns.lineplot(data=crop_soybean_1k_ton, x="TIME", y="Value", hue="LOCATION", style="LOCATION")
sns.move_legend(
    ax, "lower center", #lower center
    bbox_to_anchor=(.9, 1),
    ncol=3, title=None, frameon=False,
)
plt.xlabel("Year")
plt.ylabel("Yield (1000/Ton)")
plt.title("Soybean Yield in 48 countries from 1990 to 2021", size=18)


# In[ ]:


#Country name were coded in alpha3.
#pycountry was used to get original country name
#pip install pycountry_convert before running

import pycountry_convert as pc
import pycountry

#Get unique country id
country_id = crop_rice_ton_ha.LOCATION.unique()

#Rename dataframe
crop_rice_1k_ton2 = crop_rice_1k_ton
crop_wheat_1k_ton2 = crop_wheat_1k_ton
crop_maize_1k_ton2 = crop_maize_1k_ton
crop_soybean_1k_ton2 = crop_soybean_1k_ton

#Get alpha2 and alpha3 country code from pycountry
list_alpha_2 = [i.alpha_2 for i in list(pycountry.countries)]
list_alpha_3 = [i.alpha_3 for i in list(pycountry.countries)]    

#Function to get country name
def country_flag(df):
    if (len(df['LOCATION'])==2 and df['LOCATION'] in list_alpha_2):
        return pycountry.countries.get(alpha_2=df['LOCATION']).name
    elif (len(df['LOCATION'])==3 and df['LOCATION'] in list_alpha_3):
        return pycountry.countries.get(alpha_3=df['LOCATION']).name
    else:
        return df['LOCATION']

#Get country name and save in datafrmae
#Note that EU28 is not a country
#EU28 is a combination of all EU countries. Therefore, it was returned as EU28.

#Rice
crop_rice_1k_ton2['country_name']=crop_rice_1k_ton2.apply(country_flag, axis = 1)
print(crop_rice_1k_ton2.head())

#Wheat
crop_wheat_1k_ton2['country_name']=crop_wheat_1k_ton2.apply(country_flag, axis = 1)
print(crop_wheat_1k_ton2.head())

#Rice
crop_maize_1k_ton2['country_name']=crop_maize_1k_ton2.apply(country_flag, axis = 1)
print(crop_maize_1k_ton2.head())

#Rice
crop_soybean_1k_ton2['country_name']=crop_soybean_1k_ton2.apply(country_flag, axis = 1)
print(crop_soybean_1k_ton2.head())


# In[ ]:


#Get continent for country
#pip install pycountry_convert before running

import pycountry_convert as pc
import pandas as pd

def country_to_continent(country_name):
    try:
        country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
        return country_continent_name
    except:
        return country_name

continent = []
for line in crop_rice_1k_ton2.index:
    country = crop_rice_1k_ton2['country_name'][line]
    continent.append(country_to_continent(country))
crop_rice_1k_ton2['continent'] = continent
print(crop_rice_1k_ton2.head())

continent = []
for line in crop_wheat_1k_ton2.index:
    country = crop_wheat_1k_ton2['country_name'][line]
    continent.append(country_to_continent(country))
crop_wheat_1k_ton2['continent'] = continent
print(crop_wheat_1k_ton2.head())

continent = []
for line in crop_maize_1k_ton2.index:
    country = crop_maize_1k_ton2['country_name'][line]
    continent.append(country_to_continent(country))
crop_maize_1k_ton2['continent'] = continent
print(crop_maize_1k_ton2.head())

continent = []
for line in crop_soybean_1k_ton2.index:
    country = crop_soybean_1k_ton2['country_name'][line]
    continent.append(country_to_continent(country))
crop_soybean_1k_ton2['continent'] = continent
print(crop_soybean_1k_ton2.head())


# In[ ]:


#Plot of rice yield by continent

import seaborn as sns
sns.set(rc={'figure.figsize':(15,8.27)})
ax = sns.lineplot(data=crop_rice_1k_ton2, x="TIME", y="Value", hue="continent", style="continent")
sns.move_legend(
    ax, "lower center", #lower center
    bbox_to_anchor=(.95, 1),
    ncol=3, title=None, frameon=False,
)
plt.xlabel("Year")
plt.ylabel("Yield (1000/Ton)")
plt.title("Rice Yield Five continents from 1990 to 2021", size=18)


# In[ ]:


#Plot of wheat yield by continent

import seaborn as sns
sns.set(rc={'figure.figsize':(15,8.27)})
ax = sns.lineplot(data=crop_wheat_1k_ton2, x="TIME", y="Value", hue="continent", style="continent")
sns.move_legend(
    ax, "lower center", #lower center
    bbox_to_anchor=(.95, 1),
    ncol=3, title=None, frameon=False,
)
plt.xlabel("Year")
plt.ylabel("Yield (1000/Ton)")
plt.title("Wheat Yield Five continents from 1990 to 2021", size=18)


# In[ ]:


#Plot of maize yield by continent

import seaborn as sns
sns.set(rc={'figure.figsize':(15,8.27)})
ax = sns.lineplot(data=crop_maize_1k_ton2, x="TIME", y="Value", hue="continent", style="continent")
sns.move_legend(
    ax, "lower center", #lower center
    bbox_to_anchor=(.95, 1),
    ncol=3, title=None, frameon=False,
)
plt.xlabel("Year")
plt.ylabel("Yield (1000/Ton)")
plt.title("Maize Yield Five continents from 1990 to 2021", size=18)


# In[ ]:


#Plot of soybean yield by continent

import seaborn as sns
sns.set(rc={'figure.figsize':(15,8.27)})
ax = sns.lineplot(data=crop_soybean_1k_ton2, x="TIME", y="Value", hue="continent", style="continent")
sns.move_legend(
    ax, "lower center", #lower center
    bbox_to_anchor=(.95, 1),
    ncol=3, title=None, frameon=False,
)
plt.xlabel("Year")
plt.ylabel("Yield (1000/Ton)")
plt.title("Soybean Yield Five continents from 1990 to 2021", size=18)


# In[ ]:


#Aggregate yield by country name

def mean_loc(dat):
    dat_mean = dat.groupby(['SUBJECT','country_name','continent'])['Value'].agg('mean')
    dat_mean.columns = ['Value']
    dat_mean = dat_mean.reset_index()
    return dat_mean
    
crop_rice_1k_ton_mean = mean_loc(crop_rice_1k_ton2)
print(crop_rice_1k_ton_mean.head())
    
crop_wheat_1k_ton_mean = mean_loc(crop_wheat_1k_ton2)
print(crop_wheat_1k_ton_mean.head())

crop_maize_1k_ton_mean = mean_loc(crop_maize_1k_ton2)
print(crop_maize_1k_ton_mean.head())

crop_soybean_1k_ton_mean = mean_loc(crop_soybean_1k_ton2)
print(crop_soybean_1k_ton_mean.head())


# In[ ]:


#function to get longitude and latitude data from country name
#pip install geopy.geocoders before running

#Function for getting geolcation for each country
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="my_request")
def geolocate(country_name):
    try:
        # Geolocate the center of the country
        loc = geolocator.geocode(country_name)
        # And return latitude and longitude
        return (loc.latitude, loc.longitude)
    except:
        if country_name == 'EU28':
            loc = geolocator.geocode('Europe')
            return (loc.latitude, loc.longitude)
        else:
            # Return missing value
            return (np.nan,np.nan)


# In[ ]:


#Get cordinates
#Warning: Take about 45 to 60 seconds

def get_cordinates(dat):
    latitude = []
    longitude = []
    
    #Get unique country ids
    country_id = dat["country_name"]
    
    #Get cordinates
    for country in country_id:
        geo_loc = geolocate(country)
        latitude.append(geo_loc[0])
        longitude.append(geo_loc[1])
    
    #make new dataframe of cordinate and country id
    right_df = pd.DataFrame({'country_name': country_id,
                        'latitude': latitude,
                        'longitude': longitude
                       })
    
    #left join new dataframe with orif=ginal data by country name
    dat = dat.merge(right_df, on='country_name', how='left')
    print("---------------------Get Country Cordinate complete---------------------")
    return dat

#Rice
crop_rice_1k_ton_mean2 = get_cordinates(crop_rice_1k_ton_mean)
print(crop_rice_1k_ton_mean2.head())

#Wheat
crop_wheat_1k_ton_mean2 = get_cordinates(crop_wheat_1k_ton_mean)
print(crop_wheat_1k_ton_mean2.head())

#Maize
crop_maize_1k_ton_mean2 = get_cordinates(crop_maize_1k_ton_mean)
print(crop_maize_1k_ton_mean2.head())

#Soybean
crop_soybean_1k_ton_mean2 = get_cordinates(crop_soybean_1k_ton_mean)
print(crop_soybean_1k_ton_mean2.head())


# In[ ]:


#Save data
import os
os.makedirs('/Users/paul.adunola/Desktop', exist_ok=True)
crop_rice_1k_ton_mean2.to_csv('/Users/paul.adunola/Desktop/Rice_yield.csv')
crop_wheat_1k_ton_mean2.to_csv('/Users/paul.adunola/Desktop/Wheat_yield.csv')
crop_maize_1k_ton_mean2.to_csv('/Users/paul.adunola/Desktop/Maize_yield.csv')
crop_soybean_1k_ton_mean2.to_csv('/Users/paul.adunola/Desktop/Soybean_yield.csv')


# In[ ]:


#Create a world map to show distributions of yield 
#pip install folium before running

import folium
from folium.plugins import MarkerCluster

def yield_map(dat):
    # create empty map
    world_map= folium.Map(tiles="cartodbpositron")
    marker_cluster = MarkerCluster().add_to(world_map)
    
    #for each coordinate, create circlemarker of yield for each country
    for i in range(len(dat)):
        lat = dat.iloc[i]['latitude']
        long = dat.iloc[i]['longitude']
        country = dat.iloc[i]['country_name']
        Value = dat.iloc[i]['Value']
        radius=5
        folium.vector_layers.CircleMarker(location = [lat, long], radius=radius, 
                            color='yellow',fill=True,fill_color='blue',fill_opacity=0.6,
                            tooltip = str(country)+','+str(float(round(Value,2)))+','+str("1000/ton")).add_to(marker_cluster)
    #show the map
    return world_map


# In[ ]:


# World map for rice
#Zoom in and pan for better visualization and actual yield
#Mouse over to get actual yield in 1000/ton

yield_map(crop_rice_1k_ton_mean2)


# In[ ]:


# World map for wheat
#Zoom in and pan for better visualization and actual yield
#Mouse over to get actual yield in 1000/ton

yield_map(crop_wheat_1k_ton_mean2)


# In[ ]:


# World map for maize
#Zoom in and pan for better visualization and actual yield
#Mouse over to get actual yield in 1000/ton

yield_map(crop_maize_1k_ton_mean2)


# In[ ]:


# World map for soybean
#Zoom in and pan for better visualization and actual yield
#Mouse over to get actual yield in 1000/ton

yield_map(crop_soybean_1k_ton_mean2)


# In[ ]:


#Save map
os.makedirs('/Users/paul.adunola/Desktop', exist_ok=True)
yield_map(crop_rice_1k_ton_mean2).save("/Users/paul.adunola/Desktop/Rice_yield.html")
yield_map(crop_wheat_1k_ton_mean2).save("/Users/paul.adunola/Desktop/Wheat_yield.html")
yield_map(crop_maize_1k_ton_mean2).save("/Users/paul.adunola/Desktop/Maize_yield.html")
yield_map(crop_soybean_1k_ton_mean2).save("/Users/paul.adunola/Desktop/Soybean_yield.html")
