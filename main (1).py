#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import urllib.request, json 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
from geopandas import GeoSeries
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
import time
import builtins

start = time.time()
print('Start')

# Корректируем функцию print, чтобы всегда писала время исполнения
def print(*args, **kwargs):
    if type(args) == tuple:
        builtins.print(format(int(time.time() - start), "3d") + 's: ' + ''.join(map(str, args)))
    elif type(args) == int:
        builtins.print(format(int(time.time() - start), "3d") + 's: ' + str(args))
    elif type(args) == int:
        builtins.print(format(int(time.time() - start), "3d") + 's: ' + args)
    

def plot_map(df_permits, df_police, df_permits_centers, df_police_centers):    
    # Скачаем очертания региона
    counties = gpd.read_file(
        'data/map/wgs84/townct_37800_0000_2010_s100_census_1_shp_wgs84.shp')
    # Подготовим точки
    geom_permits = [Point(xy) for xy in zip(df_permits['lg'], df_permits['lt'])]
    geom_permits_centers = [Point(xy) for xy in zip(df_permits_centers['lg'],
                                                        df_permits_centers['lt'])]
    geom_police = [Point(xy) for xy in zip(df_police['lg'], df_police['lt'])]
    geom_police_centers = [Point(xy) for xy in zip(df_police_centers['lg'],
                                                       df_police_centers['lt'])]
    
    
    gdf_permits = gpd.GeoDataFrame(df_permits, geometry=geom_permits, crs="EPSG:4326")
    gdf_permits_centers = gpd.GeoDataFrame(df_permits_centers,
                                               geometry=geom_permits_centers, crs="EPSG:4326")
    gdf_police = gpd.GeoDataFrame(df_police, geometry=geom_police, crs="EPSG:4326")
    gdf_police_centers = gpd.GeoDataFrame(df_police_centers,
                                              geometry=geom_police_centers, crs="EPSG:4326")
    city = counties[counties["NAME10"] == "Hartford"]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    city.plot(ax=ax, color='white', edgecolor='black')
    gdf_police.plot(ax=ax, marker='.', color='red', markersize=1)
    gdf_permits.plot(ax=ax, marker='.', color='blue', markersize=1)
    gdf_police_centers.plot(ax=ax, marker='o', color='red',
                                edgecolor='black', linewidth=2, markersize=500)
    gdf_permits_centers.plot(ax=ax, marker='o', color='blue',
                                 edgecolor='black', linewidth=2, markersize=500)
    
    plt.show()
    print('Plot - done')

def find_best_clusters(df, maximum_K):    
    clusters_centers = []
    k_values = []
    
    for k in range(1, maximum_K):        
        kmeans_model = KMeans(n_clusters = k)
        kmeans_model.fit(df)        
        clusters_centers.append(kmeans_model.inertia_)
        k_values.append(k)
    print('Best K - found')
    return clusters_centers, k_values

def generate_elbow_plot(clusters_centers, k_values):    
    figure = plt.subplots(figsize = (10, 10))
    plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Cluster Inertia")
    plt.title("Elbow Plot of KMeans")
    print('Elbow plot - done')
    plt.show()

# Скачиваем данные о разрешениях на строительство/продажу недвижимости
with urllib.request.urlopen(
    "https://data.hartford.gov/api/views/p2vw-4aab/rows.json?accessType=DOWNLOAD") as url:
    data = json.load(url)['data']
print('Building and Trades Permits - downloaded')

# Достаем из всех полей только номер дела, дату, широту и долготу. Все заворачиваем в список
datalist = []
for row in data:
    if float(row[24][1]) > 45:
        continue
    d = {}
    lt = float(row[24][1])
    lg = float(row[24][2])
    d['lt'] = lt
    d['lg'] = lg
    if not 38 < lt < 45 or not -75 < lg < -70:
        continue
    datalist.append(d)

# Получившийся список собираем в DataFrame pandas
df_permits = pd.DataFrame(data=datalist)
print('Building and Trades Permits dataframe - done')

# Аналогично - огромный json с записями о полицейских инцидентах
with urllib.request.urlopen("https://data.hartford.gov/api/views/889t-nwfu/rows.json?accessType=DOWNLOAD") as url:
    data = json.load(url)['data']
print('Police records - downloaded')

# Достаем из всех полей только номер дела, дату, широту и долготу. Все заворачиваем в список
datalist = []
for row in data:    
    d = {}
    lt = float(row[19][1])
    lg = float(row[19][2])
    d['lt'] = lt
    d['lg'] = lg
    if not 38 < lt < 45 or not -75 < lg < -70:
        continue
    datalist.append(d)

# Получившийся список собираем в DataFrame pandas
df_police = pd.DataFrame(data=datalist)
print('Police records dataframe - done')

# Сделаем кластеризацию k-means, чтобы найти 
# ключевые центры строительства/продажи недвижимости и криминала

# Сначала посмотрим, как от кол-ва кластеров зависит ошибка,
# чтобы выбрать этот параметр
#clusters_centers, k_values = find_best_clusters(df_permits, 12)
#generate_elbow_plot(clusters_centers, k_values)

#Видно, что почти гипербола. Пусть k = 6
kmeans_model = KMeans(n_clusters = 6)
kmeans_model.fit(df_permits)

# Припишем к каждой записи номер кластера
df_permits["clusters"] = kmeans_model.labels_
# На будущее сохраним центры кластеров
df_permits_centers = pd.DataFrame(data = kmeans_model.cluster_centers_)
df_permits_centers = df_permits_centers.rename(columns={0: "lt", 1: "lg"})
print('Permits clusters - found')

# То же самое для криминала
#clusters_centers, k_values = find_best_clusters(df_police, 12)
#generate_elbow_plot(clusters_centers, k_values)

#Видно, что почти гипербола. Пусть k = 7
kmeans_model = KMeans(n_clusters = 7)
kmeans_model.fit(df_police)

# Припишем к каждой записи номер кластера
df_police["clusters"] = kmeans_model.labels_
# На будущее сохраним центры кластеров
df_police_centers = pd.DataFrame(data = kmeans_model.cluster_centers_)
df_police_centers = df_police_centers.rename(columns={0: "lt", 1: "lg"})
print('Police records clusters - found')

# Посмотрим карту
plot_map(df_permits, df_police, df_permits_centers, df_police_centers)










