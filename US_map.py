
def get_capitals():
    import pandas as pd
    import numpy as np
    
    capitals = pd.read_csv(r'C:\Users\mark\Documents\L4-Project-Data\us-state-capitals.csv')
    capitals_numpy = np.array([[capitals['LAT'], capitals['LONG'], capitals['CAPITAL']]])
    coords = capitals_numpy.transpose()
    return coords, capitals


def draw_map(perc_loss):
    import geopandas
    import pandas as pd
    
    # Load the shape file using geopandas
    geo_usa = geopandas.read_file(r'C:\Users\mark\Documents\L4-Project-Data\cb_2018_us_state_20m\cb_2018_us_state_20m.shp')
    #geo_cut = geo_usa.drop([7,25,48]) # Cut rubbish states
    
    # Add % loss to capitals dataframe
    coords, capitals = get_capitals()
    capitals['PERC_LOSS'] = pd.Series(perc_loss)
    
    # Merge usa_state data and geo_usa shapefile
    geo_merge=geo_usa.merge(capitals,on='NAME')

    return(geo_merge)


# geo_merge = draw_map(1) 
# geo_merge.plot(column='PERC_LOSS',figsize=(25, 15),legend=True, cmap='Blues')
# plt.xlim(-130,-60)
# plt.ylim(20,55)
# for i in range(len(geo_merge)):
#     plt.text(geo_merge.LONG[i],geo_merge.LAT[i],"{}\n{}".format(geo_merge.NAME[i],geo_merge.PERC_LOSS[i]),size=10)
# plt.title('% Loss due to cloud cover',fontsize=25)
# plt.show()