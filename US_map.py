
def county_map():
    import geopandas
    
    # Load the shape file using geopandas
    geo_usa = geopandas.read_file(r'C:\Users\mark\OneDrive - Durham University\L4 Project\L4-Project-Data\Grid\cb_2018_us_county_500k\cb_2018_us_county_500k.shp')

    # Add % loss to capitals dataframe
    geo_usa['LAT'] = geo_usa.centroid.map(lambda p: p.y)
    geo_usa['LONG'] = geo_usa.centroid.map(lambda p: p.x)
    
    # Select states
    #df_new = geo_usa[(geo_usa['STATEFP'] == '06')]
    
    df_new = geo_usa[(geo_usa['STATEFP'] == '06') | # CA
                      (geo_usa['STATEFP'] == '32') | # NV
                      (geo_usa['STATEFP'] == '04') | # AR
                      (geo_usa['STATEFP'] == '49') | # UT
                      (geo_usa['STATEFP'] == '08') | # CO
                      (geo_usa['STATEFP'] == '35')]  # NM

    return(df_new)


def state_map():
    import geopandas
    
    # Load the shape file using geopandas
    geo_usa = geopandas.read_file(r'C:\Users\mark\OneDrive - Durham University\L4 Project\L4-Project-Data\Grid\cb_2018_us_state_500k\cb_2018_us_state_500k.shp', index_col=['STATEFP'], usecols=['geometry'])
    geo_usa.index = geo_usa.STATEFP
    
    df_new = geo_usa.drop(index = ['02', '11', '15', '72', '78', '60', '66', '69'])
    # Alaska 02, Delaware 10, DC 11, Hawaii 15, New Hampshire 33, Puerto Rico 72, Virgin Islands 78


    return(df_new)

