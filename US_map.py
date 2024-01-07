
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
