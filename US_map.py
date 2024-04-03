
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

def hexagons_map():

    import geopandas as gpd
    from shapely.ops import unary_union
    import h3pandas
    
    # Read shapefile
    gdf = gpd.read_file("C:/Users/mark/OneDrive - Durham University/L4 Project/L4-Project-Data/Grid/cb_2018_us_state_500k/cb_2018_us_state_500k.shp")
    
    # Get US without territories / Alaska + Hawaii
    usa = gdf[~gdf.NAME.isin(["Hawaii", "Alaska", "American Samoa", 
                             "United States Virgin Islands", "Guam",
                             "Commonwealth of the Northern Mariana Islands",
                             "Puerto Rico"])]
    
    # Convert to EPSG 4326 for compatibility with H3 Hexagons
    usa = usa.to_crs(epsg=4326)
    # Get union of the shape (whole US)
    union_poly = unary_union(usa.geometry)
    
    resolution = 3
    hexagons = usa.h3.polyfill_resample(resolution)
    
    # Centres
    hexagons['LAT'] = hexagons.centroid.map(lambda p: p.y)
    hexagons['LONG'] = hexagons.centroid.map(lambda p: p.x)

    return hexagons