# L4-Project

Some required data stored in the Data folder, including:
    One NSRDB/FARMS file (Camp_Fire),
    Spectral response curves (for 8 materials),
    Spec sheet data (for standard 5 materials),
    US State census map .shp for plotting maps,
    Some old (and redundant) file opening tools in open_data,
    
FARMS data used for states and hexagons was too big to be stored in the GitHub
repo so is left out and must be downloaded separately. The supplied NSRDB_API
in the Functions folder provides a semi-automated method for this. The API
queues requests (and reattempts on failures) for FARMS data. Sadly NSRDB then
send a download link via email which adds manual tedium.

###

There are 3 main scripts available:

spectral_efficiency_hexagons computes everything from the project report. The
API has a script currently setup to download data as in the report (~600 files)
which then have to be ran through Rename_NSRDB_Files to allow direct use in
spectral_efficiency_hexagons.

spectral_efficiency_single demonstrates similar at a single location (Camp
Fire 2018) so should be instantly runnable.

spectral_efficiency_states is an earlier iteration of hexagons which uses ~50
locations instead (note - requires tweeking of API script).

###

The validation scripts are also provided, but are very rough so will not work
without significant tweeks and getting the relevant data from NREL and Sandia
(see project report). Following Sandia's procedures is a good starting point
for you own validation:
https://pvpmc.sandia.gov/model-validation/model-validation-procedure/

Other used functions:
    1) spec_response_function calculates the spectral mismatch from a FARMS csv
       for 8 materials.
    2) US_map handles the creation of geopandas coordinates from the census
       files.
    
Other useful scripts:
    1) plot_spectral_response demonstrates briefly the above 1)
    2) hexagons_plot demonstrates briefly the above 2)
    
