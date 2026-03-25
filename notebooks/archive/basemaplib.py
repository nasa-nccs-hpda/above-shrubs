import folium

# Get a basemap
tiler_basemap_icesat2boreal = 'https://titiler.maap-project.org/mosaics/623f8f82-ffe7-4348-ab48-d920e4b34763/tiles/{z}/{x}/{y}@1x?rescale=0%2C30&bidx=1&colormap_name=inferno' # Height 2020 updated mask
tiler_basemap_googleterrain = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}'
tiler_basemap_gray =          'http://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'
tiler_basemap_image =         'https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
tiler_basemap_natgeo =        'https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}'
basemaps = {
   'Boreal Height' : folium.TileLayer(
    tiles = tiler_basemap_icesat2boreal,
    attr = 'MAAP',
    name = 'Boreal Height',
    overlay = False,
    control = True
   ),
   'Google Terrain' : folium.TileLayer(
    tiles = tiler_basemap_googleterrain,
    attr = 'Google',
    name = 'Google Terrain',
    overlay = False,
    control = True
   ),
    'basemap_gray' : folium.TileLayer(
        tiles=tiler_basemap_gray,
        opacity=1,
        name="ESRI gray",
        attr="MAAP",
        overlay=False
    ),
    'Imagery' : folium.TileLayer(
        tiles=tiler_basemap_image,
        opacity=1,
        name="ESRI imagery",
        attr="MAAP",
        overlay=False
    ),
    'ESRINatGeo' : folium.TileLayer(
    tiles=tiler_basemap_natgeo,
    opacity=1,
    name='ESRI Nat. Geo.',
    attr='ESRI',
    overlay=False
    )
}