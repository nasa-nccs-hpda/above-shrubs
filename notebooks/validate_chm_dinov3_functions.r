# First, here's the parse function you need:
parse_source_filename <- function(filename) {
  basename <- basename(filename)
  parts <- strsplit(basename, "_")[[1]]
  
  sensor_lookup <- c(
    "WV01" = "Worldview-1", "WV02" = "Worldview-2", "WV03" = "Worldview-3", "WV04" = "Worldview-4",
    "GE01" = "GeoEye-1", "QB02" = "QuickBird", "IK01" = "IKONOS",
    "SPOT5" = "SPOT 5", "SPOT6" = "SPOT 6", "SPOT7" = "SPOT 7",
    "PHR1A" = "Pleiades-1A", "PHR1B" = "Pleiades-1B",
    "LG01" = "Legion-1", "LG02" = "Legion-2", "LG03" = "Legion-3", 
    "LG04" = "Legion-4", "LG05" = "Legion-5", "LG06" = "Legion-6"
  )
  
  sensor_code <- parts[1]
  sensor_name <- sensor_lookup[sensor_code]
  if (is.na(sensor_name)) sensor_name <- sensor_code
  
  date_str <- parts[2]
  year <- substr(date_str, 1, 4)
  month <- as.integer(substr(date_str, 5, 6))
  day <- as.integer(substr(date_str, 7, 8))
  formatted_date <- paste0(month, "/", day, "/", year)
  date_obj <- as.Date(date_str, format = "%Y%m%d")
  
  return(list(
    sensor = sensor_name,
    sensor_code = sensor_code,
    date = formatted_date,
    date_obj = date_obj,
    year = as.integer(year),
    month = month,
    day = day
  ))
}

get_lvis_mosaic_for_catid <- function(catid, 
                                      val_csv_dir,
                                      lvis_grid_dir,
                                      crop_extent = NULL,
                                      target_crs = NULL,
                                      verbose = TRUE) {
  #' Create a mosaic of LVIS reference grids for a given catid
  #' 
  #' @param catid The catid value to search for in CSV filenames
  #' @param val_csv_dir Directory containing validation CSV files
  #' @param lvis_grid_dir Directory containing LVIS grid TIF files
  #' @param crop_extent Optional terra extent object to crop the mosaic
  #' @param target_crs Optional CRS to reproject the mosaic to
  #' @param verbose Logical, print debug information
  #' @return A terra SpatRaster object (mosaic of LVIS grids), or NULL if none found
  
  library(terra)
  library(dplyr)
  
  if (verbose) {
    cat("\n=== LVIS Mosaic Creation ===\n")
    cat("Searching for catid:", catid, "in", val_csv_dir, "\n")
  }
  
  # Find all CSV files containing the catid
  csv_files <- list.files(val_csv_dir, pattern = catid, full.names = TRUE, recursive = FALSE)
  
  if (length(csv_files) == 0) {
    if (verbose) cat("No validation CSV files found for catid:", catid, "\n")
    return(NULL)
  }
  
  if (verbose) {
    cat("Found", length(csv_files), "validation CSV files\n")
  }
  
  # Extract unique LVIS flightline identifiers from filenames
  lvis_ids <- c()
  
  for (csv_file in csv_files) {
    csv_basename <- basename(csv_file)
    
    # Split filename by underscore
    # Format: WV03_20190730_M1BS_104001004FBF8F00-chm-dm-10m_LVISF2_ABoVE2019_0722_R2003_070510__val_1344.csv
    # We want: LVISF2_ABoVE2019_0722_R2003_070510
    
    # Find the LVISF2 part
    parts <- strsplit(csv_basename, "_")[[1]]
    
    # Find index where "LVISF2" starts
    lvis_start_idx <- which(parts == "LVISF2")
    
    if (length(lvis_start_idx) > 0) {
      # Extract LVISF2_ABoVE2019_0722_R2003_070510
      # This is 5 parts: LVISF2, ABoVE2019, 0722, R2003, 070510
      lvis_id <- paste(parts[lvis_start_idx:(lvis_start_idx + 4)], collapse = "_")
      lvis_ids <- c(lvis_ids, lvis_id)
    }
  }
  
  # Get unique LVIS IDs
  lvis_ids <- unique(lvis_ids)
  
  if (length(lvis_ids) == 0) {
    if (verbose) cat("No LVIS identifiers found in CSV filenames\n")
    return(NULL)
  }
  
  if (verbose) {
    cat("Found", length(lvis_ids), "unique LVIS flightlines:\n")
    for (id in lvis_ids) cat("  ", id, "\n")
  }
  
  # Find corresponding RH098 TIF files
  lvis_tif_paths <- c()
  
  for (lvis_id in lvis_ids) {
    # Construct expected filename: LVISF2_ABoVE2019_0722_R2003_070510_RH098_mean_30m.tif
    tif_name <- paste0(lvis_id, "_RH098_mean_30m.tif")
    tif_path <- file.path(lvis_grid_dir, tif_name)
    
    if (file.exists(tif_path)) {
      lvis_tif_paths <- c(lvis_tif_paths, tif_path)
      if (verbose) cat("Found LVIS grid:", tif_name, "\n")
    } else {
      if (verbose) cat("WARNING: LVIS grid not found:", tif_name, "\n")
    }
  }
  
  if (length(lvis_tif_paths) == 0) {
    if (verbose) cat("No LVIS grid TIF files found\n")
    return(NULL)
  }
  
  if (verbose) cat("Loading", length(lvis_tif_paths), "LVIS grid(s)...\n")
  
  # Load all LVIS rasters
  lvis_rasters <- lapply(lvis_tif_paths, rast)
  
  # Create mosaic if multiple grids
  if (length(lvis_rasters) == 1) {
    lvis_mosaic <- lvis_rasters[[1]]
    if (verbose) cat("Single LVIS grid loaded\n")
  } else {
    if (verbose) cat("Creating mosaic from", length(lvis_rasters), "grids...\n")
    lvis_mosaic <- do.call(mosaic, c(lvis_rasters, fun = "mean"))
    if (verbose) cat("Mosaic created\n")
  }
  
  if (verbose) {
    cat("Mosaic dimensions:", dim(lvis_mosaic)[1:2], "\n")
    cat("Mosaic extent:", as.vector(ext(lvis_mosaic)), "\n")
    cat("Mosaic CRS:", crs(lvis_mosaic, describe = TRUE)$name, "\n")
  }
  
  # Reproject to target CRS if specified
  if (!is.null(target_crs)) {
    if (verbose) cat("Reprojecting mosaic to target CRS...\n")
    lvis_mosaic <- project(lvis_mosaic, target_crs, method = "bilinear")
    if (verbose) cat("Reprojection complete\n")
  }
  
  # Crop to extent if specified
  if (!is.null(crop_extent)) {
    if (verbose) cat("Cropping mosaic to specified extent...\n")
    lvis_mosaic <- crop(lvis_mosaic, crop_extent, snap = "out")
    if (verbose) {
      cat("Cropped mosaic dimensions:", dim(lvis_mosaic)[1:2], "\n")
    }
  }
  
  if (verbose) cat("LVIS mosaic ready\n\n")
  
  return(lvis_mosaic)
}


# Now the main function - EXACTLY matching the structure from your attached file
map_raster_by_catid <- function(catid, 
                                chm_dir,
                                source_dir = NULL,
                                val_csv_dir = NULL,  # ADD THIS
                                lvis_grid_dir = NULL,  # ADD THIS
                                colorbar = "plasma",
                                fill_label = "Value",
                                plot_title = NULL,
                                SCALE_FACTOR = 1,
                                use_forest_ht_cmap = FALSE,
                                include_source = TRUE,
                                verbose = TRUE,
                                crop_center_lon = NULL,
                                crop_center_lat = NULL,
                                crop_width_pixels = NULL,
                                crop_height_pixels = NULL,
                                chm_max_value = 10,
                                edge_buffer = 1.15,
                                SHOW_REF = FALSE,
                                ref_gpkg_path = "/explore/nobackup/projects/above/misc/ABoVE_Shrubs/validation/chm/dinov3/4.3.2.5/dm_10m/val_intersect_footprints_LVIS_20260127.gpkg") {
  
  library(ggplot2)
  library(terra)
  library(sf)
  library(dplyr)
  library(patchwork)
  library(scales)
  
  has_ggspatial <- requireNamespace("ggspatial", quietly = TRUE)
  
  get_crs_name <- function(raster) {
    crs_str <- crs(raster, describe = TRUE)
    if (!is.null(crs_str$name) && crs_str$name != "") {
      return(crs_str$name)
    } else {
      proj_str <- as.character(crs(raster))
      if (grepl("UTM", proj_str, ignore.case = TRUE)) {
        zone <- sub(".*zone=([0-9]+).*", "\\1", proj_str)
        return(paste0("UTM Zone ", zone))
      } else if (grepl("4326", proj_str)) {
        return("WGS84")
      } else {
        return("Custom CRS")
      }
    }
  }
  
  forest_ht_colors <- c('#636363','#fc8d59','#fee08b','#ffffbf',
                        '#d9ef8b','#91cf60','#1a9850','#005a32')
  
  # Search for CHM file
  if (verbose) {
    cat("\n=== CHM Processing ===\n")
    cat("Searching for catid:", catid, "in", chm_dir, "\n")
  }
  
  chm_files <- list.files(chm_dir, pattern = catid, full.names = TRUE, recursive = FALSE)
  
  if (length(chm_files) == 0) {
    stop(paste("No CHM files found containing catid:", catid))
  }
  
  if (length(chm_files) > 1) {
    warning(paste("Multiple CHM files found for catid", catid, "- using the first one"))
    if (verbose) {
      cat("Found files:\n")
      for (f in chm_files) cat("  ", basename(f), "\n")
    }
  }
  
  raster_path <- chm_files[1]
  
  if (verbose) {
    cat("Using CHM file:", basename(raster_path), "\n")
  }
  
  # Parse raster filename
  parsed_filename <- parse_source_filename(basename(raster_path))
  
  # Load the CHM raster
  r <- rast(raster_path)
  
  if (verbose) {
    cat("CHM CRS:", get_crs_name(r), "\n")
    cat("CHM dimensions:", dim(r)[1:2], "\n")
  }
  
  chm_crs <- crs(r)
  display_ext <- NULL
  
  # Apply cropping if specified
  if (!is.null(crop_center_lon) && !is.null(crop_center_lat) && 
      !is.null(crop_width_pixels) && !is.null(crop_height_pixels)) {
    
    if (verbose) cat("Applying crop with buffer:", crop_width_pixels, "x", crop_height_pixels, "pixels\n")
    
    crop_center_pt <- vect(cbind(crop_center_lon, crop_center_lat), crs = "EPSG:4326")
    crop_center_native <- project(crop_center_pt, crs(r))
    crop_coords <- crds(crop_center_native)
    
    chm_res <- res(r)
    
    display_half_width <- (crop_width_pixels * chm_res[1]) / 2
    display_half_height <- (crop_height_pixels * chm_res[2]) / 2
    
    display_ext <- ext(crop_coords[1,1] - display_half_width,
                       crop_coords[1,1] + display_half_width,
                       crop_coords[1,2] - display_half_height,
                       crop_coords[1,2] + display_half_height)
    
    crop_half_width <- display_half_width * edge_buffer
    crop_half_height <- display_half_height * edge_buffer
    
    crop_ext <- ext(crop_coords[1,1] - crop_half_width,
                    crop_coords[1,1] + crop_half_width,
                    crop_coords[1,2] - crop_half_height,
                    crop_coords[1,2] + crop_half_height)
    
    if (verbose) {
      cat("Display extent:", as.vector(display_ext), "\n")
      cat("Crop extent (with buffer):", as.vector(crop_ext), "\n")
    }
    
    r <- crop(r, crop_ext)
    
    if (verbose) {
      cat("Cropped CHM dimensions:", dim(r)[1:2], "\n")
    }
  }
  
  if (SCALE_FACTOR != 1) {
    r <- r * SCALE_FACTOR
  }
  
  r_native <- r
  
  if (verbose) {
    cat("CHM extent:", paste(round(as.vector(ext(r_native)), 0), collapse = ", "), "\n")
  }
  
  raster_df <- as.data.frame(r_native, xy = TRUE, na.rm = TRUE)
  colnames(raster_df) <- c("x", "y", "value")
  raster_df$value <- pmin(raster_df$value, chm_max_value)
  
  if (!is.null(display_ext)) {
    xlim_display <- c(display_ext[1], display_ext[2])
    ylim_display <- c(display_ext[3], display_ext[4])
  } else {
    xlim_display <- range(raster_df$x, na.rm = TRUE)
    ylim_display <- range(raster_df$y, na.rm = TRUE)
  }
  
  if (is.null(plot_title)) {
    plot_title <- paste("Predicted vegetation height")
  }
  
  native_crs <- crs(r_native, proj = TRUE)
  
  # Create CHM plot
  p_chm <- ggplot() +
    geom_raster(data = raster_df, aes(x = x, y = y, fill = value))
  
  if (use_forest_ht_cmap) {
    p_chm <- p_chm + scale_fill_gradientn(colors = forest_ht_colors, 
                                          na.value = "transparent",
                                          limits = c(0, chm_max_value),
                                          oob = scales::squish)
  } else {
    p_chm <- p_chm + scale_fill_viridis_c(option = colorbar, 
                                          na.value = "transparent",
                                          limits = c(0, chm_max_value),
                                          oob = scales::squish)
  }
  
  p_chm <- p_chm +
    coord_sf(crs = native_crs,
             xlim = xlim_display,
             ylim = ylim_display,
             expand = FALSE,
             datum = sf::st_crs(4326))
  
  if (has_ggspatial) {
    p_chm <- p_chm +
      ggspatial::annotation_scale(location = "bl", width_hint = 0.3, 
                                   style = "bar", line_width = 1,
                                   height = unit(0.15, "cm"),
                                   text_cex = 0.8)
  }
  
  p_chm <- p_chm +
    labs(
      fill = fill_label,
      x = NULL,
      y = NULL,
      title = plot_title #, subtitle = catid
    ) +
    theme_bw() +
    theme(
      panel.grid = element_line(color = "gray90"),
      axis.text = element_text(size = 9),
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      aspect.ratio = diff(ylim_display) / diff(xlim_display),
        plot.caption = element_text(size = 6, color = "gray40")
      )
  
  # If include_source is FALSE or source_dir is NULL, return CHM only
  if (!include_source || is.null(source_dir)) {
    if (verbose) cat("\nReturning CHM only\n")
    return(p_chm)
  }
  
  # --- Create false color composite map ---
  
  if (verbose) cat("\n=== Source Processing ===\n")
  
  # Search for source file containing catid
  if (verbose) {
    cat("Searching for catid:", catid, "in", source_dir, "\n")
  }
  
  source_files <- list.files(source_dir, pattern = catid, full.names = TRUE, recursive = FALSE)
  
  if (length(source_files) == 0) {
    cat("WARNING: No source files found containing catid:", catid, "- returning CHM only\n\n")
    return(p_chm)
  }
  
  if (length(source_files) > 1) {
    warning(paste("Multiple source files found for catid", catid, "- using the first one"))
    if (verbose) {
      cat("Found files:\n")
      for (f in source_files) cat("  ", basename(f), "\n")
    }
  }
  
  source_path <- source_files[1]
  
  if (verbose) {
    cat("Using source file:", basename(source_path), "\n")
  }
  
  tryCatch({
    # Load source raster
    if (verbose) cat("Loading source raster...\n")
    r_source <- rast(source_path)
    
    if (verbose) {
      cat("Source CRS:", get_crs_name(r_source), "\n")
      cat("Source dimensions:", dim(r_source)[1:2], "\n")
      cat("Source bands:", nlyr(r_source), "\n")
    }
    
    same_crs <- compareGeom(r_source, r_native, crs = TRUE, ext = FALSE, stopOnError = FALSE)
    if (verbose) cat("Source and CHM in same CRS:", same_crs, "\n")
    
    if (verbose) cat("Cropping source to buffered CHM extent...\n")
    
    if (same_crs) {
      source_crop_ext <- ext(r_native)
      r_source_cropped <- crop(r_source, source_crop_ext, snap = "out")
    } else {
      # Transform CHM extent to source CRS
      chm_ext_buffered <- ext(r_native)
      chm_poly <- as.polygons(chm_ext_buffered, crs = chm_crs)
      chm_poly_source_crs <- project(chm_poly, crs(r_source))
      crop_extent <- ext(chm_poly_source_crs)
      
      if (verbose) cat("CHM extent in source CRS:", as.vector(crop_extent), "\n")
      
      r_source_cropped <- crop(r_source, crop_extent, snap = "out")
      
      if (verbose) cat("Reprojecting cropped source to CHM CRS...\n")
      r_source_cropped <- project(r_source_cropped, chm_crs, method = "bilinear")
    }
    
    if (verbose) {
      cat("Cropped source dimensions:", dim(r_source_cropped)[1:2], "\n")
    }
    
    # Get band names
    band_names <- names(r_source_cropped)
    nir_idx <- grep("BAND-N|NIR", band_names, ignore.case = TRUE)
    red_idx <- grep("BAND-R|RED", band_names, ignore.case = TRUE)
    green_idx <- grep("BAND-G|GREEN", band_names, ignore.case = TRUE)
    
    if (verbose) {
      cat("Using bands: NIR=", nir_idx[1], ", Red=", red_idx[1], ", Green=", green_idx[1], "\n", sep="")
    }
    
    if (verbose) cat("Extracting and processing bands...\n")
    nir_band <- r_source_cropped[[nir_idx[1]]]
    red_band <- r_source_cropped[[red_idx[1]]]
    green_band <- r_source_cropped[[green_idx[1]]]
    
    nir_band[nir_band < 0] <- NA
    red_band[red_band < 0] <- NA
    green_band[green_band < 0] <- NA
    
    stretch_band <- function(band) {
      vals <- values(band, na.rm = TRUE)
      if (length(vals) == 0 || all(is.na(vals))) return(band)
      q <- quantile(vals, probs = c(0.02, 0.98), na.rm = TRUE)
      band_stretched <- (band - q[1]) / (q[2] - q[1])
      band_stretched <- clamp(band_stretched, lower = 0, upper = 1)
      return(band_stretched)
    }
    
    if (verbose) cat("Applying stretch and creating RGB composite...\n")
    
    nir_stretched <- stretch_band(nir_band)
    red_stretched <- stretch_band(red_band)
    green_stretched <- stretch_band(green_band)
    
    nir_df <- as.data.frame(nir_stretched, xy = TRUE, na.rm = TRUE)
    red_df <- as.data.frame(red_stretched, xy = TRUE, na.rm = TRUE)
    green_df <- as.data.frame(green_stretched, xy = TRUE, na.rm = TRUE)
    
    colnames(nir_df) <- c("x", "y", "nir")
    colnames(red_df) <- c("x", "y", "red")
    colnames(green_df) <- c("x", "y", "green")
    
    rgb_df <- nir_df %>%
      inner_join(red_df, by = c("x", "y")) %>%
      inner_join(green_df, by = c("x", "y")) %>%
      filter(!is.na(nir), !is.na(red), !is.na(green)) %>%
      mutate(color = rgb(nir, red, green, maxColorValue = 1))
    
    if (verbose) {
      cat("Valid RGB pixels:", nrow(rgb_df), "\n")
    }
    
    if (verbose) cat("Creating false color composite plot...\n")
    
    p_fcc <- ggplot() +
      geom_raster(data = rgb_df, aes(x = x, y = y, fill = color)) +
      scale_fill_identity(na.value = "transparent") +
      coord_sf(crs = native_crs,
               xlim = xlim_display,
               ylim = ylim_display,
               expand = FALSE,
               datum = sf::st_crs(4326))
    
    if (has_ggspatial) {
      p_fcc <- p_fcc +
        ggspatial::annotation_scale(location = "bl", width_hint = 0.3, 
                                     style = "bar", line_width = 1,
                                     height = unit(0.15, "cm"),
                                     text_cex = 0.8)
    }
    
    p_fcc <- p_fcc +
      labs(
        x = NULL,
        y = NULL,
        title = paste("Commercial VHR surface reflectance"), 
          subtitle = paste0(parsed_filename$sensor, ': ', parsed_filename$date_obj, ' ', catid)
      ) +
      theme_bw() +
      theme(
        panel.grid = element_line(color = "gray90"),
        axis.text = element_text(size = 9),
        axis.text.x = element_text(angle = 0, hjust = 0.5),
        aspect.ratio = diff(ylim_display) / diff(xlim_display),
        plot.caption = element_text(size = 6, color = "gray40")
      )


    if (verbose) cat("Combining plots...\n")
    # Combine plots side by side
    combined_plot <- p_fcc + p_chm + plot_layout(ncol = 2)
      
    # Check for LVIS reference
    p_lvis <- NULL
    if (SHOW_REF && !is.null(display_ext)) {
      if (verbose) cat("\n=== LVIS Reference CHM Processing ===\nval_csv_dir = ", val_csv_dir,
        "\nlvis_grid_dir = ", lvis_grid_dir)
      
      # Use get_lvis_mosaic_for_catid to find and mosaic LVIS grids
      r_lvis <- get_lvis_mosaic_for_catid(
        catid = catid,
        val_csv_dir = val_csv_dir,
        lvis_grid_dir = lvis_grid_dir,
        crop_extent = ext(r_native),
        target_crs = chm_crs,
        verbose = verbose
      )
      
      if (!is.null(r_lvis)) {
        # LVIS mosaic found and processed
        
        # Don't apply scale factor to LVIS
        # if (SCALE_FACTOR != 1) {
        #   r_lvis <- r_lvis * SCALE_FACTOR
        # }
        
        # Convert to dataframe
        lvis_df <- as.data.frame(r_lvis, xy = TRUE, na.rm = TRUE)
        colnames(lvis_df) <- c("x", "y", "value")
        lvis_df$value <- pmin(lvis_df$value, chm_max_value)
        
        if (verbose) cat("LVIS valid pixels:", nrow(lvis_df), "\n")
        
        # Create LVIS CHM plot
        p_lvis <- ggplot() +
          geom_raster(data = lvis_df, aes(x = x, y = y, fill = value))
        
        if (use_forest_ht_cmap) {
          p_lvis <- p_lvis + scale_fill_gradientn(colors = forest_ht_colors, 
                                                  na.value = "transparent",
                                                  limits = c(0, chm_max_value),
                                                  oob = scales::squish)
        } else {
          p_lvis <- p_lvis + scale_fill_viridis_c(option = colorbar, 
                                                  na.value = "transparent",
                                                  limits = c(0, chm_max_value),
                                                  oob = scales::squish)
        }
        
        p_lvis <- p_lvis +
          coord_sf(crs = native_crs,
                   xlim = xlim_display,
                   ylim = ylim_display,
                   expand = FALSE,
                   datum = sf::st_crs(4326))
        
        if (has_ggspatial) {
          p_lvis <- p_lvis +
            ggspatial::annotation_scale(location = "bl", width_hint = 0.3, 
                                         style = "bar", line_width = 1,
                                         height = unit(0.15, "cm"),
                                         text_cex = 0.8)
        }
        
        p_lvis <- p_lvis +
          labs(
            fill = fill_label,
            x = NULL,
            y = NULL,
            title = "Reference vegetation height", subtitle = 'airborne LIDAR: 30m LVIS mean RH98 (2017/2019)'
          ) +
          theme_bw() +
          theme(
            panel.grid = element_line(color = "gray90"),
            axis.text = element_text(size = 9),
            axis.text.x = element_text(angle = 0, hjust = 0.5),
            aspect.ratio = diff(ylim_display) / diff(xlim_display),
            plot.caption = element_text(size = 6, color = "gray40")
          )
        
        # Remove legend from predicted CHM for 3-panel layout
        p_chm <- p_chm + guides(fill = "none")
        
        if (verbose) cat("Returning 3-panel plot\n\n")
        return(p_fcc + p_chm + p_lvis + plot_layout(ncol = 3))
      } else {
        if (verbose) cat("No LVIS mosaic found for this catid\n")
      }
    }
    
    if (verbose) cat("SUCCESS - but there is no LVIS for this site\n\n")
    
    return(combined_plot)
    
  }, error = function(e) {
    cat("ERROR:", conditionMessage(e), "\n")
    cat("ERROR with LVIS for this site: returning combined_plot with no LVIS\n\n")
    return(combined_plot)
  })
}