# ------------------------------------------------------------
# 0) LIBRARIES
# ------------------------------------------------------------
library(dplyr)
library(sf)
library(INLA)
library(spdep)

# ------------------------------------------------------------
# 1) DATA (RAW)
# ------------------------------------------------------------
obs_exp <- read.csv("D:/data.csv")

obs_exp$DANE <- as.character(obs_exp$DANE)
obs_exp$period <- as.character(obs_exp$period)

# ------------------------------------------------------------
# 2) SHAPEFILE AND GRAPH (ORIGINAL)
# ------------------------------------------------------------
shp <- st_read("D:/map/Col.shp",
               quiet = TRUE)
shp <- st_make_valid(shp)
shp$DANE <- as.character(shp$DANE)

# ORDERING (CRITICAL)
shp <- shp %>% arrange(DANE)

# ------------------------------------------------------------
# 3) MAKE SHAPEFILE AND DATA CONSISTENT
# ------------------------------------------------------------
# Valid intersection
valid_danes <- intersect(shp$DANE, unique(obs_exp$DANE))
cat("Valid municipalities:", length(valid_danes), "\n")

# Filter shapefile
shp2 <- shp %>%
  filter(DANE %in% valid_danes) %>%
  arrange(DANE)

nb <- poly2nb(shp2, queen = TRUE)
adj_file <- tempfile(fileext = ".adj")
nb2INLA(adj_file, nb)
g <- inla.read.graph(adj_file)

# Create consistent spatial index
shp2 <- shp2 %>%
  mutate(spatial_idx = row_number())

# Filter data
obs_exp_filtered <- obs_exp %>%
  filter(DANE %in% valid_danes)

# ------------------------------------------------------------
# 4) BUILD MUNICIPALITY–PERIOD DATASET
# ------------------------------------------------------------
model_data <- obs_exp_filtered %>%
  left_join(
    shp2 %>% st_drop_geometry() %>% select(DANE, spatial_idx),
    by = "DANE"
  ) %>%
  arrange(DANE, period)

cat("NAs in spatial_idx:", sum(is.na(model_data$spatial_idx)), "\n")
stopifnot(sum(is.na(model_data$spatial_idx)) == 0)
stopifnot(length(unique(model_data$spatial_idx)) == g$n)

# ------------------------------------------------------------
# 5) TEMPORAL INDICES AND INTERACTION
# ------------------------------------------------------------
# Temporal index
model_data <- model_data %>%
  mutate(
    time_idx = as.numeric(as.factor(period))
  )

# IID spatial index
model_data <- model_data %>%
  mutate(
    spatial_idx_iid = spatial_idx
  )

# Space-time interaction
model_data <- model_data %>%
  mutate(
    interaction_idx = interaction(spatial_idx, time_idx, drop = TRUE) %>%
      as.numeric()
  )

# Offset
model_data <- model_data %>%
  mutate(
    log_E = log(exp_rural + 0.001)
  )

# Summary
cat("Total rows:", nrow(model_data), "\n")
cat("Municipalities:", length(unique(model_data$spatial_idx)), "\n")
cat("Periods:", length(unique(model_data$time_idx)), "\n")

# ------------------------------------------------------------
# 6) SPATIO-TEMPORAL MODEL (BYM + TIME)
# ------------------------------------------------------------
formula_st <- cases_rural ~ 1 +
  
  # 🔹 Structured spatial effect (ICAR)
  f(spatial_idx,
    model = "besag",
    graph = g,
    scale.model = TRUE,
    hyper = list(
      prec = list(prior = "loggamma", param = c(1, 0.01))
    )
  ) +
  
  # 🔹 Unstructured spatial effect
  f(spatial_idx_iid,
    model = "iid",
    hyper = list(
      prec = list(prior = "loggamma", param = c(1, 0.01))
    )
  ) +
  
  # 🔹 Temporal effect (RW1)
  f(time_idx,
    model = "rw1",
    hyper = list(
      prec = list(prior = "loggamma", param = c(1, 0.01))
    )
  ) +
  
  # 🔹 Space-time interaction
  f(interaction_idx,
    model = "iid",
    hyper = list(
      prec = list(prior = "loggamma", param = c(1, 0.01))
    )
  )

# ------------------------------------------------------------
# 7) MODEL FITTING
# ------------------------------------------------------------
fit_st <- inla(
  formula = formula_st,
  family = "nbinomial",
  data = model_data,
  offset = log_E,
  control.predictor = list(compute = TRUE),
  control.compute = list(
    dic = TRUE,
    waic = TRUE,
    config = TRUE
  ),
  verbose = FALSE
)

summary(fit_st)

# ------------------------------------------------------------
# 8) POSTERIOR SIR
# ------------------------------------------------------------
lp <- fit_st$summary.linear.predictor

# Check column names
print(names(lp))

# SIR = exp(eta) = exp(log(mu) - log(E))
model_data$SIR_mean <- exp(lp$mean - model_data$log_E)
model_data$SIR_lwr95 <- exp(lp$`0.025quant` - model_data$log_E)
model_data$SIR_upr95 <- exp(lp$`0.975quant` - model_data$log_E)

# Excess probability (approximation)
model_data$excess <- as.integer(model_data$SIR_lwr95 > 1)

# Summary
summary(model_data$SIR_mean)
str(model_data)

# ------------------------------------------------------------
# 9) MERGE RESULTS BACK TO ORIGINAL DATA
# ------------------------------------------------------------

# Select only the new variables to add
results_to_add <- model_data %>%
  select(DANE_Year, 
         spatial_idx,
         time_idx,
         spatial_idx_iid,
         interaction_idx,
         log_E,
         SIR_mean,
         SIR_lwr95,
         SIR_upr95,
         excess)

# Join results with original data using existing DANE_Year
obs_exp_updated <- obs_exp %>%
  left_join(results_to_add, by = "DANE_Year")

# Check join
cat("Original rows:", nrow(obs_exp), "\n")
cat("Updated rows:", nrow(obs_exp_updated), "\n")
cat("Matches:", sum(!is.na(obs_exp_updated$SIR_mean)), "\n")

# ------------------------------------------------------------
# 10) SAVE UPDATED DATA
# ------------------------------------------------------------
write.csv(obs_exp_updated, 
          "D:/data.csv", 
          row.names = FALSE)

cat("Results saved to data.csv with new variables:\n")
cat("- spatial_idx\n")
cat("- time_idx\n")
cat("- spatial_idx_iid\n")
cat("- interaction_idx\n")
cat("- log_E\n")
cat("- SIR_mean\n")
cat("- SIR_lwr95\n")
cat("- SIR_upr95\n")
cat("- excess\n")