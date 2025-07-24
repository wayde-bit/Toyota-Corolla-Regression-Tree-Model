# =============================================================================
# CSIS 657 - Toyota Corolla Price Prediction using Regression Trees
# Student: [Name]
# Date: [Date]
# Purpose: Build and evaluate regression tree models for Toyota Corolla pricing
# =============================================================================

# Load required libraries
library(readxl)
library(rpart)
library(rpart.plot)
library(dplyr)
library(ggplot2)

# Clear workspace
rm(list=ls())

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

# Load the Toyota Corolla dataset
toyota <- read_excel("~/Desktop/ToyotaCorolla.xlsx")

# Clean column names (replace spaces and hyphens with underscores)
names(toyota) <- gsub(" ", "_", names(toyota))
names(toyota) <- gsub("-", "_", names(toyota))

# Display basic dataset information
cat("Dataset dimensions:", dim(toyota), "\n")
cat("First few rows:\n")
print(head(toyota))

# =============================================================================
# DESCRIPTIVE STATISTICS AND VISUALIZATION
# =============================================================================

cat("\n=== DESCRIPTIVE STATISTICS FOR PRICE ===\n")
print(summary(toyota$Price))

# Additional descriptive statistics
price_stats <- data.frame(
  Mean = mean(toyota$Price, na.rm = TRUE),
  Std_Dev = sd(toyota$Price, na.rm = TRUE),
  Variance = var(toyota$Price, na.rm = TRUE),
  IQR = IQR(toyota$Price, na.rm = TRUE)
)
print(price_stats)

# Create and save boxplot
png("price_boxplot.png", width = 800, height = 600)
boxplot(toyota$Price, 
        main = "Toyota Corolla Price Distribution", 
        ylab = "Price (Euros)", 
        col = "lightblue",
        outline = TRUE)
dev.off()

# Display boxplot in console
boxplot(toyota$Price, 
        main = "Toyota Corolla Price Distribution", 
        ylab = "Price (Euros)", 
        col = "lightblue")

# =============================================================================
# DATA PREPARATION FOR MODELING
# =============================================================================

# Define required variables for modeling
vars_needed <- c("Price", "Age_08_04", "Fuel_Type", "KM", "HP", "Automatic", 
                 "Doors", "Quarterly_Tax", "Mfr_Guarantee", "Guarantee_Period", 
                 "Airco", "Automatic_airco", "CD_Player", "Powered_Windows", 
                 "Sport_Model", "Tow_Bar")

# Create modeling dataset with selected variables
model_data <- toyota %>% 
    select(all_of(vars_needed)) %>% 
    na.omit()

cat("\nModeling dataset dimensions:", dim(model_data), "\n")

# =============================================================================
# DATA PARTITIONING
# =============================================================================

# Set seed for reproducibility
set.seed(123)

# Create 60/40 train/test split
n <- nrow(model_data)
train_size <- floor(0.6 * n)
train_indices <- sample(seq_len(n), size = train_size)

train_data <- model_data[train_indices, ]
test_data <- model_data[-train_indices, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")

# =============================================================================
# REGRESSION TREE MODELING
# =============================================================================

# Model 1: Initial specifications (minbucket=5, maxdepth=20, cp=0.01)
cat("\n=== BUILDING REGRESSION TREE MODELS ===\n")

tree_model_1 <- rpart(Price ~ ., 
                      data = train_data, 
                      method = "anova",
                      control = rpart.control(minbucket = 5, 
                                              maxdepth = 20, 
                                              cp = 0.01))

pred_1 <- predict(tree_model_1, test_data)
rmse_1 <- sqrt(mean((test_data$Price - pred_1)^2))

# Model 2: Alternative settings (smaller cp, smaller minbucket)
tree_model_2 <- rpart(Price ~ ., 
                      data = train_data, 
                      method = "anova",
                      control = rpart.control(minbucket = 3, 
                                              maxdepth = 25, 
                                              cp = 0.005))

pred_2 <- predict(tree_model_2, test_data)
rmse_2 <- sqrt(mean((test_data$Price - pred_2)^2))

# Model 3: Conservative settings (larger minbucket, smaller maxdepth)
tree_model_3 <- rpart(Price ~ ., 
                      data = train_data, 
                      method = "anova",
                      control = rpart.control(minbucket = 8, 
                                              maxdepth = 15, 
                                              cp = 0.007))

pred_3 <- predict(tree_model_3, test_data)
rmse_3 <- sqrt(mean((test_data$Price - pred_3)^2))

# Model 4: Very flexible settings (smallest cp, highest maxdepth)
tree_model_4 <- rpart(Price ~ ., 
                      data = train_data, 
                      method = "anova",
                      control = rpart.control(minbucket = 4, 
                                              maxdepth = 30, 
                                              cp = 0.003))

pred_4 <- predict(tree_model_4, test_data)
rmse_4 <- sqrt(mean((test_data$Price - pred_4)^2))

# =============================================================================
# MODEL COMPARISON AND SELECTION
# =============================================================================

# Compare all models
rmse_values <- c(rmse_1, rmse_2, rmse_3, rmse_4)
model_names <- c("Model 1 (Initial)", "Model 2 (Flexible)", 
                 "Model 3 (Conservative)", "Model 4 (Very Flexible)")

# Create comparison table
model_comparison <- data.frame(
  Model = model_names,
  RMSE = round(rmse_values, 2),
  stringsAsFactors = FALSE
)

cat("\n=== MODEL COMPARISON ===\n")
print(model_comparison)

# Select best model
best_model_index <- which.min(rmse_values)
best_rmse <- min(rmse_values)

# Assign best model
model_list <- list(tree_model_1, tree_model_2, tree_model_3, tree_model_4)
best_tree_model <- model_list[[best_model_index]]

cat("\nBest model:", model_names[best_model_index], "\n")
cat("Best RMSE:", round(best_rmse, 2), "\n")

# =============================================================================
# BEST MODEL ANALYSIS AND VISUALIZATION
# =============================================================================

cat("\n=== BEST MODEL DETAILS ===\n")
print(best_tree_model)

# Create and save tree visualization
png("best_tree_model.png", width = 1200, height = 800)
rpart.plot(best_tree_model, 
           main = "Best Toyota Corolla Price Prediction Tree", 
           type = 4, 
           extra = 101, 
           under = TRUE,
           cex = 0.8)
dev.off()

# Display tree in console
rpart.plot(best_tree_model, 
           main = "Best Toyota Corolla Price Prediction Tree", 
           type = 4, 
           extra = 101, 
           under = TRUE)

# =============================================================================
# FINAL RESULTS
# =============================================================================

cat("\n=== FINAL RESULTS ===\n")
cat("Final RMSE on holdout set:", round(best_rmse, 2), "\n")
cat("Model performance relative to price variance:\n")
cat("Price standard deviation:", round(sd(test_data$Price), 2), "\n")
cat("RMSE as % of std dev:", round((best_rmse / sd(test_data$Price)) * 100, 1), "%\n")

# Variable importance (if available)
if(length(best_tree_model$variable.importance) > 0) {
  cat("\nVariable Importance:\n")
  print(round(best_tree_model$variable.importance, 2))
}

cat("\n=== ANALYSIS COMPLETE ===\n")
