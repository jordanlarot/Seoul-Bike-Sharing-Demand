# Load libraries
library(readxl)
library(lubridate) 
library(dplyr)     
library(ggplot2)
library(Metrics)
library(boot)
library(xgboost)
library(randomForest)
library(caret)
library(rpart)

# Specify data url 
DATA_URL <- 'https://raw.githubusercontent.com/jordanlarot/Seoul-Bike-Sharing-Demand/main/data.csv'

# Load data 
df <- read.csv(DATA_URL)

### Exploratory Data Analysis ####

# View structure of the data
str(df)

# Convert columns to the correct data type
df <- df %>%
  mutate(
    Date = dmy(Date)  
  )
df$Seasons <- as.factor(df$Seasons)
df$Holiday <- as.factor(df$Holiday)
df$Functioning.Day <- as.factor(df$Functioning.Day)

# Summary statistics 
summary(df)

# Daily Rented Bike Count
ggplot(df, aes(x=Date, y=Rented.Bike.Count)) + 
  geom_line() + 
  theme_minimal() + 
  labs(title="Daily Rented Bike Count",
       x="Date",
       y="Rented Bike Count")

# Boxplot of Rented Bike Count
boxplot(df$`Rented.Bike.Count`, 
        main = "Boxplot of Rented Bike Count", 
        ylab = "Rented Bike Count", 
        col = "grey", 
        border = "black")

# Countplot by Season
ggplot(df, aes(x=Seasons, fill=Seasons)) +  
  geom_bar(colour="black") +  
  theme_minimal() +
  theme(
  ) +
  labs(title="Countplot by Season",
       x="Season",
       y="Count") +
  scale_x_discrete(limits=c("Spring", "Summer", "Autumn", "Winter")) + 
  scale_fill_manual(values=c("Spring"="green", "Summer"="yellow", "Autumn"="orange", "Winter"="blue"))


# Total Number of Bikes Rented Per Hour
ggplot(df, aes(x = as.factor(Hour), y = Rented.Bike.Count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Total Number of Bikes Rented Per Hour",
       x = "Hour of the Day",
       y = "Number of Bikes Rented") +
  theme_minimal()

### Feature Engineering ####

# Remove days that are Non-functioning
df <- df %>%
  filter(Functioning.Day != 'No')

# Drop Functioning.Day since all days rows have 'Yes'
df <- select(df, -Functioning.Day)

# Extract day of the week and month from 'Date'
df <- df %>%
  mutate(
    
    # Temporal features
    Day.of.Week = as.factor(weekdays(Date)),
    Month = as.factor(month(Date)),
    weekend = as.factor(ifelse(Day.of.Week %in% c("Saturday", "Sunday"), 1, 0)),
    
    # Weather features
    is_rainy = Rainfall > 0,  
    is_snowy = Snowfall > 0,
    is_rainy = ifelse(is_rainy, 1, 0), 
    is_snowy = ifelse(is_snowy, 1, 0),
    weather_condition = as.factor(case_when(
      Temperature < 5 ~ "Cold",  # Temperatures below 41°F are 'Cold'
      Temperature >= 5 & Temperature <= 25 ~ "Mild",  # Between 41°F and 77°F is 'Mild'
      Temperature > 25 ~ "Hot"  # Above 77°F is 'Hot'
    ))
)

### Data Preprocessing ####

# Set seed
set.seed(42)

# Specify training size
training_size <- 0.8

# Generate indexes for train/test split
index <- sample(nrow(df),nrow(df) * training_size)

# Split data into train and test sets
df_train <- df[index, ]
df_test <- df[-index, ]

# Drop 'Date'
df_train <- select(df_train, -Date)
df_test <- select(df_test, -Date)

### Linear Regression (Baseline) ####

# Set starting points
nullmodel <- lm(Rented.Bike.Count ~ 1, 
                data = df_train)
fullmodel <- lm(Rented.Bike.Count ~ ., 
                data = df_train)

# Perform background elimination
model.step <- step(fullmodel, 
                   direction="backward")

# Create best linear model
lin_reg <- lm(Rented.Bike.Count ~ 
                    Hour + 
                    Temperature + 
                    Humidity + 
                    Wind.speed + 
                    Visibility + 
                    Solar.Radiation + 
                    Rainfall + 
                    Snowfall + 
                    Holiday + 
                    Day.of.Week + 
                    Month + 
                    is_rainy + 
                    is_snowy + 
                    weather_condition,
                  data=df_train)

# View model summary
lin_reg_summary <- summary(lin_reg)
lin_reg_summary # Adjusted R-squared: 0.6052

# Make predictions on testing set 
predictions_lin_reg <- predict(lin_reg, df_test)

# Calculate MSE 
final_model.mse <- mse(actual = df_test$Rented.Bike.Count, 
                       predicted = predictions_lin_reg)
final_model.mse #149882

# Calculate RMSE
final_model.rmse <- rmse(actual = df_test$Rented.Bike.Count, 
                         predicted = predictions_lin_reg)
final_model.rmse #387.146

# Calculate MAE
final_model.mae <- mae(actual = df_test$Rented.Bike.Count, 
                       predicted = predictions_lin_reg)
final_model.mae #297.3499

### Decision Tree ####

# Train decision tree
dt_model <- rpart(Rented.Bike.Count ~ ., 
                  df_train)

# Print cp table 
printcp(dt_model)

# Prune tree
dt_model_pruned <- prune(dt_model, 
                         cp = 0.0175)

# Get variable importance
importance <- as.data.frame(dt_model_pruned$variable.importance)

# Name columns
importance$Feature <- rownames(importance)
colnames(importance)[1] <- "Importance"

# Order the data by importance
importance <- importance[order(-importance$Importance),]

# View feature importance
importance # remove features below 106797128 threshold

# Train new decision tree model
dt_model_final <- rpart(Rented.Bike.Count ~
                          Hour +
                          Month + 
                          Dew.point.temperature +
                          weather_condition +
                          Seasons +
                          Humidity +
                          Solar.Radiation +
                          Rainfall, 
                        df_train)


# out of sample prediction 
dt_pred <- predict(dt_model_final, 
                   df_test)

# Calculate RMSE
dt.rmse <- rmse(actual = df_test$Rented.Bike.Count, 
                predicted = dt_pred)
dt.rmse #362.3567

# Calculate MAE
dt.mse <- mse(actual = df_test$Rented.Bike.Count, 
              predicted = dt_pred)
dt.mse #131302.3

### Random Forest ####

# Train a base RandomForest model; note: default mtry = p/3, nodesize=5, ntree=500
rf<- randomForest(Rented.Bike.Count ~., 
                  data=df_train, 
                  do.trace=50, 
                  keep.inbag=TRUE)

# Get MSE from our model
mse_data <- data.frame(Trees= 1:rf$ntree, MSE=rf$mse)

# Create plot of MSE to see where MSE stabilizes to determine optimal number of trees to use
mse_plot <- ggplot(mse_data, aes(x = Trees, y=MSE)) +
  geom_line()+
  ggtitle ("MSE with different 'ntree' values") +
  xlab ("Number of Trees") +
  ylab ("Mean Squared Error")
mse_plot

# Try RandomForest with different number of trees 
rf_1<- randomForest(Rented.Bike.Count ~., 
                    data=df_train, 
                    ntree=1000, 
                    do.trace= 100, 
                    keep.inbag = TRUE)
rf_1

# Get MSE from our model
mse_data_1 <- data.frame(Trees= 1:rf_1$ntree, MSE=rf_1$mse)

mse_plot_1 <- ggplot(mse_data_1, aes(x = Trees, y=MSE)) +
  geom_line()+
  ggtitle ("MSE with different 'ntree' values") +
  xlab ("Number of Trees") +
  ylab ("Mean Squared Error")
mse_plot_1
# MSE seems to stabilize around 500; there is minimal decrease in MSE beyond 
# Find the optimal number of variables at each internal node in the tree 

# Select mtry with minimum OOB error 
mtry <- tuneRF(df_train[-1],df_train$Rented.Bike.Count, ntreeTry=500,
               stepFactor=2,improve=0.01, trace=TRUE, plot=TRUE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m) # the optimal mtry= 10 

# Build model using optimal ntree and mtry
rf_best <- randomForest(Rented.Bike.Count ~ .,
                        data=df_train, 
                        ntree=500, 
                        mtry=best.m, 
                        importance=TRUE)

# see which variables are more important 
importance(rf_best) # Hour, temperature, and humidity are the top predictors
varImpPlot(rf_best)

# Build model using optimal ntree and mtry dropping the three variables less than 2000000
rf_best <- randomForest(Rented.Bike.Count ~
                          Hour +
                          Temperature +
                          Month +
                          Humidity +
                          Solar.Radiation +
                          weather_condition +
                          Day.of.Week +
                          Rainfall +
                          is_rainy +
                          Seasons +
                          Dew.point.temperature +
                          weekend + 
                          Wind.speed + 
                          Visibility +
                          Holiday, 
                        data=df_train, 
                        ntree=500, 
                        mtry=best.m, 
                        importance= TRUE)


# Make predictions using best Random Forest model
predictions_rf <- predict(rf_best, 
                          df_test)

# Extract vector of actual values
actual_rf <- df_test$Rented.Bike.Count

# Calculate RMSE 
rmse_value_rf <- rmse(actual_rf, predictions_rf)
rmse_value_rf #152.667

# Calculate MSE
mse_value_rf <- mse(actual_rf, predictions_rf)
mse_value_rf #23307.2

# Calculate MAE
mae_value_rf <- mae(actual_rf, predictions_rf)
mae_value_rf #94.18526

# see which variables are more important 
importance(rf_best) 

# Hour, Temperature, Month, Humidity, Solar.Radiation were the most important features that 
# impacted Rented Bike Count

### XGBoost ####
set.seed(42)

# Convert factor features to numeric (Seasons, Holiday, Day.of.Week, Month, weekend, weather_condition)
df_train$Seasons <- as.numeric(factor(df_train$Seasons))
df_train$Holiday <- as.numeric(factor(df_train$Holiday))
df_train$Day.of.Week <- as.numeric(factor(df_train$Day.of.Week))
df_train$Month <- as.numeric(factor(df_train$Month))
df_train$weekend <- as.numeric(factor(df_train$weekend))
df_train$weather_condition <- as.numeric(factor(df_train$weather_condition))
df_test$Seasons <- as.numeric(factor(df_test$Seasons))
df_test$Holiday <- as.numeric(factor(df_test$Holiday))
df_test$Day.of.Week <- as.numeric(factor(df_test$Day.of.Week))
df_test$Month <- as.numeric(factor(df_test$Month))
df_test$weekend <- as.numeric(factor(df_test$weekend))
df_test$weather_condition <- as.numeric(factor(df_test$weather_condition))

# Create numeric matrix
train_feature_matrix <- as.matrix(df_train[, -which(names(df_train) == "Rented.Bike.Count")])
train_target_matrix <- as.matrix(df_train$Rented.Bike.Count)
test_feature_matrix <- as.matrix(df_test[, -which(names(df_test) == "Rented.Bike.Count")])
test_target_matrix <- as.matrix(df_test$Rented.Bike.Count)

# Define parameters
params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",  # or "binary:logistic" for classification
  eta = 0.05,
  max_depth = 8,
  min_child_weight = 4,
  subsample = 0.9,
  colsample_bytree = 0.9,
  alpha=0.1,
  gamma=0.2,
  lambda=1
)

# Number of boosting rounds
nrounds <- 500

# Train the model using variable names
xgboost_model <- xgboost(data = train_feature_matrix, 
                         label = train_target_matrix, 
                         nrounds = nrounds, 
                         params = params, 
                         verbose = 0) 

# Make predictions using test feature matrix
predictions_xg <- predict(xgboost_model, 
                          newdata = test_feature_matrix)

# Calculate RMSE 
rmse_value_xg <- rmse(test_target_matrix, predictions_xg)
rmse_value_xg #135.9434

# Calculate MSE
mse_value_xg <- mse(test_target_matrix, predictions_xg)
mse_value_xg #18480.61

# Calculate MAE
mae_value_xg <- mae(test_target_matrix, predictions_xg)
mae_value_xg #77.58121

# Calculate feature importance
importance_matrix <- xgb.importance(feature_names = colnames(train_feature_matrix[-train_target_matrix]), 
                                    model = xgboost_model)

# Print the feature importance
importance_matrix

# Drop is_rainy, Snowfall, is_snowy 
# because they are substantially less important based upon the importance matrix 

# Specify columns to drop
cols_to_drop <- c("is_snowy", "is_rainy", "Snowfall")

# Drop columns from df_train and df_test
df_train <- df_train[, !(names(df_train) %in% cols_to_drop)]
df_test <- df_test[, !(names(df_test) %in% cols_to_drop)]

# Create numeric matrix
train_feature_matrix <- as.matrix(df_train[, -which(names(df_train) == "Rented.Bike.Count")])
train_target_matrix <- as.matrix(df_train$Rented.Bike.Count)
test_feature_matrix <- as.matrix(df_test[, -which(names(df_test) == "Rented.Bike.Count")])
test_target_matrix <- as.matrix(df_test$Rented.Bike.Count)

# Define parameters
params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",  
  eta = 0.025,
  max_depth = 8,
  min_child_weight = 3,
  subsample = 0.85,
  colsample_bytree = 0.9,
  alpha=0.1,
  gamma=0.3,
  lambda=1
)

# Number of boosting rounds
nrounds <- 2500

# Train the model using variable names
xgboost_best <- xgboost(data = train_feature_matrix, 
                         label = train_target_matrix, 
                         nrounds = nrounds, 
                         params = params, 
                         verbose = 0) 

# Make predictions using test feature matrix
predictions_xg_best <- predict(xgboost_best, 
                          newdata = test_feature_matrix)

# Calculate RMSE 
rmse_value_xg <- rmse(test_target_matrix, predictions_xg_best)
rmse_value_xg #134.8099

# Calculate MSE
mse_value_xg <- mse(test_target_matrix, predictions_xg_best)
mse_value_xg #18173.71

# Calculate MAE
mae_value_xg <- mae(test_target_matrix, predictions_xg_best)
mae_value_xg #76.80349

# Calculate feature importance
importance_matrix_best <- xgb.importance(feature_names = colnames(train_feature_matrix[-train_target_matrix]), 
                                    model = xgboost_best)

# Print the feature importance
importance_matrix_best

# Sort the dataframe by 'Gain' in descending order and select the top 7
top_features <- importance_matrix_best[order(-importance_matrix_best$Gain), ][1:7, ]

# Plot the top 7 features
ggplot(top_features, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", color = "black", fill = "steelblue") +
  labs(title = "Top 7 Most Important Features",
       x = "Feature",
       y = "Importance (Gain)") +
  theme_minimal() +
  scale_fill_gradient(low = "steelblue", high = "darkblue")

