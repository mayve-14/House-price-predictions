train_data <- train
#Data exploration and preprocessing
head(train_data)
test_data<- test
sum(is.na(train_data))
sum(is.na(test_data))
summary(train_data)
summary(test_data)
library(skimr)
skim(train_data)
skim(test_data)
#drop columns with too many null values
train_data= subset(train_data, select = -c(Alley,FireplaceQu,PoolQC,Fence,MiscFeature,Utilities))
test_data=subset(test_data, select = -c(Alley,FireplaceQu,PoolQC,Fence,MiscFeature,Utilities))
#replace null values in other columns
train_data$LotFrontage <- ifelse(is.na(train_data$LotFrontage), 64, train_data$LotFrontage)
test_data$LotFrontage  <- ifelse(is.na(test_data$LotFrontage), 63, test_data$LotFrontage)
# Remove null  & NA values
train_data <- na.omit(train_data)
test_data <- na.omit(test_data)
#Visualization
library(tidyr)
library(tidyverse)
ggplot(data=train_data, aes(x=SalePrice)) +
geom_histogram(fill="steelblue", color="black") +
ggtitle("Histogram of Sale Price Values")
ggscatter(train_data, x = "TotRmsAbvGrd", y = "SalePrice", 
add = "reg.line", conf.int = TRUE, 
cor.coef = TRUE, cor.method = "pearson",
xlab = "  TotRmsAbvGrd  ", ylab = "SalePrice")
#categorical features handling
library(caret)
dmy <- dummyVars(" ~ .", data = train_data, fullRank = T)
train_data<- data.frame(predict(dmy, newdata = train_data))
dmytest <- dummyVars(" ~ .", data = test_data, fullRank = T)
test_data<- data.frame(predict(dmytest, newdata = test_data))
corelation<-cor(train_data)
train_data= subset(train_data, select = c(Id,OverallQual,YearBuilt,YearRemodAdd,TotalBsmtSF,X1stFlrSF,GrLivArea,FullBath,TotRmsAbvGrd,GarageCars,GarageArea,KitchenQualTA,TotRmsAbvGrd,SalePrice))
test_data= subset(test_data, select = c(Id,OverallQual,YearBuilt,YearRemodAdd,TotalBsmtSF,X1stFlrSF,GrLivArea,FullBath,TotRmsAbvGrd,GarageCars,GarageArea,KitchenQualTA,TotRmsAbvGrd))
corelation<-cor(train_data)
SalePrice<-train_data$SalePrice
# normalizing data
ss <- preProcess(as.data.frame(subset(train_data, select = -c(SalePrice,Id))), method=c("range"))
train_data <- predict(ss, as.data.frame(subset(train_data, select = -c(SalePrice,Id))))
test_data <- predict(ss, as.data.frame(test_data))
train_data$SalePrice <- SalePrice
# Splitting the dataset into the Training set and validation set
library(caTools)
set.seed(123)
split = sample.split(train_data$SalePrice, SplitRatio = 0.6)
training_set = subset(train_data, split == TRUE)
validation_set = subset(train_data, split == FALSE)
#Modeling
#  polynomial regression 
poly_model <- lm(SalePrice ~ poly(OverallQual+YearBuilt+YearRemodAdd+TotalBsmtSF+GrLivArea+FullBath+TotRmsAbvGrd+GarageCars,5),
                 data = training_set)
validation_poly<-predict(poly_model, data.frame(validation_set[-13]))
plot(validation_poly,validation_set$SalePrice)
# Evaluate the model on the test data
predictions <- predict(poly_model, newdata = validation_set)
R2=cor(validation_set$SalePrice,predictions)^2

RMSE <- sqrt(mean((predictions - validation_set$SalePrice)^2))
cat("Root Mean Squared Error (RMSE) on test data:", RMSE)
MAE <- mean(abs(predictions - validation_set$SalePrice))
cat("Mean Absolute Error (MAE) on validation data:", MAE)
MSE <- mean((predictions - validation_set$SalePrice)^2)
cat("Mean Squared Error (MSE) on validation data:", MSE)
# Make predictions
predictions <- poly_model %>% predict(test_data)
solution2 <- data.frame(Id =test_data$Id,SalePrice  = predictions)
write.csv(solution2, file = 'polynomial regression.csv', row.names = F)# error on kaggle 0.2


#install.packages("randomForest")
library(randomForest)
# Load the necessary libraries
library(caret)

# Create a Random Forest model
rf_model <- randomForest(SalePrice ~ ., data = train_data, ntree = 500)
# Evaluate the model on the test data
predictions <- predict(rf_model, newdata = validation_set)
R2_rf=cor(validation_set$SalePrice,predictions)^2
plot(predictions,validation_set$SalePrice)

RMSE <- sqrt(mean((predictions - validation_set$SalePrice)^2))
cat("Root Mean Squared Error (RMSE) on test data:", RMSE)
MAE <- mean(abs(predictions - validation_set$SalePrice))
cat("Mean Absolute Error (MAE) on validation data:", MAE)
MSE <- mean((predictions - validation_set$SalePrice)^2)
cat("Mean Squared Error (MSE) on validation data:", MSE)
# Generate predictions on the test data
predictions <- predict(rf_model, newdata = test_data)
# Combine the predictions with the test data
solution <- data.frame(Id = test_data$Id, SalePrice = predictions)
# Write the predictions to a CSV file
write.csv(solution, file = "randomforest.csv", row.names = FALSE)  # error on kaggle 0.15

