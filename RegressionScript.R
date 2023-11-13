#Loading the packages
library(caret)
library(readr)
library(corrplot)
library(e1071)
library(gbm)
library(ggplot2)

#dumifying existing product
RawExistingProduct<- read.csv("existingproductattributes2017.csv")
tempExist <- dummyVars(" ~ .", data = RawExistingProduct)
existingProduct <- data.frame(predict(tempExist, newdata = RawExistingProduct))
existingProductB <- data.frame(predict(tempExist, newdata = RawExistingProduct))

#dumifying raw product
RawNewProduct <- read.csv("newproductattributes2017.csv")
tempNew <- dummyVars(" ~ .", data = RawNewProduct)
newProduct <- data.frame(predict(tempNew, newdata = RawNewProduct))

#removing nulls from existing product
str(existingProduct)
summary(existingProduct)
existingProduct$BestSellersRank<- NULL

#Alt where we keep best sellers rank
existingProductB$BestSellersRank[is.na(existingProductB$BestSellersRank)] <- mean(existingProductB$BestSellersRank, na.rm = TRUE)
existingProductB$BestSellersRank

#removing nulls from new product
str(newProduct)
summary(newProduct)
newProduct$BestSellersRank <- NULL

#We take a look at our correlation plot and will remove the data that we feel we do not need
corrData <- cor(existingProduct)
corrData
corrplot(corrData)

corrDataB <- cor(existingProductB)
corrDataB
corrplot(corrDataB)
#Best Sellers rank does not have any significant effect for volume, so we can exclude it.

#Since Shipping weight, product depth/width/height and profit margins don't seem to affect volume, we will remove them
existingProduct$ShippingWeight <- NULL
existingProduct$ProductDepth <- NULL
existingProduct$ProductWidth <- NULL
existingProduct$ProductHeight <- NULL
existingProduct$ProfitMargin <- NULL
existingProduct$ProductNum <- NULL

#We must repeat this for newProduct
newProduct$ShippingWeight <- NULL
newProduct$ProductDepth <- NULL
newProduct$ProductWidth <- NULL
newProduct$ProductHeight <- NULL
newProduct$ProfitMargin <- NULL
existingProduct$ProductNum <- NULL

#setting our seed
set.seed(123)

#separating our data so that we can have training data and testing data
inTraining <- createDataPartition(existingProduct$Volume, p = .8, list = FALSE)
training <- existingProduct[inTraining,]
testing <- existingProduct[-inTraining,]
fitControl <- trainControl(method = "repeatedcv", number = 3, repeats = 1)

#We will do linear Regression
linear <- train(Volume~., data = training, method = "lm", trControl=fitControl)
linear

#This is the result when we used linear model, it did not do well at all as RMSE is minuscule and R squared is a one which doesnt sound right
#  RMSE          Rsquared  MAE         
#5.329679e-13  1         3.14352e-13


#We will try random forest
rfGrid <- expand.grid(mtry = c(1,2,3,4,5))
randomForest <- train(Volume~., data = training, method = "rf", trControl=fitControl, tuneGrid=rfGrid)
randomForest
summary(randomForest)

#These are the results for random forest, now these seem more real, but have low rsquared value
# mtry  RMSE      Rsquared   MAE     
# 1     1039.622  0.5920195  500.8054
# 2     1035.596  0.6302385  377.4122
# 3     1061.123  0.6402024  356.3261
# 4     1049.563  0.6517623  331.4490
# 5     1081.188  0.6598864  330.1750

rfTest <- predict(randomForest, testing)
postResample(rfTest, testing$Volume)
#When we run prediction, it does have relative high accuracy 
#        RMSE     Rsquared          MAE 
#1102.7282019    0.8550209  378.6198607 


#We will try GBM now
GBM <-  train(Volume~.,
                  data = training, 
                  method = "gbm",
                  trControl = fitControl,
                  tuneLength = 5)
GBM$finalModel$tuneValue
GBM$finalModel$tree
summary(GBM)
plot(GBM)
GBM
varImp(GBM) 
#Results for GBM
# interaction.depth  n.trees  RMSE       Rsquared 
# 1                   50      1046.4931  0.5169286
# 1                  100      1167.9241  0.4925352
# 1                  150      1215.7024  0.4969874
# 1                  200      1204.2463  0.5059083
# 1                  250      1254.2466  0.4798946
# 2                   50       987.0066  0.5547620
# 2                  100      1083.0180  0.4830115
# 2                  150      1168.7536  0.4878268
# 2                  200      1210.1532  0.4498034
# 2                  250      1228.5446  0.4440902
# 3                   50      1008.6345  0.5051922
# 3                  100      1113.3227  0.4717633
# 3                  150      1178.2033  0.4359067
# 3                  200      1268.7830  0.4159755
# 3                  250      1245.9327  0.4249619
# 4                   50      1037.1392  0.5361747
# 4                  100      1140.1775  0.4812736
# 4                  150      1172.9143  0.4668447
# 4                  200      1237.8208  0.4422102
# 4                  250      1345.0807  0.4175399
# 5                   50       960.8076  0.5996973
# 5                  100      1089.9466  0.5128583
# 5                  150      1120.8552  0.4747243
# 5                  200      1220.0302  0.4465416
# 5                  250      1305.1087  0.4304874

GBMTest <- predict(GBM, testing)
postResample(GBMTest, testing$Volume)
#RMSE     Rsquared          MAE 
#1357.1542551    0.4367781  536.0079182
#These were even worse than the one when we did random forest


#We will try svm now
SVM <- svm(Volume ~., data = existingProduct)
SVM


SVMTest <- predict(SVM, testing)
postResample(SVMTest, testing$Volume)
#RMSE     Rsquared          MAE 
#1128.5257017    0.8699854  413.2459728 

#SVM actually works well here as the it has the highest Rsquared value!

#check for neg values
SVMTest
#17         18         22         27         34 
#168.57243  934.50935 1424.33214  140.41337 1092.48358 
#43         46         49         57         60 
#240.48385 1187.49764   23.61306  578.06766  374.53674 
#61         64         66         72         73 
#383.55980   75.99764  147.81033  119.34579 2700.54562 

#There are none so we are good to move forward with this model

finalPred <- predict(SVM, newProduct)
postResample(finalPred, testing$Volume)

output <- newProduct
output$predictions <- finalPred
write.csv(output, file="predictions.csv", row.names = TRUE)


ggplot(existingProduct, aes(x=PositiveServiceReview, y=Volume)) + 
  geom_point() +
  geom_smooth(method=lm, se=FALSE) +
  labs(title = "Positive Service Reviews vs Volume Sold", x = "Positive Service Reviews", y = " Volume Sold")

ggplot(existingProduct, aes(x=NegativeServiceReview, y=Volume)) + 
  geom_point() +
  geom_smooth(method=lm, se=FALSE) +
  labs(title = "Negative Service Reviews vs Volume Sold", x = "Negative Service Reviews", y = " Volume Sold")


