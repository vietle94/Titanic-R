library(tidyverse)
library(doParallel)
library(caret)

# Load data ---------------------------------------------------------------


train <- read_csv("./titanic/train.csv",
                  col_types = cols(
                    Survived = col_factor(ordered = FALSE, include_na = T),
                    Pclass = col_factor(),
                    Sex = col_factor(),
                    Embarked = col_factor()
                  ))
test <- read_csv("./titanic/test.csv",
                 col_types = cols(
                   Pclass = col_factor(),
                   Sex = col_factor(),
                   Embarked = col_factor()
                 ))

# Tidy --------------------------------------------------------------------

test <- test %>% select(-Cabin, -Ticket, -PassengerId, -Name)

train$Survived <- fct_recode(train$Survived, "Lived" = "1", "Died" = "0") %>% 
  fct_relevel("Lived")
train <- train %>% select(-Cabin, -Ticket, -PassengerId, -Name)

# Impute Embarked ---------------------------------------------------------

train %>% map_dbl(~sum(is.na(.)))
table(train$Embarked)
train[is.na(train$Embarked),]$Embarked <- "S"

# Dummy variables ---------------------------------------------------------

dummy <- dummyVars(~., data = train[,-1])
dummy_train <- predict(dummy, newdata=train[,-1]) %>% as_tibble()
dummy_train$Survived <- train$Survived

# Prepare model building ----------------------------------------------------------
set.seed(1234)
inTrain <- createDataPartition(dummy_train$Survived, p = 0.8, list = F)
traindata <- dummy_train[inTrain,]
testdata <- dummy_train[-inTrain,]
trCtr <- trainControl(method = "repeatedcv", repeats = 10, 
                      classProbs = T, summaryFunction = twoClassSummary)
library(doParallel)
cl <- makeCluster(7)
registerDoParallel(cl)

# rf ----------------------------------------------------------------------

set.seed(1234)
rfmod <- train(Survived ~., data = traindata, trControl = trCtr,
               method = "rf", metric = "ROC", 
               preProcess = c("bagImpute"), na.action = na.pass,
               tuneLength = 8)

plot(rfmod)
rftest <- predict(rfmod, newdata = testdata, na.action = na.pass)
rftestProb <- predict(rfmod, newdata = testdata, na.action = na.pass, type = "prob") 

confusionMatrix(testdata$Survived, rftest)

# Logistics ---------------------------------------------------------------
set.seed(1234)
bayes <- train(Survived ~., data = traindata, trControl = trCtr,
               method = "bayesglm", metric = "ROC",
               preProcess = c("bagImpute"), na.action = na.pass)
bayestest <- predict(bayes, newdata = testdata,
                     na.action = na.pass)
confusionMatrix(testdata$Survived, bayestest)

bayestestProb <- predict(bayes, newdata = testdata, type = "prob",
                         na.action = na.pass)

# rpart -------------------------------------------------------------------

set.seed(1234)
rpart <- train(Survived ~., data = traindata, trControl = trCtr,
               method = "rpart", metric = "ROC", tuneLength = 30,
               na.action = na.pass)
rpart
rpartTest <- predict(rpart, newdata = testdata,
                     na.action = na.pass)
rpartTestProb <- predict(rpart, newdata = testdata, type = "prob",
                     na.action = na.pass)

confusionMatrix(testdata$Survived, rpartTest)

# rpart with bagImpute ----------------------------------------------------

set.seed(1234)
rpartbagImpute <- train(Survived~., data = traindata, trControl = trCtr,
                        method = "rpart", metric = "ROC", tuneLength = 30,
                        preProcess = "bagImpute", na.action = na.pass)
rpartbagImpute
rpartbagImputeTest <- predict(rpartbagImpute, newdata = testdata, 
                              na.action = na.pass)
rpartbagImputeTestProb <- predict(rpartbagImpute, newdata = testdata, type = "prob",
                                  na.action = na.pass)

confusionMatrix(testdata$Survived, rpartbagImputeTest)

# c5.0 --------------------------------------------------------------------

set.seed(1234)
C5.0 <- train(Survived~., data = traindata, trControl = trCtr,
              method = "C5.0", metric = "ROC", )

# ROC curve ---------------------------------------------------------------

library(pROC)
rfROC <- roc(predictor = rftestProb$Lived,
             response = testdata$Survived,
             levels = rev(levels(testdata$Survived)), plot = TRUE,
             print.auc = TRUE) #need to reverse level of factor
# Can extract parameters in rfROC here and
# Can use this threshold to modify predict in rfProbs

bayesROC <- roc(predictor = bayestestProb$Lived,
                response = testdata$Survived,
                levels = rev(levels(testdata$Survived)),
                plot = TRUE,
                print.auc = T, add = T, col = "#9E0142", print.auc.y = 0.6)

rpartROC <- roc(predictor = rpartTestProb$Lived,
                response = testdata$Survived,
                levels = rev(levels(testdata$Survived)),
                plot = T,
                print.auc = T, add = T, col = "blue", print.auc.y = 0.7)
rpartbagImputeROC <- roc(predictor = rpartbagImputeTestProb$Lived,
                         response = testdata$Survived,
                         levels = rev(levels(testdata$Survived)),
                         plot = TRUE,
                         print.auc = T, add = T, col = "yellow", print.auc.y = 0.4)


# Extract data from ROC ---------------------------------------------------


df <- tibble(trueposi = rfROC$sensitivities, 
             falposi = 1- rfROC$specificities, thred = rfROC$thresholds) 

ggplot(df) + geom_path(aes(falposi, trueposi)) + scale_y_continuous(breaks = seq(0,1,0.2))

view(df)
# exit --------------------------------------------------------------------



on.exit(stopCluster(cl))
