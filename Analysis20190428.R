library(tidyverse)
library(doParallel)
library(caret)

# Load data ---------------------------------------------------------------


train0 <- read_csv("./titanic/train.csv",
                   col_types = cols(
                     Survived = col_factor(ordered = FALSE, include_na = T),
                     Pclass = col_factor(),
                     Sex = col_factor(),
                     Embarked = col_factor()
                   ))
intrain <- dim(train0)[1]
test0 <- read_csv("./titanic/test.csv",
                  col_types = cols(
                    Pclass = col_factor(),
                    Sex = col_factor(),
                    Embarked = col_factor()
                  ))

# Tidy --------------------------------------------------------------------
# Combine test and train for feature selection
combined <- bind_rows(train0, test0)

# Extract Title
combined <- combined %>% mutate(Title = str_extract(Name, ", (\\w)+")) %>% 
  mutate(Title = str_extract(Title, "\\w+"))
table(combined$Title)

combined %>% mutate(normTitle = Title %in% c("Miss", "Master", "Mr", "Mrs")) %>% 
  filter(normTitle == F) %>%
  ggplot() + geom_count(aes(Title, Survived))

# Remove those title different from Miss, Master, Mr and Mrs since they are low frequency,
# Combine them into Others

combined <- combined %>%
  mutate(Title = ifelse(Title %in% c("Miss", "Master", "Mr", "Mrs"), Title,"Others"))

# Make Missingage predictor
combined <- combined %>% mutate(MissingAge = is.na(Age))

# We can use Name to predict Age?
# Master are underage male children(Fact)
# Take mean Age from train set and use it to replace NA from all data with Master title

meanMasterAge <- combined %>%
  slice(1:dim(train0)[1]) %>% 
  filter(Title == "Master") %>% 
  summarise(mean(Age, na.rm = T)) %>% 
  pull()
combined <- combined %>% 
  mutate(Age = ifelse(Title == "Master" & is.na(Age), meanMasterAge ,Age ))


combined$Survived <- fct_recode(combined$Survived, "Lived" = "1", "Died" = "0") %>% 
  fct_relevel("Lived")

# Remove bad columns
combined <- combined %>% select(-PassengerId, -Name, -Cabin, -Ticket)

# Impute Embarked 
combined[1:intrain,] %>% map_dbl(~sum(is.na(.)))
combined[is.na(combined$Embarked),]$Embarked <- "S"

# Impute Age
imputemod <- preProcess(combined[1:intrain,-1], method = "bagImpute")
imputed <- predict(imputemod, newdata = combined[1:intrain,-1])
imputed %>% map_df(~ sum(is.na(.)))
combined[1:intrain,]$Age <- imputed$Age
imputed_test <- predict(imputemod, newdata = combined[-(1:intrain),-1])
combined[-(1:intrain),]$Age <- imputed_test$Age
combined[-(1:intrain),]$Fare <- imputed_test$Fare
combined %>% map_df( ~sum(is.na(.)))

combined <- combined %>% mutate(Age = cut(Age, seq(0,80,10)))

# Adding Features ---------------------------------------------------------

combined <- combined %>% mutate(Fare = log10(Fare +1)) 

# Dummy coded columns -------------------------------------------------------------
dummy <- dummyVars(~., data = combined[,-1]) 
dummy_combined <- predict(dummy, newdata = combined[,-1]) %>% as_tibble()
dummy_combined$Survived <- combined$Survived

# Prepare model building ----------------------------------------------------------
set.seed(12345)
notinholdout <- createDataPartition(train0$Survived, p = 0.8, list = F)
length(notinholdout)

# train and test set
train <- combined[1:intrain,] %>% slice(notinholdout) 
holdout <- combined[1:intrain,] %>% slice(-notinholdout) 
test <- combined[-(1:intrain),]

trainDummy <- dummy_combined[1:intrain,] %>% slice(notinholdout) 
holdoutDummy <- dummy_combined[1:intrain,] %>% slice(-notinholdout)
testDummy <- dummy_combined[-(1:intrain),] 

# Get 10 fold repeated 10 times train control
trCtr <- trainControl(method = "repeatedcv", repeats = 3, 
                      classProbs = T)

trCtr_search <- trainControl(method = "repeatedcv", repeats = 3, 
                             classProbs = T,
                             search = "random")
trCtr_none <- trainControl(method = "none", classProbs = T)

library(doParallel)
cl <- makeCluster(7)
registerDoParallel(cl)

# SVM ---------------------------------------------------------------------
SVM <- train(Survived~., data = trainDummy, trControl = trCtr,
             method = "svmRadial", tuneLength = 20, 
             preProcess = c("center", "scale"),
             na.action = na.pass)

SVMf <- train(Survived~., data = dummy_combined[1:intrain,], trControl = trCtr_none,
              method = "svmRadial", tuneGrid = SVM$bestTune,
              preProcess = c("center", "scale"),
              na.action = na.pass)
SVMtest <- predict(SVM, newdata = holdoutDummy, na.action = na.pass)
SVMtestProb <- predict(SVM, newdata = holdoutDummy, na.action = na.pass,
                       type = "prob")
confusionMatrix(holdoutDummy$Survived, SVMtest)


# xgb ---------------------------------------------------------------------

xgb <- train(Survived ~., data = trainDummy, trControl = trCtr_search,
             method = "xgbTree", na.action = na.pass,
             tuneLength = 30)
xgb$bestTune

xgbf <- train(Survived ~., data = dummy_combined[1:intrain,], trControl = trCtr_none,
              method = "xgbTree", na.action = na.pass,
              tuneGrid = xgb$bestTune)

xgbf

xgbtest <- predict(xgb, newdata = holdoutDummy, na.action = na.pass)
xgbtestProb <- predict(xgb, newdata = holdoutDummy, na.action = na.pass,
                       type = "prob")
confusionMatrix(holdoutDummy$Survived, xgbtest)

# rf ----------------------------------------------------------------------

set.seed(1234)
rfmod <- train(Survived ~., data = trainDummy, trControl = trCtr,
               method = "rf", na.action = na.pass,
               tuneLength = 8)

rfmodf <- train(Survived ~., data = dummy_combined[1:intrain,], trControl = trCtr_none,
                method = "rf", na.action = na.pass, tuneGrid = rfmod$bestTune)

plot(rfmod)
rftest <- predict(rfmod, newdata = holdoutDummy, na.action = na.pass)
rftestProb <- predict(rfmod, newdata = holdoutDummy, na.action = na.pass, type = "prob") 

confusionMatrix(holdoutDummy$Survived, rftest)

# Logistics ---------------------------------------------------------------
set.seed(1234)
bayes <- train(Survived ~., data = trainDummy, trControl = trCtr,
               method = "bayesglm", na.action = na.pass)

bayesf <- train(Survived ~., data = dummy_combined[1:intrain,], trControl = trCtr_none,
                method = "bayesglm", na.action = na.pass)

bayestest <- predict(bayes, newdata = holdoutDummy,
                     na.action = na.pass)
confusionMatrix(holdoutDummy$Survived, bayestest)

bayestestProb <- predict(bayes, newdata = holdoutDummy, type = "prob",
                         na.action = na.pass)

# rpart -------------------------------------------------------------------

set.seed(1234)
rpart <- train(Survived ~., data = train, trControl = trCtr,
               method = "rpart", tuneLength = 30,
               na.action = na.pass)
rpart
rpartf <- train(Survived ~., data = combined[1:intrain,], trControl = trCtr_none,
                method = "rpart", tuneGrid = rpart$bestTune,
                na.action = na.pass)
rpartTest <- predict(rpart, newdata = holdout,
                     na.action = na.pass)
rpartTestProb <- predict(rpart, newdata = holdout, type = "prob",
                         na.action = na.pass)

confusionMatrix(holdout$Survived, rpartTest)

# c5.0 --------------------------------------------------------------------

set.seed(1234)
C5.0 <- train(Survived~., data = train, trControl = trCtr_search,
              method = "C5.0", tuneLength = 30,
              na.action = na.pass)
C5.0f <- train(Survived~., data = combined[1:intrain,], trControl = trCtr_none,
               method = "C5.0", tuneGrid = C5.0$bestTune,
               na.action = na.pass)

C5.0
C5.0Test <- predict(C5.0, newdata = holdout, na.action = na.pass)
C5.0TestProb <- predict(C5.0, newdata = holdout, type = "prob",
                        na.action = na.pass)
confusionMatrix(holdout$Survived, C5.0Test)


# Compare models ----------------------------------------------------------

cvValues <- resamples(list(C5.0 = C5.0, xgb = xgb, rpart = rpart, SVM = SVM,
                           bayes = bayes, rf = rfmod))
summary(cvValues)
xyplot(cvValues, metric = "Accuracy")
parallelplot(cvValues, metric = "Accuracy")
dotplot(cvValues, metric = "Accuracy")
dotplot(cvValues, metric = "Accuracy")
rocDiffs <- diff(cvValues, metric = "Accuracy")
summary(rocDiffs)


# final -------------------------------------------------------------------

all <- list(C5.0 = C5.0f, rpart = rpartf )
all_dummy <- list(xgb = xgbf, rf = rfmodf, bayes = bayesf, SVM = SVMf)

finaltest <- predict(all, newdata = test, na.action = na.pass)
finaltestDummy <- predict(all_dummy, newdata = testDummy, na.action = na.pass)

library(lubridate)

prefix <- "accuracy_catAge"
for (i in names(finaltest)){
  bind_cols(PassengerID = test0$PassengerId, Survived = finaltest[[i]]) %>% 
    mutate(Survived = fct_recode(Survived, "1" = "Lived", "0" = "Died")) %>% 
    write_csv(str_c("./prediction/", prefix,"_" ,i,"_", today(), ".csv"))
}
for (i in names(finaltestDummy)){
  bind_cols(PassengerID = test0$PassengerId, Survived = finaltestDummy[[i]]) %>% 
    mutate(Survived = fct_recode(Survived, "1" = "Lived", "0" = "Died")) %>% 
    write_csv(str_c("./prediction/", prefix,"_" ,i,"_", today(), ".csv"))
}
