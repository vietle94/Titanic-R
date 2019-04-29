library(tidyverse)
library(doParallel)
library(caret)
library(partykit)
library(lubridate)
library(patchwork)
library(caretEnsemble)

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

# Ticket -----------------------------------------------------
sameTicketTrain <- combined[1:intrain,] %>% 
  count(Ticket) %>% filter(n != 1) %>% 
  pull(Ticket)

combined$companion <- NA

combined[1:intrain,]$companion <- combined[1:intrain,] %>% 
  mutate(companion = ifelse(Ticket %in% sameTicketTrain, T, F)) %>% 
  pull(companion)

sameTicketTest <- combined %>% 
  count(Ticket) %>% filter(n != 1) %>% 
  pull(Ticket)

combined[-(1:intrain),]$companion <- combined[-(1:intrain),] %>% 
  mutate(companion = ifelse(Ticket %in% sameTicketTest, T, F)) %>% 
  pull(companion)

# Fare ---------------------------------------------------------

combined <- combined %>% mutate(Fare = log10(Fare +1))

# Remove bad columns ---------------------------------------------
combined <- combined %>% select(-PassengerId, -Name, -Cabin, -Ticket)

# Impute Embarked --------------------------------------------------
combined %>% map_dbl(~sum(is.na(.)))
combined[is.na(combined$Embarked),]$Embarked <- "S"

# Dummy coded columns -------------------------------------------------------------
dummy <- dummyVars(~., data = combined[,-1]) 
dummy_combined <- predict(dummy, newdata = combined[,-1]) %>% as_tibble()
dummy_combined$Survived <- combined$Survived
dummy_combined <- dummy_combined %>% 
  select(Survived, everything())

# Impute Age ----------------------------------------------------------
imputemod <- preProcess(dummy_combined[1:intrain,-1], method = "bagImpute")

imputed <- predict(imputemod, newdata = dummy_combined[1:intrain,-1])
dummy_combined[1:intrain,]$Age <- imputed$Age

imputed_test <- predict(imputemod, newdata = dummy_combined[-(1:intrain),-1])
dummy_combined[-(1:intrain),]$Age <- imputed_test$Age
dummy_combined[-(1:intrain),]$Fare <- imputed_test$Fare

dummy_combined %>% map_df( ~sum(is.na(.)))
rm(imputed)
rm(imputed_test)
rm(imputemod)

# dummy_combined <- dummy_combined %>% mutate(Age = cut(Age, seq(0,80,10)))


# Prepare model building ----------------------------------------------------------
set.seed(12345)
notinholdout <- createDataPartition(train0$Survived, p = 0.8, list = F)
length(notinholdout)

# train and test set
train <- dummy_combined[1:intrain,] %>% slice(notinholdout) 
holdout <- dummy_combined[1:intrain,] %>% slice(-notinholdout) 
test <- dummy_combined[-(1:intrain),]

# Get 10 fold repeated 3 times train control
trCtr <- trainControl(method = "repeatedcv", repeats = 5, 
                      classProbs = T, savePredictions = T)

trCtr_search <- trainControl(method = "repeatedcv", repeats = 5, 
                             classProbs = T, savePredictions = T,
                             search = "random")
trCtr_none <- trainControl(method = "none", classProbs = T)
  
cl <- makeCluster(7)
registerDoParallel(cl)

# Get model info ----------------------------------------------------------

getModelInfo()$svmRadial$parameters
getModelInfo()$xgbTree$parameters
getModelInfo()$rf$parameters



# Formula ----------------------------------------------------------------

formula <- as.formula(Survived ~ Fare + Age + TitleMr + Pclass.1 +
                      Sex.female + Sex.male + Pclass.3 + SibSp + Parch)

# SVM ---------------------------------------------------------------------

SVM <- train(formula, data = train, trControl = trCtr,
             method = "svmRadial", tuneLength = 10, 
             preProcess = c("center", "scale"),
             na.action = na.pass)

# Train all 
SVMf <- train(Survived~., data = dummy_combined[1:intrain,], trControl = trCtr_none,
              method = "svmRadial", tuneGrid = SVM$bestTune,
              preProcess = c("center", "scale"),
              na.action = na.pass)

# Evaluate
SVMtest <- predict(SVM, newdata = holdout, na.action = na.pass)
SVMtestProb <- predict(SVM, newdata = holdout, na.action = na.pass,
                       type = "prob")
confusionMatrix(holdout$Survived, SVMtest)


# xgb ---------------------------------------------------------------------

xgb <- train(formula, data = train, trControl = trCtr_search,
             method = "xgbTree", na.action = na.pass,
             tuneLength = 30)

xgbTune <- train(formula, data = train, trControl = trCtr,
                 method = "xgbTree", na.action = na.pass,
                 tuneGrid = xgb$bestTune)

xgbf <- train(formula, data = dummy_combined[1:intrain,], trControl = trCtr_none,
              method = "xgbTree", na.action = na.pass,
              tuneGrid = xgb$bestTune)

plot(xgb)

xgbtest <- predict(xgb, newdata = holdout, na.action = na.pass)
xgbtestProb <- predict(xgb, newdata = holdout, na.action = na.pass,
                       type = "prob")
confusionMatrix(holdout$Survived, xgbtest)

# rf ----------------------------------------------------------------------

rfmod <- train(formula, data = train, trControl = trCtr,
               method = "rf", na.action = na.pass,
               tuneLength = 8)

rfmodf <- train(formula, data = dummy_combined[1:intrain,], trControl = trCtr_none,
                method = "rf", na.action = na.pass, tuneGrid = rfmod$bestTune)

plot(rfmod)
rftest <- predict(rfmod, newdata = holdout, na.action = na.pass)
rftestProb <- predict(rfmod, newdata = holdout, na.action = na.pass, type = "prob") 

confusionMatrix(holdout$Survived, rftest)

# Logistics ---------------------------------------------------------------

bayes <- train(formula, data = train, trControl = trCtr,
               method = "bayesglm", na.action = na.pass)

bayesf <- train(formula, data = dummy_combined[1:intrain,], trControl = trCtr_none,
                method = "bayesglm", na.action = na.pass)

bayestest <- predict(bayes, newdata = holdout,
                     na.action = na.pass)
confusionMatrix(holdout$Survived, bayestest)

bayestestProb <- predict(bayes, newdata = holdout, type = "prob",
                         na.action = na.pass)

# rpart -------------------------------------------------------------------

rpart <- train(formula, data = train, trControl = trCtr,
               method = "rpart", tuneLength = 30,
               na.action = na.pass)
rpart
rpartf <- train(formula, data = dummy_combined[1:intrain,], trControl = trCtr_none,
                method = "rpart", tuneGrid = rpart$bestTune,
                na.action = na.pass)
rpartTest <- predict(rpart, newdata = holdout,
                     na.action = na.pass)
rpartTestProb <- predict(rpart, newdata = holdout, type = "prob",
                         na.action = na.pass)

confusionMatrix(holdout$Survived, rpartTest)

# c5.0 --------------------------------------------------------------------

C5.0 <- train(formula, data = train, trControl = trCtr_search,
              method = "C5.0", tuneLength = 30,
              na.action = na.pass)
C5.0f <- train(formula, data = dummy_combined[1:intrain,], trControl = trCtr_none,
               method = "C5.0", tuneGrid = C5.0$bestTune,
               na.action = na.pass)

C5.0
C5.0Test <- predict(C5.0, newdata = holdout, na.action = na.pass)
C5.0TestProb <- predict(C5.0, newdata = holdout, type = "prob",
                        na.action = na.pass)
confusionMatrix(holdout$Survived, C5.0Test)


# Compare models ----------------------------------------------------------
# C5.0 = C5.0,rpart = rpart, SVM = SVM, bayes = bayes, 
# xgb and rf
cvValues <- resamples(list( xgb = xgb,
                            C5.0 = C5.0,rpart = rpart, 
                            SVM = SVM, bayes = bayes, rf = rfmod))
summary(cvValues)
xyplot(cvValues, metric = "Accuracy")
parallelplot(cvValues, metric = "Accuracy")
dotplot(cvValues, metric = "Accuracy")
dotplot(cvValues, metric = "Accuracy")
rocDiffs <- diff(cvValues, metric = "Accuracy")
summary(rocDiffs)

modelCor(cvValues) %>% as_tibble()


# final -------------------------------------------------------------------

all <- list(C5.0 = C5.0f, rpart = rpartf,
            xgb = xgbf, rf = rfmodf, bayes = bayesf, SVM = SVMf )

finaltest <- predict(all, newdata = test, na.action = na.pass)

prefix <- "companion"
for (i in names(finaltest)){
  bind_cols(PassengerID = test0$PassengerId, Survived = finaltest[[i]]) %>% 
    mutate(Survived = fct_recode(Survived, "1" = "Lived", "0" = "Died")) %>% 
    write_csv(str_c("./prediction/", prefix,"_" ,i,"_", today(), ".csv"))
}

p1 <- ggplot(varImp(rfmodf))
p2 <- ggplot(varImp(xgbf))
p3 <- ggplot(varImp(C5.0f))
p4 <- ggplot(varImp(rpartf))

p1 + p2 + p3 + p4


plot(as.party(rpart$finalModel))
plot(xgboost::xgb.dump(xgb$finalModel))

xgb$pred %>% mutate(Accuracy = ifelse(pred == obs, T, F)) %>% 
  group_by( eta, max_depth, colsample_bytree, nrounds, gamma, subsample) %>% 
  summarise(Accuracy = mean(Accuracy)) %>% view()

plot(xgbf)
