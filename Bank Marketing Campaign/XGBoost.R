
bank <- bank1[,c(1,2,3,4)]

dmy <- dummyVars("~.",data = bank)
bank_dmy <- data.frame(predict(dmy,newdata = bank))

label <- bank1$y

label <- as.matrix(label)


label[label=='no'] <- 1

label[label=='yes'] <- 0


bank_dmy$y <- label


#To Split Test and train


smp_size <- floor(0.75 * nrow(bank_dmy))


set.seed(123)

train_bank <- sample(seq_len(nrow(bank_dmy)), size = smp_size)

train <- bank_dmy[train_bank, ]
test <- bank_dmy[-train_bank, ]


#To Split the Predictors and Lables


train_pred <- data.matrix(train[,-20])
test_pred <- data.matrix(test[,-20])



train_label <- data.matrix(train[,20])
test_label <- data.matrix(test[,20])



#data <- data.matrix(train_dmy[,-20])




model <- xgboost(data = train_pred, label = train_label, objective = "binary:logistic",nrounds = 2, method ="class")

model <- xgboost(data = train_pred, label = train_label, objective ="binary:logistic", nrounds = 2, method ="class")


pred <- predict(model,test_pred,type = "class")

pred



#To Check The accuracy

pred1 <- as.matrix(pred)

pred1[pred1 >= 0.644] <- 1

pred1[pred1 <= 0.644] <- 0

confusionMatrix(pred1,test_label)

recall(pred1,test_label)

