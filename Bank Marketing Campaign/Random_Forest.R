#Model Using RandomForest


library(randomForest)
library(caret)


bankf <- bankfull

bank1 <- bankfull[,c(7,11,12,16,17)]

bank1$housing <- as.factor(bank1$housing)
bank1$month <- as.factor(bank1$month)
bank1$duration <- as.numeric(bank1$duration)
bank1$poutcome <- as.factor(bank1$poutcome)
bank1$y <- as.factor(bank1$y)
bank1$age <- as.numeric(bankfull$age)
bank1$balance <- as.numeric(bankfull$balance)
bank1$day <- as.numeric(bankfull$day)
bank1$campaign <- as.numeric(bankfull$campaign)
bank1$previous <- as.numeric(bankfull$previous) 
bank1$pdays <- as.numeric(bankfull$pdays)
bank1$loan <- as.factor(bankfull$loan)
bank1$job <- as.factor(bankfull$job)
bank1$marital <- as.factor(bankfull$marital)
bank1$education <- as.factor(bankfull$education)
bank1$default <- as.factor(bankfull$default)



#To Split Test and train


smp_size <- floor(0.75 * nrow(bank1))


set.seed(123)

train_bank <- sample(seq_len(nrow(bank1)), size = smp_size)

train <- bank1[train_bank, ]
test <- bank1[-train_bank, ]


ran <- randomForest(y~.,data = train, ntree = 100,mtry=3)


#to Predict

ran_pre <- predict(ran,test)


# Accuracy Measures

confusionMatrix(ran_pre,test$y)

recall(ran_pre,test$y)


precision(ran_pre,test$y)



















































