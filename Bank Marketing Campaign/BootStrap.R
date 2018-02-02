
#Using Bootstrap Aggregation Method


library(readr)
bankfull <- read_csv("C:/Users/karth/Desktop/DESKTOP_1/CLASS NOTES/Q3/Data Mining/Project/Bank/bank-full.csv")
View(bankfull)

library(tree)
library(caret)

bankf <- bankfull

bankf$age <- as.numeric(bankfull$age)
bankf$job <- as.factor(bankfull$job)
bankf$marital <- as.factor(bankfull$marital)
bankf$education <- as.factor(bankfull$education)
bankf$default <- as.factor(bankfull$default)
bankf$balance <- as.numeric(bankfull$balance)
bankf$housing <- as.factor(bankfull$housing)
bankf$loan <- as.factor(bankfull$loan)
bankf$day <- as.numeric(bankfull$day)
bankf$month <- as.factor(bankfull$month)
bankf$duration <- as.numeric(bankfull$duration)
bankf$campaign <- as.numeric(bankfull$campaign)
bankf$pdays <- as.numeric(bankfull$pdays)
bankf$previous <- as.numeric(bankfull$previous)
bankf$poutcome <- as.factor(bankfull$poutcome)
bankf$y <- as.factor(bankfull$y)



#To Split Test and train


smp_size <- floor(0.75 * nrow(bankf))


set.seed(123)

train_bank <- sample(seq_len(nrow(bankf)), size = smp_size)

train <- bankf[train_bank, ]
test <- bankf[-train_bank, ]




#Modeling with Training set


bank_boot <- bagging(y~., data = train)


#Testing with the test data


boot_test <- predict(bank_boot,test)


#Checking the Accuracy


confusionMatrix(boot_test,test$y)

recall(boot_test,test$y)

precision(boot_test,test$y)

F_meas(boot_test,test$y)







