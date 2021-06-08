library(reshape2)
library(caret)
library(rpart.plot)
library(class)

# Read deletion information

df = read.table("deletion.tsv.gz", header=T)
print(summary(df))
print(dim(df))
unknown = df[is.na(df$status), ]
dim(unknown)

# Keep ID as unique rowname
rownames(df) = df$id
df = df[,!colnames(df) %in% c("chr_start_end","id")]
unknown = df[is.na(df$status),]   # unlabeled data
df = df[!is.na(df$status),]       # labeled data
df$status = factor(df$status)
print("Labeled data")
print(table(df$status))
print(dim(df))


# Split training and testing
train = createDataPartition(y = df$status, p=0.8, list=F)
training = df[train,]
print("Training dimensions")
print(dim(training))
print("Training status")
print(table(training$status))
testing = df[-train,]
print("Testing dimensions")
print(dim(testing))
print("Testing status")
print(table(testing$status))




# 10-fold cross-validation
trctrl = trainControl(
  method="repeatedcv", 
  number=10, 
  repeats=10, 
  search ="grid")

# difrent values of mtry (number of randomly sampled variables)
tunegrid <- expand.grid(.mtry = c(1: 14))

# Build depth one decision tree
# fit = train(status ~ ., data=training, method="rpart", 
#             parms = list(split="information"), 
#             trControl=trctrl, tuneLength=10, 
#             control = list(maxdepth = 1))

# tune mtry of random forest using grid search defined in trctrl and tunegrid
fit2 = train(status ~ ., data=training, method="rf", 
            parms = list(split="information"), 
            metric = "Accuracy",
            tuneGrid = tunegrid,
            trControl=trctrl, tuneLength=10, 
            control = list(maxdepth = 1))

# print(fit2)

# save the best accuracy and number of randomly sampled variables
best_tune_mtry = data.frame(
  accuracy = max(fit2$results$Accuracy),
  mtry = fit2$bestTune$mtry
)

#select the number of randomly sampled parameters with the best accuracy
tunegrid <- expand.grid(.mtry = best_tune_mtry$mtry)

# find the best nodesize for random forest
maxnodes = c(1:20) 
for (node in c(1:20)) {
  tmp_maxnode = train(status ~ ., data=training, method="rf", 
                      parms = list(split="information"), 
                      metric = "Accuracy",
                      tuneGrid = tunegrid,
                      nodesize = node,
                      trControl=trctrl, tuneLength=10, 
                      ntree = 800,
                      control = list(maxdepth = 1)) 
  maxnodes[node] <- tmp_maxnode$results$Accuracy
}

#save the nodesize for the best accuracy of random forest  
best_tune_nodesize = data.frame(
  accuracy = max(maxnodes),
  nodesize = which(maxnodes == max(maxnodes) )[1]
)

# set the node size by saved value
nodesize =  best_tune_nodesize$nodesize  

# find out the best tree size for random forest
maxtrees_tree = c(1:36) 
maxtrees_accuracy = c(1:36) 
index = 1
for (ntree in seq(250, 2000, 50)) {
  tmp_maxntree = train(status ~ ., data=training, method="rf", 
                      parms = list(split="information"), 
                      metric = "Accuracy",
                      tuneGrid = tunegrid,
                      nodesize = nodesize,
                      trControl=trctrl, tuneLength=10, 
                      ntree = ntree,
                      control = list(maxdepth = 1)) 
  maxtrees_tree[index] = ntree
  maxtrees_accuracy[index] = tmp_maxntree$results$Accuracy
  index = index + 1
}
# maxtrees_tree
# maxtrees_accuracy


# save the size of the tree with the best accuracy 
best_tune_ntree = data.frame(
  accuracy = max(maxtrees_accuracy),
  ntree = maxtrees_tree[which(maxtrees_accuracy == max(maxtrees_accuracy) )[1]]
)
# set tree size for random forest by saved value
ntree = best_tune_ntree$ntree


# train final model using saved parameters
fit_rf = train(status ~ ., data=training, method="rf", 
                    parms = list(split="information"), 
                    metric = "Accuracy",
                    tuneGrid = tunegrid,
                    nodesize = nodesize,
                    trControl=trctrl, tuneLength=10, 
                    ntree = ntree,
                    control = list(maxdepth = 1)) 


# prp(fit$finalModel, tweak=1.2)

# vizualize the final model 
plot(fit_rf$finalModel)


# Evaluate decision tree
# pred = predict(fit, newdata=testing)
# print(caret::confusionMatrix(pred, testing$status, positive="1"))

pred_rf = predict(fit_rf, newdata = testing)
print(caret::confusionMatrix(pred_rf, testing$status, positive="1"))

# Apply to unknown data
# pred = predict(fit, newdata=unknown)
print("Outcome prediction of unlabeled data")
# print(table(pred))
# unknown$status = pred

pred_rf = predict(fit_rf, newdata = unknown)
print(table(pred_rf))
unknown$status = pred_rf

# Write the predicted data to a file
df = rbind(df, unknown)
df$id = rownames(df)
write.table(df, "predictions.tsv", quote=F, row.names=F, col.names=T, sep="\t")

