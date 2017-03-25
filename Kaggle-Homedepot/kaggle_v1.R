#library(readr)
library(SnowballC)
#Read the input csv files
cat("Reading Data\n")
train <- read.csv('input/train.csv')
test <- read.csv('input/test.csv')
attributes <- read.csv('input/attributes.csv')
pd_desc <- read.csv('input/product_descriptions.csv')

#Count of Training data
train_count <- dim(train)

##Get the Brand Values and process the data 
product_brand <- subset(attributes, name=="MFG Brand Name", select=c(product_uid, value))
colnames(product_brand) <- c("product_uid", "brand_name")
product_brand[product_brand=="N/A" | product_brand==".N/A" | product_brand=="n/a" ]<-NA

#Combine product descriptions, brand name into product details.
product_details <- merge(pd_desc,product_brand,by="product_uid")
product_details[is.na(product_details)] <- ""

y_train <- train$relevance
train <- train[,train$relevance,drop=TRUE]

all_details <- rbind(train)
