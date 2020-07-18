###k-means clustering

#recognize the arguments inputed
args <- commandArgs(trailingOnly=TRUE)

#read dataset
data <- read.table(args[1])


#transpose columns and rows
data <- t(data)


#the number of clusters
k=7

#necessary functions

#compute the cluster controid
compute_centroid <- function(data_grouped){
  centroid <- NULL
  for (i in 1:k){
    #for each cluster, extract the data belonging to this cluster
    clusteri <- data_grouped[data_grouped[,'group'] == i,]
    #calculate the mean data to get the data of centroid
    if (is.matrix(clusteri)){
      centroidi <- t(colMeans(clusteri))[-1]
    } else {
      centroidi <- clusteri
    }
    #combind all of the centroids together
    centroid <- rbind(centroid,centroidi)
  }
  return(centroid)
}


#define cluster
define_clu <- function(datapoint,cen){
  c <- NULL
  #calculate the distance between a specific datapoint and the centroids
  for (i in 1:nrow(cen)){
    c <- append(c,sqrt(sum((cen[i,]-datapoint)^2)))
  }
  #get the position of the centroid with minimum distance from a datapoint
  c_posi <- which.min(c)
  return(c_posi)
}

#fix centroid, so that this function can be used in apply function for datapoints
define_clu_2 <- function(datapoint){
  cen <- centroid
  define_clu(datapoint,cen)
}




#set a null dataframe to store the results from 25 random starts
final_output <- NULL

#25 random starts
itera=0
while(itera<25) {
  #check if there is situations that no data points assigned to a cluster
  error = "off"
  #random start
  l <- 0
  
  #check if there are 7 type of numbers
  while (l != 7){
  group <- sample(1:k,size=nrow(data),replace = TRUE)
  l <- length(unique(group))}

  #combine the data and its corresponding cluster
  data_grouped <- cbind(group,data)

  #set converge starting from a non-zero number
  converge <- 1
  #iterate until the group will not change
  #the difference between previous classification and new classification is zero
  while(converge != 0){
    #compute the centroid
    centroid <- compute_centroid(data_grouped)
    #get the new classification
    cluster_new <- apply(data, 1, define_clu_2)
    
    #when the situation that no data point assigned to a cluster happens, set variable error as "on"
    if(length(unique(cluster_new)) != 7){error = "on"}
    
    #calculate the difference between previous classification and new classification
    converge <- sum((group - cluster_new)^2)
    #assign the new cluster to variable 'group' and begin next loop 
    group <- cluster_new
    #get new grouped data again
    data_grouped <- cbind(group,data)
  }

  #if error is on, skip this loop and start again
  if(error == "on") next
  
  #add the iteration times
  itera = itera + 1
  
  #calculate the objective function
  difference <- NULL
  for (i in 1:k){
    #for each datapoint, calculate the Euclidean Distance from the centroid of its cluster
    groupi <- data_grouped[data_grouped[,'group'] == i,]
    if (is.matrix(groupi)){
      for (n in 1:nrow(groupi)){
        diff <- groupi[n,2:ncol(groupi)] - centroid[i,]
        diff <- sqrt(sum(diff^2))
        difference <- append(difference,diff)
      }
    } else {
      diff <- groupi[2:length(groupi)] - centroid[i,]
      diff <- sqrt(sum(diff^2))
      difference <- append(difference,diff)
    }
    
    #sum their together
    difference <- sum(difference)
  }    
  #divide the sum by the number of datapoints 
  objective <- difference/nrow(data)
  
  #add the cluster assignment and the objective function
  group <- append(group,objective)
  
  #combine all of the outputs together
  final_output <- cbind(final_output,group)

}


#find the minimun objective funtion
min <- which.min(final_output[nrow(data)+1,])

#output the cluster assignment with the minimum objective function
output <- final_output[,min]

#write the output file
write.table(output, file=args[2], row.names = FALSE, col.names = FALSE)








###global k-means clustering

#recognize the arguments inputed
args <- commandArgs(trailingOnly=TRUE)

#read dataset
data <- read.table(args[1])
#data <- read.table("input_data_example2.txt")

#transpose columns and rows
data <- t(data)


#when k=1, the centroid is the average of all the data points
centroid1 <- t(colMeans(data))

#compute the cluster controid
compute_centroid <- function(data_grouped){
  centroid <- NULL
  for (i in 1:k){
    #for each cluster, extract the data belonging to this cluster
    clusteri <- data_grouped[data_grouped[,'group'] == i,]
    #calculate the mean data to get the data of centroid
    if (is.matrix(clusteri)){
      centroidi <- t(colMeans(clusteri))[-1]
    } else {
      centroidi <- clusteri[-1]
    }
    #combind all of the centroids together
    centroid <- rbind(centroid,centroidi)
  }
  return(centroid)
}


#define cluster
define_clu <- function(datapoint,cen){
  c <- NULL
  #calculate the distance between a specific datapoint and the centroids
  for (i in 1:nrow(cen)){
    c <- append(c,sqrt(sum((cen[i,]-datapoint)^2)))
  }
  #get the position of the centroid with minimum distance from a datapoint
  c_posi <- which.min(c)
  return(c_posi)
}

#fix centroid, so that this function can be used in apply function for datapoints
define_clu_2 <- function(datapoint){
  cen <- centroid
  define_clu(datapoint,cen)
}




objective_function <- function(data_grouped,centroid){
#calculate the objective function
difference <- NULL
for (i in 1:k){
  #for each datapoint, calculate the Euclidean Distance from the centroid of its cluster
  groupi <- data_grouped[data_grouped[,'group'] == i,]
  if (is.matrix(groupi)){
    groupi <-  groupi[,-1]
    for (n in 1:nrow(groupi)){
      diff <- groupi[n,] - centroid[i,]
      diff <- sqrt(sum(diff^2))
      difference <- append(difference,diff)
    }
  } else {
    diff <- groupi[2:length(groupi)] - centroid[i,]
    diff <- sqrt(sum(diff^2))
    difference <- append(difference,diff)
  }
}
#divide the sum by the number of datapoints 
objective <- sum(difference)/nrow(data)

return(objective)
}

#store the centroids
center<-c()
k=2
while(k <= 4){
  print(paste("Calculating the", k ,"metroid"))
  
  final_output <- NULL
  for (i in 1:nrow(data)){
    #add each data point to the centroid
    centroid <- rbind(centroid1,data[i,])
    #assign the culster to data points
    group <- apply(data, 1, define_clu_2)
    data_grouped <- cbind(group,data)
  
    #get the best assignment
    converge = 1
    while(converge != 0){
      centroid <- compute_centroid(data_grouped)
      #get the new classification
      cluster_new <- apply(data, 1, define_clu_2)
      #calculate the difference between previous classification and new classification
      converge <- sum((group - cluster_new)^2)
      #assign the new cluster to variable 'group' and begin next loop 
      group <- cluster_new
      #get new grouped data again
      data_grouped <- cbind(group,data)
    }
    
    #calculate the minimum objective function
    objective <- objective_function(data_grouped,centroid)
    
    
    #add the objective function to the clusters
    group_combine <- append(group,objective)
    
    #combine all of the results together
    final_output <- cbind(final_output,group_combine)
    
    center[[i]] <- centroid
  } 
  
  
  #get the cluster assignment which results in the minimum objective function
  posi <- unname(which.min(final_output[nrow(data)+1,]))
  
  #get the best cluster assignment
  final_clu <- final_output[,posi]
  
  #get the final grouped data
  data_for_final <- cbind(final_clu[-1],data)
  colnames(data_for_final)[1] <- "group"
  
  #get the best centroids
  centroid1 <- center[[posi]]

  k = k+1
}


#output <- append(clu,objective_new)
output <- unname(final_clu)

#write the output file
write.table(output, file=args[2], row.names = FALSE, col.names = FALSE)




