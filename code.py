import json

import numpy as np 
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from math import ceil
import random 
from itertools import combinations

#put all data in a list so that index order is fixed          
def toList(data):
    temp=list(data.values())
    fin=[]
    for i in temp:
        for j in i:
            fin.append(j)
    return fin

def getTitlelist(productList):
     #enter dict to get the titles
  empty=[]
  normal=[]
  for item in productList:
      temp= re.sub('Inch|inches|"|-inch|inch|in|inch|-Inch| Inch', 'inch ', item.get("title"))
      temp = re.sub('Hertz|hertz|Hz|HZ|hz|-hz|hz', 'hz', temp)
      temp = re.sub('Diagonal|Diagonal Size|diagonal|diag.|diagonally' ,'Diag.', temp)

      temp = temp.replace('(', ' ')
      temp = temp.replace("'", 'inch') 
      temp = temp.replace('"', 'inch') 


      temp = temp.replace(')', ' ')
      temp = temp.replace('-', ' ')
      temp = temp.replace('/', ' ')
      temp = temp.replace(',', '')
      temp=temp.upper()
      normal.append(temp)

      
      empty.append(list(filter(None, temp.split(" "))) )
  return [empty,normal]

def getSubstring(titles):
    model_words= set()
    
    for product in titles:
        for item in product:
            model_words.add(item)
       
    return  sorted(model_words)
def sparseMatrix(substring,title):
    matrix=[]
    for item in title:
        vector=[]
       
        for x in item:
            
             vector.append(substring.index(x))
           
        
        matrix.append(vector)
    return matrix

def getSignatureMatrix(substring,sparse,numperm):
    matrixPerm=[]
    sign=[]
    perm=list(range(len(substring)))
    for i in range(numperm):
        
        matrixPerm.append(random.sample(perm, len(substring)))
    for item in sparse:
        tempSign=[]
        for row in matrixPerm:
            i=0
            for num in row:
                if num in item:
                    tempSign.append(i)
                    break
                i+=1
                
                
        sign.append(tempSign)
    return sign


def chunk_into_n(lst, n):
  size = ceil(len(lst) / n)
  return list(
    map(lambda x: lst[x * size:x * size + size],
    list(range(n)))
  )

def getBand(signature, numBands):
    temp=[]
    for item in signature:
        temp.append(chunk_into_n(item,numBands))
    return temp
def convert(list):
    # Converting integer list to string list
    s = [str(i) for i in list]
     
    # Join list items using join()
    res = int("".join(s))
     
    return(res)
def intoBucket(bands):
    bucket=[]
    for i in range(len(bands[0])):
        temp= [ [] for _ in range(1_000_005) ]
        for item in range(len(bands)):
            index=convert(bands[item][i])
            
            temp[index % 1_000_003].append(item)
        list2 = [x for x in temp if x != []]
       
        bucket.append(list2)
    return bucket
def extractPairs(buckets):
    temp=[]
    for row in buckets:
        for pair in row:
            if(len(pair)>1):
                temp.append(pair)
    return temp 
def sortCombinations(comb):
    sor=[]
    temp= list(combinations(comb, 2))
    for i in temp:
        sor.append(tuple(sorted(i)))
    return sor
    
    
def sortPairs(pair, brand, modelID):
    #check if brands are the same otherwise directly discard pair
  
    matrix = [[0]*len(brand) for _ in range(len(brand))]
    comb=[]
    for  item in pair:     
     comb.extend(sortCombinations(item))
    comb = list(dict.fromkeys(comb))

    for i in comb:
  
                   if(brand[i[0]]== brand[i[1]]) and ((len(modelID[i[0]])>0)  and (modelID[i[0]] == modelID[i[1]])) :
                        matrix[i[0]][i[1]]=1
                        matrix[i[1]][i[0]]=1
                   
                
    return matrix,len(comb)

def jaccard(l1,l2):
   union= list(set(l1) | set(l2))
   intersect =list(set(l1) & set(l2))
   temp= len(intersect)/len(union)
   return temp    
 
def combinePairs(matrix1,matrix2):
     temp=[[0]*len(matrix1) for _ in range(len(matrix1))]
     
     for i in range(len(matrix1)):
            for j in range(i): 
                if (matrix1[i][j]==1) or (matrix2[i][j]==1):
                    temp[i][j]=1
                    temp[j][i]=1    
     return temp



    
def similiarity(matrix, title, threshold, sparse ):  
    temp=[[0]*len(title) for _ in range(len(title))]
    for i in range(len(title)):
           for j in range(i):
              if matrix[i][j]==1:    
                        if (jaccard(sparse[i], sparse[j])>threshold):
                                         
                                   temp[i][j]=1
                                
                                   temp[j][i]=1
        

    return temp

def getBrand(productList):
  #enter dict to get the titles
  brands = ['Panasonic', 'Samsung', 'Sharp', 'Coby', 'LG', 'Sony',
        'Vizio', 'Dynex', 'Toshiba', 'HP', 'Supersonic', 'Elo',
        'Proscan', 'Westinghouse', 'SunBriteTV', 'Insignia', 'Haier',
        'Pyle', 'RCA', 'Hisense', 'Hannspree', 'ViewSonic', 'TCL',
        'Contec', 'NEC', 'Naxa', 'Elite', 'Venturer', 'Philips',
        'Open Box', 'Seiki', 'GPX', 'Magnavox', 'Hello Kitty', 'Naxa', 'Sanyo',
        'Sansui', 'Avue', 'JVC', 'Optoma', 'Sceptre', 'Mitsubishi', 'CurtisYoung', 'Compaq',
        'UpStar', 'Azend', 'Contex', 'Affinity', 'Hiteker', 'Epson', 'Viore', 'VIZIO','SIGMAC', 'Craig','ProScan', 'Apple']

  empty=[]
  for item in productList:
      temp=item.get("featuresMap").get("Brand")
      if(temp is not None):          
          empty.append(temp.upper())
      else:
          for brand in brands:
              if brand in item.get("title"):
                 temp=brand                 
                 empty.append(brand.upper())
                 break
     
  
  return empty
 
def getRealDuplicates(productList):
    empty=[[0]*len(productList) for _ in range(len(productList))]
    modelID=[]
    for product in productList:
         modelID.append(product.get("modelID"))

    for i in range(len(productList)):
        for j in range(len(productList)):
            if (modelID[i] == modelID[j]) and (i!=j):
                empty[i][j]=1
                empty[j][i]=1
            
    return empty 
def checkmodelID(productList):
    modelID=[]
    for product in productList:
        temp=product.get("modelID").replace('/','')
        temp=temp.replace('-','')
        modelID.append(temp)
    return modelID
def getmodelIDfromTitle(productList):
    empty=[]
    for item in productList:
    
         z = re.finditer(r'[a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*',  item.get("title"))
         temp=[]
         for i in z:
             temp.append(i.group())
             
         word= max(temp, key=len).replace('1080p', '')
         word=word.replace("720p", '')
         word=word.replace("1080P", '')  
         word=word.replace("600Hz", '')
         empty.append(word)
    
    return empty
def measures(scores, num_comp):
    precision =  scores[0] / ( scores[0] +     scores[1])
    recall =  scores[0] / ( scores[0] +     scores[2])
    
    if (recall + precision == 0):
        return 0
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    pq=( scores[0] +  scores[1])/num_comp
    pc=( scores[0] +     scores[1])/scores[3]
    print('pl',pq,pc)
    f1_star=2 * (pq * pc) / (pq + pc)
    return [pq,pc,f1_score,f1_star]

def accuracy(realDuplicates, predictedDuplicates):
    truePositives = 0
    falsePositives = 0
    falseNegatives = 0
    num_duplicates=0
    row1 = len(realDuplicates) 
    
    
    for i in range(row1):
        for j in range(i):
            if (realDuplicates[i][j] == predictedDuplicates[i][j]) and (realDuplicates[i][j] == 1):
                truePositives = truePositives + 1
            if (realDuplicates[i][j] < predictedDuplicates[i][j]):
                falsePositives = falsePositives + 1
            if (realDuplicates[i][j] > predictedDuplicates[i][j]):
                falseNegatives = falseNegatives + 1
            if (realDuplicates[i][j] ==1):
                  num_duplicates = num_duplicates + 1
    return[truePositives, falsePositives, falseNegatives, num_duplicates]
def countMat(matrix):
    temp=0
    for item in matrix:
        for i in item:
            temp=temp+i
    return temp
def  test(dataList):    
    n=len(dataList)
    train=[]
    test=dataList.copy()

    index=list(set(np.random.choice(n, n, replace=True)))
    for i in index:
        train.append(test[i])
    for i in sorted(index, reverse=True):
        del test[i]  
    return[train,test,index]

def factors(x):
   temp=[]
   for i in range(3, x + 1):
       if x % i == 0:
           temp.append(i) 
   return temp


 
















file = "TVs-all-merged.json"
data = json.load(open(file))  
            



#%%

num_bootstrap=5
num_perm=100
split = 1624 #63%
jaccard_threshold= [  list(range(6))[i]*.1 for i in range(6)]

full_productList=toList(data)

random.seed(123)
test_f1=[]
test_jaccard=[]
for i in range(num_bootstrap):
    print(f"\n==============================\nBOOTSTRAP {i+1} \n==============================") 
      
    split_list= test(full_productList)
    train_data=split_list[0]
    test_data=split_list[1]
    train_brandList=getBrand(  train_data)
    train_modelID=getmodelIDfromTitle(  train_data)
    
      
    train_titleList,  train_titleString=getTitlelist(  train_data)
    train_substring=getSubstring(  train_titleList)
      
    train_sparseMatrix= sparseMatrix(  train_substring,  train_titleList)
    train_REALPAIRS=getRealDuplicates(  train_data)
    f1=[]
    for jaccard_value in  jaccard_threshold:
          train_signature_matrix=getSignatureMatrix(  train_substring,  train_sparseMatrix,100)
          train_bands=getBand(  train_signature_matrix,20)
          train_buckets= intoBucket(  train_bands)
          train_extractedPairs=extractPairs(  train_buckets)
          train_candidatePairs,   train_numComparison= sortPairs(  train_extractedPairs,  train_brandList,   train_modelID )
    
          train_pairs= similiarity(  train_candidatePairs,   train_sparseMatrix,  jaccard_value,   train_sparseMatrix)
          train_accuracy=accuracy(  train_REALPAIRS,  train_pairs)
        
          print(jaccard_value,  train_accuracy)
        
        
          f1.append(measures(  train_accuracy,train_numComparison))
        
    
    
    best_jaccard=jaccard_threshold[ f1.index(max( f1))]
    test_jaccard.append(best_jaccard)
      
    test_brandList=getBrand(test_data)
    test_modelID=getmodelIDfromTitle(test_data)
       
    
    test_titleList,test_titleString=getTitlelist(test_data)
    test_substring = getSubstring(test_titleList)
    test_sparseMatrix = sparseMatrix(test_substring,test_titleList)
    test_signature_matrix = getSignatureMatrix(test_substring,test_sparseMatrix,10)
    test_bands = getBand(test_signature_matrix,2)
    test_buckets = intoBucket(test_bands)
    test_extractedPairs=extractPairs(test_buckets)
    test_candidatePairs, test_numComparison= sortPairs(test_extractedPairs,test_brandList, test_modelID )
    
    
    test_pairs= similiarity(test_candidatePairs,test_titleString, best_jaccard,test_sparseMatrix)
       
    test_REALPAIRS=getRealDuplicates(test_data)
    test_accuracy=accuracy(test_REALPAIRS, test_pairs)
    



    print(test_accuracy)
    
    test_f1.append(measures(test_accuracy,test_numComparison))
    print(measures(test_accuracy,test_numComparison))





#%%
average_jaccard=0.1
graph_accuracy=[]
num_perm=100
num_bands=factors(num_perm) 

full_productList=toList(data) 
brandList=getBrand(full_productList)

modelID=getmodelIDfromTitle( full_productList)
titleList, titleString=getTitlelist( full_productList)
substring = getSubstring( titleList)
sparseMatrix1 = sparseMatrix( substring, titleList)

REALPAIRS=getRealDuplicates( full_productList)
for num in num_bands:
     signature_matrix = getSignatureMatrix( substring, sparseMatrix1,num_perm)
     bands = getBand( signature_matrix,num)
     buckets = intoBucket( bands)
     extractedPairs=extractPairs(buckets)
     candidatePairs,n = sortPairs(extractedPairs, brandList,  modelID )
    
    
     pairs= similiarity( candidatePairs,titleList, average_jaccard,sparseMatrix1)
     print( accuracy( REALPAIRS,  pairs))

     graph_accuracy.append([n/1317876,measures(accuracy( REALPAIRS,  pairs),n),accuracy( REALPAIRS,  pairs),n])
     print(measures(accuracy( REALPAIRS,  pairs),n))
