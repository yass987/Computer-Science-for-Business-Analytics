# Computer Science for Business Analytics
Assignment 
Scalable Product Duplicate Detection

The goal of this project is to find duplicate TVs that are sold on different sites. A dataset of 1624 TVs is used and there are 4 different sites each TV can be sold on. Locality Sensitivity Hashing is performed where information from the title is used. LSH yields candidate pairs which are classified based on TV brand and modelID extracted from the title.(ModelID is not always available) As for final classification, the jaccard similiarity is calculated from the title which needs to be above a certain threshold for the pair outputed as a duplicate. Optimal threshold level is optimized.

The code works sequentially. The first cell contains all the methods and imports the data. The second cell uses bootstrapping to find the optimal threshold value. Lastly, the runs the entire algortihm with the optimal threshold for different number of bands
