#! /usr/bin/Rscript

# DisGeNET - needs a personal account (email+password)
library(disgenet2r)

disgenet_api_key <- get_disgenet_api_key(
        email = "laurahl@ucm.es", 
        password = "GraphGene21" ) 

Sys.setenv(DISGENET_API_KEY= disgenet_api_key)

# CUI codes for diseases of interest
#   C0002395 - Alzheimers Disease

# Gene-disease associations
# Genes related with Alzheimer's Disease
ad  <- c("C0002395")
data_ad  <- disease2gene(disease = ad, database = "CURATED")
results_ad <- extract(data_ad)
write.table(results_ad,
            file = 'data/AD_GDAs.tsv',
            quote = FALSE,
            sep = '\t',
            row.names = FALSE)


