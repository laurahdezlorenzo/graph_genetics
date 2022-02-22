# DisGeNET
library(disgenet2r)

disgenet_api_key <- get_disgenet_api_key(
        email = "laurahl@ucm.es", 
        password = "GraphGene21" )

Sys.setenv(DISGENET_API_KEY= disgenet_api_key)

# CUI codes for diseases of interest
#   C0002395 - Alzheimers Disease
#   C0338451 - Frontotemporal Dementia
#   C0030567 - Parkinson Disease
#   C0949664 - Taupathies
#   C0497327 - Dementia
#   C0002736 - Amyotrophic Lateral Sclerosis

# Gene-disease associations
# NDD gene set - genes related with neurodegenerative diseases
ndd <- c("C0002395", "C0338451", "C0030567", "C0949664", "C0497327", "C0002736")
data_ndd <- disease2gene(disease = ndd, database = "CURATED")
results_ndd <- extract(data_ndd)
write.table(results_ndd,
            file = 'data/ND_GDAs.tsv',
            quote = FALSE,
            sep = '\t',
            row.names = FALSE)

# AD gene set - genes related with Alzheimer's Disease
ad  <- c("C0002395")
data_ad  <- disease2gene(disease = ad, database = "CURATED")
results_ad <- extract(data_ad)
write.table(results_ad,
            file = 'data/AD_GDAs.tsv',
            quote = FALSE,
            sep = '\t',
            row.names = FALSE)
