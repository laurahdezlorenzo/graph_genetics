setwd("~/Documents/CODE/APP_genetics/ppa-graphs")

# DisGeNET
library(disgenet2r)

disgenet_api_key <- get_disgenet_api_key(
        email = "laurahl@ucm.es", 
        password = "GraphGene21" )

Sys.setenv(DISGENET_API_KEY= disgenet_api_key)

# Gene-disease associations

# CUI codes for diseases of interest
#   C0002395 - Alzheimers Disease
#   C0338451 - Frontotemporal Dementia
#   C0030567 - Parkinson Disease
#   C0949664 - Taupathies
#   C2718017 - TDP-43 Proteinopathies
#   C0497327 - Dementia
#   C0282513 - Primary Progressive Aphasia (disorder)
#   C0002736 - Amyotrophic Lateral Sclerosis

diseases <- c("C0002395", "C0338451", "C0030567", "C0949664", "C0497327", "C0002736")

# diseases <- c("C0002395")

data <- disease2gene(disease = diseases,
                     database = "CURATED")#,score = c(0.4, 1))


pdf(file = "figures/DisGeNET/gene_disease_network.pdf", width = 10, height = 10)
plot(data, class = "Network", prop = 10)
dev.off()

pdf(file = "figures/DisGeNET/heatmap.pdf", width = 10, height = 10)
plot(data, class="Heatmap", limit =30, cutoff=0.2)
dev.off()

pdf(file = "figures/DisGeNET/protein_class.pdf", width = 10, height = 10)
plot(data, class="ProteinClass")
dev.off()

# Obtain data frame with the results of the query
results <- extract(data)
# write.table(results,
#             file = 'data/AD_GDAs.tsv',
#             quote = FALSE,
#             sep = '\t',
#             row.names = FALSE)

