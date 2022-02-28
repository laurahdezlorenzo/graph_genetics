#
# Make BED files from genomic coordinates of the genes of interest.
#

infile <- 'data/AD_STRING_PPI_intprots.txt'

# BioMart
library(biomaRt)

# Create a query in BioMart with the genes of interest
hsapiens37_mart <- useEnsembl('ensembl', GRCh = 37,
                              dataset = 'hsapiens_gene_ensembl')

# Attributes - output
bed_attributes <- c('chromosome_name', 'start_position', 'end_position')

# Filters - inputs for a biomaRt query
gene_filter <- c('external_gene_name')
gene_names <- scan(infile, what = '', sep = '\n')

# getBM - main biomaRt query function
bed_df <- getBM(attributes = bed_attributes, filters = gene_filter,
                  values = gene_names, mart = hsapiens37_mart)
outfile_all <- paste0('data/BED_files/AD_PPI.genes.bed')
write.table(bed_df, sep = '\t', col.names = FALSE, row.names = FALSE,
            quote = FALSE, file = outfile_all)

# Split coordinates by chromosome
sptdf <- split(bed_df, bed_df$chromosome_name)
lapply(sptdf, function(DF){
  
  outfile <- as.character(unique(DF[['chromosome_name']]))
  outfile <- paste0('data/BED_files/AD_PPI/chr', outfile, '.genes.bed')

  write.table(DF, sep = '\t', col.names = FALSE, row.names = FALSE,
              quote = FALSE, file = outfile)
  })

# For outfile_all, then change chromosome name "X" to "23" because it is the
# chromosome numbering for Plink. We're using VCFs coming from Plink files
# from the LOAD dataset.