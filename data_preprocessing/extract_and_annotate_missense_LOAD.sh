# #!/bin/bash

graphtype='ND_PPI'

# Activate bioinformatics anaconda environment!
source /home/laura/anaconda3/etc/profile.d/conda.sh
conda activate bioinformatics

<<<<<<< HEAD
cd /media/laura/DATA/LOAD-WGS/files
=======
cd /media/laura/MyBook/TGenII-LOAD-GWAS/files
>>>>>>> 9c4be7d47d62344bf33b292471462c38f1ccc05f

################################################################################
# Extract variants in genes of interest
################################################################################

# echo "Extracting variants in genes of interest..."
# echo `date`

# mkdir selected_genes

<<<<<<< HEAD
# # Extract variants in gene locations
# echo all
# vcftools --vcf /media/laura/DATA/LOAD-WGS/files/LOAD_cohort.vcf \
#     --bed /home/laura/Documents/CODE/APP_genetics/ppa-graphs/data/BED_files/$graphtype/$graphtype.genes.bed \
#     --recode \
#     --out selected_genes/LOAD_cohort.vcf
# echo
=======
# Extract variants in gene locations
echo all
vcftools --gzvcf LOAD_cohort.vcf.gz \
    --bed /home/laura/Documents/CODE/APP_genetics/ppa-graphs/data/BED_files/$graphtype/$graphtype.genes.bed \
    --recode \
    --out selected_genes/all_chr.vcf
echo
>>>>>>> 9c4be7d47d62344bf33b292471462c38f1ccc05f

#Change chromosome notation
bcftools annotate --rename-chrs chr_name_conv.txt selected_genes/all_chr.vcf.recode.vcf > selected_genes/rename.all_chr.vcf

<<<<<<< HEAD
# # Compress and create index of new VCF file
# bgzip -c LOAD_cohort.vcf > LOAD_cohort.vcf.gz
# tabix -p vcf LOAD_cohort.vcf.gz
=======
cd selected_genes
>>>>>>> 9c4be7d47d62344bf33b292471462c38f1ccc05f

# echo "done!"
# echo `date`
# echo

# ################################################################################
# # Annotate VCF with Ensembl-VEP
# ################################################################################

# conda deactivate # VEP does not run if conda env is activated

# echo "Annotating VCF with Ensembl-VEP..."
# echo `date`

<<<<<<< HEAD
# ./vep --assembly GRCh37 \
#      --cache \
#      --check_existing \
#      --coding_only \
#      --everything \
#      --fork 6 \
#      --format vcf \
#      --input_file /media/laura/DATA/LOAD-WGS/files/rename.LOAD_cohort.vcf \
#      --merged \
#      --output_file /media/laura/DATA/LOAD-WGS/files/vep.load.vcf \
#      --species homo_sapiens \
#      --vcf
=======
vep --assembly GRCh37 \
     --cache \
     --check_existing \
     --coding_only \
     --everything \
     --fork 6 \
     --format vcf \
     --input_file rename.all_chr.vcf \
     --merged \
     --output_file vep.load.vcf \
     --species homo_sapiens \
     --vcf
>>>>>>> 9c4be7d47d62344bf33b292471462c38f1ccc05f

# echo "done!"
# echo `date`
# echo

################################################################################
# Split multiallelic sites
################################################################################

# # Activate bioinformatics anaconda environment to use bcftools!
# source /home/laura/anaconda3/etc/profile.d/conda.sh
# conda activate bioinformatics

# bcftools norm -Ov -m-any vep.load.vcf > split.vep.load.vcf ;

################################################################################
# Extract non-synonymus SNVs
################################################################################

echo "Extracting non-synonymous / missense SNVs..."
echo `date`

bcftools +split-vep split.vep.load.vcf \
	-s primary:missense \
	-f '%CHROM\t%POS\t%ID\t%REF\t%ALT\t%CSQ[\t%GT]\n' \
	-A tab \
	-i 'VARIANT_CLASS="SNV"' \
	> snvs.canonical-missense.load.tsv

bcftools +split-vep split.vep.load.vcf \
	-s worst:missense \
	-f '%CHROM\t%POS\t%ID\t%REF\t%ALT\t%CSQ[\t%GT]\n' \
	-A tab \
	-i 'VARIANT_CLASS="SNV"' \
	> snvs.worst-missense.load.tsv

bcftools +split-vep split.vep.load.vcf \
    -s worst \
    -f '%CHROM\t%POS\t%ID\t%REF\t%ALT\t%CSQ[\t%GT]\n' \
    -A tab \
    > snvs.worst.load.tsv

echo "done!"
echo `date`
echo

conda deactivate