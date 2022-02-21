# #!/bin/bash

graphtype='ND_Multi'

# Activate bioinformatics anaconda environment!
source /home/laura/anaconda3/etc/profile.d/conda.sh
conda activate bioinformatics

cd /media/laura/MyBook/ADNI-WGS

################################################################################
# Extract variants in genes of interest
################################################################################

echo "Extracting variants in genes of interest..."
echo `date`

mkdir selected_genes

# Extract variants in gene locations
for i in {1..22}; do 

    echo chr$i
	vcftools --gzvcf ADNI.808_indiv.minGQ_21.pass.ADNI_ID.chr$i.vcf.gz \
        --bed /home/laura/Documents/CODE/APP_genetics/ppa-graphs/data/BED_files/$graphtype/chr$i.genes.bed \
        --recode \
        --out selected_genes/chr$i.vcf
    echo

done

echo chrX
	vcftools --gzvcf ADNI.808_indiv.minGQ_21.pass.ADNI_ID.chr23.vcf.gz \
        --bed /home/laura/Documents/CODE/APP_genetics/ppa-graphs/data/BED_files/$graphtype/chrX.genes.bed \
        --recode \
        --out selected_genes/chrX.vcf
echo

cd selected_genes

picard GatherVcfs \
    INPUT=chr1.vcf.recode.vcf  \
    INPUT=chr2.vcf.recode.vcf  \
    INPUT=chr3.vcf.recode.vcf  \
    INPUT=chr4.vcf.recode.vcf  \
    INPUT=chr5.vcf.recode.vcf  \
    INPUT=chr6.vcf.recode.vcf  \
    INPUT=chr7.vcf.recode.vcf  \
    INPUT=chr8.vcf.recode.vcf  \
    INPUT=chr9.vcf.recode.vcf  \
    INPUT=chr10.vcf.recode.vcf  \
    INPUT=chr11.vcf.recode.vcf  \
    INPUT=chr12.vcf.recode.vcf  \
    INPUT=chr13.vcf.recode.vcf  \
    INPUT=chr14.vcf.recode.vcf  \
    INPUT=chr15.vcf.recode.vcf  \
    INPUT=chr16.vcf.recode.vcf  \
    INPUT=chr17.vcf.recode.vcf  \
    INPUT=chr18.vcf.recode.vcf  \
    INPUT=chr19.vcf.recode.vcf  \
    INPUT=chr20.vcf.recode.vcf  \
    INPUT=chr21.vcf.recode.vcf  \
    INPUT=chr22.vcf.recode.vcf  \
    INPUT=chrX.vcf.recode.vcf  \
    OUTPUT=all.808ADNI.vcf

# Compress and create index of new VCF file
bgzip -c all.808ADNI.vcf > all.808ADNI.vcf.gz
tabix -p vcf all.808ADNI.vcf.gz

echo "done!"
echo `date`
echo

################################################################################
# Annotate VCF with Ensembl-VEP
################################################################################

conda deactivate # VEP does not run if conda env is activated

echo "Annotating VCF with Ensembl-VEP..."
echo `date`

vep --assembly GRCh37 \
     --cache \
     --check_existing \
     --coding_only \
     --everything \
     --fork 6 \
     --format vcf \
     --input_file all.808ADNI.vcf \
     --merged \
     --output_file vep.adni.vcf \
     --species homo_sapiens \
     --vcf

echo "done!"
echo `date`
echo

################################################################################
# Split multiallelic sites
################################################################################

# Activate bioinformatics anaconda environment to use bcftools!
source /home/laura/anaconda3/etc/profile.d/conda.sh
conda activate bioinformatics

bcftools norm -Ov -m-any vep.adni.vcf > split.vep.adni.vcf ;

################################################################################
# Extract non-synonymus SNVs
################################################################################

echo "Extracting non-synonymous / missense SNVs..."
echo `date`

bcftools +split-vep split.vep.adni.vcf \
	-s primary:missense \
	-f '%CHROM\t%POS\t%ID\t%REF\t%ALT\t%CSQ[\t%GT]\n' \
	-A tab \
	-i 'VARIANT_CLASS="SNV"' \
	> '$graphtype'_canonical_missense.tsv

bcftools +split-vep split.vep.adni.vcf \
	-s worst:missense \
	-f '%CHROM\t%POS\t%ID\t%REF\t%ALT\t%CSQ[\t%GT]\n' \
	-A tab \
	-i 'VARIANT_CLASS="SNV"' \
	> '$graphtype'_worst_missense.tsv

bcftools +split-vep split.vep.adni.vcf \
    -s worst \
    -f '%CHROM\t%POS\t%ID\t%REF\t%ALT\t%CSQ[\t%GT]\n' \
    -A tab \
    > '$graphtype'_variants.tsv

echo "done!"
echo `date`
echo

conda deactivate